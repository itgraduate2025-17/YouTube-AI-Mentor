
# ========================
# Imports
# ========================
import os
import tempfile
import yt_dlp
import whisper
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ========================
# Globals
# ========================
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = 384
index = faiss.IndexFlatIP(EMBED_DIM)
vector_texts, vector_meta = [], []

whisper_model = whisper.load_model("base")

#  Use environment variable 
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if not os.environ["GROQ_API_KEY"]:
    raise ValueError("Missing GROQ_API_KEY. Please set it before running.")

GROQ_MODEL = "llama-3.3-70b-versatile"

# ==========================================================
# Helper: Try to fetch YouTube subtitles via yt-dlp
# ==========================================================
def get_youtube_subtitles(url):
    try:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "srt",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("requested_subtitles") or info.get("subtitles") or {}
            if not subs:
                return None

            tempdir = tempfile.mkdtemp()
            ydl_opts2 = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "srt",
                "outtmpl": os.path.join(tempdir, "%(id)s.%(ext)s"),
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts2) as ydl2:
                ydl2.extract_info(url, download=False)

            srt_text = ""
            for fname in os.listdir(tempdir):
                if fname.lower().endswith((".srt", ".vtt")):
                    with open(os.path.join(tempdir, fname), "r", encoding="utf-8", errors="ignore") as fh:
                        srt_text += fh.read() + "\n"

            if not srt_text.strip():
                return None

            lines = []
            for line in srt_text.splitlines():
                line = line.strip()
                if not line or "-->" in line or line.isdigit():
                    continue
                lines.append(line)
            return " ".join(lines)
    except Exception:
        return None

# ==========================================================
# Audio download + Whisper transcription (fallback)
# ==========================================================
def download_audio(url):
    tempdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tempdir, "video.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def transcribe_with_whisper(audio_path):
    result = whisper_model.transcribe(audio_path, fp16=False)
    return result.get("text", "")

# ==========================================================
# Chunking + Embeddings
# ==========================================================
def chunk_text(text, chunk_size_words=300, overlap_words=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap_words
    return chunks

def add_chunks_to_faiss(chunks, meta_base=None):
    if meta_base is None:
        meta_base = {}
    global index, vector_texts, vector_meta
    emb = embedder.encode(chunks, convert_to_numpy=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    index.add(emb.astype("float32"))
    for i, chunk in enumerate(chunks):
        vector_texts.append(chunk)
        meta = dict(meta_base)
        meta["chunk_index"] = len(vector_texts) - 1
        vector_meta.append(meta)
    return True

def query_faiss(question, top_k=4):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = index.search(q_emb.astype("float32"), top_k)
    hits = []
    for idx in I[0]:
        if idx < len(vector_texts):
            hits.append({
                "score": float(D[0][list(I[0]).index(idx)]),
                "text": vector_texts[idx],
                "meta": vector_meta[idx]
            })
    context = "\n\n---\n\n".join([h["text"] for h in hits])
    return context

# ==========================================================
# Groq LLaMA Chat call (SDK-based)
# ==========================================================
def call_groq_api(system_prompt, user_message):
    chat_completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )
    return chat_completion.choices[0].message.content

# ==========================================================
# Main processing function
# ==========================================================
def process_youtube(url):
    transcript = get_youtube_subtitles(url)
    if not transcript:
        audio_path = download_audio(url)
        transcript = transcribe_with_whisper(audio_path)
    if not transcript:
        return " Failed to extract transcript.", None

    chunks = chunk_text(transcript)
    add_chunks_to_faiss(chunks, {"source": url})
    return f"Transcript processed into {len(chunks)} chunks.", "Ready to chat!"

# ==========================================================
# Chat function
# ==========================================================
def chat_with_ai(question):
    if len(vector_texts) == 0:
        return "No transcript indexed yet. Please process a video first."
    context = query_faiss(question)
    system_prompt = (
        "You are an AI Mentor helping the user understand a YouTube video's content. "
        "Use the provided context to explain and elaborate clearly."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        answer = call_groq_api(system_prompt, prompt)
        return answer
    except Exception as e:
        return f"❌ Error contacting Groq API: {e}"

# ==========================================================
# Gradio UI
# ==========================================================
with gr.Blocks() as demo:
    gr.Markdown("## 🎓 YouTube AI Mentor (RAG System using Groq SDK + FAISS)")
    with gr.Tab("Process Video"):
        url_input = gr.Textbox(label="YouTube Video URL")
        process_btn = gr.Button("Process Video")
        process_status = gr.Textbox(label="Processing Status")
        ready_msg = gr.Textbox(label="Next Step", interactive=False)
        process_btn.click(fn=process_youtube, inputs=url_input, outputs=[process_status, ready_msg])

    with gr.Tab("Chat with AI Mentor"):
        question = gr.Textbox(label="Ask your question about the video")
        ask_btn = gr.Button("Ask Mentor")
        answer_box = gr.Textbox(label="AI Mentor Answer", lines=12)  # ⬅ Larger box
        ask_btn.click(fn=chat_with_ai, inputs=question, outputs=answer_box)

demo.launch()
