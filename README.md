# YouTube AI Mentor (RAG System using Groq SDK + FAISS)

This project builds an AI Mentor that summarizes and answers questions about YouTube videos.
It uses RAG (Retrieval-Augmented Generation) with:

* Groq llama-3.3-70b-versatile
* FAISS for vector search
* Whisper for audio transcription
* yt-dlp for YouTube download
* Gradio for the web interface

---

## Features

* Extract subtitles or transcribe audio from YouTube videos
* Chunk text and build FAISS index for fast retrieval
* Query video content using Groq LLM
* Interactive Gradio UI with 2 tabs: Process and Chat
* Ready for **Hugging Face deployment**
* Can be **tested directly on Google Colab**

---

## Setup Instructions

### Google Colab

Run this single command to install all dependencies and test the app in Colab:

```bash
!pip install gradio yt-dlp openai-whisper faiss-cpu sentence-transformers torch groq
```

Then run `app.py` or the notebook cells to start the interface.

### Hugging Face Deployment

This project is ready for deployment on Hugging Face Spaces.

1. Push the repository to GitHub.
2. Connect your GitHub repo to Hugging Face Spaces.
3. Set your **Groq API key** in the config.
4. The Gradio app will launch automatically on Hugging Face.

---

## Project Structure

```
YouTube-AI-Mentor/
│
├── app.py               # Main Gradio app
├── requirements.txt     # Python dependencies
├── utils/               # Helper functions (RAG, transcription, chunking)
├── videos/              # Downloaded YouTube videos
├── embeddings/          # FAISS indices
└── README.md            # Documentation
```

---

## Notes

* Only Hugging Face deployment is supported for live use.
* Google Colab can be used for testing with minimal setup.
* Replace dummy API keys with your own before running.

---

## Developer

Amjad Ali
