# 🎯 JobFit AI — RAG-Powered Job Application Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that analyzes your resume against any job description to:
- **Identify skill gaps** with an AI fit score
- **Rewrite resume bullets** with ATS-optimized language
- **Generate a tailored cover letter** in seconds

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **LLM Orchestration** | LangChain |
| **Vector Store** | FAISS *or* ChromaDB (switchable) |
| **Embeddings** | OpenAI `text-embedding-ada-002` / HuggingFace `all-MiniLM-L6-v2` |
| **LLM Backend** | OpenAI GPT-4o / GPT-3.5-turbo *or* HuggingFace `flan-t5-xxl` |
| **PDF Parsing** | PyPDF2 |

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/YOUR_USERNAME/job-application-assistant.git
cd job-application-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9+** required. Using a virtual environment is recommended:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## ⚙️ Configuration

All configuration is done **in the sidebar** of the app — no `.env` file needed.

| Setting | Options |
|---------|---------|
| LLM Provider | OpenAI GPT-4o, GPT-3.5-turbo, HuggingFace flan-t5-xxl |
| API Key | Entered securely in the sidebar (never stored) |
| Vector Store | FAISS (in-memory, fast) or ChromaDB (persistent) |
| Analysis Modules | Skill Gap / Resume Rewrites / Cover Letter (toggle each) |

### Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **HuggingFace**: https://huggingface.co/settings/tokens

---

## 🧠 How the RAG Pipeline Works

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT DOCUMENTS                     │
│         Resume (PDF / text) + Job Description           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             TEXT SPLITTING (LangChain)                  │
│    RecursiveCharacterTextSplitter (800 chars, 100 overlap)│
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    EMBEDDINGS                           │
│     OpenAI ada-002  OR  HuggingFace all-MiniLM-L6-v2    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  VECTOR STORE                           │
│              FAISS  OR  ChromaDB                        │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │  Similarity Search   │  ← Task-specific query
         │  Top-k = 6 chunks    │
         └───────────┬──────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               PROMPT ENGINEERING                        │
│   Structured prompts inject retrieved context +         │
│   full resume + full JD into the LLM                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   LLM RESPONSE                          │
│   GPT-4o / GPT-3.5 / flan-t5-xxl                       │
│   → Skill Gap Report / Rewritten Bullets / Cover Letter │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
job-application-assistant/
├── app.py                  # Streamlit UI + orchestration
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── utils/
    ├── __init__.py
    ├── pdf_processor.py    # PyPDF2-based PDF text extractor
    ├── rag_pipeline.py     # LangChain RAG pipeline class
    └── prompts.py          # Prompt templates (skill gap, rewrite, cover letter)
```

---

## 🔬 What This Proves (ATS Keywords)

> RAG · LangChain · Vector Databases · Embeddings · FAISS · ChromaDB · HuggingFace · Prompt Engineering · LLM Orchestration · OpenAI API · Streamlit · PyPDF2

This project demonstrates the **exact AI/ML engineering stack** that 2026 job descriptions require.

---

## 🛠️ Extending the Project

| Feature | How |
|---------|-----|
| Persistent storage | Replace FAISS with ChromaDB + local persistence |
| Multiple resume versions | Add a session-based version manager |
| Interview prep | Add a 4th module with likely interview questions |
| LinkedIn optimizer | Add a 5th module for LinkedIn `About` + headline rewrites |
| Batch processing | Loop over multiple JDs to find the best match |

---

## 📄 License

MIT License — free to use, modify, and distribute.
