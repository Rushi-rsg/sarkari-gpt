# 🏛️ SarkariGPT - Indian Government Schemes AI Assistant

A production-grade **Semantic Retrieval-Augmented Generation (RAG)** assistant designed to help Indian citizens discover, evaluate, and apply for central and state government schemes, subsidies, pensions, and scholarships.

---

## 🌟 Key Upgrades & Architecture

### 1. True Semantic RAG with ChromaDB & Sentence-Transformers
- **Vector Database**: Replaces naive keyword matching with a persistent **ChromaDB** vector database.
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` dense vector embeddings (384 dimensions) for deep conceptual query matching.
- **Contextual Chunking**: Schemes are split into distinct semantic chunks (Overview, Eligibility, Benefits, Application Process & Documents).

### 2. 100% Open-Source LLMs (Zero Local GPU Requirement)
- Powered by open-weights models (**Meta Llama 3.3 70B**, **Llama 3.1 8B**, or **Mixtral 8x7B**).
- Inference runs via **Groq Cloud LPU** endpoints for instantaneous token generation without taxing your local PC hardware.

### 3. Groq Whisper Voice Recognition
- Integrated with Groq's **Whisper Large v3** model for ultra-fast, robust speech-to-text capable of recognizing diverse Indian accents and regional languages.

### 4. Native Multilingual Support
- High-quality, natural responses in **English, Hindi (हिन्दी), Marathi (मराठी), Bengali (বাংলা), Tamil (தமிழ்), Telugu (తెలుగు), and Gujarati (ગુજરાતી)** without fragile web scraping translation APIs.

### 5. Curated Offline-Ready Knowledge Base
- Pre-loaded with 25+ flagship Indian schemes (PM-KISAN, Ayushman Bharat, PMAY Urban/Gramin, PM Mudra, Sukanya Samriddhi, Atal Pension, PM Vishwakarma, etc.).
- Safe, human-readable **JSON persistence** for user profiles and chat histories.

---

## 🚀 Quickstart Guide

### 1. Activate Environment
Open PowerShell and navigate to the project directory:
```powershell
cd D:\sarkari-gpt
.\.venv\Scripts\Activate.ps1
```

### 2. Run Application
```powershell
streamlit run app.py
```

### 3. Configure API Key
1. Get a free API key from [console.groq.com](https://console.groq.com).
2. Enter your key in the left sidebar under **Groq API Setup** (or add to `.streamlit/secrets.toml`).

---

## 📂 Project Structure
```
D:\sarkari-gpt\
├── app.py                     # Streamlit frontend & interactive UI
├── rag_engine.py              # ChromaDB vector store & hybrid eligibility matcher
├── ai_services.py             # Groq LLM & Whisper speech integration
├── storage.py                 # Safe JSON profile & history persistence
├── requirements.txt           # Clean Python dependencies
├── data/
│   ├── schemes_data.json      # 25+ flagship government schemes database
│   ├── user_profile.json      # Citizen profile storage
│   └── chat_history.json      # Saved interaction logs
└── chroma_db/                 # Persistent ChromaDB vector index files
```
