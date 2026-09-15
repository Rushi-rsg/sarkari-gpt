# 🏛️ SarkariGPT - AI Assistant for Indian Government Schemes

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Inference-Groq_LPU-orange.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, voice-enabled **Semantic RAG** (Retrieval-Augmented Generation) assistant that helps Indian citizens easily find, understand, and apply for government welfare schemes, loans, subsidies, and scholarships.

---

## 🎯 Why SarkariGPT?

Finding government schemes in India is often confusing due to complex eligibility rules, scattered portal links, and language barriers. 

**SarkariGPT solves this by:**
- 🔍 Checking your **exact personal eligibility** (Age, Income, State, Occupation, Category).
- 🎙️ Letting you **speak your question** in your native language (Hindi, Marathi, English, etc.).
- 📚 Providing **verified, grounded answers** using an offline-ready knowledge base of 25+ flagship schemes.
- ⚡ Running **100% open-source models** on Groq Cloud with zero GPU/hardware requirements.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🧠 **Semantic RAG Engine** | Uses **ChromaDB** with `all-MiniLM-L6-v2` dense embeddings to understand user queries conceptually (not just keyword matching). |
| 🤖 **Cloud Open-Source LLMs** | Powered by **Meta Llama 3.3 70B** and **Llama 3** hosted on Groq Cloud LPU for instant, intelligent answers without heating your computer. |
| 🎙️ **Voice In & Voice Out** | Speak naturally using **Groq Whisper Large v3** speech-to-text; listen to spoken responses with built-in audio synthesis. |
| 🌐 **Multilingual Advisory** | Direct native support for 7 languages: **English, हिन्दी (Hindi), मराठी (Marathi), বাংলা (Bengali), தமிழ் (Tamil), తెలుగు (Telugu), and ગુજરાતી (Gujarati)**. |
| 📊 **Eligibility Scoring (0–100%)** | Evaluates citizen profile against scheme criteria and displays a match score with clear explanations of why you qualify. |
| ⚖️ **Side-by-Side Comparator** | Compare benefits, income caps, documents, and application steps between any two schemes. |
| 📁 **Safe Local Persistence** | Saves citizen profiles and conversation histories safely in human-readable JSON files (no unsafe `.pkl` pickles). |

---

## 🏛️ Supported Flagship Schemes (Pre-Seeded)

Includes 25+ major central & state schemes across key sectors:
- **Agriculture**: PM-KISAN, PM Fasal Bima Yojana (PMFBY), Kisan Credit Card (KCC)
- **Healthcare**: Ayushman Bharat (PM-JAY), PM Janaushadhi Pariyojana (PMBJP)
- **Housing & Sanitation**: PMAY-Urban 2.0, PMAY-Gramin, Swachh Bharat IHHL
- **Business & Loans**: PM Mudra Yojana, PM SVANidhi, Stand-Up India, PMEGP, PM Jan Dhan
- **Social Security & Pension**: Atal Pension Yojana (APY), PM Jeevan Jyoti (PMJJBY), PMSBY, PM-SYM
- **Women & Child Welfare**: Sukanya Samriddhi (SSY), PM Ujjwala 2.0, PM Matru Vandana, Lakhpati Didi
- **Skills & Education**: PM Vishwakarma, PMKVY 4.0, National Apprenticeship (NAPS-2), Post-Matric Scholarships

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10 to 3.12** installed on your system.
- A free **Groq API Key** from [console.groq.com/keys](https://console.groq.com/keys).

### 2. Setup & Installation

Open your terminal or PowerShell:

```powershell
# Navigate to the project directory
cd D:\sarkari-gpt

# Create a virtual environment (Python 3.12 recommended)
py -3.12 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App

```powershell
streamlit run app.py
```

### 4. How to Use
1. Paste your free Groq API key in the sidebar and click **🧪 Test API Key**.
2. Select **Llama 3.3 70B** and your preferred language.
3. Fill out your details under the **👤 Citizen Profile** tab.
4. Ask any question in the **💬 AI Assistant** tab via text or microphone!

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB (Persistent)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM Inference**: Meta Llama 3.3 70B / Llama 3 (via Groq Cloud LPU)
- **Voice Recognition**: Groq Whisper Large v3
- **Audio Output**: gTTS (Google Text-to-Speech)
- **Storage**: JSON (Local, safe, file-based)

---

## 📂 Project Structure

```text
D:\sarkari-gpt\
├── app.py                     # Streamlit frontend & interactive UI
├── rag_engine.py              # ChromaDB vector store & hybrid eligibility matcher
├── ai_services.py             # Groq LLM & Whisper speech integration
├── storage.py                 # Safe JSON profile & history persistence
├── test_rag.py                # Test & validation suite
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── data/
│   ├── schemes_data.json      # 25+ curated government schemes database
│   ├── user_profile.json      # Saved citizen profile
│   └── chat_history.json      # Saved conversation history
└── chroma_db/                 # Persistent ChromaDB vector index files
```

---

## ⚠️ Disclaimer
*SarkariGPT is an AI-powered advisory tool designed for educational and informational purposes. While it references verified government sources, applicants should always confirm final eligibility and procedures on official government portals before applying.*

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
