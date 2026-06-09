# Capstone 02 – Custom Knowledge Chatbot

RAG (Retrieval-Augmented Generation) chatbot that answers questions from your own Excel and text files.

## Project Structure

```
rag_chatbot/
├── app.py               ← Streamlit web app (run this)
├── rag_pipeline.py      ← Core RAG logic (shared module)
├── RAG_Notebook.ipynb   ← Jupyter notebook (step-by-step explanation)
├── requirements.txt     ← Python dependencies
└── data/
    ├── company_faq.txt  ← Sample text file
    └── store_data.xlsx  ← Sample Excel file (Products + Customers)
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Gemini API key
Go to https://aistudio.google.com → Get API Key (free tier available)

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Use the app
1. Paste your Gemini API key in the sidebar
2. Upload your Excel or text files
3. Click "Index Files"
4. Ask questions in the chat!

## How RAG Works

```
Your Files → Parse → Chunk → TF-IDF Embed → Store
                                                 ↓
User Query → Embed → Similarity Search → Top Chunks
                                                 ↓
                           Top Chunks + Query → LLM → Answer
```

## Upgrade Path (After Capstone)

| Feature | Current | Production |
|---|---|---|
| Embeddings | TF-IDF (exact words) | sentence-transformers (meaning) |
| Vector DB | pickle file | ChromaDB / FAISS |
| LLM | Gemini API | Any LLM |
| Chunking | Word-based | Semantic chunking |

## Tech Stack
- **Python 3.10+**
- **pandas** — Excel/CSV parsing
- **scikit-learn** — TF-IDF embeddings
- **Streamlit** — Web UI
- **Gemini API** — LLM (free tier)
