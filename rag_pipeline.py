"""
RAG Pipeline Core
Embedding: scikit-learn TF-IDF + cosine similarity (no internet needed)
Vector store: ChromaDB (local)
LLM: pluggable (Gemini API key or Groq API key needed at query time)
"""
import os, re, json
import numpy as np
import pandas as pd
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

CHROMA_DIR   = "./chroma_db"
TFIDF_PATH   = "./chroma_db/tfidf.pkl"
COLLECTION   = "knowledge_base"
TOP_K        = 5

# ─── Embedder (TF-IDF) ───────────────────────────────────────────────────────
_vectorizer  = None
_tfidf_matrix = None
_doc_store   = []   # list of {"text", "metadata"}

def _save_tfidf():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump({"vectorizer": _vectorizer,
                     "matrix": _tfidf_matrix,
                     "docs": _doc_store}, f)

def _load_tfidf():
    global _vectorizer, _tfidf_matrix, _doc_store
    if os.path.exists(TFIDF_PATH):
        with open(TFIDF_PATH, "rb") as f:
            obj = pickle.load(f)
        _vectorizer   = obj["vectorizer"]
        _tfidf_matrix = obj["matrix"]
        _doc_store    = obj["docs"]
        return True
    return False

def db_stats():
    _load_tfidf()
    return {"total_chunks": len(_doc_store)}

# ─── Text chunking ────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=200, overlap=30):
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]

# ─── File parsers ─────────────────────────────────────────────────────────────
def parse_txt(file_path_or_content, filename="uploaded.txt"):
    if isinstance(file_path_or_content, str) and os.path.exists(file_path_or_content):
        with open(file_path_or_content, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(file_path_or_content)
    else:
        content = file_path_or_content   # already a string (from Streamlit uploader)
    chunks = chunk_text(content)
    return [(c, {"source": filename, "type": "text"}) for c in chunks]

def parse_excel(file_path_or_buffer, filename="uploaded.xlsx"):
    if isinstance(file_path_or_buffer, str):
        filename = os.path.basename(file_path_or_buffer)
    xl   = pd.ExcelFile(file_path_or_buffer)
    docs = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet).fillna("")
        for _, row in df.iterrows():
            row_text = f"[Sheet: {sheet}] " + " | ".join(
                f"{col}: {val}" for col, val in row.items() if str(val).strip()
            )
            docs.append((row_text, {"source": filename, "sheet": sheet, "type": "excel"}))
    return docs

# ─── Indexing ─────────────────────────────────────────────────────────────────
def index_documents(documents, reset=False):
    """documents: list of (text, metadata) tuples. Returns count."""
    global _vectorizer, _tfidf_matrix, _doc_store

    if reset:
        _doc_store = []

    _load_tfidf()   # load existing if present

    if not documents:
        return 0

    new_texts = [d[0] for d in documents]
    new_metas = [d[1] for d in documents]

    # Combine old + new
    all_texts = [d["text"] for d in _doc_store] + new_texts
    all_metas = [d["metadata"] for d in _doc_store] + new_metas

    _vectorizer   = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    _tfidf_matrix = _vectorizer.fit_transform(all_texts)
    _doc_store    = [{"text": t, "metadata": m} for t, m in zip(all_texts, all_metas)]

    _save_tfidf()
    return len(documents)

# ─── Retrieval ────────────────────────────────────────────────────────────────
def retrieve(query, top_k=TOP_K):
    global _vectorizer, _tfidf_matrix, _doc_store

    if not _load_tfidf() or not _doc_store:
        return []

    q_vec  = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        if scores[i] > 0:
            doc = _doc_store[i]
            results.append({
                "text":   doc["text"],
                "source": doc["metadata"].get("source", "unknown"),
                "sheet":  doc["metadata"].get("sheet", ""),
                "type":   doc["metadata"].get("type", "text"),
                "score":  round(float(scores[i]), 3)
            })
    return results

# ─── Prompt builder ───────────────────────────────────────────────────────────
def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}{'/ '+c['sheet'] if c['sheet'] else ''}]\n{c['text']}"
        for c in chunks
    )
    return f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the knowledge base."

Context:
{context}

Question: {query}

Answer:"""

# ─── LLM call (Gemini) ────────────────────────────────────────────────────────
def call_gemini(prompt, api_key):
    import urllib.request
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ─── Full RAG answer ──────────────────────────────────────────────────────────
def answer(query, api_key=None):
    chunks = retrieve(query)
    if not chunks:
        return "No documents indexed yet. Please upload files first.", []
    prompt = build_prompt(query, chunks)
    if api_key:
        try:
            response = call_gemini(prompt, api_key)
        except Exception as e:
            response = f"[LLM Error: {e}]\n\nRaw context retrieved:\n" + \
                       "\n---\n".join(c["text"] for c in chunks)
    else:
        # No API key: just return top retrieved chunk as answer (demo mode)
        response = f"[Demo mode — no API key]\n\nMost relevant content found:\n\n" + \
                   "\n\n---\n".join(c["text"][:300] for c in chunks[:2])
    return response, chunks
