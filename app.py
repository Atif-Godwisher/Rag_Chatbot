"""
Capstone 02 — Custom Knowledge Chatbot
Streamlit app with file upload, RAG pipeline, and chat interface
Run: streamlit run app.py
"""

import os
import json
import pickle
import tempfile
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KnowledgeBot — RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* General */
[data-testid="stAppViewContainer"] {background: #0f1117;}
[data-testid="stSidebar"] {background: #161b27; border-right: 1px solid #2a2f40;}

/* Chat messages */
.user-msg {
    background: #1e3a5f;
    border-left: 3px solid #4f8ef7;
    padding: 12px 16px;
    border-radius: 0 12px 12px 12px;
    margin: 8px 0;
    color: #e2e8f0;
}
.bot-msg {
    background: #1a2035;
    border-left: 3px solid #22c55e;
    padding: 12px 16px;
    border-radius: 0 12px 12px 12px;
    margin: 8px 0;
    color: #e2e8f0;
}
.source-tag {
    display: inline-block;
    background: #252f45;
    color: #94a3b8;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 4px 4px 0 0;
    border: 1px solid #2a3a55;
}
.stat-card {
    background: #1a2035;
    border: 1px solid #2a2f40;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    margin-bottom: 8px;
}
h1 {color: #e2e8f0 !important;}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "doc_store" not in st.session_state:
    st.session_state.doc_store = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ─── Core RAG functions ───────────────────────────────────────────────────────
def chunk_text(text, chunk_size=200, overlap=30):
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]

def parse_txt(content_bytes, filename):
    text   = content_bytes.decode("utf-8", errors="ignore")
    chunks = chunk_text(text)
    return [(c, {"source": filename, "type": "text", "sheet": ""}) for c in chunks]

def parse_excel(buffer, filename):
    xl   = pd.ExcelFile(buffer)
    docs = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet).fillna("")
        for _, row in df.iterrows():
            row_text = f"[Sheet: {sheet}] " + " | ".join(
                f"{col}: {val}" for col, val in row.items() if str(val).strip()
            )
            docs.append((row_text, {"source": filename, "type": "excel", "sheet": sheet}))
    return docs

def build_index(documents):
    texts     = [d[0] for d in documents]
    metadatas = [d[1] for d in documents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    matrix    = vectorizer.fit_transform(texts)
    doc_store = [{"text": t, "metadata": m} for t, m in zip(texts, metadatas)]
    return vectorizer, matrix, doc_store

def retrieve(query, top_k=5):
    if st.session_state.vectorizer is None or not st.session_state.doc_store:
        return []
    q_vec  = st.session_state.vectorizer.transform([query])
    scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        if scores[i] > 0:
            doc = st.session_state.doc_store[i]
            results.append({
                "text":   doc["text"],
                "source": doc["metadata"].get("source", "?"),
                "sheet":  doc["metadata"].get("sheet", ""),
                "score":  round(float(scores[i]), 4)
            })
    return results

def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}{'/' + c['sheet'] if c['sheet'] else ''}]\n{c['text']}"
        for c in chunks
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the knowledge base."

Context:
{context}

Question: {query}

Answer:"""

def call_gemini(prompt, api_key):
    import urllib.request
    url  = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:generateContent?key={api_key}"
    )
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 KnowledgeBot")
    st.markdown("*RAG-powered custom chatbot*")
    st.divider()

    # API Key — reads from Streamlit Secrets on Cloud, fallback to manual input locally
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.secrets.get("GEMINI_API_KEY", "") or st.text_input(
        "Paste your key here", type="password",
        help="Get free key at aistudio.google.com"
    )
    if not api_key:
        st.info("No key → Demo mode (shows raw retrieved context)")
    else:
        st.success("API key set ✓")

    st.divider()

    # File Upload
    st.markdown("### 📁 Upload Knowledge Files")
    uploaded = st.file_uploader(
        "Excel or Text files",
        type=["xlsx", "csv", "txt", "md"],
        accept_multiple_files=True,
        help="Upload .xlsx, .csv, .txt, or .md files"
    )

    if uploaded:
        if st.button("⚡ Index Files", use_container_width=True, type="primary"):
            all_docs = []
            with st.spinner("Parsing and indexing…"):
                for f in uploaded:
                    if f.name.endswith((".xlsx", ".csv")):
                        docs = parse_excel(f, f.name)
                    else:
                        docs = parse_txt(f.read(), f.name)
                    all_docs.extend(docs)
                    if f.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(f.name)

                # Combine with existing index
                existing = [(d["text"], d["metadata"]) for d in st.session_state.doc_store]
                combined = existing + all_docs

                v, m, ds = build_index(combined)
                st.session_state.vectorizer   = v
                st.session_state.tfidf_matrix = m
                st.session_state.doc_store    = ds

            st.success(f"Indexed {len(all_docs)} new chunks!")

    st.divider()

    # Stats
    st.markdown("### 📊 Index Stats")
    total = len(st.session_state.doc_store)
    files = len(st.session_state.indexed_files)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:24px;font-weight:700;color:#4f8ef7">{total}</div>
            <div style="font-size:11px;color:#64748b">Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:24px;font-weight:700;color:#22c55e">{files}</div>
            <div style="font-size:11px;color:#64748b">Files</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.indexed_files:
        st.markdown("**Indexed files:**")
        for fname in st.session_state.indexed_files:
            icon = "📊" if fname.endswith((".xlsx", ".csv")) else "📄"
            st.caption(f"{icon} {fname}")

    st.divider()

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset index", use_container_width=True):
            st.session_state.vectorizer   = None
            st.session_state.tfidf_matrix = None
            st.session_state.doc_store    = []
            st.session_state.indexed_files = []
            st.rerun()

# ─── Main chat area ───────────────────────────────────────────────────────────
st.markdown("# 💬 Chat with Your Knowledge Base")

if not st.session_state.doc_store:
    st.markdown("""
    <div style="background:#1a2035;border:2px dashed #2a3a55;border-radius:12px;
    padding:40px;text-align:center;margin:30px 0">
        <div style="font-size:48px">📂</div>
        <h3 style="color:#94a3b8">No files indexed yet</h3>
        <p style="color:#64748b">Upload Excel or text files in the sidebar, then click "Index Files"</p>
        <p style="color:#475569;font-size:13px">Sample files: <code>store_data.xlsx</code> and <code>company_faq.txt</code></p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Show example questions
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        example_qs = [
            "What is the return policy?",
            "How much does the Dell laptop cost?",
            "What payment methods are accepted?",
            "Who are the customers from Karachi?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(example_qs):
            if cols[i % 2].button(q, key=f"eq_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.rerun()

# Render chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 <b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 <b>Bot:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True)
        if msg.get("sources"):
            tags = "".join(
                f'<span class="source-tag">📎 {s["source"]}{"/" + s["sheet"] if s["sheet"] else ""} '
                f'({s["score"]})</span>'
                for s in msg["sources"]
            )
            st.markdown(f'<div style="margin-left:4px">{tags}</div>', unsafe_allow_html=True)

# Process last user message if no bot reply yet
if (st.session_state.chat_history and
        st.session_state.chat_history[-1]["role"] == "user" and
        st.session_state.doc_store):

    query = st.session_state.chat_history[-1]["content"]
    with st.spinner("Searching knowledge base…"):
        chunks = retrieve(query, top_k=5)
        if not chunks:
            answer_text = "I couldn't find relevant information. Try rephrasing your question."
            sources     = []
        else:
            prompt = build_prompt(query, chunks)
            if api_key:
                try:
                    answer_text = call_gemini(prompt, api_key)
                except Exception as e:
                    answer_text = f"⚠️ Gemini API error: {e}\n\nRaw context:\n\n" + \
                                  "\n\n---\n".join(c["text"][:200] for c in chunks[:2])
            else:
                answer_text = (
                    "**Demo mode** — add a Gemini API key for full answers.\n\n"
                    "**Most relevant chunks found:**\n\n" +
                    "\n\n---\n".join(f"📎 *{c['source']}*\n{c['text'][:300]}"
                                     for c in chunks[:2])
                )
            sources = chunks

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    })
    st.rerun()

# Chat input
st.markdown("<br>", unsafe_allow_html=True)
with st.container():
    user_input = st.chat_input(
        "Ask anything about your uploaded files…",
        disabled=not bool(st.session_state.doc_store)
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()
