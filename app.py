"""
Capstone 02 — Custom Knowledge Chatbot
Modern UI version
"""
import os, json, pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="KnowledgeBot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #0a0c14 !important;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #0f1220 !important;
    border-right: 1px solid #1e2540;
}

[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Sidebar title */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #1e2540;
    margin-bottom: 1.5rem;
}
.sidebar-brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.sidebar-brand-text { font-size: 15px; font-weight: 600; color: #f1f5f9; }
.sidebar-brand-sub  { font-size: 11px; color: #64748b; margin-top: 1px; }

/* Section labels */
.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #475569;
    margin-bottom: 8px;
}

/* Stat pills */
.stat-row { display: flex; gap: 8px; margin: 10px 0; }
.stat-pill {
    flex: 1; background: #141929;
    border: 1px solid #1e2540;
    border-radius: 10px; padding: 10px 8px;
    text-align: center;
}
.stat-pill-num  { font-size: 22px; font-weight: 600; color: #818cf8; line-height: 1; }
.stat-pill-label{ font-size: 10px; color: #475569; margin-top: 3px; }

/* File badge */
.file-badge {
    display: flex; align-items: center; gap: 6px;
    background: #141929; border: 1px solid #1e2540;
    border-radius: 8px; padding: 6px 10px;
    font-size: 12px; color: #94a3b8;
    margin: 4px 0;
}
.file-badge-icon { font-size: 14px; }

/* Main area */
.main-header {
    padding: 2rem 2rem 1rem;
    border-bottom: 1px solid #1a1f35;
    margin-bottom: 1.5rem;
}
.main-title { font-size: 22px; font-weight: 600; color: #f1f5f9; }
.main-sub   { font-size: 13px; color: #64748b; margin-top: 4px; }

/* Empty state */
.empty-state {
    text-align: center; padding: 4rem 2rem;
    border: 2px dashed #1e2540;
    border-radius: 16px; margin: 2rem;
}
.empty-icon  { font-size: 48px; margin-bottom: 1rem; }
.empty-title { font-size: 18px; font-weight: 500; color: #94a3b8; }
.empty-sub   { font-size: 13px; color: #475569; margin-top: 6px; }

/* Example question chips */
.chips-row { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 2rem 1.5rem; }
.chip {
    background: #141929; border: 1px solid #1e2540;
    border-radius: 20px; padding: 6px 14px;
    font-size: 12px; color: #94a3b8; cursor: pointer;
    transition: all 0.15s;
}

/* Chat messages */
.chat-area { padding: 0 2rem; }

.msg-user {
    display: flex; justify-content: flex-end; margin: 12px 0;
}
.msg-user-bubble {
    background: #312e81;
    border: 1px solid #3730a3;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; max-width: 70%;
    font-size: 14px; color: #e0e7ff; line-height: 1.5;
}

.msg-bot { display: flex; gap: 10px; margin: 12px 0; align-items: flex-start; }
.msg-bot-avatar {
    width: 32px; height: 32px; flex-shrink: 0;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}
.msg-bot-bubble {
    background: #141929; border: 1px solid #1e2540;
    border-radius: 4px 18px 18px 18px;
    padding: 12px 16px; max-width: 75%;
    font-size: 14px; color: #cbd5e1; line-height: 1.6;
}
.msg-bot-bubble p { margin: 0; }

/* Source tags */
.sources-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.source-chip {
    background: #0f172a; border: 1px solid #1e2540;
    border-radius: 6px; padding: 3px 8px;
    font-size: 11px; color: #64748b;
    display: flex; align-items: center; gap: 4px;
}
.source-score { color: #818cf8; font-weight: 500; }

/* Streamlit overrides */
[data-testid="stFileUploader"] {
    background: #141929 !important;
    border: 1px dashed #1e2540 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label { color: #94a3b8 !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.15s !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.9 !important; }

[data-testid="stTextInput"] > div > div > input {
    background: #141929 !important;
    border: 1px solid #1e2540 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

[data-testid="stChatInput"] textarea {
    background: #141929 !important;
    border: 1px solid #1e2540 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* Success/info overrides */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid #1e2540 !important;
}

/* Divider */
hr { border-color: #1e2540 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────
for k, v in {
    "chat_history": [],
    "vectorizer": None,
    "tfidf_matrix": None,
    "doc_store": [],
    "indexed_files": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── RAG core ──────────────────────────────────────────────────────────────
def chunk_text(text, size=200, overlap=30):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]).strip())
        i += size - overlap
    return [c for c in chunks if len(c) > 20]

def parse_txt(content_bytes, filename):
    text = content_bytes.decode("utf-8", errors="ignore")
    return [(c, {"source": filename, "type": "text", "sheet": ""})
            for c in chunk_text(text)]

def parse_excel(buf, filename):
    xl, docs = pd.ExcelFile(buf), []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet).fillna("")
        for _, row in df.iterrows():
            txt = f"[Sheet: {sheet}] " + " | ".join(
                f"{c}: {v}" for c, v in row.items() if str(v).strip())
            docs.append((txt, {"source": filename, "type": "excel", "sheet": sheet}))
    return docs

def build_index(docs):
    texts = [d[0] for d in docs]
    metas = [d[1] for d in docs]
    vec   = TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)
    mat   = vec.fit_transform(texts)
    store = [{"text": t, "metadata": m} for t, m in zip(texts, metas)]
    return vec, mat, store

def retrieve(query, top_k=5):
    if not st.session_state.vectorizer or not st.session_state.doc_store:
        return []
    qv     = st.session_state.vectorizer.transform([query])
    scores = cosine_similarity(qv, st.session_state.tfidf_matrix).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return [{"text": st.session_state.doc_store[i]["text"],
             "source": st.session_state.doc_store[i]["metadata"].get("source","?"),
             "sheet":  st.session_state.doc_store[i]["metadata"].get("sheet",""),
             "score":  round(float(scores[i]), 3)}
            for i in idx if scores[i] > 0]

def build_prompt(query, chunks):
    ctx = "\n\n".join(
        f"[{c['source']}{'/' + c['sheet'] if c['sheet'] else ''}]\n{c['text']}"
        for c in chunks)
    return (f"You are a helpful assistant. Answer using ONLY the context below.\n"
            f"If not found, say: 'I don't have that information in the knowledge base.'\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:")

def call_gemini(prompt, key):
    import urllib.request
    url  = ("https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-2.0-flash:generateContent?key={key}")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"]

# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🧠</div>
        <div>
            <div class="sidebar-brand-text">KnowledgeBot</div>
            <div class="sidebar-brand-sub">RAG-powered assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">API Key</div>', unsafe_allow_html=True)
    api_key = st.secrets.get("GEMINI_API_KEY", "") or st.text_input(
        "Gemini API Key", type="password", placeholder="AIza...",
        label_visibility="collapsed")
    if api_key:
        st.success("API key active", icon="✓")
    else:
        st.info("Demo mode — no key needed", icon="ℹ")

    st.divider()

    st.markdown('<div class="section-label">Knowledge Files</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["xlsx","csv","txt","md"],
                                 accept_multiple_files=True,
                                 label_visibility="collapsed")
    if uploaded:
        if st.button("⚡  Index files", use_container_width=True):
            all_docs = []
            with st.spinner("Indexing…"):
                for f in uploaded:
                    docs = (parse_excel(f, f.name)
                            if f.name.endswith((".xlsx",".csv"))
                            else parse_txt(f.read(), f.name))
                    all_docs.extend(docs)
                    if f.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(f.name)
                existing = [(d["text"], d["metadata"])
                            for d in st.session_state.doc_store]
                v, m, ds = build_index(existing + all_docs)
                st.session_state.vectorizer   = v
                st.session_state.tfidf_matrix = m
                st.session_state.doc_store    = ds
            st.success(f"Indexed {len(all_docs)} chunks")

    st.divider()

    total = len(st.session_state.doc_store)
    files = len(st.session_state.indexed_files)
    st.markdown(f"""
    <div class="section-label">Index stats</div>
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-pill-num">{total}</div>
            <div class="stat-pill-label">Chunks</div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-num">{files}</div>
            <div class="stat-pill-label">Files</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.indexed_files:
        st.markdown('<div class="section-label" style="margin-top:12px">Indexed</div>',
                    unsafe_allow_html=True)
        for fname in st.session_state.indexed_files:
            icon = "📊" if fname.endswith((".xlsx",".csv")) else "📄"
            st.markdown(f'<div class="file-badge"><span class="file-badge-icon">{icon}</span>{fname}</div>',
                        unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if c2.button("Reset all", use_container_width=True):
        for k in ["vectorizer","tfidf_matrix","doc_store","indexed_files","chat_history"]:
            st.session_state[k] = [] if k in ("doc_store","indexed_files","chat_history") else None
        st.rerun()

# ─── Main area ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">💬 Chat with your knowledge base</div>
    <div class="main-sub">Upload files → Index them → Ask anything</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.doc_store:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No files indexed yet</div>
        <div class="empty-sub">Upload Excel or text files in the sidebar, then click "Index files"</div>
    </div>
    """, unsafe_allow_html=True)
else:
    if not st.session_state.chat_history:
        st.markdown('<div class="chips-row">', unsafe_allow_html=True)
        examples = [
            "What is the return policy?",
            "How much does the Dell laptop cost?",
            "What payment methods are accepted?",
            "Who are customers from Karachi?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(examples):
            if cols[i % 2].button(q, key=f"ex_{i}"):
                st.session_state.chat_history.append({"role":"user","content":q})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Render chat
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-user-bubble">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">📎 {s["source"]}{"/" + s["sheet"] if s["sheet"] else ""} '
                f'<span class="source-score">{s["score"]}</span></span>'
                for s in msg["sources"][:3])
            sources_html = f'<div class="sources-row">{chips}</div>'
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f"""
        <div class="msg-bot">
            <div class="msg-bot-avatar">✨</div>
            <div class="msg-bot-bubble">
                <p>{content}</p>
                {sources_html}
            </div>
        </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Process last unanswered user message
if (st.session_state.chat_history and
        st.session_state.chat_history[-1]["role"] == "user" and
        st.session_state.doc_store):
    query = st.session_state.chat_history[-1]["content"]
    with st.spinner("Searching…"):
        chunks = retrieve(query)
        if not chunks:
            answer_text = "I couldn't find relevant information. Try rephrasing."
            sources = []
        else:
            prompt = build_prompt(query, chunks)
            if api_key:
                try:
                    answer_text = call_gemini(prompt, api_key)
                except Exception as e:
                    answer_text = (f"⚠️ API error: {e}\n\nRelevant content:\n\n" +
                                   "\n\n---\n".join(c["text"][:200] for c in chunks[:2]))
            else:
                answer_text = (
                    "**Demo mode** — add your Gemini API key for full answers.\n\n"
                    "**Relevant content found:**\n\n" +
                    "\n\n---\n".join(f"*{c['source']}*\n{c['text'][:300]}" for c in chunks[:2])
                )
            sources = chunks
    st.session_state.chat_history.append(
        {"role":"assistant","content":answer_text,"sources":sources})
    st.rerun()

# Chat input
user_input = st.chat_input(
    "Ask anything about your files…",
    disabled=not bool(st.session_state.doc_store))
if user_input:
    st.session_state.chat_history.append({"role":"user","content":user_input})
    st.rerun()
