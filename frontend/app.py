import os
import time
import uuid
import requests
import streamlit as st

from ui_helpers import (
    clean_answer_text,
    format_citations,
    inject_inline_citations,
    render_source_card,
)
from theme import apply_theme, LIGHT_COLORS, DARK_COLORS

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SWARA",
    page_icon="✨",
    layout="wide",
)

# Initialize dark mode state if not exists
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Apply the global CSS via theme.py
apply_theme(st.session_state.dark_mode)

# Load current colors for inline styles
theme_colors = DARK_COLORS if st.session_state.dark_mode else LIGHT_COLORS

# =========================================================
# API CONFIG
# =========================================================

API_BASE = os.environ.get(
    "SWARA_API_BASE",
    "http://localhost:8000/api/v1",
)

# =========================================================
# SESSION STATE
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0

if "sources_retrieved" not in st.session_state:
    st.session_state.sources_retrieved = 0

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("✨ SWARA")
    st.caption("Grounded AI Research Assistant")
    
    st.toggle("🌙 Dark Mode", key="dark_mode")

    st.divider()

    if st.button("✨ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.session_state.questions_asked = 0
        st.session_state.sources_retrieved = 0
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("### 📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="PDF or TXT • Drag and drop supported",
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                progress = st.empty()
                
                try:
                    progress.info("✨ Uploading your document...")
                    time.sleep(0.3)
                    progress.info("📖 Reading document contents...")
                    time.sleep(0.3)
                    progress.info("🧠 Understanding document knowledge...")
                    time.sleep(0.3)

                    response = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (file.name, file, file.type)},
                        data={"session_id": st.session_state.session_id},
                        timeout=120,
                    )

                    progress.info("🚀 Preparing SWARA for questions...")
                    time.sleep(0.4)
                    progress.empty()

                    if response.status_code == 200:
                        if file.name not in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.append(file.name)
                        st.success(f"✓ {file.name}")
                    else:
                        st.error(f"Upload failed: {response.text}")

                except Exception as e:
                    progress.empty()
                    st.error(f"Upload failed: {e}")

    if st.session_state.uploaded_files:
        st.markdown("### 📄 Uploaded Documents")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- **{fname}**")

    st.divider()
    
    st.markdown("### 📊 Session Stats")
    
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(st.session_state.uploaded_files))
    col2.metric("Questions", st.session_state.questions_asked)
    st.metric("Sources Retrieved", st.session_state.sources_retrieved)


# =========================================================
# MAIN AREA — WELCOME STATE
# =========================================================

if not st.session_state.messages:

    st.markdown(f"<h1 style='text-align: center; font-size: 56px; color: {theme_colors['accent_primary']};'>✨ SWARA</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: {theme_colors['accent_secondary']};'>Grounded AI Research Assistant</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: {theme_colors['text_muted']};'>Upload PDFs · Ask Questions · Get Verified Answers with Citations</p>", unsafe_allow_html=True)
    
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("## 📄")
            st.markdown("#### Upload Documents")
            st.caption("Upload PDFs and text files to build your knowledge base")

    with col2:
        with st.container(border=True):
            st.markdown("## 💬")
            st.markdown("#### Ask Questions")
            st.caption("Ask anything about your documents and get grounded answers")

    with col3:
        with st.container(border=True):
            st.markdown("## 📌")
            st.markdown("#### Verified Citations")
            st.caption("Every answer comes with source citations you can verify")

    st.write("")
    st.info("👋 Upload a document in the sidebar, then ask your first question below")


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📌 View Sources & Citations"):
                for idx, source in enumerate(message["sources"]):
                    render_source_card(source, idx, {})


# =========================================================
# USER INPUT
# =========================================================

prompt = st.chat_input("Ask a question about your documents...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.info("✨ Thinking...")

        try:
            payload = {
                "question": prompt,
                "top_k": 6,
                "chat_history": [
                    {
                        "role": m["role"],
                        "content": m["content"],
                    }
                    for m in st.session_state.messages[-10:]
                ],
                "session_id": st.session_state.session_id,
            }

            response = requests.post(
                f"{API_BASE}/query",
                json=payload,
                timeout=180,
            )

            if response.status_code != 200:
                st.error(f"Query failed: {response.text}")

            else:
                data = response.json()

                answer = clean_answer_text(data["answer"])
                answer = inject_inline_citations(
                    answer,
                    data.get("retrieved_chunks", []),
                )
                answer = format_citations(
                    answer,
                    data.get("retrieved_chunks", []),
                )

                thinking.empty()
                stream_placeholder = st.empty()
                streamed = ""
                words = answer.split()

                for word in words:
                    streamed += word + " "
                    stream_placeholder.markdown(streamed + "▌", unsafe_allow_html=True)
                    time.sleep(0.01)

                stream_placeholder.markdown(streamed, unsafe_allow_html=True)

                sources = data.get("retrieved_chunks", [])

                if sources:
                    with st.expander("📌 View Sources & Citations"):
                        for idx, source in enumerate(sources):
                            render_source_card(source, idx, {})

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": streamed,
                    "sources": sources,
                })

                st.session_state.questions_asked += 1
                st.session_state.sources_retrieved += len(sources)

        except Exception as e:
            thinking.empty()
            st.error(f"Query failed: {e}")