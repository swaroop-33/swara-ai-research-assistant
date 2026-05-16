import streamlit as st
import requests

from theme import apply_theme, get_colors

from ui_helpers import (
    render_source_card,
    clean_answer_text,
    format_citations,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SWARA",
    page_icon="🦥",
    layout="wide",
)

# =========================================================
# API CONFIG
# =========================================================

API_BASE = "http://localhost:8000/api/v1"

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# =========================================================
# THEME
# =========================================================

apply_theme(
    st.session_state.dark_mode
)

colors = get_colors(
    st.session_state.dark_mode
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # INTERNAL RETRIEVAL DEPTH
    # Hidden from UI now
    top_k = 6

    st.session_state.retrieval_depth = top_k

    # =====================================================
    # LOGO
    # =====================================================

    st.markdown(
        f"""
        <div style="
            padding-top: 0.3rem;
            padding-bottom: 1.2rem;
        ">
            <h1 style="
                font-size: 2.25rem;
                font-weight: 800;
                color: {colors['text']};
                margin-bottom: 0;
                letter-spacing: -0.04em;
            ">
                🦥 SWARA
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # CONTROLS
    # =====================================================

    col1, col2 = st.columns([1, 1])

    with col1:

        if st.button(
            "🌙 Dark"
            if not st.session_state.dark_mode
            else "☀️ Light",
            use_container_width=True,
        ):

            st.session_state.dark_mode = (
                not st.session_state.dark_mode
            )

            st.rerun()

    with col2:

        if st.button(
            "✨ New Chat",
            use_container_width=True,
        ):

            try:

                requests.delete(
                    f"{API_BASE}/reset",
                    timeout=30,
                )

            except Exception:
                pass

            st.session_state.messages = []
            st.session_state.uploaded_files = []

            st.rerun()

    st.write("")

    # =====================================================
    # UPLOAD SECTION
    # =====================================================

    st.markdown(
        "### Upload Document"
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    # =====================================================
    # PROCESS FILE
    # =====================================================

    if uploaded_file:

        if (
            uploaded_file.name
            not in st.session_state.uploaded_files
        ):

            files = {
                "file": uploaded_file
            }

            try:

                with st.spinner(
                    "Processing document..."
                ):

                    response = requests.post(
                        f"{API_BASE}/upload",
                        files=files,
                        timeout=300,
                    )

                if response.status_code == 200:

                    st.session_state.uploaded_files.append(
                        uploaded_file.name
                    )

                    upload_data = response.json()

                    chunks = upload_data.get(
                        "chunks_created",
                        "N/A",
                    )

                    pages = upload_data.get(
                        "pages_processed",
                        "N/A",
                    )

                    st.success(
                        f"Processed • "
                        f"{chunks} chunks • "
                        f"{pages} pages"
                    )

                else:

                    try:
                        error_message = response.json()

                    except Exception:
                        error_message = response.text

                    st.error(
                        f"Upload failed: {error_message}"
                    )

            except Exception as e:

                st.error(
                    f"Upload failed: {str(e)}"
                )

    # =====================================================
    # FILE LIST
    # =====================================================

    if st.session_state.uploaded_files:

        st.write("")

        st.markdown(
            "### Uploaded Files"
        )

        for file in st.session_state.uploaded_files:

            st.markdown(
                f"""
                <div style="
                    padding: 0.8rem 1rem;
                    border-radius: 14px;
                    margin-bottom: 0.55rem;
                    background: {colors['card_bg']};
                    border: 1px solid {colors['border']};
                    color: {colors['text']};
                    font-size: 0.92rem;
                    font-weight: 500;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                ">
                    📄 {file}
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# HERO / EMPTY STATE
# =========================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        """
        <div style="
            height: 7vh;
        "></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align:center;font-size:2.5rem;margin-bottom:0;'>🦥</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <h1 style="
            text-align:center;
            font-size:3rem;
            font-weight:800;
            color:{colors["text"]};
            margin-bottom:0.35rem;
            letter-spacing:-0.04em;
        ">
            SWARA
        </h1>

        <p style="
            text-align:center;
            font-size:1rem;
            color:{colors["muted_text"]};
            margin-bottom:2rem;
        ">
            Grounded AI research assistant.
            Upload a document and start asking questions.
        </p>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"],
            unsafe_allow_html=True,
        )

        # =================================================
        # SOURCES
        # =================================================

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):

            st.markdown(
                f"""
                <div style="
                    margin-top: 1rem;
                    margin-bottom: 0.8rem;
                    font-size: 0.9rem;
                    font-weight: 700;
                    color: {colors['muted_text']};
                    letter-spacing: 0.02em;
                    text-transform: uppercase;
                ">
                    Sources
                </div>
                """,
                unsafe_allow_html=True,
            )

            for idx, source in enumerate(
                message["sources"],
                start=1,
            ):

                render_source_card(
                    source,
                    idx,
                    colors,
                )

# =========================================================
# CHAT INPUT
# =========================================================

query = st.chat_input(
    "Ask a question about your documents..."
)

# =========================================================
# QUERY EXECUTION
# =========================================================

if query:

    # =====================================================
    # STORE USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        st.markdown(
            f"""
            <div style="
                font-size: 1rem;
                line-height: 1.7;
                font-weight: 600;
                color: {colors['text']};
            ">
                {query}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # ASSISTANT RESPONSE
    # =====================================================

    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.markdown(
            f"""
            <div style="
                padding: 0.9rem 1rem;
                border-radius: 16px;
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                color: {colors['muted_text']};
                font-size: 0.95rem;
                margin-bottom: 0.5rem;
                line-height: 1.6;
            ">
                🧠 Thinking...
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:

            response = requests.post(
                f"{API_BASE}/query",
                json={

                    "question": query,
                    "top_k": top_k,
                    "chat_history": (
                        st.session_state.messages[-6:]
                    ),
                },
                timeout=120,
            )

            thinking.empty()

            if response.status_code == 200:

                data = response.json()

                answer = data.get(
                    "answer",
                    "No response generated.",
                )

                sources = data.get(
                    "retrieved_chunks",
                    data.get(
                        "sources",
                        [],
                    ),
                )

                # =============================================
                # CLEAN ANSWER
                # =============================================

                clean_answer = clean_answer_text(
                    answer
                )

                formatted_answer = (
                    format_citations(
                        clean_answer,
                        sources,
                    )
                )

                # =============================================
                # DISPLAY ANSWER
                # =============================================

                st.markdown(
                    f"""
                    <div style="
                        line-height: 1.9;
                        font-size: 1rem;
                        color: {colors['text']};
                        margin-bottom: 0.4rem;
                    ">
                        {formatted_answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # =============================================
                # SOURCES
                # =============================================

                if sources:

                    st.markdown(
                        f"""
                        <div style="
                            margin-top: 1.2rem;
                            margin-bottom: 0.8rem;
                            font-size: 0.9rem;
                            font-weight: 700;
                            color: {colors['muted_text']};
                            text-transform: uppercase;
                            letter-spacing: 0.03em;
                        ">
                            Sources
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    for idx, source in enumerate(
                        sources,
                        start=1,
                    ):

                        render_source_card(
                            source,
                            idx,
                            colors,
                        )

                # =============================================
                # SAVE MESSAGE
                # =============================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": formatted_answer,
                        "sources": sources,
                    }
                )

            else:

                st.error(
                    f"Query failed: {response.text}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Query failed.",
                    }
                )

        except Exception as e:

            thinking.empty()

            st.error(
                f"Backend connection failed: {str(e)}"
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Backend connection failed."
                    ),
                }
            )