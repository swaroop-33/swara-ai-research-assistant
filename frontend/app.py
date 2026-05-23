import time
import requests
import streamlit as st

from ui_helpers import (
    clean_answer_text,
    format_citations,
    inject_inline_citations,
    render_source_card,
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

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🦥 SWARA")

    st.write("")

    uploaded_files = st.file_uploader(
        "Upload Document",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Limit 200MB per file • PDF, TXT",
    )

    # =====================================================
    # FILE UPLOAD
    # =====================================================

    if uploaded_files:

        for file in uploaded_files:

            if (
                file.name
                not in st.session_state.uploaded_files
            ):

                try:

                    files = {
                        "file": (
                            file.name,
                            file,
                            file.type,
                        )
                    }

                    
                    try:

                        progress_placeholder = st.empty()

                        progress_placeholder.markdown(
                            """
                            🦥 Processing document...
                            
                            ░░░░░░░░░░ 0%
                            """
                        )

                        time.sleep(0.2)

                        progress_placeholder.markdown(
                            """
                            🦥 Extracting document text...
        
                            ███░░░░░░░ 30%
                            """
                        )

                        time.sleep(0.3)

                        progress_placeholder.markdown(
                            """
                            🦥 Generating embeddings...
                            
                            ███████░░░ 70%
                            """
                        )

                        time.sleep(0.4)

                        response = requests.post(
                            f"{API_BASE}/upload",
                            files=files,
                            timeout=120,
                        )

                        progress_placeholder.markdown(
                            """
                            🦥 Finalizing vector storage...
        
                            ██████████ 100%
                            """
                        )

                        time.sleep(0.4)

                        progress_placeholder.empty()

                        if response.status_code == 200:

                            
                            if (
                                file.name
                                not in st.session_state.uploaded_files
                            ):

                                st.session_state.uploaded_files.append(
                                    file.name
                            )

                            st.success(
                                f"✓ Uploaded: {file.name}"
                            )

                        else:

                            st.error(
                                f"Upload failed: "
                                f"{response.text}"
                            )

                    except Exception as e:

                        st.error(
                            f"Upload failed: {e}"
                        )



                    if response.status_code == 200:

                        st.session_state.uploaded_files.append(
                            file.name
                        )

                    else:

                        st.error(
                            f"Upload failed: "
                            f"{response.text}"
                        )

                except Exception as e:

                    st.error(
                        f"Upload failed: {e}"
                    )

    # =====================================================
    # UPLOADED FILES
    # =====================================================

    if st.session_state.uploaded_files:

        st.write("")

        st.markdown("### Uploaded Files")

        for filename in (
            st.session_state.uploaded_files
        ):

            st.markdown(
                f"📄 {filename}"
            )

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown("# 🦥 SWARA")
if st.button(
    "🗑️ New Chat",
    use_container_width=True,
):

    st.session_state.messages = []

    st.rerun()

st.caption(
    "Grounded AI research assistant. "
    "Upload a document and start asking questions."
)

# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "▼ View Sources"
            ):

                for idx, source in enumerate(
                    message["sources"]
                ):

                    render_source_card(
                        source,
                        idx,
                        {},
                    )

# =========================================================
# USER INPUT
# =========================================================

prompt = st.chat_input(
    "Ask a question about your documents..."
)

if prompt:

    # =====================================================
    # USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # =====================================================
    # ASSISTANT MESSAGE
    # =====================================================

    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.markdown(
            "🦥 Thinking..."
        )

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
            }

            response = requests.post(
                f"{API_BASE}/query",
                json=payload,
                timeout=180,
            )

            if response.status_code != 200:

                st.error(
                    f"Query failed: "
                    f"{response.text}"
                )

            else:

                data = response.json()

                answer = clean_answer_text(
                    data["answer"]
                )

                answer = inject_inline_citations(
                    answer,
                    data.get(
                        "retrieved_chunks",
                        [],
                    ),
                )

                answer = format_citations(
                    answer,
                    data.get(
                        "retrieved_chunks",
                        [],
                    ),
                )

                # =========================================
                # STREAMING EFFECT
                # =========================================

                thinking.empty()

                stream_placeholder = (
                    st.empty()
                )

                streamed = ""

                words = answer.split()

                for word in words:

                    streamed += word + " "

                    stream_placeholder.markdown(
                        streamed + "▌"
                    )

                    time.sleep(0.01)

                stream_placeholder.markdown(
                    streamed
                )

                # =========================================
                # SOURCES
                # =========================================

                sources = data.get(
                    "retrieved_chunks",
                    [],
                )

                if sources:

                    with st.expander(
                        "▼ View Sources"
                    ):

                        for idx, source in enumerate(
                            sources
                        ):

                            render_source_card(
                                source,
                                idx,
                                {},
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": streamed,
                        "sources": sources,
                    }
                )

        except Exception as e:

            thinking.empty()

            st.error(
                f"Query failed: {e}"
            )