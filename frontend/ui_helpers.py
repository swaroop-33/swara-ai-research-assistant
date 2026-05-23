import streamlit as st
import re
import time

# =========================================================
# SOURCE CARD
# =========================================================

def render_source_card(
    source,
    idx,
    colors,
):
    """
    Premium grounded source rendering.
    """

    # =====================================================
    # EXTRACT SOURCE DATA
    # =====================================================

    if isinstance(source, dict):

        content = source.get(
            "text",
            "",
        )

        filename = source.get(
            "filename",
            "Unknown",
        )

        page = source.get(
            "page_number",
            "?",
        )

        relevance = int(
            source.get(
                "similarity_score",
                0,
            ) * 100
        )

    else:

        content = str(source)

        filename = "Unknown"

        page = "?"

        relevance = 0

    # =====================================================
    # CONTENT LIMIT
    # =====================================================

    preview_limit = 520

    if len(content) > preview_limit:

        content = (
            content[:preview_limit]
            + "..."
        )

    # =====================================================
    # RELEVANCE LABEL
    # =====================================================

    if relevance >= 85:

        relevance_label = "Very High"

    elif relevance >= 70:

        relevance_label = "High"

    elif relevance >= 55:

        relevance_label = "Medium"

    else:

        relevance_label = "Low"

    # =====================================================
    # RENDER SOURCE CARD
    # =====================================================

    st.markdown(
        f"""
---
#### 📄 Source {idx + 1}

**File:** {filename}  
**Page:** {page}  
**Relevance:** {relevance_label}

> {content}
        """
    )


# =========================================================
# CLEAN ANSWER TEXT
# =========================================================

def clean_answer_text(
    text,
):
    """
    Improve assistant readability
    while preserving markdown formatting.
    """

    if not text:

        return ""

    text = text.strip()

    # =====================================================
    # REMOVE EXCESSIVE SPACING
    # =====================================================

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # =====================================================
    # IMPROVE BULLET FORMATTING
    # =====================================================

    text = re.sub(
        r"•",
        "\n•",
        text,
    )

    # =====================================================
    # CLEAN TRAILING SPACES
    # =====================================================

    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    return text


# =========================================================
# INLINE CITATION INJECTION
# =========================================================

def inject_inline_citations(
    answer,
    sources,
):
    """
    Inject lightweight inline citations
    while preserving readability.
    """

    if not answer:

        return ""

    if not sources:

        return answer

    # Avoid duplicates
    if "[Source" in answer:

        return answer

    answer = answer.strip()

    paragraphs = answer.split("\n\n")

    enhanced = []

    for idx, para in enumerate(paragraphs):

        para = para.strip()

        if not para:

            continue

        source_num = min(
            idx + 1,
            len(sources),
        )

        # Skip tiny fragments
        if len(para) < 40:

            enhanced.append(para)

            continue

        para = (
            para
            + f" [Source {source_num}]"
        )

        enhanced.append(para)

    return "\n\n".join(enhanced)


# =========================================================
# FORMAT CITATIONS
# =========================================================

def format_citations(
    answer,
    sources,
):
    """
    Format citations for stable markdown rendering.
    """

    if not answer:

        return ""

    formatted = re.sub(
        r"\[Source (\d+)\]",
        lambda m:
            f"`[Source {m.group(1)}]`",
        answer,
    )

    return formatted


# =========================================================
# STREAM RESPONSE
# =========================================================

def stream_response(
    response_placeholder,
    text,
    delay=0.015,
):
    """
    Smooth frontend streaming effect.

    Simulates token streaming
    without backend websocket complexity.
    """

    if not text:

        response_placeholder.markdown("")

        return

    words = text.split()

    streamed = ""

    for idx, word in enumerate(words):

        streamed += word + " "

        # Preserve markdown rendering
        response_placeholder.markdown(
            streamed
        )

        # Small delay for realism
        time.sleep(delay)

    # Final clean render
    response_placeholder.markdown(
        streamed.strip()
    )