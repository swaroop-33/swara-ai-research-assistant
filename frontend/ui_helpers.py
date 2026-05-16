import streamlit as st
import re
import html

# =========================================================
# SOURCE CARD
# =========================================================

def render_source_card(
    source,
    idx,
    colors,
):
    """
    Premium source rendering.
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
    # SAFETY ESCAPE
    # =====================================================

    content = html.escape(content)

    # =====================================================
    # TRUNCATION
    # =====================================================

    preview_limit = 320

    full_content = content

    truncated = False

    if len(content) > preview_limit:

        truncated = True

        content = (
            content[:preview_limit]
            + "..."
        )

    # =====================================================
    # SOURCE LABEL COLOR
    # =====================================================

    if relevance >= 80:

        relevance_color = "#22C55E"

    elif relevance >= 60:

        relevance_color = "#F59E0B"

    else:

        relevance_color = "#EF4444"

    # =====================================================
    # MAIN SOURCE CARD
    # =====================================================

    st.markdown(
        f"""
        <div style="
            background:
                {colors['card_bg']};

            border:
                1px solid {colors['border']};

            border-radius:
                18px;

            padding:
                1rem 1rem 0.9rem 1rem;

            margin-bottom:
                0.9rem;

            backdrop-filter:
                blur(18px);

            transition:
                all 0.2s ease;

            box-shadow:
                0 4px 16px rgba(0,0,0,0.08);
        ">

            <!-- HEADER -->

            <div style="
                display:
                    flex;

                justify-content:
                    space-between;

                align-items:
                    center;

                gap:
                    1rem;

                margin-bottom:
                    0.8rem;
            ">

                <!-- FILE -->

                <div style="
                    font-weight:
                        700;

                    font-size:
                        0.95rem;

                    color:
                        {colors['text']};

                    overflow:
                        hidden;

                    text-overflow:
                        ellipsis;

                    white-space:
                        nowrap;
                ">
                    📄 {filename}
                </div>

                <!-- META -->

                <div style="
                    display:
                        flex;

                    align-items:
                        center;

                    gap:
                        0.55rem;

                    flex-shrink:
                        0;
                ">

                    <div style="
                        font-size:
                            0.78rem;

                        color:
                            {colors['muted_text']};
                    ">
                        Page {page}
                    </div>

                    <div style="
                        padding:
                            0.22rem 0.55rem;

                        border-radius:
                            999px;

                        font-size:
                            0.72rem;

                        font-weight:
                            700;

                        background:
                            rgba(255,255,255,0.06);

                        color:
                            {relevance_color};

                        border:
                            1px solid rgba(255,255,255,0.05);
                    ">
                        {relevance}%
                    </div>

                </div>

            </div>

            <!-- CONTENT -->

            <div style="
                color:
                    {colors['text']};

                font-size:
                    0.91rem;

                line-height:
                    1.75;

                white-space:
                    pre-wrap;

                opacity:
                    0.96;
            ">
                {content}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # FULL SOURCE EXPAND
    # =====================================================

    if truncated:

        st.markdown(
            f"""
            <details style="
                margin-top:
                    -0.2rem;

                margin-bottom:
                    1rem;
            ">

                <summary style="
                    cursor:
                        pointer;

                    color:
                        {colors['accent']};

                    font-size:
                        0.88rem;

                    font-weight:
                        600;

                    transition:
                        all 0.2s ease;
                ">
                    View Full Source #{idx}
                </summary>

                <div style="
                    margin-top:
                        0.75rem;

                    padding:
                        1rem;

                    border-radius:
                        16px;

                    background:
                        {colors['card_bg']};

                    border:
                        1px solid {colors['border']};

                    color:
                        {colors['text']};

                    line-height:
                        1.75;

                    white-space:
                        pre-wrap;

                    box-shadow:
                        0 4px 16px rgba(0,0,0,0.08);
                ">
                    {full_content}
                </div>

            </details>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# CLEAN ANSWER TEXT
# =========================================================

def clean_answer_text(
    text,
):
    """
    Improve readability of assistant responses
    while preserving markdown structure.
    """

    if not text:

        return ""

    # =====================================================
    # TRIM
    # =====================================================

    text = text.strip()

    # =====================================================
    # NORMALIZE EXCESSIVE SPACING
    # =====================================================

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # =====================================================
    # REMOVE TRAILING SPACES
    # =====================================================

    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    return text

# =========================================================
# FORMAT CITATIONS
# =========================================================

def format_citations(
    answer,
    sources,
):
    """
    Style source citations cleanly.
    """

    if not answer:

        return ""

    formatted = re.sub(
        r"\[Source (\d+)\]",
        lambda m:
            f"<span style='font-weight:700;'>[Source {m.group(1)}]</span>",
        answer,
    )

    return formatted