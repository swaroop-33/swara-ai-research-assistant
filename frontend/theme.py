import streamlit as st

# =========================================================
# PREMIUM LIGHT THEME
# =========================================================

LIGHT_COLORS = {
    "bg": "#F5F7FB",
    "surface": "rgba(255,255,255,0.72)",
    "sidebar": "rgba(255,255,255,0.78)",
    "card_bg": "rgba(255,255,255,0.74)",
    "border": "rgba(148,163,184,0.18)",
    "text": "#0F172A",
    "muted_text": "#64748B",
    "accent": "#7C3AED",
    "accent_soft": "rgba(124,58,237,0.12)",
}

# =========================================================
# PREMIUM DARK THEME
# =========================================================

DARK_COLORS = {
    "bg": "#050816",
    "surface": "rgba(15,23,42,0.72)",
    "sidebar": "rgba(9,15,28,0.82)",
    "card_bg": "rgba(15,23,42,0.74)",
    "border": "rgba(148,163,184,0.12)",
    "text": "#F8FAFC",
    "muted_text": "#94A3B8",
    "accent": "#8B5CF6",
    "accent_soft": "rgba(139,92,246,0.12)",
}

# =========================================================
# APPLY THEME
# =========================================================

def apply_theme(dark_mode: bool):

    colors = (
        DARK_COLORS
        if dark_mode
        else LIGHT_COLORS
    )

    st.markdown(
        f"""
        <style>

        /* =====================================================
           GLOBAL
        ===================================================== */

        .stApp {{
            background:
                radial-gradient(
                    circle at top left,
                    rgba(124,58,237,0.08),
                    transparent 30%
                ),
                radial-gradient(
                    circle at bottom right,
                    rgba(59,130,246,0.08),
                    transparent 35%
                ),
                {colors["bg"]};

            color: {colors["text"]};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, sans-serif;
            scroll-behavior: smooth;
        }}

        .main .block-container {{
            max-width: 980px;
            padding-top: 1.6rem;
            padding-left: 3rem;
            padding-right: 3rem;
            padding-bottom: 7rem;
        }}

        /* =====================================================
           SIDEBAR
        ===================================================== */

        section[data-testid="stSidebar"] {{
            width: 300px !important;
            min-width: 300px !important;

            background: {colors["sidebar"]};

            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);

            border-right: 1px solid {colors["border"]};

            box-shadow:
                0 0 30px rgba(0,0,0,0.12);

            transition: all 0.3s ease;
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 0.8rem;
        }}

        /* =====================================================
           HIDE RETRIEVAL DEPTH
        ===================================================== */

        div[data-testid="stSlider"] {{
            display: none;
        }}

        /* =====================================================
           TEXT
        ===================================================== */

        h1, h2, h3, h4, h5, h6 {{
            color: {colors["text"]} !important;
            letter-spacing: -0.03em;
            font-weight: 700;
        }}

        p, span, label {{
            color: {colors["text"]} !important;
        }}

        /* =====================================================
           BUTTONS
        ===================================================== */

        .stButton > button {{

            background: {colors["surface"]};

            color: {colors["text"]};

            border: 1px solid {colors["border"]};

            backdrop-filter: blur(18px);

            border-radius: 999px;

            height: 2.9rem;

            font-weight: 600;

            transition:
                transform 0.18s ease,
                border 0.18s ease,
                background 0.18s ease,
                box-shadow 0.18s ease;

            box-shadow:
                0 4px 18px rgba(0,0,0,0.08);
        }}

        .stButton > button:hover {{

            transform: translateY(-2px);

            border-color: {colors["accent"]};

            color: {colors["accent"]};

            box-shadow:
                0 8px 28px rgba(124,58,237,0.18);

            background:
                {colors["accent_soft"]};
        }}

        /* =====================================================
           FILE UPLOADER
        ===================================================== */

        [data-testid="stFileUploader"] {{

            background: {colors["surface"]};

            border: 1px dashed {colors["border"]};

            border-radius: 22px;

            padding: 1.1rem;

            backdrop-filter: blur(18px);

            transition: all 0.25s ease;
        }}

        [data-testid="stFileUploader"]:hover {{

            border-color: {colors["accent"]};

            box-shadow:
                0 0 30px rgba(124,58,237,0.10);
        }}

        /* =====================================================
           CHAT INPUT CONTAINER
        ===================================================== */

        [data-testid="stChatInput"] {{

            position: fixed;
            bottom: 1.2rem;
            left: 50%;
            transform: translateX(-50%);

            width: min(920px, 78vw);

            background:
                rgba(15,23,42,0.78);

            backdrop-filter: blur(24px);

            border:
                1px solid rgba(148,163,184,0.12);

            border-radius: 24px;

            padding:
                0.45rem 0.7rem;

            box-shadow:
                0 10px 40px rgba(0,0,0,0.24);

            z-index: 999;
        }}

        /* =====================================================
           CHAT INPUT FIELD
        ===================================================== */

        .stChatInput input {{

            background:
                transparent !important;

            color:
                {colors["text"]} !important;

            border:
                none !important;

            font-size:
                0.98rem !important;

            padding:
                0.95rem 0.4rem !important;

            box-shadow:
                none !important;
        }}

        /* =====================================================
           INPUT FOCUS
        ===================================================== */

        [data-testid="stChatInput"]:focus-within {{

            border:
                1px solid rgba(139,92,246,0.38);

            box-shadow:
                0 0 0 4px rgba(139,92,246,0.10),
                0 12px 48px rgba(0,0,0,0.28);
        }}

        /* =====================================================
           SEND BUTTON
        ===================================================== */

        .stChatInput button {{

            border-radius:
                999px !important;

            background:
                linear-gradient(
                    135deg,
                    #8B5CF6,
                    #7C3AED
                ) !important;

            border:
                none !important;

            color:
                white !important;

            transition:
                all 0.2s ease !important;
        }}

        .stChatInput button:hover {{

            transform:
                scale(1.05);

            box-shadow:
                0 0 18px rgba(139,92,246,0.45);
        }}

        /* =====================================================
           CHAT MESSAGES
        ===================================================== */

        div[data-testid="stChatMessage"] {{

            background: {colors["card_bg"]};

            border: 1px solid {colors["border"]};

            border-radius: 26px;

            padding: 1.25rem;

            margin-bottom: 1rem;

            line-height: 1.85;

            backdrop-filter: blur(18px);

            transition:
                transform 0.18s ease,
                border 0.18s ease,
                box-shadow 0.18s ease;

            box-shadow:
                0 4px 18px rgba(0,0,0,0.08);
        }}

        div[data-testid="stChatMessage"]:hover {{

            transform: translateY(-2px);

            border-color:
                rgba(124,58,237,0.22);

            box-shadow:
                0 12px 36px rgba(0,0,0,0.14);
        }}

        /* =====================================================
           EXPANDERS
        ===================================================== */

        details {{
            background: transparent !important;
            border-radius: 18px;
            overflow: hidden;
        }}

        summary {{
            transition: all 0.2s ease;
        }}

        summary:hover {{
            color: {colors["accent"]};
        }}

        /* =====================================================
           SCROLLBAR
        ===================================================== */

        ::-webkit-scrollbar {{
            width: 8px;
        }}

        ::-webkit-scrollbar-thumb {{
            background:
                rgba(148,163,184,0.28);

            border-radius: 999px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background:
                rgba(148,163,184,0.42);
        }}

        /* =====================================================
           ANIMATIONS
        ===================================================== */

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(8px);
            }}

            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        div[data-testid="stChatMessage"] {{
            animation: fadeIn 0.28s ease;
        }}

        /* =====================================================
           REMOVE STREAMLIT DEFAULT CLUTTER
        ===================================================== */

        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}

        header {{
            visibility: hidden;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# GET COLORS
# =========================================================

def get_colors(dark_mode: bool):

    return (
        DARK_COLORS
        if dark_mode
        else LIGHT_COLORS
    )