import streamlit as st

# =========================================================
# PREMIUM LIGHT THEME
# =========================================================

LIGHT_COLORS = {
    "bg": "#F6F8FC",
    "surface": "rgba(255,255,255,0.72)",
    "sidebar": "rgba(255,255,255,0.80)",
    "card_bg": "rgba(255,255,255,0.74)",
    "border": "rgba(148,163,184,0.16)",
    "text": "#0F172A",
    "muted_text": "#64748B",
    "accent": "#7C3AED",
    "accent_soft": "rgba(124,58,237,0.10)",
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
                    rgba(124,58,237,0.05),
                    transparent 30%
                ),
                radial-gradient(
                    circle at bottom right,
                    rgba(59,130,246,0.05),
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

            padding-top: 1.8rem;

            padding-left: 2rem;

            padding-right: 2rem;

            padding-bottom: 7rem;
        }}

        /* =====================================================
           REMOVE ALL BLUE FOCUS RINGS
        ===================================================== */

        *:focus,
        *:focus-visible,
        textarea:focus,
        input:focus,
        button:focus,
        button:focus-visible,
        textarea:focus-visible,
        input:focus-visible,
        [data-testid="stChatInput"]:focus,
        [data-testid="stChatInput"]:focus-visible,
        [data-baseweb="textarea"]:focus-within {{

            outline: none !important;

            box-shadow: none !important;
        }}

        /* =====================================================
           SIDEBAR
        ===================================================== */

        section[data-testid="stSidebar"] {{

            width: 300px !important;

            min-width: 300px !important;

            background: {colors["sidebar"]};

            backdrop-filter: blur(12px);

            -webkit-backdrop-filter: blur(12px);

            border-right:
                1px solid {colors["border"]};

            box-shadow:
                0 0 18px rgba(15,23,42,0.05);
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

            background:
                {colors["surface"]};

            color:
                {colors["text"]};

            border:
                1px solid {colors["border"]};

            border-radius:
                999px;

            height:
                2.8rem;

            font-weight:
                600;

            backdrop-filter:
                blur(10px);

            transition:
                all 0.18s ease;

            box-shadow:
                0 2px 10px rgba(15,23,42,0.04);
        }}

        .stButton > button:hover {{

            transform:
                translateY(-1px);

            border-color:
                {colors["accent"]};

            color:
                {colors["accent"]};

            background:
                {colors["accent_soft"]};

            box-shadow:
                0 6px 16px rgba(124,58,237,0.10);
        }}

        /* =====================================================
           FILE UPLOADER
        ===================================================== */

        [data-testid="stFileUploader"] {{

            background:
                {colors["surface"]};

            border:
                1px dashed {colors["border"]};

            border-radius:
                20px;

            padding:
                1rem;

            backdrop-filter:
                blur(10px);

            transition:
                all 0.18s ease;
        }}

        [data-testid="stFileUploader"]:hover {{

            border-color:
                {colors["accent"]};

            box-shadow:
                0 0 16px rgba(124,58,237,0.06);
        }}

        /* =====================================================
           CHAT INPUT CONTAINER
        ===================================================== */

        [data-testid="stChatInput"] {{

            position: fixed;

            bottom: 1rem;

            left: 50%;

            transform: translateX(-50%);

            width: min(860px, 76vw);

            background:
                rgba(255,255,255,0.96);

            border:
                1px solid rgba(203,213,225,0.55);

            border-radius:
                18px;

            padding:
                0.32rem 0.6rem;

            box-shadow:
                0 6px 20px rgba(15,23,42,0.05);

            backdrop-filter:
                blur(10px);

            -webkit-backdrop-filter:
                blur(10px);

            transition:
                all 0.18s ease;

            z-index: 999;
        }}

        [data-testid="stChatInput"]:focus-within {{

            border:
                1px solid rgba(124,58,237,0.18);

            box-shadow:
                0 8px 20px rgba(124,58,237,0.08);
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

            outline:
                none !important;

            box-shadow:
                none !important;

            font-size:
                0.96rem !important;

            padding:
                0.82rem 0.35rem !important;
        }}

        /* =====================================================
           SEND BUTTON
        ===================================================== */

        .stChatInput button {{

            width: 42px !important;

            height: 42px !important;

            min-width: 42px !important;

            border-radius:
                12px !important;

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

            display:
                flex !important;

            align-items:
                center !important;

            justify-content:
                center !important;

            transition:
                all 0.18s ease !important;

            box-shadow:
                0 4px 12px rgba(124,58,237,0.18);
        }}

        .stChatInput button:hover {{

            transform:
                translateY(-1px);

            box-shadow:
                0 4px 10px rgba(124,58,237,0.14);
        }}

        /* =====================================================
           CHAT MESSAGES
        ===================================================== */

        div[data-testid="stChatMessage"] {{

            background:
                {colors["card_bg"]};

            border:
                1px solid {colors["border"]};

            border-radius:
                22px;

            padding:
                1.25rem;

            margin-bottom:
                1rem;

            line-height:
                1.8;

            backdrop-filter:
                blur(10px);

            transition:
                all 0.18s ease;

            box-shadow:
                0 4px 16px rgba(15,23,42,0.05);
        }}

        div[data-testid="stChatMessage"]:hover {{

            transform:
                translateY(-1px);

            border-color:
                rgba(124,58,237,0.14);

            box-shadow:
                0 8px 20px rgba(15,23,42,0.06);
        }}

        /* =====================================================
           SCROLLBAR
        ===================================================== */

        ::-webkit-scrollbar {{

            width: 8px;
        }}

        ::-webkit-scrollbar-thumb {{

            background:
                rgba(148,163,184,0.22);

            border-radius:
                999px;
        }}

        ::-webkit-scrollbar-thumb:hover {{

            background:
                rgba(148,163,184,0.34);
        }}

        /* =====================================================
           MOBILE RESPONSIVENESS
        ===================================================== */

        @media (max-width: 768px) {{

            .main .block-container {{

                padding-left: 1rem;

                padding-right: 1rem;

                padding-top: 1rem;

                padding-bottom: 8rem;
            }}

            section[data-testid="stSidebar"] {{

                width: 100% !important;

                min-width: 100% !important;
            }}

            [data-testid="stChatInput"] {{

                width:
                    calc(100vw - 1.2rem);

                bottom:
                    0.7rem;

                border-radius:
                    14px;
            }}

            div[data-testid="stChatMessage"] {{

                padding:
                    1rem;

                border-radius:
                    18px;
            }}

            h1 {{

                font-size:
                    2rem !important;
            }}
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