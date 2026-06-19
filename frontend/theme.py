import streamlit as st

# =========================================================
# PREMIUM LIGHT THEME
# =========================================================

LIGHT_COLORS = {
    "app_bg": "linear-gradient(160deg, #FFFDF8 0%, #FFF7ED 30%, #FDF2F8 70%, #FFFDF8 100%)",
    "sidebar_bg": "linear-gradient(180deg, rgba(255,248,240,0.97) 0%, rgba(253,242,248,0.97) 100%)",
    "container_bg": "rgba(255,255,255,0.75)",
    "border": "rgba(236,72,153,0.15)",
    "border_hover": "rgba(236,72,153,0.30)",
    "text": "#1F2937",
    "text_muted": "#6B7280",
    "accent_primary": "#EC4899",
    "accent_secondary": "#F59E0B",
    "shadow_light": "rgba(0,0,0,0.03)",
    "shadow_hover": "rgba(236,72,153,0.08)",
    "chat_input_bg": "rgba(255,255,255,0.95)",
    "chat_input_text": "#1F2937",
    "chat_input_placeholder": "#9CA3AF",
}

# =========================================================
# PREMIUM DARK THEME
# =========================================================

DARK_COLORS = {
    "app_bg": "linear-gradient(160deg, #1A0F15 0%, #2A1520 30%, #1F101A 70%, #1A0F15 100%)",
    "sidebar_bg": "linear-gradient(180deg, rgba(30,15,24,0.97) 0%, rgba(44,22,35,0.97) 100%)",
    "container_bg": "rgba(44,22,35,0.75)",
    "border": "rgba(245,158,11,0.15)",
    "border_hover": "rgba(245,158,11,0.30)",
    "text": "#FFF7ED",
    "text_muted": "#D1D5DB",
    "accent_primary": "#F59E0B",
    "accent_secondary": "#EC4899",
    "shadow_light": "rgba(0,0,0,0.20)",
    "shadow_hover": "rgba(245,158,11,0.15)",
    "chat_input_bg": "rgba(30,15,24,0.95)",
    "chat_input_text": "#FFFFFF",
    "chat_input_placeholder": "#9CA3AF",
}

# =========================================================
# APPLY THEME
# =========================================================

def apply_theme(dark_mode: bool):

    colors = DARK_COLORS if dark_mode else LIGHT_COLORS

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── Global ────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: {colors["text"]} !important;
        }}

        .stApp {{
            background: {colors["app_bg"]} !important;
        }}

        .main .block-container {{
            max-width: 920px;
            padding-top: 1rem;
            padding-bottom: 8rem;
        }}

        /* ── Text Elements ─────────────────────────────── */
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: {colors["text"]} !important;
        }}

        /* ── Hide Streamlit chrome ─────────────────────── */
        #MainMenu, footer, header {{ visibility: hidden; }}
        div[data-testid="stDecoration"] {{ display: none; }}

        /* ── Focus rings ───────────────────────────────── */
        *:focus, *:focus-visible {{
            outline: none !important;
            box-shadow: none !important;
        }}

        /* ── Customizing the native containers ─────────── */
        div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {colors["border"]} !important;
            background: {colors["container_bg"]} !important;
            border-radius: 20px !important;
            box-shadow: 0 4px 16px {colors["shadow_light"]} !important;
            transition: all 0.2s ease;
        }}

        div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: 0 8px 24px {colors["shadow_hover"]} !important;
            border-color: {colors["border_hover"]} !important;
        }}

        /* ── Sidebar styling ───────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: {colors["sidebar_bg"]} !important;
            border-right: 1px solid {colors["border"]} !important;
        }}

        /* ── Chat Input Container ──────────────────────── */
        [data-testid="stChatInput"] {{
            position: fixed;
            bottom: 1rem;
            left: 50%;
            transform: translateX(-50%);
            width: min(840px, 74vw);
            background: {colors["chat_input_bg"]} !important;
            border: 1.5px solid {colors["border"]} !important;
            border-radius: 20px !important;
            box-shadow: 0 8px 30px {colors["shadow_hover"]} !important;
            backdrop-filter: blur(16px) !important;
            z-index: 999;
        }}
        
        /* ── Chat Input Text Area ──────────────────────── */
        [data-testid="stChatInput"] textarea {{
            color: {colors["chat_input_text"]} !important;
            caret-color: {colors["text"]} !important;
        }}

        [data-testid="stChatInput"] textarea::placeholder {{
            color: {colors["chat_input_placeholder"]} !important;
            opacity: 1 !important;
        }}

        /* ── Metric styling ────────────────────────────── */
        [data-testid="stMetricValue"] {{
            color: {colors["accent_primary"]} !important;
            font-weight: 800 !important;
        }}

        [data-testid="stMetricLabel"] {{
            font-weight: 600 !important;
            color: {colors["text_muted"]} !important;
        }}

        /* ── Assistant Message avatar customization ────── */
        div[data-testid="stChatMessage"] {{
            background: {colors["container_bg"]} !important;
            border: 1px solid {colors["border"]} !important;
            border-radius: 20px !important;
            backdrop-filter: blur(10px) !important;
        }}
        
        /* ── Expander ──────────────────────────────────── */
        div[data-testid="stExpander"] {{
            border: 1px solid {colors["border"]} !important;
            border-radius: 14px !important;
            background: {colors["container_bg"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )