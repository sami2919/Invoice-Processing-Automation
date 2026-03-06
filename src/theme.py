"""Executive Dashboard design system for the invoice processing UI."""

COLORS = {
    "primary": "#0F172A",
    "secondary": "#1E293B",
    "accent": "#3B82F6",
    "accent_hover": "#2563EB",
    "background": "#F1F5F9",
    "surface": "#FFFFFF",
    "text": "#0F172A",
    "text_secondary": "#64748B",
    "success": "#059669",
    "success_bg": "#ECFDF5",
    "danger": "#DC2626",
    "danger_bg": "#FEF2F2",
    "warning": "#D97706",
    "warning_bg": "#FFFBEB",
    "muted": "#94A3B8",
    "border": "#E2E8F0",
}


def risk_color(score: int) -> str:
    if score >= 70:
        return COLORS["danger"]
    if score >= 30:
        return COLORS["warning"]
    return COLORS["success"]


def risk_badge(score: int) -> str:
    color = risk_color(score)
    return (
        f'<span style="background:{color}; color:#fff; padding:3px 12px; '
        f'border-radius:20px; font-weight:600; font-size:0.8rem; '
        f'letter-spacing:0.02em;">'
        f'{score}/100</span>'
    )


def decision_color(decision: str) -> str:
    mapping = {
        "approved": COLORS["success"],
        "rejected": COLORS["danger"],
        "escalated": COLORS["warning"],
        "pending_human_review": COLORS["warning"],
        "error": COLORS["danger"],
    }
    return mapping.get(decision, COLORS["muted"])


def decision_badge(decision: str) -> str:
    color = decision_color(decision)
    bg_map = {
        "approved": COLORS["success_bg"],
        "rejected": COLORS["danger_bg"],
        "escalated": COLORS["warning_bg"],
        "pending_human_review": COLORS["warning_bg"],
        "error": COLORS["danger_bg"],
    }
    bg = bg_map.get(decision, "#F1F5F9")
    return (
        f'<span style="background:{bg}; color:{color}; padding:3px 12px; '
        f'border-radius:20px; font-weight:600; font-size:0.8rem; '
        f'text-transform:uppercase; letter-spacing:0.03em;">'
        f'{decision.replace("_", " ")}</span>'
    )


def kpi_card(label: str, value: str, subtitle: str = "", accent: str = "") -> str:
    border_color = accent or COLORS["border"]
    return (
        f'<div style="background:{COLORS["surface"]}; border:1px solid {COLORS["border"]}; '
        f'border-left:4px solid {border_color}; border-radius:8px; padding:20px 24px; '
        f'box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
        f'<div style="color:{COLORS["text_secondary"]}; font-size:0.75rem; font-weight:600; '
        f'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">{label}</div>'
        f'<div style="color:{COLORS["text"]}; font-size:1.75rem; font-weight:700; '
        f'line-height:1.2;">{value}</div>'
        f'{"<div style=&quot;color:" + COLORS["text_secondary"] + "; font-size:0.8rem; margin-top:4px;&quot;>" + subtitle + "</div>" if subtitle else ""}'
        f'</div>'
    )


def page_header() -> str:
    return (
        f'<div style="background:linear-gradient(135deg, {COLORS["primary"]} 0%, '
        f'{COLORS["secondary"]} 60%, {COLORS["accent"]} 100%); '
        f'margin:-1rem -1rem 0 -1rem; padding:28px 32px 24px; border-radius:0 0 0 0;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<div>'
        f'<div style="color:#FFFFFF; font-size:1.5rem; font-weight:700; '
        f'letter-spacing:-0.02em;">Invoice Processing AI</div>'
        f'<div style="color:{COLORS["muted"]}; font-size:0.85rem; margin-top:2px;">'
        f'Automated extraction, validation, fraud detection & approval</div>'
        f'</div>'
        f'<div style="display:flex; gap:8px; align-items:center;">'
        f'<span style="background:rgba(5,150,105,0.2); color:#34D399; padding:4px 12px; '
        f'border-radius:20px; font-size:0.75rem; font-weight:600;">System Online</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    return (
        f'<div style="margin:24px 0 16px;">'
        f'<div style="color:{COLORS["text"]}; font-size:1.1rem; font-weight:700;">{title}</div>'
        f'{"<div style=&quot;color:" + COLORS["text_secondary"] + "; font-size:0.82rem;&quot;>" + subtitle + "</div>" if subtitle else ""}'
        f'</div>'
    )


EXECUTIVE_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Reset & Global ── */
    html, body, [class*="css"], .stMarkdown, .stText,
    p, label, h1, h2, h3, h4, h5, h6, div, input, button, select, textarea {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}
    /* Apply Inter to spans but restore Material Symbols for Streamlit icons */
    span {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    [data-testid="stExpanderToggleIcon"],
    [class*="material-symbols"],
    [class*="material-icons"] {{
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    }}

    .stApp {{
        background-color: {COLORS["background"]};
    }}

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* ── Top padding fix since we hid the header ── */
    .stApp > div:first-child {{
        padding-top: 0 !important;
    }}
    .block-container {{
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }}

    /* ── Main content: Streamlit widgets text ── */
    /* Target only Streamlit widget wrappers, NOT custom HTML with inline styles */
    .stCheckbox label span,
    .stCheckbox p {{
        color: {COLORS["text"]} !important;
    }}
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stFileUploader label,
    .stToggle label span {{
        color: {COLORS["text"]} !important;
    }}

    /* Markdown text in main content (not inside custom HTML) */
    .stMainBlockContainer .stMarkdown p {{
        color: {COLORS["text"]};
    }}

    /* Expander summary + body text */
    div[data-testid="stExpander"] summary span {{
        color: {COLORS["text"]} !important;
    }}
    div[data-testid="stExpander"] .stMarkdown p {{
        color: {COLORS["text"]} !important;
    }}

    /* Chart axis labels and ticks */
    .vega-embed text {{
        fill: {COLORS["text"]} !important;
    }}
    .vega-embed {{
        background: transparent !important;
    }}
    .vega-embed .chart-wrapper {{
        background: transparent !important;
    }}
    div[data-testid="stVegaLiteChart"] {{
        background: {COLORS["surface"]} !important;
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 16px;
    }}
    .vega-embed svg {{
        background: transparent !important;
    }}
    .vega-embed .vega-bindings {{
        background: transparent !important;
    }}

    /* Captions in main content */
    .stMainBlockContainer .stCaption p {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {COLORS["primary"]};
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 2rem;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption p {{
        color: #94A3B8 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #F1F5F9 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.08) !important;
    }}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSlider span {{
        color: #94A3B8 !important;
    }}

    /* ── Sidebar toggle button ── */
    section[data-testid="stSidebar"] .stToggle label span {{
        color: #CBD5E1 !important;
    }}

    /* ── Sidebar buttons ── */
    section[data-testid="stSidebar"] .stButton > button {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #CBD5E1 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,0.12) !important;
    }}

    /* ── Tabs ── */
    div[data-testid="stTabs"] {{
        background: {COLORS["surface"]};
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid {COLORS["border"]};
        margin-bottom: 20px;
    }}
    button[data-baseweb="tab"] {{
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: {COLORS["text_secondary"]} !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        border-bottom: none !important;
        transition: all 0.15s ease !important;
    }}
    button[data-baseweb="tab"]:hover {{
        background: {COLORS["background"]} !important;
        color: {COLORS["text"]} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: {COLORS["accent"]} !important;
        color: #FFFFFF !important;
        border-bottom: none !important;
        box-shadow: 0 1px 3px rgba(59,130,246,0.3) !important;
    }}
    /* Hide the default tab underline */
    div[data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    div[data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* ── Primary Buttons ── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {COLORS["accent"]} !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 28px !important;
        font-size: 0.85rem !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 3px rgba(59,130,246,0.2) !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: {COLORS["accent_hover"]} !important;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3) !important;
        transform: translateY(-1px);
    }}

    /* ── Secondary Buttons ── */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {{
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        color: {COLORS["text"]} !important;
        background: {COLORS["surface"]} !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {{
        background: {COLORS["background"]} !important;
        border-color: {COLORS["muted"]} !important;
    }}

    /* ── Native metric cards override ── */
    div[data-testid="stMetric"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {{
        border-radius: 8px !important;
        border-color: {COLORS["border"]} !important;
        font-size: 0.85rem !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {COLORS["accent"]} !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
    }}

    /* ── File uploader ── */
    div[data-testid="stFileUploader"] {{
        border: 2px dashed {COLORS["border"]} !important;
        border-radius: 12px !important;
        padding: 12px !important;
        transition: border-color 0.2s !important;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: {COLORS["accent"]} !important;
    }}

    /* ── Expanders ── */
    div[data-testid="stExpander"] {{
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 10px !important;
        background: {COLORS["surface"]} !important;
        overflow: hidden;
    }}
    div[data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        background: {COLORS["surface"]} !important;
        color: {COLORS["text"]} !important;
    }}
    div[data-testid="stExpander"] details {{
        background: {COLORS["surface"]} !important;
    }}

    /* ── Containers with border ── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 10px !important;
        background: {COLORS["surface"]} !important;
    }}

    /* ── Dataframes ── */
    .stDataFrame {{
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 10px !important;
        overflow: hidden;
    }}

    /* ── Alerts ── */
    .stAlert {{
        border-radius: 10px !important;
        font-size: 0.85rem !important;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {COLORS["accent"]}, {COLORS["accent_hover"]}) !important;
        border-radius: 8px !important;
    }}

    /* ── Dividers ── */
    hr {{
        border-color: {COLORS["border"]} !important;
        opacity: 0.6;
    }}

    /* ── Download button ── */
    .stDownloadButton > button {{
        border-radius: 8px !important;
        font-size: 0.82rem !important;
    }}

    /* ── Spinner ── */
    .stSpinner > div {{
        border-top-color: {COLORS["accent"]} !important;
    }}

    /* ── Subheader styling ── */
    .stMarkdown h3 {{
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        color: {COLORS["text"]} !important;
    }}

    /* ── Caption styling ── */
    .stCaption p {{
        font-size: 0.78rem !important;
    }}
</style>
"""
