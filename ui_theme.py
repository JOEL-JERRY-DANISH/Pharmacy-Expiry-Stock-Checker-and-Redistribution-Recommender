"""
ui_theme.py — Centralized UI Design System & Styling Foundation

Provides design tokens (colors, typography, spacing), centralized CSS injection,
and reusable Streamlit UI helper components for the Pharmacy Expiry Stock Checker.
"""

from typing import Optional, Dict, Any
import streamlit as st

# =============================================================================
# 1. DESIGN TOKENS
# =============================================================================

COLORS = {
    # Brand & Primary Palette
    "primary": "#1E3A8A",          # Deep Blue (Primary brand)
    "primary_hover": "#1D4ED8",
    "primary_light": "#EFF6FF",
    "secondary": "#0F766E",        # Clinical Teal
    "secondary_light": "#F0FDFA",

    # Status / Urgency Palette
    "critical": "#DC2626",         # Red-600 (0–7 days to expiry)
    "critical_bg": "#FEF2F2",      # Red-50
    "critical_border": "#FCA5A5",  # Red-300
    "critical_text": "#991B1B",    # Red-800

    "warning": "#D97706",          # Amber-600 (8–30 days to expiry)
    "warning_bg": "#FFFBEB",       # Amber-50
    "warning_border": "#FCD34D",   # Amber-300
    "warning_text": "#92400E",     # Amber-800

    "safe": "#059669",             # Emerald-600 (Safe stock)
    "safe_bg": "#ECFDF5",          # Emerald-50
    "safe_border": "#6EE7B7",      # Emerald-300
    "safe_text": "#065F46",        # Emerald-800

    "info": "#2563EB",             # Blue-600 (Review / Info)
    "info_bg": "#EFF6FF",          # Blue-50
    "info_border": "#93C5FD",      # Blue-300
    "info_text": "#1E40AF",        # Blue-800

    # Neutral Palette
    "neutral_50": "#F8FAFC",       # Slate-50 (Page background)
    "neutral_100": "#F1F5F9",      # Slate-100 (Subtle background)
    "neutral_200": "#E2E8F0",      # Slate-200 (Dividers, light borders)
    "neutral_300": "#CBD5E1",      # Slate-300 (Input borders)
    "neutral_400": "#94A3B8",      # Slate-400 (Icons, disabled)
    "neutral_500": "#64748B",      # Slate-500 (Captions, secondary text)
    "neutral_600": "#475569",      # Slate-600 (Labels, body text)
    "neutral_700": "#334155",      # Slate-700 (Subheaders)
    "neutral_800": "#1E293B",      # Slate-800 (Headings)
    "neutral_900": "#0F172A",      # Slate-900 (Main text)
    "white": "#FFFFFF",
}

TYPOGRAPHY = {
    "font_family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "font_size_page_title": "1.75rem",
    "font_size_section": "1.20rem",
    "font_size_subsection": "1.00rem",
    "font_size_body": "0.92rem",
    "font_size_caption": "0.80rem",
}

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
}

# Urgency metadata mapping
URGENCY_THEME: Dict[str, Dict[str, str]] = {
    "critical": {
        "color": COLORS["critical"],
        "bg": COLORS["critical_bg"],
        "border": COLORS["critical_border"],
        "text": COLORS["critical_text"],
        "icon": "🔴",
        "label": "URGENT",
        "badge_class": "pill-critical",
    },
    "near-expiry": {
        "color": COLORS["warning"],
        "bg": COLORS["warning_bg"],
        "border": COLORS["warning_border"],
        "text": COLORS["warning_text"],
        "icon": "🟠",
        "label": "ACT SOON",
        "badge_class": "pill-warning",
    },
    "watch": {
        "color": "#D97706",
        "bg": "#FEF3C7",
        "border": "#FDE68A",
        "text": "#92400E",
        "icon": "🟡",
        "label": "WATCH",
        "badge_class": "pill-warning",
    },
    "review": {
        "color": COLORS["info"],
        "bg": COLORS["info_bg"],
        "border": COLORS["info_border"],
        "text": COLORS["info_text"],
        "icon": "🔵",
        "label": "REVIEW",
        "badge_class": "pill-info",
    },
    "safe": {
        "color": COLORS["safe"],
        "bg": COLORS["safe_bg"],
        "border": COLORS["safe_border"],
        "text": COLORS["safe_text"],
        "icon": "🟢",
        "label": "SAFE",
        "badge_class": "pill-safe",
    },
    "expired": {
        "color": COLORS["critical"],
        "bg": COLORS["critical_bg"],
        "border": COLORS["critical_border"],
        "text": COLORS["critical_text"],
        "icon": "❌",
        "label": "EXPIRED",
        "badge_class": "pill-critical",
    },
    "unavailable": {
        "color": COLORS["neutral_500"],
        "bg": COLORS["neutral_100"],
        "border": COLORS["neutral_300"],
        "text": COLORS["neutral_700"],
        "icon": "⚪",
        "label": "UNAVAILABLE",
        "badge_class": "pill-neutral",
    },
    "error": {
        "color": COLORS["critical"],
        "bg": COLORS["critical_bg"],
        "border": COLORS["critical_border"],
        "text": COLORS["critical_text"],
        "icon": "⚠️",
        "label": "ERROR",
        "badge_class": "pill-critical",
    },
}

# =============================================================================
# 2. CENTRALIZED CSS
# =============================================================================

CUSTOM_CSS = f"""
<style>
/* Global Page Layout & Typography */
html, body, [class*="css"] {{
    font-family: {TYPOGRAPHY['font_family']};
    color: {COLORS['neutral_900']};
}}

.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1280px !important;
}}

/* Desktop and Tablet Responsiveness */
@media (max-width: 1200px) {{
    .block-container {{
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }}
}}

@media (max-width: 992px) {{
    .block-container {{
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }}
    div[data-testid="column"] {{
        min-width: 100% !important;
        margin-bottom: 0.5rem !important;
    }}
}}

@media (max-width: 768px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }}
}}

/* Accessibility: Enhanced Focus Indicators */
button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible {{
    outline: 2px solid #1E3A8A !important;
    outline-offset: 2px !important;
}}

/* Minimum touch/click target size for interactive controls */
div.stButton > button {{
    min-height: 38px !important;
}}

/* Prevent text clipping and ensure high contrast readability */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {{
    color: #1E293B !important;
    font-size: 0.92rem !important;
    line-height: 1.55 !important;
}}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {{
    background-color: {COLORS['neutral_50']} !important;
    border-right: 1px solid {COLORS['neutral_200']} !important;
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}}

.sidebar-section-title {{
    font-size: 0.80rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLORS['neutral_600']};
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.sidebar-user-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}}

.sidebar-user-name {{
    font-weight: 600;
    font-size: 0.95rem;
    color: {COLORS['neutral_900']};
}}

.sidebar-user-branch {{
    font-size: 0.85rem;
    color: {COLORS['neutral_600']};
    margin-top: 2px;
}}

.sidebar-user-role {{
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    background: {COLORS['primary_light']};
    color: {COLORS['info_text']};
    border: 1px solid #BFDBFE;
    margin-top: 6px;
}}

/* Section Headers */
.section-header-box {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid {COLORS['neutral_200']};
    padding-bottom: 6px;
    margin-top: 1.25rem;
    margin-bottom: 1.0rem;
}}

.section-title {{
    font-size: {TYPOGRAPHY['font_size_section']};
    font-weight: 700;
    color: {COLORS['neutral_800']};
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-subtitle {{
    font-size: {TYPOGRAPHY['font_size_caption']};
    color: {COLORS['neutral_500']};
    margin-top: 2px;
    margin-bottom: 0;
}}

/* Consistent Metric Card Styling */
div[data-testid="stMetric"] {{
    background-color: {COLORS['white']} !important;
    border: 1px solid {COLORS['neutral_200']} !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}}

div[data-testid="stMetricLabel"] {{
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: {COLORS['neutral_600']} !important;
}}

div[data-testid="stMetricValue"] {{
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: {COLORS['neutral_900']} !important;
}}

/* Expander Cards */
div[data-testid="stExpander"] {{
    background-color: {COLORS['white']} !important;
    border: 1px solid {COLORS['neutral_200']} !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 12px !important;
}}

/* Streamlit Buttons */
div.stButton > button {{
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 0.95rem !important;
    transition: all 0.15s ease-in-out !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}}

div.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08) !important;
}}

div.stButton > button[kind="primary"] {{
    background-color: #1E3A8A !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
}}

div.stButton > button[kind="primary"]:hover {{
    background-color: #1D4ED8 !important;
    box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.25) !important;
}}

div.stButton > button[kind="secondary"] {{
    background-color: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    font-weight: 500 !important;
}}

div.stButton > button[kind="secondary"]:hover {{
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}}

/* Disabled button state */
div.stButton > button:disabled,
div.stButton > button[disabled] {{
    opacity: 0.55 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
    background-color: #F1F5F9 !important;
    color: #94A3B8 !important;
    border: 1px solid #E2E8F0 !important;
}}

/* Form Container & Grouping */
div[data-testid="stForm"] {{
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 16px !important;
}}

/* Form inputs & Selectbox styling */
div[data-baseweb="input"] input,
div[data-baseweb="select"] {{
    border-radius: 6px !important;
}}

div[data-baseweb="input"] > div {{
    border-color: #CBD5E1 !important;
    border-radius: 6px !important;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
}}

div[data-baseweb="input"] > div:focus-within {{
    border-color: #1E3A8A !important;
    box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.15) !important;
}}

/* Form Labels & Help text */
div[data-testid="stWidgetLabel"] label {{
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}}

div[data-testid="stWidgetLabel"] small,
div[data-testid="stCaptionContainer"] {{
    color: #64748B !important;
}}

/* Native Streamlit Alert Banners (st.error, st.warning, st.info, st.success) */
div[data-testid="stAlert"] {{
    border-radius: 8px !important;
    border-width: 1px !important;
    border-style: solid !important;
    padding: 12px 16px !important;
    font-size: 0.90rem !important;
    line-height: 1.5 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
}}

div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
    font-size: 0.90rem !important;
    line-height: 1.5 !important;
}}

/* Spinner & Processing Feedback */
div[data-testid="stSpinner"] {{
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    color: #1E3A8A !important;
    font-weight: 500 !important;
    padding: 8px 0 !important;
}}

/* Clean Pill Badges */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.80rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.025em;
    line-height: 1.25;
}}

.pill-critical {{
    background-color: {COLORS['critical_bg']};
    color: {COLORS['critical_text']};
    border: 1px solid {COLORS['critical_border']};
}}

.pill-warning {{
    background-color: {COLORS['warning_bg']};
    color: {COLORS['warning_text']};
    border: 1px solid {COLORS['warning_border']};
}}

.pill-info {{
    background-color: {COLORS['info_bg']};
    color: {COLORS['info_text']};
    border: 1px solid {COLORS['info_border']};
}}

.pill-safe {{
    background-color: {COLORS['safe_bg']};
    color: {COLORS['safe_text']};
    border: 1px solid {COLORS['safe_border']};
}}

.pill-neutral {{
    background-color: {COLORS['neutral_100']};
    color: {COLORS['neutral_700']};
    border: 1px solid {COLORS['neutral_200']};
}}

/* Header Banner Container */
.page-hero {{
    background: linear-gradient(135deg, {COLORS['neutral_50']} 0%, {COLORS['white']} 100%);
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}}

.page-hero-top {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
}}

.page-hero-title {{
    font-size: {TYPOGRAPHY['font_size_page_title']};
    font-weight: 700;
    color: {COLORS['neutral_900']};
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.page-hero-subtitle {{
    font-size: {TYPOGRAPHY['font_size_body']};
    color: {COLORS['neutral_600']};
    margin-top: 4px;
    margin-bottom: 0;
}}

.page-hero-context {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: {COLORS['neutral_100']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 500;
    color: {COLORS['neutral_700']};
}}

/* Recommendation Item Cards */
.rec-card-container {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}}

/* Clean Dataframe Container */
div[data-testid="stDataFrame"] {{
    border-radius: 8px;
    overflow-x: auto !important;
    border: 1px solid {COLORS['neutral_200']};
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}}

/* Table Typography & Row Padding for readability */
div[data-testid="stDataFrame"] div[role="grid"] {{
    font-size: 0.88rem !important;
    color: {COLORS['neutral_900']} !important;
}}

/* Empty State Container */
.empty-state-box {{
    background: {COLORS['neutral_50']};
    border: 1.5px dashed {COLORS['neutral_300']};
    border-radius: 10px;
    padding: 30px 20px;
    text-align: center;
    margin: 12px 0 16px 0;
}}

.empty-state-icon {{
    font-size: 2rem;
    margin-bottom: 8px;
    line-height: 1;
}}

.empty-state-title {{
    font-size: 0.98rem;
    font-weight: 600;
    color: {COLORS['neutral_800']};
    margin-bottom: 4px;
}}

.empty-state-desc {{
    font-size: 0.84rem;
    color: {COLORS['neutral_500']};
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.4;
}}

/* Tab Styling */
div[data-baseweb="tab-list"] {{
    gap: 8px;
    margin-bottom: 16px;
    border-bottom: 1px solid {COLORS['neutral_200']};
}}

div[data-baseweb="tab"] {{
    border-radius: 6px 6px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.90rem !important;
    padding: 8px 16px !important;
}}

/* Advisory ML Card */
.advisory-ml-box {{
    background: {COLORS['neutral_50']};
    border: 1px solid {COLORS['neutral_200']};
    border-left: 4px solid {COLORS['primary']};
    border-radius: 6px;
    padding: 10px 14px;
    margin: 8px 0;
}}

/* Risk Framework Grid */
.risk-framework-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
}}
</style>
"""

# =============================================================================
# 3. REUSABLE UI HELPERS
# =============================================================================

def inject_custom_css() -> None:
    """Inject centralized CSS into Streamlit to apply consistent design system styling."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_empty_state(title: str, message: str, icon: str = "📭") -> None:
    """Render a clean, stylized empty state container."""
    html = f"""
    <div class="empty-state-box">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-desc">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_urgency_theme(urgency: str) -> Dict[str, str]:
    """Return styling metadata (colors, labels, icons) for a given urgency level."""
    normalized = str(urgency).strip().lower()
    return URGENCY_THEME.get(normalized, URGENCY_THEME["review"])


def format_urgency_badge(urgency: str) -> str:
    """
    Return a standardized, scannable text badge combining an icon and text label.
    Guarantees status is never communicated solely through color.
    """
    theme = get_urgency_theme(urgency)
    return f"{theme['icon']} {theme['label']}"


def format_shelf_life(days: int) -> str:
    """Format days to expiry into a scannable, dual-encoded string (icon + text)."""
    if days < 0:
        return f"❌ Expired ({abs(days)}d ago)"
    elif days <= 7:
        return f"🔴 Critical ({days}d remaining)"
    elif days <= 30:
        return f"🟠 Act Soon ({days}d remaining)"
    elif days <= 60:
        return f"🟡 Watch ({days}d remaining)"
    else:
        return f"🟢 Safe ({days}d remaining)"


def format_risk_badge(level: str, text: Optional[str] = None) -> str:
    """
    Format a semantic risk badge ensuring text is always preserved alongside iconography.
    Semantic tiers: 'safe', 'warning', 'critical', 'unavailable'/'error'.
    """
    norm = str(level).strip().lower()
    if norm in ("safe", "low"):
        theme = URGENCY_THEME["safe"]
        lbl = text or "SAFE / LOW RISK"
    elif norm in ("warning", "near-expiry", "medium", "moderate"):
        theme = URGENCY_THEME["near-expiry"]
        lbl = text or "WARNING / ACT SOON"
    elif norm in ("watch",):
        theme = URGENCY_THEME["watch"]
        lbl = text or "WATCH (31–60d)"
    elif norm in ("critical", "high", "urgent", "expired"):
        theme = URGENCY_THEME["critical"]
        lbl = text or ("EXPIRED" if norm == "expired" else "CRITICAL / HIGH RISK")
    else:
        theme = URGENCY_THEME.get("unavailable", URGENCY_THEME["review"])
        lbl = text or "UNAVAILABLE / MANUAL REVIEW"
    return f"{theme['icon']} {lbl}"


def render_page_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    context_info: Optional[str] = None,
) -> None:
    """Render a standardized, polished page hero header with optional contextual info."""
    icon_html = f"<span>{icon}</span> " if icon else ""
    subtitle_html = f"<p class='page-hero-subtitle'>{subtitle}</p>" if subtitle else ""
    context_html = f"<div class='page-hero-context'>{context_info}</div>" if context_info else ""
    html = f"""
    <div class="page-hero">
        <div class="page-hero-top">
            <div>
                <h1 class="page-hero-title">{icon_html}{title}</h1>
                {subtitle_html}
            </div>
            {context_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    badge_text: Optional[str] = None,
) -> None:
    """Render a consistent section header with divider, optional subtitle, and badge."""
    icon_html = f"<span>{icon}</span> " if icon else ""
    sub_html = f"<p class='section-subtitle'>{subtitle}</p>" if subtitle else ""
    badge_html = f"<span class='status-pill pill-neutral'>{badge_text}</span>" if badge_text else ""
    html = f"""
    <div class="section-header-box">
        <div>
            <h2 class="section-title">{icon_html}{title}</h2>
            {sub_html}
        </div>
        {badge_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_user_badge(name: str, branch: str) -> None:
    """Render a clean user profile badge for the sidebar."""
    is_admin = branch == "All branches"
    role_icon = "🛡️" if is_admin else "💊"
    role_label = "Admin" if is_admin else "Pharmacist"
    html = f"""
    <div class="sidebar-user-card">
        <div class="sidebar-user-name">👤 {name}</div>
        <div class="sidebar-user-branch">📍 {branch}</div>
        <span class="sidebar-user-role">{role_icon} {role_label}</span>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)


def logout_user() -> None:
    """Centralized user logout and session cleanup handler."""
    st.session_state.logged_in    = False
    st.session_state.current_user = None
    st.session_state.login_time   = None
    st.session_state.pop("confirmed", None)
    st.session_state.pop("overridden", None)
    st.session_state.pop("barcode_registry", None)
    st.rerun()


def render_sidebar_account(user: Optional[Dict[str, Any]]) -> None:
    """Render sidebar account card and logout action."""
    if not user:
        return
    st.sidebar.markdown("<div class='sidebar-section-title'>👤 Account</div>", unsafe_allow_html=True)
    render_user_badge(user.get("name", ""), user.get("branch", ""))
    if st.sidebar.button("🚪 Log Out", key="sidebar_logout_btn", use_container_width=True):
        logout_user()


def render_sidebar_nav(current_page: str, user: Optional[Dict[str, Any]] = None) -> None:
    """Render clear sidebar navigation links."""
    st.sidebar.markdown("<div class='sidebar-section-title'>🧭 Navigation</div>", unsafe_allow_html=True)
    is_admin = bool(user and user.get("branch") == "All branches")

    # Native page navigation links
    if current_page == "recommender":
        st.sidebar.markdown("👉 **💊 Expiry Stock Checker** *(Active)*")
        if is_admin:
            try:
                st.sidebar.page_link("pages/admin_dashboard.py", label="Admin Dashboard", icon="📊")
            except Exception:
                pass
    elif current_page == "admin_dashboard":
        try:
            st.sidebar.page_link("app.py", label="Expiry Stock Checker", icon="💊")
        except Exception:
            pass
        st.sidebar.markdown("👉 **📊 Admin Dashboard** *(Active)*")


def render_urgency_pill(urgency: str, custom_text: Optional[str] = None) -> str:
    """Return HTML for a status badge pill matching the design system."""
    meta = get_urgency_theme(urgency)
    label = custom_text or meta["label"]
    icon = meta["icon"]
    badge_cls = meta["badge_class"]
    return f"<span class='status-pill {badge_cls}'>{icon} {label}</span>"
