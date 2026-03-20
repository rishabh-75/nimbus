"""
ui/theme.py
───────────
NIMBUS Emerald Slate v5 — Design System
Colour tokens, platform font detection, QSS master stylesheet.

Every value here is locked and matches the roadmap spec exactly.
Do not substitute, simplify, or "modernise" any of these values.
"""

from __future__ import annotations

import platform

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR TOKENS
# ══════════════════════════════════════════════════════════════════════════════

BG = "#080C10"  # main background
SURFACE = "#0D1117"  # card / panel background
S2 = "#111827"  # slightly lighter surface (KPI tile bg, inputs)
S3 = "#162032"  # hover / selected row
BORDER = "#1E2937"  # all borders and dividers

EM = "#10B981"  # emerald — primary accent, NIMBUS brand, BULLISH, LIVE
EM_DK = "#059669"  # darker emerald for pressed states
RED = "#EF4444"  # bearish, error, AVOID, EXIT signals
GOLD = "#F59E0B"  # warning, HALF size, LATE W%R phase, STALE state
VIOLET = "#8B5CF6"  # max pain levels, secondary accent
WHITE = "#E2E8F0"  # primary text
MUTED = "#64748B"  # secondary text, labels, metadata
GREEN = "#22C55E"  # FULL size, positive delta, strong viability

UP = "#26A69A"  # bullish candles
DOWN = "#EF5350"  # bearish candles
BB_LINE = "#BEBEC8"  # Bollinger Band lines (neutral grey, not green)
BB_FILL = "rgba(190,190,210,0.06)"  # BB channel fill
BLUE = "#60A5FA"  # WATCH/QUARTER sizing tier, informational

# ══════════════════════════════════════════════════════════════════════════════
# BADGE COLOUR PAIRINGS  (bg, text, border)
# ══════════════════════════════════════════════════════════════════════════════

BADGE_STYLES = {
    "BULLISH": {"bg": "#0D2B20", "text": EM, "border": EM},
    "BEARISH": {"bg": "#2B0D0D", "text": RED, "border": RED},
    "FRESH": {"bg": "#0D2B20", "text": EM, "border": EM},
    "DEVELOPING": {"bg": "#1A2010", "text": "#A3E635", "border": "#A3E635"},
    "LATE": {"bg": "#1F1A0D", "text": GOLD, "border": GOLD},
    "NO CROSS": {"bg": S2, "text": MUTED, "border": BORDER},
    "SQUEEZE": {"bg": "#1A1040", "text": VIOLET, "border": VIOLET},
    "EXPANDED": {"bg": "#200D1A", "text": "#F472B6", "border": "#F472B6"},
    "PINNING": {"bg": "#200D1A", "text": "#F472B6", "border": "#F472B6"},
    "WATCH": {"bg": "#0D1A2B", "text": BLUE, "border": BLUE},
    "NEUTRAL": {"bg": S2, "text": MUTED, "border": BORDER},
}

# Viability sizing badge pairings
SIZING_STYLES = {
    "FULL": {"bg": "#0A1F15", "text": GREEN, "border": GREEN},
    "HALF": {"bg": "#1F1A0A", "text": GOLD, "border": GOLD},
    "SKIP": {"bg": "#1F0A0A", "text": RED, "border": RED},
    "AVOID": {"bg": "#1F0A0A", "text": RED, "border": RED},
    "WATCH": {"bg": "#0D1A2B", "text": BLUE, "border": BLUE},
    "QTR": {"bg": "#0D1A2B", "text": BLUE, "border": BLUE},
    "ZERO": {"bg": "#1F0A0A", "text": RED, "border": RED},
}

# KPI tile value colours by state
KPI_COLORS = {
    "positive": EM,
    "negative": RED,
    "neutral": GOLD,
    "quarter": BLUE,
    "pinning": VIOLET,
    "na": MUTED,
}

# ══════════════════════════════════════════════════════════════════════════════
# FONTS — platform detection
# ══════════════════════════════════════════════════════════════════════════════

# if platform.system() == "Darwin":
#     FONT_UI = ""
#     FONT_DISPLAY = "SF Pro Display"
# else:
#     FONT_UI = "Inter, Arial, sans-serif"
#     FONT_DISPLAY = "Inter, Arial, sans-serif"

# # JetBrains Mono is bundled cross-platform (assets/) — no fallback needed
# FONT_MONO = "JetBrains Mono"

FONT_UI = "JetBrains Mono"
FONT_DISPLAY = "JetBrains Mono"
FONT_MONO = "JetBrains Mono"


# ══════════════════════════════════════════════════════════════════════════════
# QSS MASTER STYLESHEET
# ══════════════════════════════════════════════════════════════════════════════

QSS = f"""
/* ── Base ──────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG};
    color: {WHITE};
    font-family: "{FONT_UI}";
    font-size: 13px;
}}
QMainWindow {{
    background-color: {BG};
}}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {SURFACE};
}}
QTabBar::tab {{
    background: {S2};
    color: {MUTED};
    padding: 10px 28px;
    border-bottom: 2px solid transparent;
    font-size: 11pt;
    min-width: 100px;
}}
QTabBar::tab:selected {{
    color: {WHITE};
    border-bottom: 2px solid {EM};
    background: {BG};
}}
QTabBar::tab:hover {{
    color: {WHITE};
    background: {S3};
}}

/* ── Buttons ───────────────────────────────────────────────────────────── */
QPushButton {{
    background: {S2};
    color: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 14px;
    font-size: 11pt;
}}
QPushButton:hover {{
    background: {S3};
    border-color: {EM};
}}
QPushButton:pressed {{
    background: #0D2B20;
}}
QPushButton#primary {{
    background: {EM};
    color: {BG};
    border: none;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton#primary:hover {{
    background: {EM_DK};
}}
QPushButton#danger {{
    background: transparent;
    color: {RED};
    border: 1px solid {RED};
}}

/* ── Inputs ────────────────────────────────────────────────────────────── */
QLineEdit, QComboBox {{
    background: {S2};
    color: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 11pt;
}}
QLineEdit:focus {{
    border-color: {EM};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
    subcontrol-origin: padding;
    subcontrol-position: center right;
}}
QComboBox::down-arrow {{
    width: 8px;
    height: 8px;
    image: none;
}}
QComboBox QAbstractItemView {{
    background: {S2};
    color: {WHITE};
    border: 1px solid {BORDER};
    selection-background-color: {S3};
    selection-color: {WHITE};
}}

/* ── Spin Box ──────────────────────────────────────────────────────────── */
QSpinBox {{
    background: {S2};
    color: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11pt;
    font-family: "{FONT_MONO}";
}}
QSpinBox:focus {{
    border-color: {EM};
}}
QSpinBox::up-button, QSpinBox::down-button {{
    background: {BORDER};
    border: none;
    width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background: {S3};
}}

/* ── Sliders ───────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {BORDER};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {EM};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {EM_DK};
    border-radius: 2px;
}}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {SURFACE};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {EM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* ── Tables ────────────────────────────────────────────────────────────── */
QTableView {{
    background: {SURFACE};
    color: {WHITE};
    gridline-color: {BORDER};
    selection-background-color: {S3};
    selection-color: {WHITE};
    font-family: "{FONT_MONO}";
    font-size: 12px;
    border: none;
}}
QTableView::item {{
    padding: 6px 8px;
    border-bottom: 1px solid {BORDER};
}}
QHeaderView::section {{
    background: #0A1520;
    color: {EM};
    font-size: 10pt;
    border: 1px solid {BORDER};
    padding: 6px 10px;
}}

/* ── Status bar ────────────────────────────────────────────────────────── */
QStatusBar {{
    background: #0A1520;
    color: {MUTED};
    border-top: 1px solid {BORDER};
    font-family: "{FONT_MONO}";
    font-size: 11px;
}}

/* ── Labels ────────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {WHITE};
}}
QLabel#muted {{
    color: {MUTED};
}}
QLabel#section-header {{
    color: {MUTED};
    font-size: 9px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    font-family: "{FONT_UI}";
}}
QLabel#nimbus-logo {{
    font-family: "{FONT_MONO}";
    font-size: 18px;
    font-weight: 800;
    color: {EM};
    letter-spacing: 3px;
}}

/* ── Frames (cards, panels) ────────────────────────────────────────────── */
QFrame#card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#kpi-tile {{
    background: {S2};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* ── Progress bar ──────────────────────────────────────────────────────── */
QProgressBar {{
    background: {BORDER};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    font-size: 9px;
    color: {MUTED};
}}
QProgressBar::chunk {{
    background: {EM};
    border-radius: 3px;
}}

/* ── Tooltips ──────────────────────────────────────────────────────────── */
QToolTip {{
    background: {S2};
    color: {WHITE};
    border: 1px solid {BORDER};
    padding: 4px 8px;
    font-size: 11px;
}}

/* ── Checkbox ──────────────────────────────────────────────────────────── */
QCheckBox {{
    color: {WHITE};
    spacing: 6px;
    font-size: 11pt;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {S2};
}}
QCheckBox::indicator:checked {{
    background: {EM};
    border-color: {EM};
}}
QCheckBox::indicator:disabled {{
    background: {BORDER};
    border-color: {MUTED};
}}
QCheckBox:disabled {{
    color: {MUTED};
}}
"""


def score_color(score: int) -> str:
    """Return the colour hex for a viability score value."""
    if score >= 70:
        return GREEN
    elif score >= 50:
        return GOLD
    elif score >= 30:
        return BLUE
    else:
        return RED


def wr_color(wr_value: float) -> str:
    """Return the colour hex for a Williams %R value."""
    if wr_value >= -20:
        return EM
    elif wr_value >= -50:
        return GOLD
    else:
        return RED
