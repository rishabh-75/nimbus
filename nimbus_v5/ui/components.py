"""
ui/components.py
────────────────
Production-grade NIMBUS UI components.
Designed for institutional trading terminal aesthetics:
  - Information density: every pixel carries signal
  - Typography hierarchy: monospace numbers, sans-serif labels
  - Color encodes state: green=go, red=stop, gold=caution
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QGridLayout,
)
from PyQt6.QtGui import QFont

from ui.theme import (
    BG,
    SURFACE,
    S2,
    S3,
    BORDER,
    EM,
    RED,
    GOLD,
    MUTED,
    WHITE,
    GREEN,
    BLUE,
    VIOLET,
    FONT_MONO,
    FONT_UI,
    BADGE_STYLES,
    SIZING_STYLES,
)


# ══════════════════════════════════════════════════════════════════════════════
# KPI TILE — value + delta subtitle
# ══════════════════════════════════════════════════════════════════════════════


class KPITile(QFrame):
    """
    Compact metric tile: value (large mono) + delta (small colored).
    Matches Streamlit st.metric() output.
    """

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._oid = f"kpi-{id(self)}"
        self.setObjectName(self._oid)
        self._set_border(BORDER)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(1)

        # Label (top, small, muted, uppercase)
        self._name = QLabel(label.upper())
        self._name.setFont(QFont(FONT_UI, 6))
        self._name.setStyleSheet(f"color: {MUTED}; letter-spacing: 0.8px;")
        lay.addWidget(self._name)

        # Value (large, monospace, bold)
        self._value = QLabel("--")
        self._value.setFont(QFont(FONT_MONO, 10))
        self._value.setStyleSheet(f"color: {WHITE}; font-weight: bold;")
        lay.addWidget(self._value)

        # Delta / subtitle (small, colored)
        self._delta = QLabel("")
        self._delta.setFont(QFont(FONT_MONO, 7))
        self._delta.setStyleSheet(f"color: {MUTED};")
        lay.addWidget(self._delta)

    def set_value(self, text: str, color: str = WHITE, delta: str = ""):
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color}; font-weight: bold;")
        if delta:
            self._delta.setText(delta)
            self._delta.setVisible(True)
        else:
            self._delta.setVisible(False)

    def set_title(self, text: str):
        """Rename the tile label at runtime (e.g. PCR OI → VSR for ETFs)."""
        self._name.setText(text.upper())

    def set_delta(self, text: str, color: str = MUTED):
        self._delta.setText(text)
        self._delta.setStyleSheet(f"color: {color};")
        self._delta.setVisible(bool(text))

    def set_stale(self, stale: bool):
        self._set_border(GOLD if stale else BORDER)

    def _set_border(self, color: str):
        self.setStyleSheet(
            f"QFrame#{self._oid} {{ background: {S2}; "
            f"border: 1px solid {color}; border-radius: 6px; }}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# BADGE — coloured state pill
# ══════════════════════════════════════════════════════════════════════════════


class Badge(QLabel):

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setFont(QFont(FONT_UI, 7))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(18)
        self.setMaximumWidth(220)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.apply_style("NEUTRAL")

    def apply_style(self, key: str):
        styles = {**BADGE_STYLES, **SIZING_STYLES}
        s = styles.get(key, BADGE_STYLES.get("NEUTRAL", {}))
        bg, tc, bc = s.get("bg", S2), s.get("text", MUTED), s.get("border", BORDER)
        self.setStyleSheet(
            f"background: {bg}; color: {tc}; border: 1px solid {bc}; "
            f"border-radius: 3px; padding: 1px 6px; font-weight: 600;"
        )

    def set_badge(self, text: str, style_key: str = ""):
        display = text if len(text) <= 28 else text[:26] + "…"
        self.setText(display)
        self.setToolTip(text)
        self.apply_style(style_key or text)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL FRAME — card with accent border
# ══════════════════════════════════════════════════════════════════════════════


class PanelFrame(QFrame):

    def __init__(self, title: str = "", accent: str = BORDER, parent=None):
        super().__init__(parent)
        self._accent = accent
        self._oid = f"pf-{id(self)}"
        self.setObjectName(self._oid)
        self._apply_style()

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(5)

        if title:
            hdr = QLabel(title.upper())
            hdr.setFont(QFont(FONT_UI, 7))
            hdr.setStyleSheet(f"color: {MUTED}; letter-spacing: 1px;")
            self._layout.addWidget(hdr)

    def _apply_style(self):
        self.setStyleSheet(
            f"QFrame#{self._oid} {{ background: {SURFACE}; "
            f"border: 1px solid {BORDER}; border-radius: 6px; "
            f"border-top: 2px solid {self._accent}; }}"
        )

    def set_accent(self, color: str):
        self._accent = color
        self._apply_style()

    def content_layout(self) -> QVBoxLayout:
        return self._layout


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL ROW — for Key Levels grid (name | value | pct)
# ══════════════════════════════════════════════════════════════════════════════


class LevelRow(QWidget):
    """Single key-level row: RESISTANCE  25,000  +5.3%"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(6)

        self._name = QLabel("")
        self._name.setFont(QFont(FONT_UI, 7))
        self._name.setStyleSheet(f"color: {MUTED}; letter-spacing: 0.5px;")
        self._name.setFixedWidth(90)
        lay.addWidget(self._name)

        self._val = QLabel("")
        self._val.setFont(QFont(FONT_MONO, 9))
        self._val.setStyleSheet(f"color: {WHITE}; font-weight: 600;")
        self._val.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        lay.addWidget(self._val, stretch=1)

        self._pct = QLabel("")
        self._pct.setFont(QFont(FONT_MONO, 7))
        self._pct.setMinimumWidth(60)
        self._pct.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        lay.addWidget(self._pct)

        # Bottom border
        self.setStyleSheet(f"border-bottom: 1px solid {BORDER};")

    def set_data(self, name: str, value: float, color: str = WHITE, pct: str = ""):
        self._name.setText(name.upper())
        self._val.setText(f"{value:,.0f}")
        self._val.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._pct.setText(pct)
        self._pct.setStyleSheet(f"color: {color};")


# ══════════════════════════════════════════════════════════════════════════════
# CHECKLIST ROW — icon + detail + implication
# ══════════════════════════════════════════════════════════════════════════════


class ChecklistRow(QWidget):

    _ICONS = {
        "pass": ("✓", GREEN),
        "warn": ("!", GOLD),
        "fail": ("✗", RED),
        "neutral": ("-", MUTED),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 3, 0, 3)
        outer.setSpacing(8)

        self._icon = QLabel("-")
        self._icon.setFixedWidth(14)
        self._icon.setFont(QFont(FONT_MONO, 9))
        self._icon.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        outer.addWidget(self._icon)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        self._detail = QLabel("")
        self._detail.setFont(QFont(FONT_UI, 8))
        self._detail.setStyleSheet(f"color: {WHITE};")
        self._detail.setWordWrap(True)
        col.addWidget(self._detail)

        self._impl = QLabel("")
        self._impl.setFont(QFont(FONT_UI, 7))
        self._impl.setStyleSheet(f"color: {MUTED};")
        self._impl.setWordWrap(True)
        col.addWidget(self._impl)

        outer.addLayout(col, stretch=1)
        self.setStyleSheet(f"border-bottom: 1px solid {BORDER};")

    def set_data(self, status: str, item: str, detail: str, implication: str = ""):
        icon, color = self._ICONS.get(status, ("-", MUTED))
        self._icon.setText(icon)
        self._icon.setStyleSheet(f"color: {color};")
        self._detail.setText(detail)
        self._impl.setText(implication)
        self._impl.setVisible(bool(implication))


# ══════════════════════════════════════════════════════════════════════════════
# RISK NOTE — styled warning box
# ══════════════════════════════════════════════════════════════════════════════


class RiskNote(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.2); "
            f"border-left: 2px solid {RED}; border-radius: 3px; padding: 3px 8px;"
        )
        from PyQt6.QtWidgets import QSizePolicy

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setMinimumWidth(0)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lbl = QLabel(f"! {text}")
        lbl.setFont(QFont(FONT_UI, 8))
        lbl.setStyleSheet("color: #fca5a5;")
        lbl.setWordWrap(True)
        lbl.setMinimumWidth(0)
        lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lay.addWidget(lbl, stretch=1)
