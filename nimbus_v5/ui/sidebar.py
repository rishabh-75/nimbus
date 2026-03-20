"""
ui/sidebar.py
─────────────
NIMBUS sidebar: symbol selector, lot size, indicator parameter sliders,
refresh button.

Fixed width 220px. Refresh button has fixed width and never resizes.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QSpinBox,
    QSlider, QFrame, QSizePolicy,
)
from PyQt6.QtGui import QFont

from ui.theme import (
    BG, SURFACE, S2, BORDER, EM, MUTED, WHITE, RED, GOLD, GREEN, BLUE,
    FONT_MONO, FONT_UI,
)

logger = logging.getLogger(__name__)

# Default symbol universe (loaded at startup, refreshed by UniverseWorker)
_DEFAULT_SYMBOLS = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK",
    "ICICIBANK", "INFY", "SBIN", "AXISBANK", "LT",
    "BAJFINANCE", "KOTAKBANK", "ADANIENT", "ADANIPORTS",
    "TATAMOTORS", "TATASTEEL", "SUNPHARMA", "WIPRO",
]


class Sidebar(QWidget):
    """
    Fixed-width sidebar with trading controls.

    Signals:
        symbol_changed(str)      — emitted when user selects/enters a new symbol
        lot_size_changed(int)    — emitted when lot size spinner changes
        bb_period_changed(int)   — BB period slider
        bb_std_changed(float)    — BB std dev slider (emits value / 10.0)
        wr_period_changed(int)   — WR period slider
        wr_thresh_changed(float) — WR threshold slider
        refresh_clicked()        — refresh button pressed
    """

    symbol_changed    = pyqtSignal(str)
    lot_size_changed  = pyqtSignal(int)
    bb_period_changed = pyqtSignal(int)
    bb_std_changed    = pyqtSignal(float)
    wr_period_changed = pyqtSignal(int)
    wr_thresh_changed = pyqtSignal(float)
    refresh_clicked   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("nimbus-sidebar")
        self.setFixedWidth(236)
        self.setStyleSheet(
            f"QWidget#nimbus-sidebar {{ background: {SURFACE}; "
            f"border-right: 1px solid {BORDER}; }}"
        )
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 12)
        layout.setSpacing(6)

        # ── NIMBUS logo ───────────────────────────────────────────────────────
        logo = QLabel("NIMBUS")
        logo.setObjectName("nimbus-logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(logo)

        subtitle = QLabel("EMERALD SLATE")
        subtitle.setStyleSheet(
            f"color: {MUTED}; font-size: 8px; letter-spacing: 2px;"
        )
        layout.addWidget(subtitle)
        layout.addSpacing(16)

        # ── Symbol selector ───────────────────────────────────────────────────
        layout.addWidget(self._section_label("SYMBOL"))

        sym_row = QHBoxLayout()
        sym_row.setSpacing(4)

        self._symbol_combo = QComboBox()
        self._symbol_combo.setEditable(False)
        self._symbol_combo.addItems(_DEFAULT_SYMBOLS)
        self._symbol_combo.setCurrentText("NIFTY")
        self._symbol_combo.currentTextChanged.connect(self._on_combo_changed)
        sym_row.addWidget(self._symbol_combo, stretch=1)

        layout.addLayout(sym_row)

        # Manual entry row
        manual_row = QHBoxLayout()
        manual_row.setSpacing(4)

        self._symbol_input = QLineEdit()
        self._symbol_input.setPlaceholderText("Enter symbol…")
        self._symbol_input.setMaxLength(20)
        self._symbol_input.returnPressed.connect(self._on_manual_entry)
        manual_row.addWidget(self._symbol_input, stretch=1)

        go_btn = QPushButton("GO")
        go_btn.setObjectName("primary")
        go_btn.setFixedSize(48, 30)
        go_btn.setFont(QFont(FONT_UI, 10))
        go_btn.setToolTip("Load symbol")
        go_btn.clicked.connect(self._on_manual_entry)
        manual_row.addWidget(go_btn)

        layout.addLayout(manual_row)
        layout.addSpacing(12)

        # ── Lot size ──────────────────────────────────────────────────────────
        layout.addWidget(self._section_label("LOT SIZE"))

        self._lot_spin = QSpinBox()
        self._lot_spin.setRange(1, 50000)
        self._lot_spin.setValue(75)
        self._lot_spin.setSingleStep(25)
        self._lot_spin.valueChanged.connect(self.lot_size_changed.emit)
        layout.addWidget(self._lot_spin)
        layout.addSpacing(12)

        # ── Separator ─────────────────────────────────────────────────────────
        layout.addWidget(self._separator())
        layout.addSpacing(8)

        # ── Dual-Mode Signal Panel ────────────────────────────────────────────
        layout.addWidget(self._section_label("DUAL MODE"))
        layout.addSpacing(4)

        self._dm_segment_lbl = QLabel("Segment: —")
        self._dm_segment_lbl.setFont(QFont(FONT_MONO, 8))
        self._dm_segment_lbl.setStyleSheet(f"color: {WHITE};")
        layout.addWidget(self._dm_segment_lbl)

        self._dm_mode_lbl = QLabel("Mode: —")
        self._dm_mode_lbl.setFont(QFont(FONT_MONO, 8))
        self._dm_mode_lbl.setStyleSheet(f"color: {WHITE};")
        layout.addWidget(self._dm_mode_lbl)

        self._dm_wr_lbl = QLabel("WR: —")
        self._dm_wr_lbl.setFont(QFont(FONT_MONO, 8))
        self._dm_wr_lbl.setStyleSheet(f"color: {WHITE};")
        layout.addWidget(self._dm_wr_lbl)

        self._dm_adx_lbl = QLabel("MFI: —")
        self._dm_adx_lbl.setFont(QFont(FONT_MONO, 8))
        self._dm_adx_lbl.setStyleSheet(f"color: {WHITE};")
        layout.addWidget(self._dm_adx_lbl)

        self._dm_sma_lbl = QLabel("SMA: — | DD: — | —")
        self._dm_sma_lbl.setFont(QFont(FONT_MONO, 8))
        self._dm_sma_lbl.setStyleSheet(f"color: {WHITE};")
        layout.addWidget(self._dm_sma_lbl)

        layout.addSpacing(4)

        self._dm_score_lbl = QLabel("Score: —")
        self._dm_score_lbl.setFont(QFont(FONT_MONO, 9))
        self._dm_score_lbl.setStyleSheet(f"color: {EM}; font-weight: bold;")
        layout.addWidget(self._dm_score_lbl)

        self._dm_breakdown_lbl = QLabel("")
        self._dm_breakdown_lbl.setFont(QFont(FONT_MONO, 7))
        self._dm_breakdown_lbl.setStyleSheet(f"color: {MUTED};")
        self._dm_breakdown_lbl.setWordWrap(True)
        layout.addWidget(self._dm_breakdown_lbl)

        layout.addSpacing(4)

        self._dm_data_lbl = QLabel("")
        self._dm_data_lbl.setFont(QFont(FONT_MONO, 7))
        self._dm_data_lbl.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self._dm_data_lbl)

        layout.addSpacing(12)
        layout.addWidget(self._separator())
        layout.addSpacing(8)

        # ── Refresh button ────────────────────────────────────────────────────
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("primary")
        self._refresh_btn.setFixedWidth(208)  # fixed width, never resizes
        self._refresh_btn.setFixedHeight(36)
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.clicked.connect(self.refresh_clicked.emit)
        layout.addWidget(self._refresh_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Push everything up
        layout.addStretch(1)

        # ── Version label ─────────────────────────────────────────────────────
        version = QLabel("v5 · Qt · Emerald Slate")
        version.setStyleSheet(f"color: {MUTED}; font-size: 8px;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section-header")
        lbl.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; letter-spacing: 1.4px;"
            f" font-family: '{FONT_UI}';"
        )
        return lbl

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {BORDER}; border: none;")
        return line

    def _add_slider(self, layout, label_text, min_val, max_val, default,
                    callback) -> tuple:
        """Add a labelled slider with a value display. Returns (slider, value_label)."""
        row = QHBoxLayout()
        row.setSpacing(4)

        name_lbl = QLabel(label_text)
        name_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        row.addWidget(name_lbl)

        row.addStretch(1)

        val_lbl = QLabel(str(default))
        val_lbl.setStyleSheet(
            f"color: {WHITE}; font-size: 10px; font-family: '{FONT_MONO}';"
        )
        val_lbl.setMinimumWidth(32)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(val_lbl)

        layout.addLayout(row)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(callback)
        layout.addWidget(slider)
        layout.addSpacing(6)

        return slider, val_lbl

    # ──────────────────────────────────────────────────────────────────────────
    # SLOTS
    # ──────────────────────────────────────────────────────────────────────────

    def _on_combo_changed(self, text: str):
        if text:
            logger.info("Symbol selected from combo: %s", text)
            self.symbol_changed.emit(text.strip().upper())

    def _on_manual_entry(self):
        text = self._symbol_input.text().strip().upper()
        if text:
            logger.info("Manual symbol entry: %s", text)
            # Add to combo if not present
            idx = self._symbol_combo.findText(text)
            if idx < 0:
                self._symbol_combo.addItem(text)
            self._symbol_combo.setCurrentText(text)
            self._symbol_input.clear()
            self.symbol_changed.emit(text)

    def _on_bb_period(self, val: int):
        pass  # Legacy — sliders removed, kept for signal compatibility

    def _on_bb_std(self, val: int):
        pass

    def _on_wr_period(self, val: int):
        pass

    def _on_wr_thresh(self, val: int):
        pass

    def update_dual_mode(self, dm_sig):
        """Update sidebar panel from unified mean-reversion signal."""
        try:
            # Tier
            tier = dm_sig.tier
            tier_c = {
                "PRIMARY": EM, "SECONDARY": GOLD, "NONE": MUTED
            }.get(tier, MUTED)
            self._dm_segment_lbl.setText(f"Tier: {tier}")
            self._dm_segment_lbl.setStyleSheet(f"color: {tier_c};")

            self._dm_mode_lbl.setText("Mean Reversion")
            self._dm_mode_lbl.setStyleSheet(f"color: {WHITE};")

            # WR(30)
            wr_val = dm_sig.wr_30
            wr_c = EM if wr_val < -50 else (GREEN if wr_val < -30 else MUTED)
            self._dm_wr_lbl.setText(f"WR(30): {wr_val:.0f}")
            self._dm_wr_lbl.setStyleSheet(f"color: {wr_c};")

            # MFI + vs SMA combined
            mfi_c = EM if dm_sig.mfi >= 50 else (GREEN if dm_sig.mfi >= 30 else RED)
            self._dm_adx_lbl.setText(f"MFI: {dm_sig.mfi:.0f}")
            self._dm_adx_lbl.setStyleSheet(f"color: {mfi_c};")

            pct = dm_sig.pct_from_sma
            sma_c = RED if not dm_sig.above_sma else EM
            self._dm_sma_lbl.setText(
                f"SMA: {pct:+.1f}% | DD: {dm_sig.dd_from_high:.1f}% | {dm_sig.red_streak}d red"
            )
            self._dm_sma_lbl.setStyleSheet(f"color: {sma_c};")

            # Score
            sc = dm_sig.dual_score
            from ui.theme import score_color
            sc_c = score_color(sc)
            self._dm_score_lbl.setText(
                f"Score: {sc}  [{dm_sig.dual_label}]  {dm_sig.dual_sizing}"
            )
            self._dm_score_lbl.setStyleSheet(f"color: {sc_c}; font-weight: bold;")

            # Breakdown
            parts = [f"base={dm_sig.base_score}"]
            if dm_sig.options_overlay != 0:
                parts.append(f"opt={dm_sig.options_overlay:+d}")
            if dm_sig.filing_overlay != 0:
                parts.append(f"fil={dm_sig.filing_overlay:+d}")
            if dm_sig.is_trap:
                parts.append("⚠ TRAP")
            self._dm_breakdown_lbl.setText(" + ".join(parts) + f" → {sc}")

            # Data quality
            if dm_sig.data_sufficient:
                self._dm_data_lbl.setText(
                    f"{dm_sig.input_interval} → {dm_sig.daily_bars_used}d bars"
                )
                self._dm_data_lbl.setStyleSheet(f"color: {MUTED};")
            else:
                self._dm_data_lbl.setText("⚠ Insufficient daily bars")
                self._dm_data_lbl.setStyleSheet(f"color: {RED};")

        except Exception as exc:
            logger.debug("Sidebar update failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def set_symbols(self, symbols: list[str]):
        """Replace the combo box items (called after UniverseWorker completes)."""
        current = self._symbol_combo.currentText()
        self._symbol_combo.blockSignals(True)
        self._symbol_combo.clear()
        self._symbol_combo.addItems(symbols)
        # Restore previous selection if still in list
        idx = self._symbol_combo.findText(current)
        if idx >= 0:
            self._symbol_combo.setCurrentIndex(idx)
        else:
            self._symbol_combo.setCurrentIndex(0)
        self._symbol_combo.blockSignals(False)

    def current_symbol(self) -> str:
        return self._symbol_combo.currentText().strip().upper()

    def lot_size(self) -> int:
        return self._lot_spin.value()

    def bb_period(self) -> int:
        return self._bb_period_slider.value()

    def bb_std(self) -> float:
        return self._bb_std_slider.value() / 10.0

    def wr_period(self) -> int:
        return self._wr_period_slider.value()

    def wr_threshold(self) -> float:
        return float(self._wr_thresh_slider.value())

    def set_lot_size(self, val: int):
        self._lot_spin.setValue(val)

    def set_refresh_enabled(self, enabled: bool):
        self._refresh_btn.setEnabled(enabled)
