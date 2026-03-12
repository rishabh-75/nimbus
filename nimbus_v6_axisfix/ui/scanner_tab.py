"""
ui/scanner_tab.py
─────────────────
Scanner tab for NIMBUS Qt.

Controls:
    Run Scanner (primary), Progress bar, Universe selector,
    Filter chips (Momentum = locked/disabled), Min Viability slider

Table:
    QAbstractTableModel backed by scan results. Sortable.
    Score/Size columns colour-coded per §2.4.
    Click row → open_in_dashboard signal.

HARD GATE: Momentum filter cannot be disabled. (Lesson 8.3)
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, pyqtSignal,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QProgressBar, QTableView, QHeaderView, QCheckBox,
    QSlider, QFrame, QFileDialog, QAbstractItemView,
)
from PyQt6.QtGui import QFont, QColor, QBrush

from ui.theme import (
    BG, SURFACE, S2, S3, BORDER, EM, RED, GOLD, MUTED, WHITE, GREEN, BLUE,
    FONT_MONO, FONT_UI, score_color,
)
from ui.workers import ScanWorker
from modules.data import NIFTY100_SYMBOLS, get_universe

logger = logging.getLogger(__name__)

# ── Table columns ─────────────────────────────────────────────────────────────
_COLUMNS = [
    ("Symbol",   "symbol",            85),
    ("Price",    "last_price",         75),
    ("Score",    "viability_score",    52),
    ("Size",     "size_suggestion",    52),
    ("W%R",      "wr_50",              52),
    ("BB State", "position_state",    100),
    ("Bias",     "daily_bias",         65),
    ("Regime",   "gex_regime",         90),
    ("Res %",    "pct_to_resistance",  52),
    ("DTE",      "dte",                40),
    ("Reason",   "short_reason",        0),   # stretch
]


# ══════════════════════════════════════════════════════════════════════════════
# TABLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ScanResultModel(QAbstractTableModel):
    """Model backed by a list[dict] of scan results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[dict] = []

    def set_data(self, rows: list[dict]):
        self.beginResetModel()
        self._data = rows
        self.endResetModel()

    def append_row(self, row: dict):
        pos = len(self._data)
        self.beginInsertRows(QModelIndex(), pos, pos)
        self._data.append(row)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self._data = []
        self.endResetModel()

    def row_data(self, row: int) -> Optional[dict]:
        if 0 <= row < len(self._data):
            return self._data[row]
        return None

    # ── Qt interface ──────────────────────────────────────────────────────────

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return _COLUMNS[section][0]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row_dict = self._data[index.row()]
        col_name, col_key, _ = _COLUMNS[index.column()]

        val = row_dict.get(col_key)

        if role == Qt.ItemDataRole.DisplayRole:
            if val is None:
                return "—"
            if col_key == "last_price":
                return f"{val:,.2f}"
            if col_key == "wr_50" and isinstance(val, (int, float)):
                return f"{val:.1f}"
            if col_key == "pct_to_resistance" and isinstance(val, (int, float)):
                return f"{val:+.1f}%"
            if col_key == "dte" and isinstance(val, (int, float)):
                return f"{int(val)}d"
            return str(val)

        if role == Qt.ItemDataRole.ForegroundRole:
            if col_key == "viability_score" and isinstance(val, (int, float)):
                return QBrush(QColor(score_color(int(val))))
            if col_key == "size_suggestion":
                size_colors = {"FULL": GREEN, "HALF": GOLD, "QTR": BLUE, "SKIP": RED, "ZERO": RED}
                return QBrush(QColor(size_colors.get(str(val), MUTED)))
            if col_key == "wr_50" and isinstance(val, (int, float)):
                c = EM if val >= -20 else (GOLD if val >= -50 else RED)
                return QBrush(QColor(c))
            if col_key == "daily_bias":
                bc = {"BULLISH": EM, "BEARISH": RED, "NEUTRAL": GOLD}
                return QBrush(QColor(bc.get(str(val), MUTED)))
            if col_key == "pct_to_resistance" and isinstance(val, (int, float)):
                return QBrush(QColor(EM if val >= 5 else (GOLD if val >= 2 else RED)))

        if role == Qt.ItemDataRole.BackgroundRole:
            if col_key == "viability_score" and isinstance(val, (int, float)):
                if val >= 70:
                    return QBrush(QColor(16, 185, 129, 30))
                elif val >= 50:
                    return QBrush(QColor(245, 158, 11, 25))
                elif val >= 30:
                    return QBrush(QColor(96, 165, 250, 20))

        if role == Qt.ItemDataRole.FontRole:
            if col_key in ("viability_score", "size_suggestion"):
                f = QFont(FONT_MONO, 10)
                f.setBold(True)
                return f
            if col_key in ("last_price", "wr_50", "pct_to_resistance", "dte"):
                return QFont(FONT_MONO, 10)
            if col_key == "symbol":
                f = QFont(FONT_MONO, 10)
                f.setBold(True)
                return f

        if role == Qt.ItemDataRole.TextAlignmentRole:
            _RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if col_key in ("last_price", "viability_score", "wr_50",
                           "pct_to_resistance", "dte"):
                return _RIGHT

        return None


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER TAB
# ══════════════════════════════════════════════════════════════════════════════

class ScannerTab(QWidget):
    """
    Scanner tab. Emits row_clicked(symbol) when user clicks a result.
    """
    row_clicked = pyqtSignal(str)   # symbol

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan_worker: Optional[ScanWorker] = None
        self._model = ScanResultModel()
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._run_btn = QPushButton("Run Scanner")
        self._run_btn.setObjectName("primary")
        self._run_btn.setFixedHeight(28)
        self._run_btn.setFixedWidth(110)
        self._run_btn.clicked.connect(self._on_run)
        ctrl.addWidget(self._run_btn)

        ul = QLabel("Universe:")
        ul.setFont(QFont(FONT_UI, 9))
        ul.setStyleSheet(f"color: {MUTED};")
        ctrl.addWidget(ul)
        self._universe_combo = QComboBox()
        self._universe_combo.addItems(["NIFTY 100", "F&O Full", "Watchlist"])
        self._universe_combo.setFixedWidth(110)
        self._universe_combo.setFixedHeight(26)
        ctrl.addWidget(self._universe_combo)

        ml = QLabel("Min Score:")
        ml.setFont(QFont(FONT_UI, 9))
        ml.setStyleSheet(f"color: {MUTED};")
        ctrl.addWidget(ml)
        self._min_score_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_score_slider.setRange(0, 90)
        self._min_score_slider.setValue(50)
        self._min_score_slider.setFixedWidth(90)
        ctrl.addWidget(self._min_score_slider)
        self._min_score_lbl = QLabel("50")
        self._min_score_lbl.setFont(QFont(FONT_MONO, 9))
        self._min_score_lbl.setFixedWidth(22)
        self._min_score_slider.valueChanged.connect(
            lambda v: self._min_score_lbl.setText(str(v))
        )
        ctrl.addWidget(self._min_score_lbl)

        ctrl.addStretch(1)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setFixedHeight(26)
        self._export_btn.clicked.connect(self._on_export)
        ctrl.addWidget(self._export_btn)

        layout.addLayout(ctrl)

        # ── Filter chips ──────────────────────────────────────────────────────
        chips = QHBoxLayout()
        chips.setSpacing(4)

        # Momentum — LOCKED. Cannot be disabled. (Lesson 8.3)
        self._momentum_chk = QCheckBox("Momentum (BB+W%R)")
        self._momentum_chk.setFont(QFont(FONT_UI, 9))
        self._momentum_chk.setChecked(True)
        self._momentum_chk.setEnabled(False)  # hard gate — always on
        self._momentum_chk.setToolTip(
            "Hard gate: BB riding + W%R > -20. Cannot be disabled.\n"
            "Without both conditions, there is no trade thesis."
        )
        chips.addWidget(self._momentum_chk)

        self._filter_checks = {}
        for name in ("Structure", "Regime", "Expiry", "Bias"):
            cb = QCheckBox(name)
            cb.setFont(QFont(FONT_UI, 9))
            cb.setChecked(False)
            chips.addWidget(cb)
            self._filter_checks[name] = cb

        self._req_all_chk = QCheckBox("Require All")
        self._req_all_chk.setFont(QFont(FONT_UI, 9))
        self._req_all_chk.setChecked(False)
        chips.addWidget(self._req_all_chk)

        chips.addStretch(1)
        layout.addLayout(chips)

        # ── Progress bar ──────────────────────────────────────────────────────
        prog_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setFixedHeight(14)
        self._progress.setVisible(False)
        prog_row.addWidget(self._progress)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setFont(QFont(FONT_MONO, 8))
        self._progress_lbl.setStyleSheet(f"color: {MUTED};")
        self._progress_lbl.setVisible(False)
        prog_row.addWidget(self._progress_lbl)

        layout.addLayout(prog_row)

        # ── Results table ─────────────────────────────────────────────────────
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(26)
        self._table.setShowGrid(False)

        # Column sizing: fixed widths for data cols, stretch for Reason
        hdr = self._table.horizontalHeader()
        hdr.setMinimumSectionSize(30)
        hdr.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for i, (_, _, w) in enumerate(_COLUMNS):
            if w > 0:
                hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                hdr.resizeSection(i, w)
            else:
                hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        self._table.doubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self._table, stretch=1)

        # ── Actions row ───────────────────────────────────────────────────────
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._open_btn = QPushButton("Open in Dashboard")
        self._open_btn.setFixedHeight(26)
        self._open_btn.clicked.connect(self._on_open_in_dashboard)
        actions.addWidget(self._open_btn)

        self._count_lbl = QLabel("")
        self._count_lbl.setFont(QFont(FONT_MONO, 9))
        self._count_lbl.setStyleSheet(f"color: {MUTED};")
        actions.addStretch(1)
        actions.addWidget(self._count_lbl)

        layout.addLayout(actions)

    # ══════════════════════════════════════════════════════════════════════════
    # ACTIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_run(self):
        """Start scanner with selected universe."""
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()
            return

        # Determine symbol list
        universe = self._universe_combo.currentText()
        if universe == "NIFTY 100":
            symbols = list(NIFTY100_SYMBOLS)
        elif universe == "F&O Full":
            symbols = get_universe()
        else:
            from ui.watchlist_db import load_watchlist
            symbols = [e["symbol"] for e in load_watchlist()]

        if not symbols:
            self._count_lbl.setText("No symbols in selected universe")
            return

        min_viab = self._min_score_slider.value()
        req_all = self._req_all_chk.isChecked()

        self._model.clear()
        self._progress.setVisible(True)
        self._progress_lbl.setVisible(True)
        self._progress.setMaximum(len(symbols))
        self._progress.setValue(0)
        self._run_btn.setText("Cancel")

        self._scan_worker = ScanWorker(
            symbols=symbols,
            min_viability=min_viab,
            require_all_filters=req_all,
        )
        self._scan_worker.row_ready.connect(self._on_row_ready)
        self._scan_worker.progress.connect(self._on_progress)
        self._scan_worker.finished.connect(self._on_finished)
        self._scan_worker.start()

    def _on_row_ready(self, row: dict):
        self._model.append_row(row)

    def _on_progress(self, done: int, total: int, symbol: str):
        self._progress.setValue(done)
        self._progress_lbl.setText(f"{symbol}  {done}/{total}")

    def _on_finished(self, results: list):
        self._progress.setVisible(False)
        self._progress_lbl.setVisible(False)
        self._run_btn.setText("Run Scanner")
        self._count_lbl.setText(f"{len(results)} results")

        # Sort by score descending
        self._proxy.sort(2, Qt.SortOrder.DescendingOrder)

        logger.info("Scan complete: %d results", len(results))

    def _on_row_double_clicked(self, index: QModelIndex):
        src = self._proxy.mapToSource(index)
        row = self._model.row_data(src.row())
        if row:
            self.row_clicked.emit(row["symbol"])

    def _on_open_in_dashboard(self):
        indices = self._table.selectionModel().selectedRows()
        if indices:
            src = self._proxy.mapToSource(indices[0])
            row = self._model.row_data(src.row())
            if row:
                self.row_clicked.emit(row["symbol"])

    def _on_export(self):
        if not self._model._data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Scanner Results", "nimbus_scan.csv", "CSV Files (*.csv)"
        )
        if path:
            df = pd.DataFrame(self._model._data)
            df.to_csv(path, index=False)
            logger.info("Exported %d rows to %s", len(df), path)
