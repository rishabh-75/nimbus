"""
ui/scanner_tab.py — Scanner with RS/IVR columns, styled filter chips,
single-click populate, double-click navigate, stale banner.

Patch v5.3 (FIX-ETF):
  - Added "ETFs" universe option to combo box
  - _on_run() routes "ETFs" → NSE_ETF_SYMBOLS list
  - ETF rows display "ETF" in Regime column and "—" for DTE/Res%
    (both already handled by existing display logic: None → "—")
"""

from __future__ import annotations
import logging, time
from typing import Optional
import pandas as pd
from PyQt6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QProgressBar,
    QTableView,
    QHeaderView,
    QCheckBox,
    QSlider,
    QSpinBox,
    QFrame,
    QFileDialog,
    QAbstractItemView,
)
from PyQt6.QtGui import QFont, QColor, QBrush
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
    FONT_MONO,
    FONT_UI,
    score_color,
)
from ui.workers import ScanWorker
from modules.data import NIFTY100_SYMBOLS, get_universe
from modules.etf_analyzer import NSE_ETF_SYMBOLS  # FIX-ETF

logger = logging.getLogger(__name__)

_COLUMNS = [
    ("Symbol", "symbol", 110),
    ("Price", "last_price", 90),
    ("Score", "viability_score", 62),
    ("Size", "size_suggestion", 65),
    ("W%R", "wr_50", 65),
    ("BB State", "position_state", 130),
    ("Bias", "daily_bias", 80),
    ("Regime", "gex_regime", 120),
    ("Res %", "pct_to_resistance", 75),
    ("DTE", "dte", 55),
    ("Reason", "short_reason", 0),  # stretch
]


def _chip_style(checked: bool) -> str:
    if checked:
        return (
            f"QPushButton {{ background: #0D2B20; border: 1px solid {EM}; "
            f"color: {EM}; border-radius: 4px; padding: 3px 10px; font-size: 9pt; }}"
        )
    return (
        f"QPushButton {{ background: #111827; border: 1px solid {BORDER}; "
        f"color: {MUTED}; border-radius: 4px; padding: 3px 10px; font-size: 9pt; }}"
    )


class ScanResultModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[dict] = []

    def set_data(self, rows):
        self.beginResetModel()
        self._data = rows
        self.endResetModel()

    def append_row(self, row):
        pos = len(self._data)
        self.beginInsertRows(QModelIndex(), pos, pos)
        self._data.append(row)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self._data = []
        self.endResetModel()

    def row_data(self, row):
        return self._data[row] if 0 <= row < len(self._data) else None

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return _COLUMNS[section][0]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row_dict = self._data[index.row()]
        _, col_key, _ = _COLUMNS[index.column()]
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
            if col_key in ("rs_vs_market", "rs_vs_sector"):
                if val is None:
                    return "N/A"
                pp = val * 100
                return f"+{pp:.1f}pp" if pp > 0 else f"{pp:.1f}pp"
            if col_key == "ivr_state":
                return str(val) if val else "N/A"
            # FIX-ETF: "ETF" regime label gets a friendly display
            if col_key == "gex_regime" and val == "ETF":
                return "ETF ✦"
            return str(val)

        if role == Qt.ItemDataRole.ForegroundRole:
            if col_key == "viability_score" and isinstance(val, (int, float)):
                return QBrush(QColor(score_color(int(val))))
            if col_key == "size_suggestion":
                sc = {
                    "FULL": GREEN,
                    "HALF": GOLD,
                    "QTR": BLUE,
                    "SKIP": RED,
                    "ZERO": RED,
                    "AVOID": RED,
                }
                return QBrush(QColor(sc.get(str(val), MUTED)))
            if col_key == "wr_50" and isinstance(val, (int, float)):
                return QBrush(
                    QColor(EM if val >= -20 else (GOLD if val >= -50 else RED))
                )
            if col_key == "daily_bias":
                return QBrush(
                    QColor(
                        {"BULLISH": EM, "BEARISH": RED, "NEUTRAL": GOLD}.get(
                            str(val), MUTED
                        )
                    )
                )
            if col_key == "pct_to_resistance" and isinstance(val, (int, float)):
                return QBrush(QColor(EM if val >= 5 else (GOLD if val >= 2 else RED)))
            if col_key in ("rs_vs_market", "rs_vs_sector"):
                if val is None:
                    return QBrush(QColor(MUTED))
                pp = val * 100
                if pp > 0.5:
                    return QBrush(QColor(EM))
                if pp < -0.5:
                    return QBrush(QColor(RED))
                return QBrush(QColor(MUTED))
            if col_key == "ivr_state":
                c = {"CHEAP": EM, "RICH": RED, "FAIR": MUTED, "N/A": MUTED}
                return QBrush(QColor(c.get(str(val), MUTED)))
            # FIX-ETF: ETF regime label in blue
            if col_key == "gex_regime" and val == "ETF":
                return QBrush(QColor(BLUE))

        if role == Qt.ItemDataRole.BackgroundRole:
            if col_key == "viability_score" and isinstance(val, (int, float)):
                if val >= 70:
                    return QBrush(QColor(16, 185, 129, 30))
                elif val >= 50:
                    return QBrush(QColor(245, 158, 11, 25))

        if role == Qt.ItemDataRole.FontRole:
            if col_key in ("viability_score", "size_suggestion", "symbol"):
                f = QFont(FONT_MONO, 10)
                f.setBold(True)
                return f
            if col_key in ("last_price", "wr_50", "pct_to_resistance", "dte"):
                return QFont(FONT_MONO, 10)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if col_key in (
                "last_price",
                "viability_score",
                "wr_50",
                "pct_to_resistance",
                "dte",
            ):
                return RIGHT

        return None


class ScannerTab(QWidget):
    row_clicked = pyqtSignal(str)  # double-click → open in dashboard
    symbol_selected = pyqtSignal(str)  # single-click → populate sidebar

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan_worker: Optional[ScanWorker] = None
        self._last_scan_ts: Optional[float] = None
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

        ctrl.addWidget(self._lbl("Universe:"))
        self._universe_combo = QComboBox()
        # FIX-ETF: added "ETFs" universe option
        self._universe_combo.addItems(["NIFTY 100", "F&O Full", "ETFs", "Watchlist"])
        self._universe_combo.setFixedWidth(110)
        self._universe_combo.setFixedHeight(26)
        ctrl.addWidget(self._universe_combo)

        ctrl.addWidget(self._lbl("Min Score:"))
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

        mom = QPushButton("Momentum (BB+W%R)")
        mom.setCheckable(True)
        mom.setChecked(True)
        mom.setEnabled(False)
        mom.setStyleSheet(_chip_style(True))
        mom.setToolTip("Hard gate: cannot be disabled")
        chips.addWidget(mom)

        self._filter_checks = {}
        for name in ("Structure", "Regime", "Expiry", "Bias"):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.setStyleSheet(_chip_style(False))
            btn.toggled.connect(lambda c, b=btn: b.setStyleSheet(_chip_style(c)))
            chips.addWidget(btn)
            self._filter_checks[name] = btn

        self._req_all = QPushButton("Require All")
        self._req_all.setCheckable(True)
        self._req_all.setChecked(False)
        self._req_all.setStyleSheet(_chip_style(False))
        self._req_all.toggled.connect(
            lambda c: self._req_all.setStyleSheet(_chip_style(c))
        )
        chips.addWidget(self._req_all)

        chips.addStretch(1)
        layout.addLayout(chips)

        # ── Stale banner ──────────────────────────────────────────────────────
        self._stale_banner = QFrame()
        self._stale_banner.setStyleSheet(
            "QFrame { background: #1A1A0D; border-left: 4px solid #F59E0B; padding: 4px 10px; }"
        )
        bl = QHBoxLayout(self._stale_banner)
        bl.setContentsMargins(2, 0, 2, 0)
        self._stale_lbl = QLabel("")
        self._stale_lbl.setFont(QFont(FONT_UI, 8))
        self._stale_lbl.setStyleSheet(f"color: {GOLD};")
        bl.addWidget(self._stale_lbl)
        rb = QPushButton("Refresh Now")
        rb.setFixedHeight(20)
        rb.clicked.connect(self._on_run)
        bl.addWidget(rb)
        bl.addStretch()
        self._stale_banner.setVisible(False)
        layout.addWidget(self._stale_banner)

        # ── Progress ──────────────────────────────────────────────────────────
        pr = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setFixedHeight(14)
        self._progress.setVisible(False)
        pr.addWidget(self._progress)
        self._progress_lbl = QLabel("")
        self._progress_lbl.setFont(QFont(FONT_MONO, 8))
        self._progress_lbl.setStyleSheet(f"color: {MUTED};")
        self._progress_lbl.setVisible(False)
        pr.addWidget(self._progress_lbl)
        layout.addLayout(pr)

        # ── Table ─────────────────────────────────────────────────────────────
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(32)
        self._table.setShowGrid(False)
        self._table.setMouseTracking(True)

        hdr = self._table.horizontalHeader()
        hdr.setMouseTracking(True)
        hdr.setMinimumSectionSize(30)
        hdr.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        for i, (_, _, w) in enumerate(_COLUMNS):
            if w > 0:
                hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                hdr.resizeSection(i, w)
            else:
                hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        self._table.clicked.connect(self._on_row_single_click)
        self._table.doubleClicked.connect(self._on_row_double_click)
        layout.addWidget(self._table, stretch=1)

        # ── Actions ───────────────────────────────────────────────────────────
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._open_btn = QPushButton("Open in Dashboard")
        self._open_btn.setFixedHeight(26)
        self._open_btn.clicked.connect(self._on_open_in_dashboard)
        actions.addWidget(self._open_btn)

        self._count_lbl = QLabel("")
        self._count_lbl.setFont(QFont(FONT_MONO, 8))
        self._count_lbl.setStyleSheet(f"color: {MUTED};")
        actions.addStretch(1)
        actions.addWidget(self._count_lbl)
        layout.addLayout(actions)

    def _lbl(self, text):
        l = QLabel(text)
        l.setFont(QFont(FONT_UI, 9))
        l.setStyleSheet(f"color: {MUTED};")
        return l

    # ══════════════════════════════════════════════════════════════════════════
    def _on_run(self):
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()
            return

        universe = self._universe_combo.currentText()
        if universe == "NIFTY 100":
            symbols = list(NIFTY100_SYMBOLS)
        elif universe == "F&O Full":
            symbols = get_universe()
        elif universe == "ETFs":  # FIX-ETF
            symbols = list(NSE_ETF_SYMBOLS)  # FIX-ETF
        else:
            from ui.watchlist_db import load_watchlist

            symbols = [e["symbol"] for e in load_watchlist()]

        if not symbols:
            self._count_lbl.setText("No symbols")
            return

        self._model.clear()
        self._stale_banner.setVisible(False)
        self._progress.setVisible(True)
        self._progress_lbl.setVisible(True)
        self._progress.setMaximum(len(symbols))
        self._progress.setValue(0)
        self._run_btn.setText("Cancel")

        self._scan_worker = ScanWorker(
            symbols=symbols,
            min_viability=self._min_score_slider.value(),
            require_all_filters=self._req_all.isChecked(),
        )
        self._scan_worker.row_ready.connect(self._on_row_ready)
        self._scan_worker.progress.connect(self._on_progress)
        self._scan_worker.finished.connect(self._on_finished)
        self._scan_worker.start()

    def _on_row_ready(self, row):
        self._model.append_row(row)

    def _on_progress(self, done, total, symbol):
        self._progress.setValue(done)
        self._progress_lbl.setText(f"{symbol} {done}/{total}")

    def _on_finished(self, results):
        self._progress.setVisible(False)
        self._progress_lbl.setVisible(False)
        self._run_btn.setText("Run Scanner")
        self._last_scan_ts = time.time()

        total = getattr(self._scan_worker, "total_scanned", len(results))
        self._count_lbl.setText(f"{len(results)} results from {total} symbols scanned")
        self._proxy.sort(2, Qt.SortOrder.DescendingOrder)
        logger.info("Scan complete: %d results", len(results))

    def _on_row_single_click(self, index):
        src = self._proxy.mapToSource(index)
        row = self._model.row_data(src.row())
        if row:
            self.symbol_selected.emit(row["symbol"])

    def _on_row_double_click(self, index):
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
            self, "Export", "nimbus_scan.csv", "CSV (*.csv)"
        )
        if path:
            pd.DataFrame(self._model._data).to_csv(path, index=False)
