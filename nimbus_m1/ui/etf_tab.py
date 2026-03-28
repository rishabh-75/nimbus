"""
ui/etf_tab.py — ETF Momentum Scanner tab.

Scans NSE ETFs for pullback-in-trend momentum entries.
Walk-forward validated: Sharpe 2.15, Win 60%, Avg +2.91%.
"""

from __future__ import annotations
import logging, time
from PyQt6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    pyqtSignal,
    QThread,
)
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QPushButton,
    QLabel,
    QSlider,
    QTableView,
    QHeaderView,
    QAbstractItemView,
)
from PyQt6.QtGui import QFont, QColor, QBrush

from ui.theme import (
    BG,
    SURFACE,
    S2,
    BORDER,
    EM,
    MUTED,
    WHITE,
    RED,
    GOLD,
    GREEN,
    BLUE,
    FONT_MONO,
    FONT_UI,
    score_color,
)
from ui.components import Badge

logger = logging.getLogger(__name__)

# ETF universe
ETF_SYMBOLS = [
    "GOLDBEES",
    "GOLDCASE",
    "GOLDETF",
    "IVZINGOLD",
    "SILVERBEES",
    "SILVERIETF",
    "NIFTYBEES",
    "SETFNIF50",
    "JUNIORBEES",
    "NV20IETF",
    "BANKBEES",
    "SETFNIFBK",
    "PSUBNKBEES",
    "PVTBANIETF",
    "FINIETF",
    "ITBEES",
    "PHARMABEES",
    "INFRABEES",
    "AUTIBEES",
    "CPSEETF",
    "AUTOIETF",
    "FMCGIETF",
    "HEALTHIETF",
    "HEALTHY",
    "ENERGY",
    "MOENERGY",
    "COMMOIETF",
    "METALIETF",
    "OILIETF",
    "MOM100",
    "LOWVOLIETF",
    "MIDCAPIETF",
    "MON100",
    "MAFANG",
    "MAHKTECH",
    "HNGSNGBEES",
    "MODEFENCE",
    "ICICIB22",
    "NIF100BEES",
    "CONSUMBEES",
    "MID150BEES",
    "MOALPHA50",
    "TOP100CASE",
    "TOP10ADD",
    "MID100CASE",
]

_COLUMNS = [
    ("Symbol", "symbol", 110),
    ("Price", "last_price", 90),
    ("Score", "score", 55),
    ("Verdict", "verdict", 240),
    ("Size", "sizing", 55),
    ("ADTV ₹Cr", "adtv", 70),
    ("Trend", "trend", 55),
    ("WR(20)", "wr", 55),
    ("Dip", "dip", 55),
    ("ROC%", "roc", 55),
    ("MFI", "mfi", 50),
    ("ADX", "adx", 50),
    ("vs SMA50", "pct_sma50", 70),
    ("Reason", "reason", 0),  # stretch last
]


# ══════════════════════════════════════════════════════════════════════════════
# SCAN WORKER
# ══════════════════════════════════════════════════════════════════════════════


class ETFScanWorker(QThread):
    row_ready = pyqtSignal(dict)
    finished_all = pyqtSignal(list)

    def __init__(self, symbols, min_score=0):
        super().__init__()
        self.symbols = symbols
        self.min_score = min_score
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from modules.data import get_price_daily
        from modules.etf_momentum import compute_etf_momentum, momentum_verdict

        results = []
        for sym in self.symbols:
            if self._cancelled:
                break
            try:
                df, msg = get_price_daily(sym, days=400)
                if df.empty or len(df) < 60:
                    continue
                sig = compute_etf_momentum(df, symbol=sym)
                if not sig.data_sufficient:
                    continue

                # Compute ADTV (20-day avg traded value in ₹ Cr)
                if "Volume" in df.columns and "Close" in df.columns:
                    traded_val = df["Close"].tail(20) * df["Volume"].tail(20)
                    adtv_cr = float(traded_val.mean()) / 1e7  # ₹ → Cr
                else:
                    adtv_cr = 0.0

                row = {
                    "symbol": sym,
                    "last_price": sig.close,
                    "score": sig.momentum_score,
                    "verdict": momentum_verdict(sig),
                    "sizing": sig.sizing if adtv_cr >= 1.0 else "SKIP",
                    "adtv": round(adtv_cr, 1),
                    "trend": "▲ UP" if sig.in_uptrend else "▼ DN",
                    "wr": round(sig.wr_20, 1),
                    "dip": f"{sig.wr_min_10:.0f}" if sig.dip_detected else "—",
                    "roc": round(sig.roc_10, 1),
                    "mfi": round(sig.mfi, 0),
                    "adx": round(sig.adx, 0),
                    "pct_sma50": round(sig.pct_from_sma50, 1),
                    "reason": sig.entry_reason,
                    "entry": sig.entry_triggered and adtv_cr >= 1.0,
                    "in_uptrend": sig.in_uptrend,
                    "dip_detected": sig.dip_detected,
                    "recovered": sig.recovered,
                    "illiquid": adtv_cr < 1.0,
                }
                # Override verdict for illiquid ETFs
                if adtv_cr < 1.0:
                    row["verdict"] = f"⚠ ILLIQUID ({adtv_cr:.1f} Cr) — avoid"
                if sig.momentum_score >= self.min_score:
                    results.append(row)
                    self.row_ready.emit(row)
            except Exception as e:
                logger.debug("ETF scan %s failed: %s", sym, e)
        self.finished_all.emit(results)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE MODEL
# ══════════════════════════════════════════════════════════════════════════════


class ETFModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._data: list[dict] = []

    def set_results(self, rows):
        self.beginResetModel()
        self._data = rows
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = []
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            return _COLUMNS[section][0]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._data[index.row()]
        _, col_key, _ = _COLUMNS[index.column()]
        val = row.get(col_key)

        if role == Qt.ItemDataRole.DisplayRole:
            if val is None:
                return "—"
            if col_key == "last_price":
                return f"{val:,.2f}"
            if col_key == "wr" and isinstance(val, (int, float)):
                return f"{val:.0f}"
            if col_key == "roc" and isinstance(val, (int, float)):
                return f"{val:+.1f}%"
            if col_key == "pct_sma50" and isinstance(val, (int, float)):
                return f"{val:+.1f}%"
            if col_key == "mfi" and isinstance(val, (int, float)):
                return f"{val:.0f}"
            if col_key == "adx" and isinstance(val, (int, float)):
                return f"{val:.0f}"
            if col_key == "adtv" and isinstance(val, (int, float)):
                if val >= 100:
                    return f"{val:.0f}"
                if val >= 10:
                    return f"{val:.0f}"
                return f"{val:.1f}"
            if col_key == "score" and isinstance(val, (int, float)):
                entry = row.get("entry", False)
                return f"{'▶' if entry else ''}{int(val)}"
            return str(val)

        if role == Qt.ItemDataRole.ForegroundRole:
            if col_key == "score" and isinstance(val, (int, float)):
                return QBrush(QColor(score_color(int(val))))
            if col_key == "verdict" and val:
                v = str(val)
                if "ILLIQUID" in v:
                    return QBrush(QColor(RED))
                if "BUY" in v:
                    return QBrush(QColor(GREEN))
                if "WAIT" in v:
                    return QBrush(QColor(GOLD))
                if "WATCH" in v:
                    return QBrush(QColor(BLUE))
                if "HOLD" in v:
                    return QBrush(QColor(EM))
                return QBrush(QColor(MUTED))
            if col_key == "sizing":
                c = {"FULL": GREEN, "HALF": GOLD, "SKIP": MUTED}
                return QBrush(QColor(c.get(str(val), MUTED)))
            if col_key == "trend":
                return QBrush(QColor(GREEN if "UP" in str(val) else RED))
            if col_key == "adtv" and isinstance(val, (int, float)):
                if val >= 50:
                    return QBrush(QColor(GREEN))
                if val >= 10:
                    return QBrush(QColor(EM))
                if val >= 1:
                    return QBrush(QColor(GOLD))
                return QBrush(QColor(RED))
            if col_key == "wr" and isinstance(val, (int, float)):
                return QBrush(
                    QColor(RED if val < -60 else (GOLD if val < -30 else GREEN))
                )
            if col_key == "roc" and isinstance(val, (int, float)):
                return QBrush(QColor(GREEN if val > 0 else RED))
            if col_key == "mfi" and isinstance(val, (int, float)):
                return QBrush(
                    QColor(GREEN if val >= 50 else (MUTED if val >= 30 else RED))
                )
            if col_key == "pct_sma50" and isinstance(val, (int, float)):
                return QBrush(QColor(GREEN if val > 0 else RED))

        if role == Qt.ItemDataRole.BackgroundRole:
            if col_key == "score" and row.get("entry"):
                return QBrush(QColor(16, 185, 129, 40))

        if role == Qt.ItemDataRole.FontRole:
            if col_key in ("score", "sizing", "symbol", "verdict"):
                f = QFont(FONT_MONO, 10)
                f.setBold(True)
                return f
            if col_key in (
                "last_price",
                "wr",
                "roc",
                "mfi",
                "adx",
                "pct_sma50",
                "adtv",
            ):
                return QFont(FONT_MONO, 10)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col_key in (
                "last_price",
                "score",
                "wr",
                "roc",
                "mfi",
                "adx",
                "pct_sma50",
                "adtv",
            ):
                return Qt.AlignmentFlag.AlignCenter

        if role == Qt.ItemDataRole.UserRole:
            # Raw values for sorting
            if val is None:
                return -9999
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                # Score column has ▶ prefix
                try:
                    return float(val.replace("▶", "").replace("▷", "").strip())
                except ValueError:
                    pass
            return str(val)

        return None


# ══════════════════════════════════════════════════════════════════════════════
# ETF TAB
# ══════════════════════════════════════════════════════════════════════════════


class ETFMomentumTab(QWidget):
    symbol_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 4)
        root.setSpacing(5)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._run_btn = QPushButton("Scan ETFs")
        self._run_btn.setFixedWidth(100)
        self._run_btn.setStyleSheet(
            f"background: {EM}; color: {BG}; font-weight: bold; border-radius: 4px; padding: 5px;"
        )
        self._run_btn.clicked.connect(self._on_run)
        ctrl.addWidget(self._run_btn)

        ctrl.addWidget(QLabel("Min Score:"))
        self._score_slider = QSlider(Qt.Orientation.Horizontal)
        self._score_slider.setRange(0, 80)
        self._score_slider.setValue(30)
        self._score_slider.setFixedWidth(120)
        ctrl.addWidget(self._score_slider)
        self._score_val = QLabel("30")
        self._score_val.setFont(QFont(FONT_MONO, 9))
        self._score_slider.valueChanged.connect(
            lambda v: self._score_val.setText(str(v))
        )
        ctrl.addWidget(self._score_val)

        ctrl.addStretch(1)
        self._count_lbl = QLabel("")
        self._count_lbl.setFont(QFont(FONT_UI, 8))
        self._count_lbl.setStyleSheet(f"color: {MUTED};")
        ctrl.addWidget(self._count_lbl)

        self._progress = QLabel("")
        self._progress.setFont(QFont(FONT_UI, 8))
        self._progress.setStyleSheet(f"color: {GOLD};")
        ctrl.addWidget(self._progress)
        root.addLayout(ctrl)

        # ── Strategy strip ────────────────────────────────────────────────
        sstrip = QFrame()
        sstrip.setFixedHeight(22)
        sstrip.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 3px;"
        )
        sl = QHBoxLayout(sstrip)
        sl.setContentsMargins(8, 0, 8, 0)
        sl.setSpacing(6)
        sl.addWidget(Badge("MOMENTUM"))
        strat = QLabel(
            "Pullback-in-trend · SMA(50) trend · WR(20) dip < -60 → recover · Trail 2×ATR + SMA(20) exit · PT 10% · ADTV ≥ ₹1Cr"
        )
        strat.setFont(QFont(FONT_MONO, 7))
        strat.setStyleSheet(f"color: {MUTED};")
        sl.addWidget(strat, stretch=1)
        root.addWidget(sstrip)

        # ── Table ─────────────────────────────────────────────────────────
        self._model = ETFModel()
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(Qt.ItemDataRole.UserRole)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(
            f"QTableView {{ background: {BG}; gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QTableView::item {{ padding: 2px 4px; }}"
            f"QHeaderView::section {{ background: {SURFACE}; color: {EM}; border: 1px solid {BORDER};"
            f"  font-size: 10px; font-weight: bold; padding: 4px; }}"
        )
        hdr = self._table.horizontalHeader()
        for i, (_, _, w) in enumerate(_COLUMNS):
            if w > 0:
                hdr.resizeSection(i, w)
        hdr.setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(28)
        self._table.doubleClicked.connect(self._on_row_double_click)
        root.addWidget(self._table, stretch=1)

    # ── Actions ───────────────────────────────────────────────────────────

    def _on_run(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._run_btn.setText("Scan ETFs")
            return

        self._model.clear()
        self._run_btn.setText("Cancel")
        self._progress.setText("Scanning...")
        self._count_lbl.setText("")

        self._worker = ETFScanWorker(
            ETF_SYMBOLS,
            min_score=self._score_slider.value(),
        )
        self._worker.row_ready.connect(self._on_row)
        self._worker.finished_all.connect(self._on_done)
        self._worker.start()

    def _on_row(self, row):
        self._progress.setText(f"Scanning... {row['symbol']}")

    def _on_done(self, results):
        results.sort(key=lambda r: -r.get("score", 0))
        self._model.beginResetModel()
        self._model._data = results
        self._model.endResetModel()

        self._run_btn.setText("Scan ETFs")
        self._progress.setText("")
        entries = sum(1 for r in results if r.get("entry"))
        illiquid = sum(1 for r in results if r.get("illiquid"))
        self._count_lbl.setText(
            f"{len(results)} ETFs · {entries} entries · {illiquid} illiquid"
        )

    def _on_row_double_click(self, index):
        src = self._proxy.mapToSource(index)
        row = self._model._data[src.row()]
        sym = row.get("symbol", "")
        if sym:
            self.symbol_selected.emit(sym)
