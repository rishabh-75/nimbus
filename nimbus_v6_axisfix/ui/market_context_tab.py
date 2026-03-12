"""
ui/market_context_tab.py — Market-wide intelligence layer (Tab 4).
Polished: proper column sizing, styled tables, consistent spacing.
"""
from __future__ import annotations
import datetime, logging
from typing import Optional
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout,
    QLineEdit, QComboBox, QDateEdit, QDialogButtonBox, QAbstractItemView,
)
from PyQt6.QtGui import QFont, QColor
from ui.theme import (BG, SURFACE, S2, BORDER, EM, RED, GOLD, MUTED, WHITE,
                       GREEN, BLUE, VIOLET, FONT_MONO, FONT_UI)
from ui.components import Badge

logger = logging.getLogger(__name__)


class _FetchWorker(QThread):
    done = pyqtSignal(dict)
    def run(self):
        result = {}
        try:
            from modules.signal_engine import fetch_fii_dii
            result["fii_dii"] = fetch_fii_dii()
        except Exception as e:
            logger.warning("FII/DII fetch: %s", e); result["fii_dii"] = None
        try:
            result["sectors"] = _fetch_sectors()
        except Exception as e:
            logger.warning("Sector fetch: %s", e); result["sectors"] = []
        self.done.emit(result)


def _fetch_sectors():
    from modules.signal_engine import _get_return
    from modules.sector_map import SECTOR_TICKERS, SECTOR_NAMES, MARKET_TICKER, MARKET_FALLBACK
    mkt = _get_return(MARKET_TICKER, 10)
    if mkt is None: mkt = _get_return(MARKET_FALLBACK, 10)
    rows = []
    for t in SECTOR_TICKERS:
        ret = _get_return(t, 10)
        if ret is not None and mkt is not None and mkt != 0:
            rows.append({"name": SECTOR_NAMES.get(t, t), "return": round(ret*100, 1),
                         "vs_market": round((ret/mkt - 1)*100, 1)})
    rows.sort(key=lambda r: r["return"], reverse=True)
    return rows


def _styled_table(cols, col_widths=None):
    """Create a consistently styled QTableWidget."""
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(26)
    t.setShowGrid(False)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    hdr = t.horizontalHeader()
    if col_widths:
        for i, w in enumerate(col_widths):
            if w == 0:
                hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            else:
                hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                hdr.resizeSection(i, w)
    else:
        hdr.setStretchLastSection(True)
    return t


class MarketContextTab(QWidget):
    view_symbol = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[_FetchWorker] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(10)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        title = QLabel("MARKET CONTEXT")
        title.setFont(QFont(FONT_UI, 10)); title.setStyleSheet(f"color: {WHITE}; font-weight: bold;")
        top.addWidget(title)
        top.addStretch(1)
        self._status = QLabel("")
        self._status.setFont(QFont(FONT_MONO, 8)); self._status.setStyleSheet(f"color: {MUTED};")
        top.addWidget(self._status)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("primary"); self._refresh_btn.setFixedHeight(26)
        self._refresh_btn.clicked.connect(self._on_refresh)
        top.addWidget(self._refresh_btn)
        root.addLayout(top)

        # ── Row 1: FII/DII ────────────────────────────────────────────────────
        flow = QFrame()
        flow.setStyleSheet(f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;")
        fl = QHBoxLayout(flow); fl.setContentsMargins(16, 12, 16, 12); fl.setSpacing(30)

        # FII
        fc = QVBoxLayout(); fc.setSpacing(2)
        fc.addWidget(self._sec_lbl("FII / FPI CASH FLOW"))
        self._fii_net = QLabel("--")
        self._fii_net.setFont(QFont(FONT_MONO, 18)); self._fii_net.setStyleSheet(f"color: {MUTED}; font-weight: bold;")
        fc.addWidget(self._fii_net)
        self._fii_detail = QLabel("")
        self._fii_detail.setFont(QFont(FONT_MONO, 9)); self._fii_detail.setStyleSheet(f"color: {MUTED};")
        fc.addWidget(self._fii_detail)
        fl.addLayout(fc)

        # DII
        dc = QVBoxLayout(); dc.setSpacing(2)
        dc.addWidget(self._sec_lbl("DII CASH FLOW"))
        self._dii_net = QLabel("--")
        self._dii_net.setFont(QFont(FONT_MONO, 18)); self._dii_net.setStyleSheet(f"color: {MUTED}; font-weight: bold;")
        dc.addWidget(self._dii_net)
        self._dii_detail = QLabel("")
        self._dii_detail.setFont(QFont(FONT_MONO, 9)); self._dii_detail.setStyleSheet(f"color: {MUTED};")
        dc.addWidget(self._dii_detail)
        fl.addLayout(dc)

        fl.addStretch(1)
        self._flow_badge = Badge("--"); self._flow_badge.setFixedHeight(24)
        fl.addWidget(self._flow_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(flow)

        # ── Row 2: Sector Heatmap ─────────────────────────────────────────────
        root.addWidget(self._sec_lbl("SECTOR PERFORMANCE (10 DAY)"))
        self._sector_table = _styled_table(
            ["Sector", "Return %", "vs NIFTY 500"],
            [200, 100, 0])
        self._sector_table.setFixedHeight(220)
        root.addWidget(self._sector_table)

        # ── Row 3: Events ─────────────────────────────────────────────────────
        eh = QHBoxLayout()
        eh.addWidget(self._sec_lbl("UPCOMING EVENTS"))
        eh.addStretch(1)
        add_btn = QPushButton("+ Add Event"); add_btn.setFixedHeight(22)
        add_btn.clicked.connect(self._on_add_event)
        eh.addWidget(add_btn)
        root.addLayout(eh)

        self._event_table = _styled_table(
            ["Date", "Days", "Event", "Symbols", "Impact"],
            [100, 50, 0, 130, 65])
        root.addWidget(self._event_table, stretch=1)

    def _sec_lbl(self, text):
        l = QLabel(text.upper()); l.setFont(QFont(FONT_UI, 7))
        l.setStyleSheet(f"color: {MUTED}; letter-spacing: 1px;"); return l

    def showEvent(self, event):
        super().showEvent(event)
        if self._worker is None or not self._worker.isRunning():
            self._on_refresh()

    def _on_refresh(self):
        self._refresh_btn.setEnabled(False); self._status.setText("Loading...")
        self._worker = _FetchWorker()
        self._worker.done.connect(self._on_data)
        self._worker.start()

    def _on_data(self, data):
        self._refresh_btn.setEnabled(True)
        self._status.setText(datetime.datetime.now().strftime("Updated %H:%M:%S"))

        fd = data.get("fii_dii")
        if fd:
            fii, dii = fd.get("fii_net", 0), fd.get("dii_net", 0)
            self._fii_net.setText(f"NET: {fii:+,.0f} Cr")
            self._fii_net.setStyleSheet(f"color: {EM if fii > 0 else RED}; font-weight: bold;")
            self._fii_detail.setText(f"Buy: {fd.get('fii_buy',0):,.0f} Cr  |  Sell: {fd.get('fii_sell',0):,.0f} Cr")
            self._dii_net.setText(f"NET: {dii:+,.0f} Cr")
            self._dii_net.setStyleSheet(f"color: {EM if dii > 0 else RED}; font-weight: bold;")
            self._dii_detail.setText(f"Buy: {fd.get('dii_buy',0):,.0f} Cr  |  Sell: {fd.get('dii_sell',0):,.0f} Cr")
            sig = fd.get("signal", "N/A")
            sig_s = {"INSTITUTIONAL TAILWIND": "BULLISH", "DII SUPPORT / FII EXIT": "LATE",
                     "INSTITUTIONAL HEADWIND": "BEARISH", "FII INFLOW / DII EXIT": "WATCH"}
            self._flow_badge.set_badge(sig, sig_s.get(sig, "NEUTRAL"))

        sectors = data.get("sectors", [])
        self._sector_table.setRowCount(len(sectors))
        for i, s in enumerate(sectors):
            self._sector_table.setItem(i, 0, QTableWidgetItem(s["name"]))
            ri = QTableWidgetItem(f"{s['return']:+.1f}%")
            ri.setForeground(QColor(EM if s["return"] > 0 else RED))
            ri.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._sector_table.setItem(i, 1, ri)
            vs = s["vs_market"]
            vi = QTableWidgetItem(f"{vs:+.1f}%")
            vi.setForeground(QColor(EM if vs > 2 else (RED if vs < -2 else GOLD)))
            vi.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._sector_table.setItem(i, 2, vi)

        self._refresh_events()

    def _refresh_events(self):
        from modules.signal_engine import load_event_calendar
        today = datetime.date.today()
        upcoming = []
        for ev in load_event_calendar():
            try:
                d = datetime.date.fromisoformat(ev["date"])
                days = (d - today).days
                if -1 <= days <= 30: upcoming.append({**ev, "days_away": days})
            except: pass
        upcoming.sort(key=lambda x: x.get("days_away", 99))

        self._event_table.setRowCount(len(upcoming))
        for i, ev in enumerate(upcoming):
            self._event_table.setItem(i, 0, QTableWidgetItem(ev["date"]))
            da = QTableWidgetItem(f"{ev['days_away']}d")
            if ev["days_away"] <= 2: da.setForeground(QColor(RED))
            da.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._event_table.setItem(i, 1, da)
            self._event_table.setItem(i, 2, QTableWidgetItem(ev.get("event", "")))
            self._event_table.setItem(i, 3, QTableWidgetItem(", ".join(ev.get("symbols", []))))
            imp = QTableWidgetItem(ev.get("impact", ""))
            if ev.get("impact") == "HIGH": imp.setForeground(QColor(RED))
            imp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._event_table.setItem(i, 4, imp)

    def _on_add_event(self):
        dlg = QDialog(self); dlg.setWindowTitle("Add Event"); dlg.setMinimumWidth(350)
        form = QFormLayout(dlg)
        de = QDateEdit(); de.setCalendarPopup(True); de.setDate(datetime.date.today())
        form.addRow("Date:", de)
        ee = QLineEdit(); ee.setPlaceholderText("e.g. RBI Policy"); form.addRow("Event:", ee)
        se = QLineEdit(); se.setPlaceholderText("NIFTY,BANKNIFTY or ALL"); form.addRow("Symbols:", se)
        ic = QComboBox(); ic.addItems(["HIGH", "MEDIUM", "LOW"]); form.addRow("Impact:", ic)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); form.addRow(bb)
        if dlg.exec():
            from modules.signal_engine import load_event_calendar, save_event_calendar
            evs = load_event_calendar()
            evs.append({"date": de.date().toString("yyyy-MM-dd"), "event": ee.text().strip(),
                        "symbols": [s.strip().upper() for s in se.text().split(",") if s.strip()],
                        "impact": ic.currentText()})
            save_event_calendar(evs); self._refresh_events()
