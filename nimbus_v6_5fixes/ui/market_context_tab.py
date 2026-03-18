"""
ui/market_context_tab.py  v4
─────────────────────────────
All improvements integrated in one file:
  • 7 columns: Sector | 10D% | 1M% | 3M% | RS | Adj | Rotation
  • RS  = pure momentum score (classify_rotation weighted 3-tf)
  • Adj = quality-adjusted score (RS + VSR + NAV premium factors)
  • Rotation cell shows label + inline quality flags (PREM / THIN-VOL etc.)
  • Colour: LEADING turns gold if quality flags present
  • Interactive column resizing (drag any header divider)
  • Hover tooltip shows raw RS values, bar count, VSR, NAV delta
  • 15-min TTL cache via sector_rotation.fetch_sector_data()
  • Manual Refresh bypasses cache; tab-show uses cache
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialog,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QDateEdit,
    QDialogButtonBox,
    QAbstractItemView,
)
from PyQt6.QtGui import QFont, QColor

from ui.theme import (
    BG,
    SURFACE,
    S2,
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
)
from ui.components import Badge

logger = logging.getLogger(__name__)

_ROT_STYLE = {
    "LEADING": (EM, "LEADING"),
    "WEAKENING": (GOLD, "WEAKENING"),
    "IMPROVING": (BLUE, "IMPROVING"),
    "LAGGING": (RED, "LAGGING"),
    "UNKNOWN": (MUTED, "UNKNOWN"),
}


# ══════════════════════════════════════════════════════════════════════════════
# FETCH WRAPPER
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_sectors(force: bool = False) -> list[dict]:
    from modules.sector_rotation import fetch_sector_data

    return fetch_sector_data(force=force)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════════════


class _FetchWorker(QThread):
    done = pyqtSignal(dict)

    def __init__(self, force: bool = False, parent=None):
        super().__init__(parent)
        self._force = force

    def run(self):
        result: dict = {}
        try:
            from modules.signal_engine import fetch_fii_dii

            result["fii_dii"] = fetch_fii_dii()
        except Exception as e:
            logger.warning("FII/DII fetch: %s", e)
            result["fii_dii"] = None
        try:
            result["sectors"] = _fetch_sectors(force=self._force)
        except Exception as e:
            logger.warning("Sector fetch: %s", e)
            result["sectors"] = []
        self.done.emit(result)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _styled_table(cols: list[str], col_widths: list[int]) -> QTableWidget:
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(26)
    t.setShowGrid(False)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    hdr = t.horizontalHeader()
    hdr.setStretchLastSection(False)  # ← was True, caused the blowout

    for i, w in enumerate(col_widths):
        if w == 0:  # 0 = this column gets remaining space
            hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        else:
            hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
            hdr.resizeSection(i, w)

    return t


def _pct_item(val: Optional[float]) -> QTableWidgetItem:
    if val is None:
        itm = QTableWidgetItem("—")
        itm.setForeground(QColor(MUTED))
    else:
        itm = QTableWidgetItem(f"{val:+.1f}%")
        itm.setForeground(QColor(EM if val > 0 else (RED if val < 0 else MUTED)))
    itm.setTextAlignment(
        Qt.AlignmentFlag.AlignCenter
    )  # ← was AlignRight | AlignVCenter
    return itm


def _score_item(val: Optional[float], *, show_plus: bool = True) -> QTableWidgetItem:
    if val is None:
        itm = QTableWidgetItem("—")
        itm.setForeground(QColor(MUTED))
    else:
        fmt = f"{val:+.2f}" if show_plus else f"{val:.2f}"
        itm = QTableWidgetItem(fmt)
        color = (
            EM if val > 2.0 else (GREEN if val > 0 else (GOLD if val > -2.0 else RED))
        )
        itm.setForeground(QColor(color))
    itm.setTextAlignment(
        Qt.AlignmentFlag.AlignCenter
    )  # ← was AlignRight | AlignVCenter
    return itm


def _sector_tooltip(s: dict) -> str:
    vsr = s.get("vsr")
    nav_pct = s.get("nav_pct")
    lines = [
        f"Ticker    : {s.get('ticker', '—')}",
        f"Bars      : {s.get('valid_bars', '—')}",
        "─────────────────────────",
        (
            f"RS 10D    : {s['rs_10d']:+.3f}"
            if s.get("rs_10d") is not None
            else "RS 10D    : n/a"
        ),
        (
            f"RS  1M    : {s['rs_1m']:+.3f}"
            if s.get("rs_1m") is not None
            else "RS  1M    : n/a"
        ),
        (
            f"RS  3M    : {s['rs_3m']:+.3f}"
            if s.get("rs_3m") is not None
            else "RS  3M    : n/a"
        ),
        "─────────────────────────",
        f"RS Score  : {s.get('rs_score', 0):+.2f}  (pure momentum)",
        f"Adj Score : {s.get('adj_score', 0):+.2f}  (w/ quality)",
        "─────────────────────────",
        (f"VSR       : {vsr:.2f}" if vsr is not None else "VSR       : n/a"),
        (f"NAV Delta : {nav_pct:+.2f}%" if nav_pct is not None else "NAV Delta : n/a"),
    ]
    if s.get("flags"):
        lines += ["─────────────────────────", "Flags     : " + "  ".join(s["flags"])]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WIDGET
# ══════════════════════════════════════════════════════════════════════════════


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

        # Top bar
        top = QHBoxLayout()
        title = QLabel("MARKET CONTEXT")
        title.setFont(QFont(FONT_UI, 10))
        title.setStyleSheet(f"color: {WHITE}; font-weight: bold;")
        top.addWidget(title)
        top.addStretch(1)
        self._status = QLabel("")
        self._status.setFont(QFont(FONT_MONO, 8))
        self._status.setStyleSheet(f"color: {MUTED};")
        top.addWidget(self._status)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("primary")
        self._refresh_btn.setFixedHeight(26)
        self._refresh_btn.clicked.connect(self._on_refresh)
        top.addWidget(self._refresh_btn)
        root.addLayout(top)

        # FII/DII
        flow = QFrame()
        flow.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;"
        )
        fl = QHBoxLayout(flow)
        fl.setContentsMargins(16, 12, 16, 12)
        fl.setSpacing(30)

        fc = QVBoxLayout()
        fc.setSpacing(2)
        fc.addWidget(self._sec_lbl("FII / FPI CASH FLOW"))
        self._fii_net = QLabel("--")
        self._fii_net.setFont(QFont(FONT_MONO, 18))
        self._fii_net.setStyleSheet(f"color: {MUTED}; font-weight: bold;")
        fc.addWidget(self._fii_net)
        self._fii_detail = QLabel("")
        self._fii_detail.setFont(QFont(FONT_MONO, 9))
        self._fii_detail.setStyleSheet(f"color: {MUTED};")
        fc.addWidget(self._fii_detail)
        fl.addLayout(fc)

        dc = QVBoxLayout()
        dc.setSpacing(2)
        dc.addWidget(self._sec_lbl("DII CASH FLOW"))
        self._dii_net = QLabel("--")
        self._dii_net.setFont(QFont(FONT_MONO, 18))
        self._dii_net.setStyleSheet(f"color: {MUTED}; font-weight: bold;")
        dc.addWidget(self._dii_net)
        self._dii_detail = QLabel("")
        self._dii_detail.setFont(QFont(FONT_MONO, 9))
        self._dii_detail.setStyleSheet(f"color: {MUTED};")
        dc.addWidget(self._dii_detail)
        fl.addLayout(dc)

        fl.addStretch(1)
        self._flow_badge = Badge("--")
        self._flow_badge.setFixedHeight(24)
        fl.addWidget(self._flow_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(flow)

        # Sector rotation header
        sh = QHBoxLayout()
        sh.addWidget(self._sec_lbl("SECTOR ROTATION — vs NIFTY 500"))
        sh.addStretch(1)
        for lbl_text, color in [
            ("LEADING", EM),
            ("WEAKENING", GOLD),
            ("IMPROVING", BLUE),
            ("LAGGING", RED),
        ]:
            chip = QLabel(f"● {lbl_text}")
            chip.setFont(QFont(FONT_MONO, 7))
            chip.setStyleSheet(f"color: {color};")
            sh.addWidget(chip)
        root.addLayout(sh)

        # 7 columns: Sector | 10D% | 1M% | 3M% | RS | Adj | Rotation(stretch)
        # RS  = pure momentum score
        # Adj = quality-adjusted (RS + VSR + NAV factors)
        self._sector_table = _styled_table(
            ["Sector", "10D %", "1M %", "3M %", "RS", "Adj", "Signal", "Rotation"],
            [155, 75, 75, 75, 75, 75, 80, 120],  # all fixed
        )
        self._sector_table.setFixedHeight(430)
        root.addWidget(self._sector_table)

        # Events
        eh = QHBoxLayout()
        eh.addWidget(self._sec_lbl("UPCOMING EVENTS"))
        eh.addStretch(1)
        add_btn = QPushButton("+ Add Event")
        add_btn.setFixedHeight(22)
        add_btn.clicked.connect(self._on_add_event)
        eh.addWidget(add_btn)
        root.addLayout(eh)

        self._event_table = _styled_table(
            ["Date", "Days", "Event", "Symbols", "Impact"],
            [95, 45, 0, 140, 75],  # 0 = Event stretches
        )
        root.addWidget(self._event_table, stretch=1)

    def _sec_lbl(self, text: str) -> QLabel:
        l = QLabel(text.upper())
        l.setFont(QFont(FONT_UI, 7))
        l.setStyleSheet(f"color: {MUTED}; letter-spacing: 1px;")
        return l

    # Lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        if self._worker is None or not self._worker.isRunning():
            self._on_refresh(force=False)

    def _on_refresh(self, force: bool = True):
        self._refresh_btn.setEnabled(False)
        self._status.setText("Loading...")
        self._worker = _FetchWorker(force=force)
        self._worker.done.connect(self._on_data)
        self._worker.start()

    # Data callback
    def _on_data(self, data: dict):
        self._refresh_btn.setEnabled(True)
        self._status.setText(datetime.datetime.now().strftime("Updated %H:%M:%S"))

        fd = data.get("fii_dii")
        if fd:
            fii = fd.get("fii_net", 0)
            dii = fd.get("dii_net", 0)
            self._fii_net.setText(f"NET: {fii:+,.0f} Cr")
            self._fii_net.setStyleSheet(
                f"color: {EM if fii > 0 else RED}; font-weight: bold;"
            )
            self._fii_detail.setText(
                f"Buy: {fd.get('fii_buy',0):,.0f} Cr | Sell: {fd.get('fii_sell',0):,.0f} Cr"
            )
            self._dii_net.setText(f"NET: {dii:+,.0f} Cr")
            self._dii_net.setStyleSheet(
                f"color: {EM if dii > 0 else RED}; font-weight: bold;"
            )
            self._dii_detail.setText(
                f"Buy: {fd.get('dii_buy',0):,.0f} Cr | Sell: {fd.get('dii_sell',0):,.0f} Cr"
            )
            sig = fd.get("signal", "N/A")
            sig_s = {
                "INSTITUTIONAL TAILWIND": "BULLISH",
                "DII SUPPORT / FII EXIT": "LATE",
                "INSTITUTIONAL HEADWIND": "BEARISH",
                "FII INFLOW / DII EXIT": "WATCH",
            }
            self._flow_badge.set_badge(sig, sig_s.get(sig, "NEUTRAL"))

        _ENTRY_STYLE = {
            "STRONG": EM,
            "GOOD": GREEN,
            "CAUTION": GOLD,
            "AVOID": RED,
        }

        # Sector rotation
        sectors = data.get("sectors", [])
        self._sector_table.setRowCount(len(sectors))
        for i, s in enumerate(sectors):
            # Col 0 — name + tooltip
            name_itm = QTableWidgetItem(s["name"])
            name_itm.setFont(QFont(FONT_MONO, 9))
            name_itm.setToolTip(_sector_tooltip(s))
            self._sector_table.setItem(i, 0, name_itm)

            # Col 1-3 — absolute returns
            self._sector_table.setItem(i, 1, _pct_item(s.get("ret_10d")))
            self._sector_table.setItem(i, 2, _pct_item(s.get("ret_1m")))
            self._sector_table.setItem(i, 3, _pct_item(s.get("ret_3m")))

            # Col 4 — RS Score (pure momentum, 3-tf weighted)
            self._sector_table.setItem(i, 4, _score_item(s.get("rs_score")))

            # Col 5 — Adj Score (RS + quality)
            self._sector_table.setItem(i, 5, _score_item(s.get("adj_score")))

            # Col 6 — Entry Signal (price structure viability from analyze_price_only)
            entry = s.get("entry_label")
            if entry:
                e_itm = QTableWidgetItem(entry)
                e_itm.setForeground(QColor(_ENTRY_STYLE.get(entry, MUTED)))
            else:
                e_itm = QTableWidgetItem("—")
                e_itm.setForeground(QColor(MUTED))
            e_itm.setFont(QFont(FONT_MONO, 8))
            e_itm.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._sector_table.setItem(i, 6, e_itm)

            # Col 7 — Rotation label + inline flags
            rot = s.get("rotation", "—")
            flags = s.get("flags", [])
            color, label_text = _ROT_STYLE.get(rot, (MUTED, rot))
            if flags and color in (EM, BLUE):
                color = GOLD
            display = label_text + ("  " + "  ".join(flags) if flags else "")
            rot_itm = QTableWidgetItem(display)
            rot_itm.setForeground(QColor(color))
            rot_itm.setFont(QFont(FONT_MONO, 8))
            rot_itm.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._sector_table.setItem(i, 7, rot_itm)

        self._refresh_events()

    # Events calendar
    def _refresh_events(self):
        from modules.signal_engine import load_event_calendar

        today = datetime.date.today()
        upcoming = []
        for ev in load_event_calendar():
            try:
                d = datetime.date.fromisoformat(ev["date"])
                days = (d - today).days
                if -1 <= days <= 30:
                    upcoming.append({**ev, "days_away": days})
            except Exception:
                pass
        upcoming.sort(key=lambda x: x.get("days_away", 99))
        self._event_table.setRowCount(len(upcoming))
        for i, ev in enumerate(upcoming):
            self._event_table.setItem(i, 0, QTableWidgetItem(ev["date"]))
            da = QTableWidgetItem(f"{ev['days_away']}d")
            if ev["days_away"] <= 2:
                da.setForeground(QColor(RED))
            da.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._event_table.setItem(i, 1, da)
            self._event_table.setItem(i, 2, QTableWidgetItem(ev.get("event", "")))
            self._event_table.setItem(
                i, 3, QTableWidgetItem(", ".join(ev.get("symbols", [])))
            )
            imp = QTableWidgetItem(ev.get("impact", ""))
            if ev.get("impact") == "HIGH":
                imp.setForeground(QColor(RED))
            imp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._event_table.setItem(i, 4, imp)

    def _on_add_event(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Event")
        dlg.setMinimumWidth(350)
        form = QFormLayout(dlg)
        de = QDateEdit()
        de.setCalendarPopup(True)
        de.setDate(datetime.date.today())
        form.addRow("Date:", de)
        ee = QLineEdit()
        ee.setPlaceholderText("e.g. RBI Policy")
        form.addRow("Event:", ee)
        se = QLineEdit()
        se.setPlaceholderText("NIFTY,BANKNIFTY or ALL")
        form.addRow("Symbols:", se)
        ic = QComboBox()
        ic.addItems(["HIGH", "MEDIUM", "LOW"])
        form.addRow("Impact:", ic)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec():
            from modules.signal_engine import load_event_calendar, save_event_calendar

            evs = load_event_calendar()
            evs.append(
                {
                    "date": de.date().toString("yyyy-MM-dd"),
                    "event": ee.text().strip(),
                    "symbols": [
                        s.strip().upper() for s in se.text().split(",") if s.strip()
                    ],
                    "impact": ic.currentText(),
                }
            )
            save_event_calendar(evs)
            self._refresh_events()
