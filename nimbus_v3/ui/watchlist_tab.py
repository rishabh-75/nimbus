"""
ui/watchlist_tab.py
───────────────────
Watchlist tab for NIMBUS Qt.

Features:
    - SQLite-backed table (QAbstractTableModel)
    - Add / Remove controls
    - Alert banners (MID_BAND_BROKEN, FIRST_DIP, low viability)
    - View in Dashboard button → switches tab + loads data
    - Refresh all watchlist entries with live data
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, pyqtSignal,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableView, QHeaderView, QFrame,
    QAbstractItemView, QMessageBox,
)
from PyQt6.QtGui import QFont, QColor, QBrush

from ui.theme import (
    BG, SURFACE, S2, S3, BORDER, EM, RED, GOLD, MUTED, WHITE, GREEN, BLUE,
    FONT_MONO, FONT_UI, score_color,
)
from ui.watchlist_db import (
    load_watchlist, add_entry, remove_entry, save_watchlist,
)
from ui.workers import WatchlistWorker
from modules.watchlist import detect_alerts

logger = logging.getLogger(__name__)

_COLUMNS = [
    ("Symbol",   "symbol",          80),
    ("Added",    "added_at",        130),
    ("Entry",    "entry_price",     80),
    ("Stop",     "stop_price",      80),
    ("Target",   "target_price",    80),
    ("Score",    "live_score",      55),
    ("Tier",     "live_tier",       50),
    ("Size",     "live_size",       55),
    ("WR",       "live_wr",         50),
    ("MFI",      "live_mfi",        50),
    ("Notes",    "notes",           160),
]


# ══════════════════════════════════════════════════════════════════════════════
# TABLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class WatchlistModel(QAbstractTableModel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[dict] = []

    def set_data(self, entries: list[dict]):
        self.beginResetModel()
        self._data = entries
        self.endResetModel()

    def row_data(self, row: int) -> Optional[dict]:
        return self._data[row] if 0 <= row < len(self._data) else None

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
        entry = self._data[index.row()]
        col_name, col_key, _ = _COLUMNS[index.column()]
        live = entry.get("live", {})

        # Resolve value
        if col_key == "live_score":
            val = live.get("dm_score") or live.get("viability_score")
        elif col_key == "live_size":
            val = live.get("dm_sizing") or live.get("size_suggestion")
        elif col_key == "live_wr":
            val = live.get("dm_wr") or live.get("wr_50")
        elif col_key == "live_tier":
            val = live.get("dm_tier")
        elif col_key == "live_mfi":
            val = live.get("dm_mfi")
        else:
            val = entry.get(col_key)

        if role == Qt.ItemDataRole.DisplayRole:
            if val is None:
                return "—"
            if col_key in ("entry_price", "stop_price", "target_price") and isinstance(val, (int, float)):
                return f"{val:,.2f}"
            if col_key == "live_wr" and isinstance(val, (int, float)):
                return f"{val:.0f}"
            if col_key == "live_mfi" and isinstance(val, (int, float)):
                return f"{val:.0f}"
            if col_key == "live_tier":
                return {"PRIMARY": "PRI", "SECONDARY": "SEC", "NONE": "—"}.get(str(val), str(val))
            if col_key == "added_at" and isinstance(val, str):
                return val[:16]
            return str(val)

        if role == Qt.ItemDataRole.ForegroundRole:
            if col_key == "live_score" and isinstance(val, (int, float)):
                return QBrush(QColor(score_color(int(val))))
            if col_key == "live_size":
                sc = {"FULL": GREEN, "HALF": GOLD, "SKIP": RED, "AVOID": RED}
                return QBrush(QColor(sc.get(str(val), MUTED)))
            if col_key == "live_wr" and isinstance(val, (int, float)):
                return QBrush(QColor(EM if val < -50 else (GREEN if val < -30 else MUTED)))
            if col_key == "live_mfi" and isinstance(val, (int, float)):
                return QBrush(QColor(EM if val >= 50 else (GREEN if val >= 30 else RED)))
            if col_key == "live_tier":
                sc = {"PRIMARY": GREEN, "SECONDARY": GOLD, "NONE": MUTED}
                return QBrush(QColor(sc.get(str(val), MUTED)))

        if role == Qt.ItemDataRole.FontRole:
            if col_key in ("live_score", "live_size", "symbol"):
                f = QFont(FONT_MONO, 11)
                if col_key == "symbol":
                    f.setBold(True)
                return f

        return None


# ══════════════════════════════════════════════════════════════════════════════
# ALERT BANNER
# ══════════════════════════════════════════════════════════════════════════════

class AlertBanner(QFrame):
    """Dismissible alert banner. Severity: error (red) or warn (gold)."""

    dismissed = pyqtSignal()

    def __init__(self, message: str, severity: str = "warn", parent=None):
        super().__init__(parent)
        color = RED if severity == "error" else GOLD
        bg = "#2B0D0D" if severity == "error" else "#1F1A0D"
        self.setStyleSheet(
            f"background: {bg}; border-left: 4px solid {color}; "
            f"border-radius: 4px; padding: 4px 12px;"
        )
        self.setFixedHeight(32)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)

        icon = "●" if severity == "error" else "⚠"
        lbl = QLabel(f"{icon}  {message}")
        lbl.setFont(QFont(FONT_UI, 10))
        lbl.setStyleSheet(f"color: {WHITE}; font-weight: 600; border: none;")
        lay.addWidget(lbl, stretch=1)

        dismiss = QPushButton("Dismiss")
        dismiss.setFixedWidth(60)
        dismiss.setStyleSheet(
            f"color: {MUTED}; border: none; font-size: 9px;"
        )
        dismiss.clicked.connect(self._on_dismiss)
        lay.addWidget(dismiss)

    def _on_dismiss(self):
        self.setVisible(False)
        self.dismissed.emit()


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST TAB
# ══════════════════════════════════════════════════════════════════════════════

class WatchlistTab(QWidget):
    """Watchlist tab. Emits view_symbol(str) to switch to Dashboard."""

    view_symbol = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[dict] = []
        self._model = WatchlistModel()
        self._worker: Optional[WatchlistWorker] = None
        self._build_ui()
        self._load()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # ── Alert area ────────────────────────────────────────────────────────
        self._alert_area = QVBoxLayout()
        self._alert_area.setSpacing(4)
        layout.addLayout(self._alert_area)

        # ── Add controls ──────────────────────────────────────────────────────
        add_row = QHBoxLayout()
        add_row.setSpacing(6)

        add_row.addWidget(QLabel("Add Symbol:"))
        self._add_input = QLineEdit()
        self._add_input.setPlaceholderText("e.g. RELIANCE")
        self._add_input.setMaxLength(20)
        self._add_input.setFixedWidth(140)
        self._add_input.returnPressed.connect(self._on_add)
        add_row.addWidget(self._add_input)

        self._add_btn = QPushButton("+ Add")
        self._add_btn.setObjectName("primary")
        self._add_btn.clicked.connect(self._on_add)
        add_row.addWidget(self._add_btn)

        add_row.addSpacing(16)

        self._refresh_btn = QPushButton("Refresh All")
        self._refresh_btn.clicked.connect(self._on_refresh)
        add_row.addWidget(self._refresh_btn)

        add_row.addStretch(1)

        self._count_lbl = QLabel("")
        self._count_lbl.setFont(QFont(FONT_MONO, 9))
        self._count_lbl.setStyleSheet(f"color: {MUTED};")
        add_row.addWidget(self._count_lbl)

        layout.addLayout(add_row)

        # ── Table ─────────────────────────────────────────────────────────────
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(30)
        self._table.horizontalHeader().setStretchLastSection(True)

        hdr = self._table.horizontalHeader()
        for i, (_, _, w) in enumerate(_COLUMNS):
            hdr.resizeSection(i, w)

        self._table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, stretch=1)

        # ── Action buttons ────────────────────────────────────────────────────
        actions = QHBoxLayout()

        self._view_btn = QPushButton("View in Dashboard")
        self._view_btn.clicked.connect(self._on_view)
        actions.addWidget(self._view_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setObjectName("danger")
        self._remove_btn.clicked.connect(self._on_remove)
        actions.addWidget(self._remove_btn)

        actions.addStretch(1)
        layout.addLayout(actions)

    # ══════════════════════════════════════════════════════════════════════════
    # DATA
    # ══════════════════════════════════════════════════════════════════════════

    def _load(self):
        self._entries = load_watchlist()
        self._model.set_data(self._entries)
        self._count_lbl.setText(f"{len(self._entries)} symbols")

    def _on_add(self):
        sym = self._add_input.text().strip().upper()
        if not sym:
            return
        self._entries = add_entry(self._entries, sym)
        self._model.set_data(self._entries)
        self._add_input.clear()
        self._count_lbl.setText(f"{len(self._entries)} symbols")
        logger.info("Watchlist: added %s", sym)

    def _on_remove(self):
        indices = self._table.selectionModel().selectedRows()
        if not indices:
            return
        row = self._model.row_data(indices[0].row())
        if not row:
            return
        sym = row["symbol"]
        reply = QMessageBox.question(
            self, "Remove from Watchlist",
            f"Remove {sym} from watchlist?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._entries = remove_entry(self._entries, sym)
            self._model.set_data(self._entries)
            self._count_lbl.setText(f"{len(self._entries)} symbols")
            logger.info("Watchlist: removed %s", sym)

    def _on_view(self):
        indices = self._table.selectionModel().selectedRows()
        if indices:
            row = self._model.row_data(indices[0].row())
            if row:
                self.view_symbol.emit(row["symbol"])

    def _on_double_click(self, index: QModelIndex):
        row = self._model.row_data(index.row())
        if row:
            self.view_symbol.emit(row["symbol"])

    def _on_refresh(self):
        """Refresh all watchlist entries with live data."""
        if not self._entries:
            return
        if self._worker and self._worker.isRunning():
            return

        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setText("Refreshing…")

        self._worker = WatchlistWorker(self._entries)
        self._worker.wl_ready.connect(self._on_refresh_done)
        self._worker.error.connect(self._on_refresh_error)
        self._worker.start()

    def _on_refresh_done(self, entries: list):
        self._entries = entries
        self._model.set_data(self._entries)
        self._refresh_btn.setEnabled(True)
        self._refresh_btn.setText("Refresh All")
        self._update_alerts()
        logger.info("Watchlist refresh complete: %d entries", len(entries))

    def _on_refresh_error(self, msg: str):
        self._refresh_btn.setEnabled(True)
        self._refresh_btn.setText("Refresh All")
        logger.error("Watchlist refresh error: %s", msg)

    # ══════════════════════════════════════════════════════════════════════════
    # ALERTS
    # ══════════════════════════════════════════════════════════════════════════

    def _update_alerts(self):
        """Show alert banners for symbols needing attention."""
        # Clear existing alerts
        while self._alert_area.count():
            item = self._alert_area.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        alerts = detect_alerts(self._entries)
        for alert in alerts[:5]:  # max 5 banners
            banner = AlertBanner(
                alert["message"],
                severity=alert["severity"],
            )
            self._alert_area.addWidget(banner)
