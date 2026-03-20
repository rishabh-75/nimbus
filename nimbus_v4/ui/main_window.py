"""
ui/main_window.py
─────────────────
NIMBUS QMainWindow: sidebar (fixed 220px) + QTabWidget (3 tabs).

Phase 1: DataManager wired, sidebar signals connected, status bar live.
Symbol change → immediate pipeline execution (Lesson 8.6).
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

from PyQt6.QtCore import Qt, QSize, QSettings
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTabWidget,
    QLabel,
    QStatusBar,
    QInputDialog,
)
from PyQt6.QtGui import QShortcut, QKeySequence

from ui.theme import BG, SURFACE, BORDER, EM, RED, GOLD, MUTED, WHITE, BLUE, FONT_MONO, QSS
from ui.sidebar import Sidebar
from ui.data_manager import DataManager
from ui.dashboard_tab import DashboardTab
from ui.scanner_tab import ScannerTab
from ui.watchlist_tab import WatchlistTab
from ui.market_context_tab import MarketContextTab
from ui.watchlist_db import init_watchlist
# KiteTicker removed — yfinance polling is the sole data source

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Top-level window for NIMBUS Qt.

    Layout:
        ┌────────────┬────────────────────────────────────────┐
        │  Sidebar    │  QTabWidget                            │
        │  (220px)    │    Tab 0: Dashboard                    │
        │             │    Tab 1: Scanner                      │
        │             │    Tab 2: Watchlist                     │
        └────────────┴────────────────────────────────────────┘
        │  Status bar                                          │
        └──────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIMBUS · Emerald Slate · Qt")
        self.setMinimumSize(QSize(1200, 760))
        self.resize(1480, 900)

        # Apply master stylesheet
        self.setStyleSheet(QSS)

        # ── Data manager ──────────────────────────────────────────────────────
        self.data_mgr = DataManager(parent=self)

        # ── Watchlist SQLite init ─────────────────────────────────────────────
        init_watchlist()

        # ── Signal tracker init ────────────────────────────────────────────────
        from modules.signal_tracker import init_tracker
        init_tracker()

        self._build_ui()
        self._connect_signals()

        # ── Fetch universe on startup ─────────────────────────────────────────
        self.data_mgr.fetch_universe()

        # ── Auto-load initial symbol after window shows ───────────────────────
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(500, self._auto_load_initial)

        # ── Keyboard shortcuts (§6.2) ─────────────────────────────────────────
        self._setup_shortcuts()

        # ── Restore window geometry ───────────────────────────────────────────
        self._restore_state()

        logger.info("MainWindow initialised — yfinance polling active")

    def _auto_load_initial(self):
        """Trigger pipeline for the default symbol on startup."""
        symbol = self.sidebar.current_symbol()
        if symbol:
            logger.info("Auto-loading initial symbol: %s", symbol)
            self._on_symbol_changed(symbol)

    # ──────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────────
        self.sidebar = Sidebar(self)
        root.addWidget(self.sidebar)

        # ── Tab widget ────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        # Prevent Qt from truncating tab labels
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideNone)
        self.tabs.tabBar().setExpanding(False)

        # Phase 2: real Dashboard tab
        self._dashboard_tab = DashboardTab(self)
        self._scanner_tab = ScannerTab(self)
        self._watchlist_tab = WatchlistTab(self)
        self._context_tab = MarketContextTab(self)

        self.tabs.addTab(self._dashboard_tab, "Dashboard")
        self.tabs.addTab(self._scanner_tab, "Scanner")
        self.tabs.addTab(self._watchlist_tab, "Watchlist")
        self.tabs.addTab(self._context_tab, "Market Context")

        root.addWidget(self.tabs, stretch=1)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; padding: 0 4px;"
        )
        self._status_bar.addWidget(self._status_dot)

        self._status_label = QLabel("Ready · Select a symbol to begin")
        self._status_label.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; font-family: '{FONT_MONO}';"
        )
        self._status_bar.addWidget(self._status_label, stretch=1)

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL WIRING
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        """Wire sidebar → DataManager → status bar."""

        # Sidebar → pipeline
        self.sidebar.symbol_changed.connect(self._on_symbol_changed)
        self.sidebar.refresh_clicked.connect(self._on_refresh)

        # Sidebar → DataManager indicator params
        self.sidebar.bb_period_changed.connect(self.data_mgr.set_bb_period)
        self.sidebar.bb_std_changed.connect(self.data_mgr.set_bb_std)
        self.sidebar.wr_period_changed.connect(self.data_mgr.set_wr_period)
        self.sidebar.wr_thresh_changed.connect(self.data_mgr.set_wr_threshold)
        self.sidebar.lot_size_changed.connect(self.data_mgr.set_lot_size)

        # DataManager → status bar
        self.data_mgr.freshness_changed.connect(self._on_freshness_changed)
        self.data_mgr.error_occurred.connect(self._on_error)
        self.data_mgr.spot_updated.connect(self._on_spot_updated)

        # DataManager → universe → sidebar
        self.data_mgr.universe_ready.connect(self._on_universe_ready)

        # DataManager → dashboard tab
        self.data_mgr.price_updated.connect(self._on_price_for_dashboard)
        self.data_mgr.context_updated.connect(self._dashboard_tab.on_context_updated)
        self.data_mgr.ps_updated.connect(self._dashboard_tab.on_ps_updated)
        self.data_mgr.spot_updated.connect(self._dashboard_tab.on_spot_updated)
        self.data_mgr.filing_ready.connect(self._dashboard_tab.on_filing_updated)

        # DataManager → dual-mode signal
        self.data_mgr.dual_mode_updated.connect(self._dashboard_tab.on_dual_mode_updated)
        self.data_mgr.dual_mode_updated.connect(
            lambda sym, sig: self.sidebar.update_dual_mode(sig)
        )

        # DataManager → analytics complete (logged)
        self.data_mgr.context_updated.connect(self._on_context_updated)
        self.data_mgr.ps_updated.connect(self._on_ps_updated)

        # Scanner → Dashboard navigation (Lesson 8.7)
        self._scanner_tab.row_clicked.connect(self.open_symbol_in_dashboard)

        # Scanner → populate sidebar (single click, no tab switch)
        self._scanner_tab.symbol_selected.connect(
            lambda sym: self.sidebar._symbol_combo.setCurrentText(sym)
        )

        # Watchlist → Dashboard navigation
        self._watchlist_tab.view_symbol.connect(self.open_symbol_in_dashboard)

        # Market Context → Dashboard navigation
        self._context_tab.view_symbol.connect(self.open_symbol_in_dashboard)

    # ──────────────────────────────────────────────────────────────────────────
    # SLOTS: Sidebar actions
    # ──────────────────────────────────────────────────────────────────────────

    def _on_symbol_changed(self, symbol: str):
        """
        CRITICAL (Lesson 8.6): Symbol change → immediate pipeline execution.
        (1) Clear cached DataFrames
        (2) Set needs_refresh flag
        (3) Directly trigger PriceWorker and OptionsWorker
        Never rely on a timer or passive state check.
        """
        logger.info("Symbol changed → immediate pipeline: %s", symbol)
        self.sidebar.set_refresh_enabled(False)
        self._set_status("LIVE", f"Loading {symbol}…")

        # Update lot size display
        from modules.data import NSE_LOT_SIZES

        lot = NSE_LOT_SIZES.get(symbol, 75)
        self.sidebar.set_lot_size(lot)

        # IMMEDIATE pipeline execution
        self.data_mgr.refresh(symbol, fetch_options=True)

    def _on_refresh(self):
        symbol = self.sidebar.current_symbol()
        if not symbol:
            return
        logger.info("Manual refresh: %s", symbol)
        self.sidebar.set_refresh_enabled(False)
        self._set_status("LIVE", f"Refreshing {symbol}…")
        self.data_mgr.refresh(symbol, fetch_options=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SLOTS: DataManager callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _on_freshness_changed(self, state: str):
        """Update status bar per §3.2 format strings."""
        symbol = self.data_mgr.active_symbol
        spot = self.data_mgr.spot

        if state == "LIVE":
            ts = self.data_mgr.last_refresh_time()
            nxt = self.data_mgr.next_refresh_time()
            ts_str = ts.strftime("%H:%M:%S") if ts else "—"
            nxt_str = nxt.strftime("%H:%M:%S") if nxt else "—"
            spot_str = f"{spot:,.2f}" if spot > 0 else "—"
            self._set_status(
                "LIVE",
                f"LIVE  |  {symbol} {spot_str}  |  "
                f"Last refresh: {ts_str}  |  Next: {nxt_str}",
            )
            self.sidebar.set_refresh_enabled(True)

        elif state == "CACHED":
            spot_str = f"{spot:,.2f}" if spot > 0 else "—"
            self._set_status(
                "CACHED",
                f"CACHED  |  {symbol} {spot_str}  |  Market closed — using last available data",
            )
            self.sidebar.set_refresh_enabled(True)

        elif state == "STALE":
            ts = self.data_mgr.last_refresh_time()
            if ts:
                ago = datetime.datetime.now() - ts
                mins = int(ago.total_seconds() // 60)
                self._set_status(
                    "STALE",
                    f"STALE · {mins}m ago  |  {symbol}  |  Retry available",
                )
            else:
                self._set_status("STALE", f"STALE  |  {symbol}")
            self.sidebar.set_refresh_enabled(True)

        elif state == "ERROR":
            self._set_status("ERROR", f"ERROR  |  Fetch failed for {symbol}  |  Retry")
            self.sidebar.set_refresh_enabled(True)

    def _on_error(self, symbol: str, msg: str):
        logger.error("Error for %s: %s", symbol, msg)
        self.sidebar.set_refresh_enabled(True)

    def _on_spot_updated(self, symbol: str, spot: float):
        logger.info("Spot updated: %s = %.2f", symbol, spot)

    def _on_price_for_dashboard(self, symbol: str, df):
        """Feed price data to dashboard header + chart."""
        self._dashboard_tab.on_price_updated(symbol, df)
        self._dashboard_tab.set_price_df(df)

    def _on_universe_ready(self, symbols: list):
        """Populate sidebar combo with fetched universe."""
        logger.info("Universe loaded: %d symbols", len(symbols))
        self.sidebar.set_symbols(symbols)

    def _on_context_updated(self, symbol: str, ctx):
        """Analytics complete — log summary. Dashboard wiring in Phase 3."""
        v = ctx.viability
        regime_str = getattr(getattr(ctx, "regime", None), "regime", "ETF")
        logger.info(
            "Context ready: %s — score=%d sizing=%s label=%s regime=%s",
            symbol,
            v.score,
            v.sizing,
            v.label,
            regime_str,
        )
        self.sidebar.set_refresh_enabled(True)

    def _on_ps_updated(self, symbol: str, ps):
        """PriceSignals ready — log summary. Dashboard wiring in Phase 3."""
        logger.info(
            "PriceSignals ready: %s — WR=%.1f bias=%s pos=%s vol=%s",
            symbol,
            ps.wr_value if ps.wr_value is not None else -999,
            ps.daily_bias,
            ps.position_state,
            ps.vol_state,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API (for future phases)
    # ──────────────────────────────────────────────────────────────────────────

    def open_symbol_in_dashboard(self, symbol: str):
        """
        Called by scanner tab (Phase 5): switch to Dashboard + run pipeline.
        Clean: setCurrentIndex(0) + trigger pipeline. No JS injection needed.
        (Lesson 8.7)
        """
        logger.info("open_symbol_in_dashboard: %s", symbol)
        self.sidebar._symbol_combo.setCurrentText(symbol)
        self.tabs.setCurrentIndex(0)
        self.data_mgr.refresh(symbol, fetch_options=True)

    def _set_status(self, state: str, message: str):
        """Update status bar dot + label."""
        colors = {
            "LIVE": EM,
            "STALE": GOLD,
            "CACHED": BLUE,
            "ERROR": RED,
            "READY": MUTED,
        }
        color = colors.get(state, MUTED)
        self._status_dot.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 0 4px;"
        )
        self._status_label.setText(message)

    # ──────────────────────────────────────────────────────────────────────────
    # KEYBOARD SHORTCUTS (§6.2)
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        """R = refresh, Cmd+1/2/3 = tab switch, Cmd+F = symbol search."""
        QShortcut(QKeySequence("R"), self).activated.connect(self._on_refresh)
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(
            lambda: self.tabs.setCurrentIndex(0)
        )
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(
            lambda: self.tabs.setCurrentIndex(1)
        )
        QShortcut(QKeySequence("Ctrl+3"), self).activated.connect(
            lambda: self.tabs.setCurrentIndex(2)
        )
        QShortcut(QKeySequence("Ctrl+4"), self).activated.connect(
            lambda: self.tabs.setCurrentIndex(3)
        )
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(
            lambda: self.sidebar._symbol_input.setFocus()
        )

    # ──────────────────────────────────────────────────────────────────────────
    # WINDOW STATE PERSISTENCE (§6.2)
    # ──────────────────────────────────────────────────────────────────────────

    def _restore_state(self):
        """Restore window geometry and last symbol from QSettings."""
        settings = QSettings("NIMBUS", "EmeraldSlate")
        geo = settings.value("window/geometry")
        if geo:
            self.restoreGeometry(geo)
        last_sym = settings.value("window/last_symbol", "NIFTY")
        if last_sym:
            self.sidebar._symbol_combo.setCurrentText(last_sym)

    def _save_state(self):
        """Save window geometry and current symbol to QSettings."""
        settings = QSettings("NIMBUS", "EmeraldSlate")
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/last_symbol", self.sidebar.current_symbol())

    def closeEvent(self, event):
        """Persist state on close."""
        self._save_state()
        super().closeEvent(event)


