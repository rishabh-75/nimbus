"""
ui/data_manager.py
──────────────────
Singleton data orchestrator for NIMBUS Qt.

Spawns PriceWorker + OptionsWorker, on completion calls analyze() and
compute_price_signals(), emits context_updated + ps_updated.

Manages auto-refresh via QTimer during market hours (5 min interval).
Tracks DataFreshnessState and emits freshness_changed.

CRITICAL RULE (Lesson 8.6):
    Symbol change → immediate pipeline re-execution.
    Never rely on a timer or passive state check.
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Optional

import pandas as pd

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from modules.indicators import compute_price_signals, PriceSignals
from modules.analytics import analyze, OptionsContext
from modules.data import infer_spot, NSE_LOT_SIZES, is_market_open

from ui.workers import (
    PriceWorker, OptionsWorker, UniverseWorker,
    DataFreshnessState,
)

logger = logging.getLogger(__name__)

# Freshness thresholds (seconds)
_LIVE_THRESHOLD  = 300    # 5 min
_STALE_THRESHOLD = 900    # 15 min
_REFRESH_MS      = 300_000  # 5 min auto-refresh


class DataManager(QObject):
    """
    Central data orchestrator.

    Signals:
        price_updated(symbol, DataFrame)       — raw price df with BB + WR
        options_updated(symbol, DataFrame)     — raw options chain
        context_updated(symbol, OptionsContext) — full analytics context
        ps_updated(symbol, PriceSignals)       — price-derived signals
        spot_updated(symbol, float)            — current spot price
        error_occurred(symbol, message)        — any fetch error
        freshness_changed(state_string)        — LIVE / STALE / ERROR
        universe_ready(list)                   — symbol universe loaded
    """

    price_updated     = pyqtSignal(str, object)     # symbol, DataFrame
    options_updated   = pyqtSignal(str, object)     # symbol, DataFrame
    context_updated   = pyqtSignal(str, object)     # symbol, OptionsContext
    ps_updated        = pyqtSignal(str, object)     # symbol, PriceSignals
    spot_updated      = pyqtSignal(str, float)      # symbol, spot price
    error_occurred    = pyqtSignal(str, str)         # symbol, error msg
    freshness_changed = pyqtSignal(str)              # LIVE / STALE / ERROR
    universe_ready    = pyqtSignal(list)             # symbol list

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── State ─────────────────────────────────────────────────────────────
        self._active_symbol:  str = ""
        self._price_df:       Optional[pd.DataFrame] = None
        self._options_df:     Optional[pd.DataFrame] = None
        self._ctx:            Optional[OptionsContext] = None
        self._ps:             Optional[PriceSignals] = None
        self._spot:           float = 0.0

        self._last_refresh:   Optional[float] = None
        self._freshness:      DataFreshnessState = DataFreshnessState.STALE

        # ── Indicator parameters ──────────────────────────────────────────────
        self.bb_period:  int   = 20
        self.bb_std:     float = 1.0
        self.wr_period:  int   = 50
        self.wr_thresh:  float = -20.0
        self.lot_size:   int   = 75

        # ── Worker references (prevent GC) ────────────────────────────────────
        self._price_worker:   Optional[PriceWorker] = None
        self._options_worker: Optional[OptionsWorker] = None
        self._universe_worker: Optional[UniverseWorker] = None

        # ── Auto-refresh timer ────────────────────────────────────────────────
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._on_auto_refresh)

        # ── Freshness check timer (every 30s) ────────────────────────────────
        self._freshness_timer = QTimer(self)
        self._freshness_timer.setInterval(30_000)
        self._freshness_timer.timeout.connect(self._check_freshness)
        self._freshness_timer.start()

        logger.info("DataManager initialised")

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def refresh(self, symbol: str, fetch_options: bool = True):
        """
        Start full data pipeline for a symbol.

        CRITICAL (Lesson 8.6): This is called IMMEDIATELY on symbol change.
        It clears cached data, spawns workers, and starts the pipeline.
        Never called passively via timer for symbol changes.
        """
        symbol = symbol.strip().upper()
        logger.info("DataManager.refresh: %s (fetch_options=%s)", symbol, fetch_options)

        # Clear cached state for the new symbol
        self._active_symbol = symbol
        self._price_df  = None
        self._options_df = None
        self._ctx = None
        self._ps  = None

        # Update lot size from known table
        self.lot_size = NSE_LOT_SIZES.get(symbol, 75)

        # ── Spawn PriceWorker ─────────────────────────────────────────────────
        self._price_worker = PriceWorker(
            symbol=symbol,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            wr_period=self.wr_period,
        )
        self._price_worker.price_ready.connect(self._on_price_ready)
        self._price_worker.error.connect(self._on_price_error)
        self._price_worker.start()

        # ── Spawn OptionsWorker ───────────────────────────────────────────────
        if fetch_options:
            self._options_worker = OptionsWorker(symbol=symbol, max_expiries=3)
            self._options_worker.options_ready.connect(self._on_options_ready)
            self._options_worker.error.connect(self._on_options_error)
            self._options_worker.start()

        # Start / restart auto-refresh timer
        self._refresh_timer.start()

    def fetch_universe(self):
        """Load the F&O symbol universe asynchronously."""
        self._universe_worker = UniverseWorker()
        self._universe_worker.universe_ready.connect(self._on_universe_ready)
        self._universe_worker.error.connect(
            lambda msg: logger.warning("Universe fetch failed: %s", msg)
        )
        self._universe_worker.start()

    @property
    def active_symbol(self) -> str:
        return self._active_symbol

    @property
    def price_df(self) -> Optional[pd.DataFrame]:
        return self._price_df

    @property
    def options_df(self) -> Optional[pd.DataFrame]:
        return self._options_df

    @property
    def context(self) -> Optional[OptionsContext]:
        return self._ctx

    @property
    def price_signals(self) -> Optional[PriceSignals]:
        return self._ps

    @property
    def spot(self) -> float:
        return self._spot

    @property
    def freshness(self) -> DataFreshnessState:
        return self._freshness

    def last_refresh_time(self) -> Optional[datetime.datetime]:
        if self._last_refresh is None:
            return None
        return datetime.datetime.fromtimestamp(self._last_refresh)

    def next_refresh_time(self) -> Optional[datetime.datetime]:
        if self._last_refresh is None:
            return None
        return datetime.datetime.fromtimestamp(self._last_refresh + _REFRESH_MS / 1000)

    # ══════════════════════════════════════════════════════════════════════════
    # WORKER CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_price_ready(self, df: pd.DataFrame, msg: str):
        """Called when PriceWorker emits price_ready."""
        symbol = self._active_symbol
        logger.info("Price ready: %s — %s", symbol, msg)

        self._price_df = df
        self._last_refresh = time.time()
        self._freshness = DataFreshnessState.LIVE
        self.freshness_changed.emit("LIVE")  # always emit, even if already LIVE

        if not df.empty:
            self._spot = float(df.iloc[-1]["Close"])
            self.spot_updated.emit(symbol, self._spot)

        self._ps = compute_price_signals(df, wr_thresh=self.wr_thresh)
        self.price_updated.emit(symbol, df)
        self.ps_updated.emit(symbol, self._ps)

        # Always run analytics with whatever we have (price-only is fine)
        self._run_analytics()

    def _on_price_error(self, msg: str):
        symbol = self._active_symbol
        logger.error("Price error: %s — %s", symbol, msg)
        self._set_freshness(DataFreshnessState.ERROR)
        self.error_occurred.emit(symbol, f"Price: {msg}")

    def _on_options_ready(self, df: pd.DataFrame, msg: str):
        """Called when OptionsWorker emits options_ready."""
        symbol = self._active_symbol
        logger.info("Options ready: %s — %s", symbol, msg)

        self._options_df = df
        self.options_updated.emit(symbol, df)

        # If price already arrived, run analytics
        if self._price_df is not None:
            self._run_analytics()

    def _on_options_error(self, msg: str):
        symbol = self._active_symbol
        logger.warning("Options error: %s — %s", symbol, msg)
        self.error_occurred.emit(symbol, f"Options: {msg}")

        # Run analytics with price-only (no options) if price is available
        if self._price_df is not None and self._ctx is None:
            self._run_analytics()

    def _on_universe_ready(self, symbols: list):
        logger.info("Universe ready: %d symbols", len(symbols))
        self.universe_ready.emit(symbols)

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════

    def _run_analytics(self):
        """
        Run analyze() with current price + options data.
        Single source of truth: analytics.analyze() → ctx.viability.
        Never write a second scoring function. (Lesson 8.1)
        """
        symbol = self._active_symbol
        spot = self._spot

        try:
            # Use options-inferred spot if available (more accurate)
            if self._options_df is not None and not self._options_df.empty:
                opt_spot = infer_spot(self._options_df)
                if opt_spot and opt_spot > 0:
                    spot = opt_spot
                    self._spot = spot

            # Build options context via the SINGLE scoring function
            options_df = self._options_df if self._options_df is not None else pd.DataFrame()
            self._ctx = analyze(
                options_df,
                spot=spot,
                lot_size=self.lot_size,
                price_signals=self._ps,
                room_thresh=3.0,
            )

            logger.info(
                "Analytics complete: %s — viability=%d sizing=%s label=%s",
                symbol, self._ctx.viability.score,
                self._ctx.viability.sizing, self._ctx.viability.label,
            )
            self.context_updated.emit(symbol, self._ctx)
        except Exception as exc:
            logger.error("Analytics error for %s: %s", symbol, exc)

        # Always update freshness to LIVE after pipeline completes
        self._freshness = DataFreshnessState.LIVE
        self.freshness_changed.emit("LIVE")

    # ══════════════════════════════════════════════════════════════════════════
    # FRESHNESS & AUTO-REFRESH
    # ══════════════════════════════════════════════════════════════════════════

    def _set_freshness(self, state: DataFreshnessState):
        if state != self._freshness:
            self._freshness = state
            self.freshness_changed.emit(state.value)
            logger.info("Freshness → %s", state.value)

    def _check_freshness(self):
        """Periodic check: downgrade LIVE → STALE if data is old."""
        if self._last_refresh is None:
            return
        age = time.time() - self._last_refresh
        if self._freshness == DataFreshnessState.LIVE and age > _LIVE_THRESHOLD:
            self._set_freshness(DataFreshnessState.STALE)

    def _on_auto_refresh(self):
        """Timer-driven auto-refresh — only during market hours."""
        if not self._active_symbol:
            return
        if not is_market_open():
            logger.info("Auto-refresh skipped: market closed")
            return
        logger.info("Auto-refresh triggered for %s", self._active_symbol)
        self.refresh(self._active_symbol, fetch_options=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PARAMETER UPDATES
    # ══════════════════════════════════════════════════════════════════════════

    def set_bb_period(self, val: int):
        self.bb_period = val

    def set_bb_std(self, val: float):
        self.bb_std = val

    def set_wr_period(self, val: int):
        self.wr_period = val

    def set_wr_threshold(self, val: float):
        self.wr_thresh = val

    def set_lot_size(self, val: int):
        self.lot_size = val
