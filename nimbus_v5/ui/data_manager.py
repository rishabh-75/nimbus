"""
ui/data_manager.py  (patched — ETF routing)
Changes vs original:
  1. refresh()        — skip OptionsWorker for ETF symbols (saves time + avoids empty options error)
  2. _run_analytics() — for ETF symbols calls analyze_etf() instead of analytics.analyze()
                        emits ETFContext via context_updated (same signal, different type)
  ZERO other changes.
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
from modules.etf_analyzer import analyze_etf, NSE_ETF_SYMBOLS  # ← NEW

from ui.workers import (
    PriceWorker,
    OptionsWorker,
    UniverseWorker,
    DataFreshnessState,
)

logger = logging.getLogger(__name__)

_LIVE_THRESHOLD = 300
_STALE_THRESHOLD = 900
_REFRESH_MS = 300_000


class DataManager(QObject):
    price_updated = pyqtSignal(str, object)
    options_updated = pyqtSignal(str, object)
    context_updated = pyqtSignal(str, object)  # OptionsContext OR ETFContext
    ps_updated = pyqtSignal(str, object)
    spot_updated = pyqtSignal(str, float)
    error_occurred = pyqtSignal(str, str)
    freshness_changed = pyqtSignal(str)
    universe_ready = pyqtSignal(list)
    filing_ready = pyqtSignal(str, object)
    dual_mode_updated = pyqtSignal(str, object)  # DualModeSignal

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_symbol: str = ""
        self._price_df: Optional[pd.DataFrame] = None
        self._options_df: Optional[pd.DataFrame] = None
        self._ctx = None
        self._ps: Optional[PriceSignals] = None
        self._spot: float = 0.0
        self._last_refresh: Optional[float] = None
        self._freshness: DataFreshnessState = DataFreshnessState.STALE
        self.bb_period: int = 20
        self.bb_std: float = 1.0
        self.wr_period: int = 50
        self.wr_thresh: float = -20.0
        self.lot_size: int = 75
        self._price_worker: Optional[PriceWorker] = None
        self._options_worker: Optional[OptionsWorker] = None
        self._universe_worker: Optional[UniverseWorker] = None
        # Options cache: {symbol: (DataFrame, timestamp)} — survives symbol changes
        # Used to show last-known options data when market is closed
        self._options_cache: dict[str, tuple] = {}
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._on_auto_refresh)
        self._freshness_timer = QTimer(self)
        self._freshness_timer.setInterval(30_000)
        self._freshness_timer.timeout.connect(self._check_freshness)
        self._freshness_timer.start()
        self._prewarm_nifty500()
        self._filing_worker = None
        self._filing_fv = None  # last filing variance for dual-mode overlay
        logger.info("DataManager initialised")

    # ── PUBLIC API ─────────────────────────────────────────────────────────────
    def _prewarm_nifty500(self):
        """
        Pre-populate the NIFTY500 cache in filings_v2.py on DataManager startup.
        Runs in a daemon thread so it doesn't block the UI event loop.
        Without this, the first FilingsWorker run would block on NSE I/O
        for ~2s before the actual enrichment loop even starts.
        """
        import threading

        def _warm():
            try:
                from modules.filings_v2 import get_nifty500

                syms = get_nifty500()
                logger.info("NIFTY500 pre-warm: %d symbols cached", len(syms))
            except Exception as exc:
                logger.warning("NIFTY500 pre-warm failed (non-fatal): %s", exc)

        threading.Thread(target=_warm, daemon=True, name="nifty500-prewarm").start()

    def _start_filing_worker(self, symbol: str, adv_cr: float = 0.0):
        from ui.workers import SingleFilingWorker

        if self._filing_worker and self._filing_worker.isRunning():
            self._filing_worker.quit()
        self._filing_worker = SingleFilingWorker(symbol=symbol, adv_cr=adv_cr)
        self._filing_worker.filing_ready.connect(self._on_filing_ready)
        self._filing_worker.start()

    def _on_filing_ready(self, sym: str, fv):
        self._filing_fv = fv
        self.filing_ready.emit(sym, fv)
        # Re-emit dual-mode with filing overlay now available
        self._emit_dual_mode()

    def refresh(self, symbol: str, fetch_options: bool = True):
        symbol = symbol.strip().upper()
        logger.info("DataManager.refresh: %s", symbol)
        self._active_symbol = symbol
        self._price_df = None
        self._options_df = None
        self._ctx = None
        self._ps = None
        self._filing_fv = None
        self.lot_size = NSE_LOT_SIZES.get(symbol, 75)

        market_open = is_market_open()

        self._price_worker = PriceWorker(
            symbol=symbol,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            wr_period=self.wr_period,
        )
        self._price_worker.price_ready.connect(self._on_price_ready)
        self._price_worker.error.connect(self._on_price_error)
        self._price_worker.start()

        # ── Always fetch options (NSE returns last trading day's EOD data
        # even after hours and weekends). Only skip for ETFs.
        is_etf = symbol in NSE_ETF_SYMBOLS
        if fetch_options and not is_etf:
            self._options_worker = OptionsWorker(symbol=symbol, max_expiries=3)
            self._options_worker.options_ready.connect(self._on_options_ready)
            self._options_worker.error.connect(self._on_options_error)
            self._options_worker.start()

        # Only auto-refresh timer during market hours (no point polling after close)
        if market_open:
            self._refresh_timer.start()
        else:
            self._refresh_timer.stop()

    def fetch_universe(self):
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
    def context(self):
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

    def last_refresh_time(self):
        return (
            datetime.datetime.fromtimestamp(self._last_refresh)
            if self._last_refresh
            else None
        )

    def next_refresh_time(self):
        return (
            datetime.datetime.fromtimestamp(self._last_refresh + _REFRESH_MS / 1000)
            if self._last_refresh
            else None
        )

    # ── WORKER CALLBACKS ───────────────────────────────────────────────────────

    def _on_price_ready(self, df: pd.DataFrame, msg: str):
        symbol = self._active_symbol
        logger.info("Price ready: %s — %s", symbol, msg)
        self._price_df = df
        self._last_refresh = time.time()
        # Mark CACHED when market closed, LIVE when open
        if is_market_open():
            self._freshness = DataFreshnessState.LIVE
            self.freshness_changed.emit("LIVE")
        else:
            self._freshness = DataFreshnessState.CACHED
            self.freshness_changed.emit("CACHED")
        if not df.empty:
            self._spot = float(df.iloc[-1]["Close"])
            self.spot_updated.emit(symbol, self._spot)
        self._ps = compute_price_signals(df, wr_thresh=self.wr_thresh)
        self.price_updated.emit(symbol, df)
        self.ps_updated.emit(symbol, self._ps)

        # Dual-mode runs after _run_analytics (needs options ctx)
        self._run_analytics()

    def _on_price_error(self, msg: str):
        symbol = self._active_symbol
        logger.error("Price error: %s — %s", symbol, msg)
        self._set_freshness(DataFreshnessState.ERROR)
        self.error_occurred.emit(symbol, f"Price: {msg}")

    def _on_options_ready(self, df: pd.DataFrame, msg: str):
        symbol = self._active_symbol
        logger.info("Options ready: %s — %s", symbol, msg)
        self._options_df = df
        # Cache for after-hours use
        if df is not None and not df.empty:
            import time as _t
            self._options_cache[symbol] = (df, _t.time())
            logger.info("Options cached: %s (%d rows)", symbol, len(df))
        self.options_updated.emit(symbol, df)
        if self._price_df is not None:
            self._run_analytics()

    def _on_options_error(self, msg: str):
        symbol = self._active_symbol
        logger.warning("Options error: %s — %s", symbol, msg)
        self.error_occurred.emit(symbol, f"Options: {msg}")
        if self._price_df is not None and self._ctx is None:
            self._run_analytics()

    def _on_universe_ready(self, symbols: list):
        logger.info("Universe ready: %d symbols", len(symbols))
        self.universe_ready.emit(symbols)

    # ── ANALYTICS ─────────────────────────────────────────────────────────────

    def _run_analytics(self):
        from modules.etf_analyzer import NSE_ETF_SYMBOLS, analyze_etf

        symbol = self._active_symbol

        spot = self._spot

        if symbol in NSE_ETF_SYMBOLS:
            try:
                ctx = analyze_etf(
                    symbol=symbol,
                    price_df=self._price_df,
                    price_signals=self._ps,
                    fetch_nav=True,
                )
                if ctx is None:
                    logger.warning("analyze_etf returned None for %s", symbol)
                    return
                self._ctx = ctx
                logger.info(
                    "ETF analytics: %s — score=%d sizing=%s label=%s",
                    symbol,
                    ctx.viability.score,
                    ctx.viability.sizing,
                    ctx.viability.label,
                )
                self.context_updated.emit(symbol, ctx)
            except Exception as exc:
                logger.error("ETF analytics error %s: %s", symbol, exc)

            adv_cr = getattr(self._ps, "adv_cr", 0.0) if self._ps else 0.0
            self._start_filing_worker(symbol, adv_cr=adv_cr)
            self._emit_dual_mode()
            # Don't override CACHED state set by _on_price_ready
            return

        # ── F&O path (unchanged) ──────────────────────────────────────────────
        try:
            if self._options_df is not None and not self._options_df.empty:
                opt_spot = infer_spot(self._options_df)
                if opt_spot and opt_spot > 0:
                    spot = opt_spot
                    self._spot = spot
            options_df = (
                self._options_df if self._options_df is not None else pd.DataFrame()
            )
            self._ctx = analyze(
                options_df,
                spot=spot,
                lot_size=self.lot_size,
                price_signals=self._ps,
                room_thresh=3.0,
            )
            logger.info(
                "Analytics complete: %s — viability=%d sizing=%s label=%s",
                symbol,
                self._ctx.viability.score,
                self._ctx.viability.sizing,
                self._ctx.viability.label,
            )
            self.context_updated.emit(symbol, self._ctx)
        except Exception as exc:
            logger.error("Analytics error for %s: %s", symbol, exc)
        adv_cr = getattr(self._ps, "adv_cr", 0.0) if self._ps else 0.0
        self._start_filing_worker(symbol, adv_cr=adv_cr)
        self._emit_dual_mode()
        # Don't override CACHED state set by _on_price_ready

    # ── DUAL-MODE SIGNAL ───────────────────────────────────────────────────

    def _emit_dual_mode(self):
        """
        Compute and emit dual-mode signal.
        Called after analytics (has options ctx) and again after filing arrives.
        Input is the daily price_df — dual_mode.py computes all indicators on daily.
        """
        symbol = self._active_symbol
        if self._price_df is None or self._price_df.empty:
            return
        try:
            from modules.dual_mode import compute_dual_mode
            dm_sig = compute_dual_mode(
                self._price_df,
                symbol=symbol,
                options_ctx=self._ctx,          # may be None
                filing_variance=self._filing_fv, # may be None
            )
            self.dual_mode_updated.emit(symbol, dm_sig)
            logger.info(
                "DualMode: %s — tier=%s base=%d opt=%+d fil=%+d → final=%d %s [%s]",
                symbol, dm_sig.tier,
                dm_sig.base_score, dm_sig.options_overlay, dm_sig.filing_overlay,
                dm_sig.dual_score, dm_sig.dual_label, dm_sig.input_interval,
            )

            # Auto-log entry signals for forward validation
            if dm_sig.entry_triggered:
                try:
                    from modules.signal_tracker import log_signal
                    log_signal(dm_sig)
                except Exception as log_exc:
                    logger.debug("Signal tracking failed: %s", log_exc)
        except Exception as exc:
            logger.debug("DualMode failed: %s", exc)

    # ── FRESHNESS & AUTO-REFRESH ───────────────────────────────────────────────

    def _set_freshness(self, state: DataFreshnessState):
        if state != self._freshness:
            self._freshness = state
            self.freshness_changed.emit(state.value)
            logger.info("Freshness → %s", state.value)

    def _check_freshness(self):
        if self._last_refresh is None:
            return
        age = time.time() - self._last_refresh
        if self._freshness == DataFreshnessState.LIVE and age > _LIVE_THRESHOLD:
            self._set_freshness(DataFreshnessState.STALE)

    def _on_auto_refresh(self):
        if not self._active_symbol:
            return
        if not is_market_open():
            logger.info("Auto-refresh skipped: market closed")
            return
        self.refresh(self._active_symbol, fetch_options=True)

    # ── PARAMETER UPDATES ─────────────────────────────────────────────────────

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
