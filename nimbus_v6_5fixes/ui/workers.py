"""
ui/workers.py
─────────────
QThread subclasses for all async data operations.
DataFreshnessState enum tracks LIVE / STALE / ERROR.

Every worker logs start, completion time, and errors per §0.3.
Workers never touch the UI — they emit signals only.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Optional

import pandas as pd

from PyQt6.QtCore import QThread, pyqtSignal

from modules.data import (
    get_price_4h,
    download_options,
    get_universe,
    NSE_LOT_SIZES,
)
from modules.indicators import add_bollinger, add_williams_r, compute_price_signals
from modules.analytics import analyze
from ui.watchlist_db import load_watchlist
from modules.scanner import analyze_symbol

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FRESHNESS STATE
# ══════════════════════════════════════════════════════════════════════════════


class DataFreshnessState(enum.Enum):
    LIVE = "LIVE"  # last refresh < 5 min ago
    STALE = "STALE"  # last refresh 5-15 min ago
    ERROR = "ERROR"  # fetch failed


# ══════════════════════════════════════════════════════════════════════════════
# PRICE WORKER
# ══════════════════════════════════════════════════════════════════════════════


class PriceWorker(QThread):
    """
    Fetch price data via yfinance, compute BB + WR indicators.
    Emits the fully-enriched DataFrame ready for charting and signal extraction.
    """

    price_ready = pyqtSignal(pd.DataFrame, str)  # df, status_msg
    error = pyqtSignal(str)  # error message

    def __init__(
        self,
        symbol: str,
        bb_period: int = 20,
        bb_std: float = 1.0,
        wr_period: int = 50,
        parent=None,
    ):
        super().__init__(parent)
        self.symbol = symbol
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.wr_period = wr_period

    def run(self):
        logger.info("PriceWorker start: %s", self.symbol)
        t0 = time.time()
        try:
            df, msg = get_price_4h(self.symbol)
            if df is not None and not df.empty:
                df = add_bollinger(df, self.bb_period, self.bb_std)
                df = add_williams_r(df, self.wr_period)
                elapsed = time.time() - t0
                logger.info(
                    "PriceWorker done: %s in %.1fs (%d bars)",
                    self.symbol,
                    elapsed,
                    len(df),
                )
                self.price_ready.emit(df, msg)
            else:
                logger.warning("PriceWorker empty: %s — %s", self.symbol, msg)
                self.error.emit(msg or f"No price data for {self.symbol}")
        except Exception as exc:
            logger.error("PriceWorker error: %s: %s", self.symbol, exc)
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS WORKER
# ══════════════════════════════════════════════════════════════════════════════


class OptionsWorker(QThread):
    """
    Download NSE options chain for a symbol.
    NSE session pattern is inside data.py — unchanged.
    """

    options_ready = pyqtSignal(pd.DataFrame, str)  # df, status_msg
    error = pyqtSignal(str)  # error message

    def __init__(self, symbol: str, max_expiries: int = 3, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.max_expiries = max_expiries

    def run(self):
        logger.info("OptionsWorker start: %s", self.symbol)
        t0 = time.time()
        try:
            df, msg = download_options(self.symbol, max_expiries=self.max_expiries)
            elapsed = time.time() - t0
            if df is not None and not df.empty:
                logger.info(
                    "OptionsWorker done: %s in %.1fs (%d rows)",
                    self.symbol,
                    elapsed,
                    len(df),
                )
                self.options_ready.emit(df, msg)
            else:
                logger.warning("OptionsWorker empty: %s — %s", self.symbol, msg)
                self.error.emit(msg or f"No options data for {self.symbol}")
        except Exception as exc:
            logger.error("OptionsWorker error: %s: %s", self.symbol, exc)
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# SCAN WORKER
# ══════════════════════════════════════════════════════════════════════════════


class ScanWorker(QThread):
    """
    Run analyze_symbol() over a list of symbols.

    HARD GATE: passes_momentum is checked unconditionally before emitting
    any row. This is the lesson from Bug B (ADANIPORTS with W%R=-92).
    A symbol not riding the upper BB or with W%R <= -20 has no trade thesis
    and must never appear in scanner output.
    """

    row_ready = pyqtSignal(dict)  # single result row
    progress = pyqtSignal(int, int, str)  # done, total, current_symbol
    finished = pyqtSignal(list)  # all results
    error = pyqtSignal(str)

    def __init__(
        self,
        symbols: list[str],
        min_viability: int = 50,
        require_all_filters: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.total_scanned: int = 0
        self.symbols = symbols
        self.min_viability = min_viability
        self.require_all_filters = require_all_filters
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        logger.info(
            "ScanWorker start: %d symbols, min_viab=%d",
            len(self.symbols),
            self.min_viability,
        )
        t0 = time.time()
        results = []
        total = len(self.symbols)
        self.total_scanned = total

        for i, sym in enumerate(self.symbols):
            if self._cancelled:
                self.total_scanned = i
                logger.info("ScanWorker cancelled at %d/%d", i, total)
                break

            self.progress.emit(i + 1, total, sym)

            try:
                row = analyze_symbol(sym)
            except Exception as exc:
                logger.warning("ScanWorker: %s failed: %s", sym, exc)
                continue

            if row is None:
                continue

            # ── HARD GATE — momentum (BB riding + W%R > -20) ──────────────
            # Enforced unconditionally. No flag, no toggle can override this.
            # (Lesson 8.3: three stacked failures let ADANIPORTS through)
            if not row["passes_momentum"]:
                continue

            if row["viability_score"] < self.min_viability:
                continue

            if self.require_all_filters and not row["all_filters_pass"]:
                continue

            results.append(row)
            self.row_ready.emit(row)

        results.sort(key=lambda r: r["viability_score"], reverse=True)
        elapsed = time.time() - t0
        logger.info(
            "ScanWorker done: %d/%d passed in %.1fs",
            len(results),
            total,
            elapsed,
        )
        self.finished.emit(results)


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE WORKER
# ══════════════════════════════════════════════════════════════════════════════


class UniverseWorker(QThread):
    """Fetch F&O symbol universe from NSE (or cache/fallback)."""

    universe_ready = pyqtSignal(list)  # list of symbol strings
    error = pyqtSignal(str)

    def run(self):
        logger.info("UniverseWorker start")
        t0 = time.time()
        try:
            symbols = get_universe()
            logger.info(
                "UniverseWorker done: %d symbols in %.1fs",
                len(symbols),
                time.time() - t0,
            )
            self.universe_ready.emit(symbols)
        except Exception as exc:
            logger.error("UniverseWorker error: %s", exc)
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST WORKER
# ══════════════════════════════════════════════════════════════════════════════


class WatchlistWorker(QThread):
    """
    Refresh all watchlist entries by running analyze_symbol() on each.
    Uses modules/watchlist.refresh_watchlist() which does the heavy lifting.
    """

    wl_ready = pyqtSignal(list)  # enriched entries list
    progress = pyqtSignal(int, int, str)  # done, total, symbol
    error = pyqtSignal(str)

    def __init__(self, entries: list[dict], parent=None):
        super().__init__(parent)
        self.entries = entries

    def run(self):
        logger.info("WatchlistWorker start: %d entries", len(self.entries))
        t0 = time.time()
        try:
            from modules.watchlist import refresh_watchlist

            enriched = refresh_watchlist(
                self.entries,
                progress_cb=lambda done, total, sym: self.progress.emit(
                    done, total, sym
                ),
            )
            logger.info(
                "WatchlistWorker done: %d entries in %.1fs",
                len(enriched),
                time.time() - t0,
            )
            self.wl_ready.emit(enriched)
        except Exception as exc:
            logger.error("WatchlistWorker error: %s", exc)
            self.error.emit(str(exc))
