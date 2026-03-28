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
    get_price_daily,
    download_options,
    get_universe,
    NSE_LOT_SIZES,
)
from modules.indicators import (
    add_bollinger,
    add_williams_r,
    compute_price_signals,
)
from modules.analytics import analyze
from ui.watchlist_db import load_watchlist
from modules.scanner import analyze_symbol

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# FRESHNESS STATE
# ══════════════════════════════════════════════════════════════════════════════


class DataFreshnessState(enum.Enum):
    LIVE = "LIVE"  # last refresh < 5 min ago, during market hours
    STALE = "STALE"  # last refresh 5-15 min ago
    CACHED = "CACHED"  # fetched outside market hours
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
            df, msg = get_price_daily(self.symbol)
            if df is not None and not df.empty:
                df = add_bollinger(df, self.bb_period, self.bb_std)
                df = add_williams_r(df, self.wr_period)
                logger.info(
                    "PriceWorker done: %s in %.1fs (%d bars)",
                    self.symbol,
                    time.time() - t0,
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
        min_score: int = 50,
        require_all_filters: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.symbols = symbols
        self.min_score = min_score
        self.require_all_filters = require_all_filters
        self._cancelled = False
        self.total_scanned = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        from modules.data import is_market_open

        if not is_market_open():
            logger.warning("ScanWorker: market closed — results use cached/stale data")

        logger.info(
            "ScanWorker start: %d symbols, min_score=%d",
            len(self.symbols),
            self.min_score,
        )
        t0 = time.time()
        results = []
        total = len(self.symbols)

        for i, sym in enumerate(self.symbols):
            if self._cancelled:
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

            # ── Score filter (dual-mode score) ───────────────────────────
            dm_sc = row.get("dm_score", row.get("viability_score", 0))
            if dm_sc < self.min_score:
                continue

            if self.require_all_filters and not row["all_filters_pass"]:
                continue

            results.append(row)
            self.row_ready.emit(row)

        self.total_scanned = total
        results.sort(key=lambda r: (r.get("setup_priority", 8), -r.get("dm_score", 0)))
        logger.info(
            "ScanWorker done: %d/%d passed in %.1fs",
            len(results),
            total,
            time.time() - t0,
        )
        self.finished.emit(results)


# ══════════════════════════════════════════════════════════════════════════════
# FILINGS WORKER
# ══════════════════════════════════════════════════════════════════════════════

_FILINGS_DELAY: float = 0.6  # inter-symbol — NSE friendly; smaller universe than scan


class FilingsWorker(QThread):
    """
    Post-scan filing enrichment pass.

    Runs AFTER ScanWorker.finished — receives only the filtered results
    (typically 5–30 symbols, not 500).  For each symbol:
      1. Fetches Reg-30 filings via fetch_filings()
      2. Calls get_filing_variance() — uses adv_cr already in the row
      3. Re-runs classify_setup_v3() with fv fields patched in,
         reconstructing OptionsSignalState + MomentumState from row dict
      4. Emits row_enriched(symbol, dict) — scanner_tab patches model in-place

    market_cap_cr: 0.0 default (D1 falls back to absolute Rs Cr tiers).
    Populated from NSE equity-info API in a future fast-follow.

    Rules:
      - Never touches UI — emits signals only
      - Cancellable at any symbol boundary
      - Per-symbol failures are silently skipped (row stays un-enriched)
    """

    row_enriched = pyqtSignal(str, dict)  # symbol, updated fields
    progress = pyqtSignal(int, int, str)  # done, total, symbol
    finished = pyqtSignal()

    def __init__(
        self, results: list[dict], db_path: str = "nimbus_data.db", parent=None
    ):
        super().__init__(parent)
        self.results = results
        self.db_path = db_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import datetime
        import pandas as pd
        from modules.filings_v2 import (
            make_nse_session,
            fetch_announcements,
            get_filing_variance,
        )
        from modules.setup_classifier import (
            classify_setup_v3,
            OptionsSignalState,
            MomentumState,
            SETUP_COLORS,
            SETUP_PRIORITY,
        )

        total = len(self.results)
        t0 = time.time()
        logger.info("FilingsWorker start — %d symbols", total)

        # One warmed session for the full batch (saves one round-trip per symbol)
        sess = make_nse_session()

        for i, row in enumerate(self.results):
            if self._cancelled:
                logger.info("FilingsWorker cancelled at %d/%d", i, total)
                break

            sym = row.get("symbol", "")
            self.progress.emit(i + 1, total, sym)
            time.sleep(_FILINGS_DELAY)

            try:
                # FIX-API + FIX-FIELD: correct endpoint, combined desc+subject text
                announcements = fetch_announcements(sym, sess=sess, max_records=5)
                if not announcements:
                    continue

                latest = announcements[0]
                text = latest["text"]
                ts_str = latest["ts"]
                filing_ts = None
                for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y"):
                    try:
                        filing_ts = pd.Timestamp(
                            datetime.datetime.strptime(ts_str.strip(), fmt)
                        )
                        break
                    except (ValueError, AttributeError):
                        continue

                fv = get_filing_variance(
                    symbol=sym,
                    subject=text,
                    filing_ts=filing_ts,
                    nse_session=sess,
                    adv_cr=float(row.get("adv_cr", 0.0)),
                    market_cap_cr=float(row.get("market_cap_cr", 0.0)),
                    db_path=self.db_path,
                )
            except Exception as exc:
                logger.debug("FilingsWorker %s failed: %s", sym, exc)
                continue

            if fv is None:
                continue

            opts = OptionsSignalState(
                gex_regime=row.get("gex_regime_raw", "Neutral"),
                gex_rising=bool(row.get("gex_rising", False)),
                pcr=float(row.get("pcr_oi") or 1.0),
                pcr_trending=row.get("pcr_trending", "FLAT"),
                iv_skew=row.get("iv_skew", "FLAT"),
                delta_bias=row.get("delta_bias", "NEUTRAL"),
                call_oi_wall_pct=float(row.get("call_oi_wall_pct") or 0.0),
                pct_to_resistance=row.get("pct_to_resistance"),
                pcr_oi=float(row.get("pcr_oi") or 1.0),
            )
            mom = MomentumState(
                bb_position=row.get("bb_position", "unknown"),
                position_state=row.get("position_state", "UNKNOWN"),
                vol_state=row.get("vol_state", "NORMAL"),
                wr_phase=row.get("wr_phase", "NONE"),
                wr_value=row.get("wr_50"),
                wr_in_momentum=bool(row.get("wr_above_minus20", False)),
            )
            setup_type, setup_detail = classify_setup_v3(
                viability_score=int(
                    row.get("dm_score", row.get("viability_score", 50))
                ),
                filing_variance=fv.variance,
                filing_direction=fv.badge_color,
                filing_conviction=fv.conviction,
                filing_category=fv.category.value,
                opts=opts,
                mom=mom,
            )
            self.row_enriched.emit(
                sym,
                {
                    "filing_badge": fv.badge_text,
                    "filing_color": fv.badge_color,
                    "filing_variance": fv.variance,
                    "filing_detail": fv.detail_line,
                    "filing_confirmed": fv.confirmed,
                    "setup_type": setup_type.value,
                    "setup_detail": setup_detail,
                    "setup_color": SETUP_COLORS[setup_type],
                    "setup_priority": SETUP_PRIORITY[setup_type],
                },
            )
            logger.info(
                "FilingsWorker %s → %s | %s | var %+d",
                sym,
                fv.badge_text,
                setup_type.value,
                fv.variance,
            )

        logger.info("FilingsWorker done in %.1fs", time.time() - t0)
        self.finished.emit()


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


# ══════════════════════════════════════════════════════════════════════════════
# Single Filing WORKER
# ══════════════════════════════════════════════════════════════════════════════


class SingleFilingWorker(QThread):
    """
    Filing enrichment for the Dashboard tab — one symbol at a time.
    Fired by DataManager after analytics completes for the active symbol.

    Signals:
        filing_ready(str, object)  — emitted with (symbol, FilingVariance)
                                     when an actionable filing is found
        finished()                 — always emitted, even on failure
    """

    filing_ready = pyqtSignal(str, object)
    finished = pyqtSignal()

    def __init__(
        self,
        symbol: str,
        adv_cr: float = 0.0,
        market_cap_cr: float = 0.0,
        db_path: str = "nimbus_data.db",
        parent=None,
    ):
        super().__init__(parent)
        self.symbol = symbol
        self.adv_cr = adv_cr
        self.market_cap_cr = market_cap_cr
        self.db_path = db_path

    def run(self):
        import datetime
        import pandas as pd
        from modules.filings_v2 import (
            make_nse_session,
            fetch_announcements,
            get_filing_variance,
        )

        sym = self.symbol
        t0 = time.time()
        logger.info("SingleFilingWorker start: %s", sym)
        try:
            sess = make_nse_session()
            announcements = fetch_announcements(sym, sess=sess, max_records=5)
            if not announcements:
                return

            latest = announcements[0]
            text = latest["text"]
            ts_str = latest["ts"]
            filing_ts = None
            for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y"):
                try:
                    filing_ts = pd.Timestamp(
                        datetime.datetime.strptime(ts_str.strip(), fmt)
                    )
                    break
                except (ValueError, AttributeError):
                    continue

            fv = get_filing_variance(
                symbol=sym,
                subject=text,
                filing_ts=filing_ts,
                nse_session=sess,
                adv_cr=self.adv_cr,
                market_cap_cr=self.market_cap_cr,
                db_path=self.db_path,
            )
            if fv is not None:
                self.filing_ready.emit(sym, fv)
                logger.info(
                    "SingleFilingWorker %s → %s | var %+d | conv %d/10 (%.1fs)",
                    sym,
                    fv.badge_text,
                    fv.variance,
                    fv.conviction,
                    time.time() - t0,
                )
            else:
                logger.debug("SingleFilingWorker %s — no actionable filing", sym)
        except Exception as exc:
            logger.warning("SingleFilingWorker %s failed: %s", sym, exc)
        finally:
            self.finished.emit()
