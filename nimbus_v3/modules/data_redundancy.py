"""
modules/data_redundancy.py
───────────────────────────
Fallback chains for missing data in the NIMBUS pipeline.

Each function attempts multiple data sources in priority order
and returns a result with a staleness/confidence indicator.

Fallback chains:
  Options:  NSE live → NSE CSV upload → last cached chain (with age discount) → None
  MFI:      Stock MFI → Sector-ETF MFI proxy → None
  Spot:     Options underlying → yfinance last close → cached spot → None
  Sector:   yfinance batch → individual download → cached → None

Every result carries a DataQuality tag so downstream consumers
know how much to trust it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataResult:
    """Wrapper around any data result with quality metadata."""
    data: object = None
    source: str = "NONE"        # LIVE | CACHED | PROXY | FALLBACK | NONE
    age_seconds: float = 0.0    # time since data was fetched
    confidence: float = 0.0     # 0-1 confidence factor
    error: str = ""

    @property
    def available(self) -> bool:
        return self.data is not None and self.source != "NONE"

    @property
    def stale(self) -> bool:
        return self.age_seconds > 900  # > 15 minutes

    def discount_factor(self) -> float:
        """Score discount based on data quality. 1.0 = full trust, 0.0 = no trust."""
        source_discount = {
            "LIVE": 1.0,
            "CACHED": 0.85,
            "PROXY": 0.7,
            "FALLBACK": 0.5,
            "NONE": 0.0,
        }
        base = source_discount.get(self.source, 0.0)
        # Age discount: linear decay after 5 minutes
        if self.age_seconds > 300:
            age_penalty = min(0.3, (self.age_seconds - 300) / 3600 * 0.3)
            base -= age_penalty
        return max(0.0, base)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS CHAIN FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

_options_cache: dict[str, tuple[pd.DataFrame, float]] = {}


def get_options_with_fallback(
    symbol: str,
    max_expiries: int = 3,
) -> DataResult:
    """
    Options chain with fallback chain:
      1. NSE live download
      2. Cached chain (if < 30 min old)
      3. None with clear indication

    Stores successful fetches in cache for #2.
    """
    from modules.data import download_options, is_market_open

    # Try live download (only during market hours)
    if is_market_open():
        try:
            df, msg = download_options(symbol, max_expiries=max_expiries)
            if df is not None and not df.empty:
                _options_cache[symbol] = (df, time.time())
                return DataResult(data=df, source="LIVE", confidence=1.0)
        except Exception as exc:
            logger.debug("Live options failed %s: %s", symbol, exc)

    # Try cache
    if symbol in _options_cache:
        cached_df, cached_ts = _options_cache[symbol]
        age = time.time() - cached_ts
        if age < 1800:  # 30 minutes
            return DataResult(
                data=cached_df, source="CACHED",
                age_seconds=age, confidence=0.85,
            )

    return DataResult(source="NONE", error=f"No options data for {symbol}")


# ══════════════════════════════════════════════════════════════════════════════
# MFI FALLBACK — SECTOR ETF PROXY
# ══════════════════════════════════════════════════════════════════════════════


def get_mfi_with_fallback(
    ps,  # PriceSignals from the stock
    symbol: str,
) -> DataResult:
    """
    MFI with fallback:
      1. Stock's own MFI (if reliable)
      2. Sector ETF MFI as proxy
      3. None

    Sector proxy MFI carries a confidence discount (0.6) because
    sector-level money flow doesn't perfectly predict stock-level.
    """
    # Primary: stock's own MFI
    if ps is not None and ps.mfi_value is not None and ps.mfi_reliable:
        return DataResult(
            data={"value": ps.mfi_value, "state": ps.mfi_state,
                  "diverge": ps.mfi_diverge},
            source="LIVE", confidence=1.0,
        )

    # Stock MFI exists but unreliable (thin volume)
    if ps is not None and ps.mfi_value is not None and not ps.mfi_reliable:
        # Try sector proxy
        sector_mfi = _get_sector_mfi(symbol)
        if sector_mfi is not None:
            return DataResult(
                data=sector_mfi, source="PROXY",
                confidence=0.6,
            )
        # Return unreliable stock MFI with low confidence
        return DataResult(
            data={"value": ps.mfi_value, "state": ps.mfi_state,
                  "diverge": ps.mfi_diverge},
            source="FALLBACK", confidence=0.4,
        )

    # No stock MFI at all — try sector
    sector_mfi = _get_sector_mfi(symbol)
    if sector_mfi is not None:
        return DataResult(data=sector_mfi, source="PROXY", confidence=0.5)

    return DataResult(source="NONE")


def _get_sector_mfi(symbol: str) -> Optional[dict]:
    """Look up sector ETF MFI from cached sector rotation data."""
    try:
        from modules.sector_rotation import get_sector_context
        ctx = get_sector_context(symbol)
        if ctx and ctx.get("mfi_value") is not None:
            return {
                "value": ctx["mfi_value"],
                "state": ctx.get("mfi_state", "NEUTRAL"),
                "diverge": ctx.get("mfi_diverge", False),
                "source_sector": ctx.get("name", "?"),
            }
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SPOT PRICE FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

_spot_cache: dict[str, tuple[float, float]] = {}


def get_spot_with_fallback(
    symbol: str,
    options_df: Optional[pd.DataFrame] = None,
    price_df: Optional[pd.DataFrame] = None,
) -> DataResult:
    """
    Spot price fallback chain:
      1. Options chain underlying value (most accurate during session)
      2. Latest close from price_df
      3. Cached spot
      4. None
    """
    from modules.data import infer_spot

    # 1. Options underlying
    if options_df is not None and not options_df.empty:
        opt_spot = infer_spot(options_df)
        if opt_spot and opt_spot > 0:
            _spot_cache[symbol] = (opt_spot, time.time())
            return DataResult(data=opt_spot, source="LIVE", confidence=1.0)

    # 2. Price DataFrame
    if price_df is not None and not price_df.empty:
        close = float(price_df.iloc[-1]["Close"])
        if close > 0:
            _spot_cache[symbol] = (close, time.time())
            return DataResult(data=close, source="LIVE", confidence=0.95)

    # 3. Cache
    if symbol in _spot_cache:
        cached_spot, cached_ts = _spot_cache[symbol]
        age = time.time() - cached_ts
        return DataResult(
            data=cached_spot, source="CACHED",
            age_seconds=age, confidence=max(0.3, 0.9 - age / 3600 * 0.3),
        )

    return DataResult(source="NONE", error=f"No spot price for {symbol}")
