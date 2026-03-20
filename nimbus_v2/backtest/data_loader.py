"""
backtest/data_loader.py
────────────────────────
Historical OHLCV data management for backtesting.

Supports:
  - yfinance download with local parquet cache
  - Synthetic data generation for offline testing
  - Batch loading of NIFTY100 universe
  - Clean timezone-normalized output

Cache: data/backtest_cache/ (parquet files, one per symbol)
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "backtest_cache"
)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE I/O
# ══════════════════════════════════════════════════════════════════════════════


def _cache_path(symbol: str, interval: str = "1d") -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{symbol}_{interval}.parquet")


def _load_cache(symbol: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, interval)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        age_days = (date.today() - df.index[-1].date()).days if len(df) > 0 else 999
        if age_days > 3:
            logger.info("Cache stale for %s (%dd old)", symbol, age_days)
            return None
        return df
    except Exception as exc:
        logger.warning("Cache read failed %s: %s", symbol, exc)
        return None


def _save_cache(df: pd.DataFrame, symbol: str, interval: str = "1d"):
    try:
        path = _cache_path(symbol, interval)
        df.to_parquet(path)
    except Exception as exc:
        logger.warning("Cache write failed %s: %s", symbol, exc)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════


def _yf_ticker(symbol: str) -> str:
    """Map NSE symbol to yfinance ticker."""
    mapping = {
        "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCPNIFTY": "^NSMIDCP", "NIFTYNXT50": "NIFTY_NEXT_50.NS",
    }
    return mapping.get(symbol, f"{symbol}.NS")


def download_history(
    symbol: str,
    years: int = 3,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV for a symbol.

    Returns a clean DataFrame with columns: Open, High, Low, Close, Volume
    Index: DatetimeIndex (timezone-naive, IST-aligned)
    """
    if use_cache:
        cached = _load_cache(symbol, interval)
        if cached is not None:
            logger.debug("Cache hit: %s (%d bars)", symbol, len(cached))
            return cached

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — returning empty DataFrame")
        return pd.DataFrame()

    ticker = _yf_ticker(symbol)
    start = (date.today() - timedelta(days=years * 365)).isoformat()

    try:
        raw = yf.download(
            tickers=ticker, start=start, interval=interval,
            auto_adjust=True, progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).strip().title().replace(" ", "_") for c in raw.columns]
        raw = raw.rename(columns={"Adj_Close": "Close"})

        # Normalize index
        raw.index = pd.to_datetime(raw.index)
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df[df["Close"] > 0].dropna(subset=["Close"])

        if not df.empty and use_cache:
            _save_cache(df, symbol, interval)

        logger.info("Downloaded %s: %d bars (%s to %s)",
                     symbol, len(df), df.index[0].date(), df.index[-1].date())
        return df

    except Exception as exc:
        logger.error("Download failed %s: %s", symbol, exc)
        return pd.DataFrame()


def download_batch(
    symbols: list[str],
    years: int = 3,
    interval: str = "1d",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for multiple symbols. Returns {symbol: DataFrame}."""
    result = {}
    for i, sym in enumerate(symbols):
        df = download_history(sym, years, interval, use_cache)
        if not df.empty:
            result[sym] = df
        if (i + 1) % 10 == 0:
            logger.info("Batch progress: %d/%d", i + 1, len(symbols))
    logger.info("Batch complete: %d/%d symbols loaded", len(result), len(symbols))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA (for offline testing / CI)
# ══════════════════════════════════════════════════════════════════════════════


def generate_synthetic(
    n_bars: int = 750,
    base_price: float = 1000.0,
    annual_return: float = 0.12,
    annual_vol: float = 0.25,
    vol_base: float = 5e6,
    seed: int = 42,
    regime_changes: bool = True,
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV with regime changes.

    Produces trending + mean-reverting regimes to test signal robustness.
    750 bars ≈ 3 years of daily data.
    """
    rng = np.random.RandomState(seed)
    daily_ret = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)

    idx = pd.bdate_range("2021-06-01", periods=n_bars)

    # Regime-switching: alternate trending / mean-reverting
    if regime_changes:
        regime_len = rng.randint(30, 80, size=20)
        regimes = []
        for i, rl in enumerate(regime_len):
            regimes.extend(["trending" if i % 2 == 0 else "ranging"] * rl)
        regimes = regimes[:n_bars]
    else:
        regimes = ["trending"] * n_bars

    closes = [base_price]
    for i in range(1, n_bars):
        if regimes[i] == "trending":
            mu = daily_ret * (1 + rng.uniform(-0.5, 1.5))
            sig = daily_vol * 0.8
        else:
            mu = 0.0
            sig = daily_vol * 1.3
        ret = mu + sig * rng.randn()
        closes.append(closes[-1] * (1 + ret))

    closes = np.array(closes)
    highs = closes * (1 + np.abs(rng.normal(0.005, 0.003, n_bars)))
    lows = closes * (1 - np.abs(rng.normal(0.005, 0.003, n_bars)))
    opens = np.roll(closes, 1) * (1 + rng.normal(0, 0.002, n_bars))
    opens[0] = base_price

    # Volume: higher in trending, lower in ranging, with spikes
    vol_mult = np.where(
        np.array(regimes[:n_bars]) == "trending",
        rng.uniform(0.8, 1.5, n_bars),
        rng.uniform(0.4, 0.9, n_bars),
    )
    # Random volume spikes (block deals)
    spike_mask = rng.random(n_bars) < 0.03
    vol_mult[spike_mask] *= rng.uniform(3, 8, spike_mask.sum())
    volumes = vol_base * vol_mult

    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes,
    }, index=idx[:n_bars])


def generate_universe(
    n_symbols: int = 20,
    n_bars: int = 750,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate a synthetic universe of N symbols with varied characteristics."""
    rng = np.random.RandomState(seed)
    universe = {}
    for i in range(n_symbols):
        sym = f"SYN{i:03d}"
        df = generate_synthetic(
            n_bars=n_bars,
            base_price=rng.uniform(200, 5000),
            annual_return=rng.uniform(-0.05, 0.25),
            annual_vol=rng.uniform(0.15, 0.45),
            vol_base=rng.uniform(1e6, 2e7),
            seed=seed + i,
        )
        universe[sym] = df
    return universe
