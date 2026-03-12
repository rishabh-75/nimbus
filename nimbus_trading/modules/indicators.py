"""
modules/indicators.py
─────────────────────
Price-based signals for NIMBUS. Computes:
  - Bollinger Bands (configurable period/std)
  - Williams %R (configurable period)
  - Daily bias (20-SMA from resampled 4H data)
  - Volatility state (SQUEEZE / NORMAL / EXPANDED)
  - Position state (Riding / First Dip / Mid-Band Broken)
  - WR momentum phase (FRESH / DEVELOPING / LATE)
  - PriceSignals dataclass — single object passed to analytics + commentary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PriceSignals:
    # Daily bias
    daily_sma: Optional[float] = None
    daily_bias: str = "NEUTRAL"  # BULLISH | BEARISH | NEUTRAL
    daily_bias_pct: float = 0.0  # spot vs SMA %

    # BB state
    bb_position: str = "unknown"  # above_upper | riding | below_mid | near_lower
    bb_pct: float = 0.5
    position_state: str = "UNKNOWN"  # see _position_state()
    upper: float = 0.0
    mid: float = 0.0
    lower: float = 0.0
    last_close: float = 0.0

    # Volatility
    vol_state: str = "NORMAL"  # SQUEEZE | NORMAL | EXPANDED
    bb_width_pct: float = 0.0
    bb_width_pctl: float = 50.0  # current width percentile (0-100)

    # Williams %R
    wr_value: Optional[float] = None
    wr_in_momentum: bool = False
    wr_trend: str = "flat"
    wr_bars_since_cross50: int = 99  # bars since WR crossed above -50
    wr_phase: str = "NONE"  # FRESH | DEVELOPING | LATE | NONE

    # Combined quick flags
    entry_valid: bool = False  # WR in zone + daily bullish
    bb_squeezing: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# CORE INDICATOR CALCULATORS
# ══════════════════════════════════════════════════════════════════════════════


def add_bollinger(
    df: pd.DataFrame, period: int = 20, std_dev: float = 1.0, col: str = "Close"
) -> pd.DataFrame:
    df = df.copy()
    mid = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    df["BB_Mid"] = mid
    df["BB_Upper"] = mid + std_dev * std
    df["BB_Lower"] = mid - std_dev * std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid.replace(0, np.nan)
    band = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Pct"] = (df[col] - df["BB_Lower"]) / band
    return df


def add_williams_r(
    df: pd.DataFrame,
    period: int = 50,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    df = df.copy()
    hh = df[high_col].rolling(period).max()
    ll = df[low_col].rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    df["WR"] = -100 * (hh - df[close_col]) / rng
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTORS (current-bar snapshots)
# ══════════════════════════════════════════════════════════════════════════════


def bb_signal(df: pd.DataFrame) -> dict:
    """Legacy dict interface — used in a few places. Prefer compute_price_signals()."""
    if df.empty or "BB_Upper" not in df.columns:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    close = float(last["Close"])
    upper = float(last["BB_Upper"])
    mid = float(last["BB_Mid"])
    lower = float(last["BB_Lower"])
    bb_pct = float(last.get("BB_Pct", 0.5) or 0.5)
    riding = bool((df["Close"].tail(4) >= df["BB_Mid"].tail(4)).all())
    broke = (float(prev["Close"]) >= float(prev["BB_Upper"])) and (close < upper)
    if close >= upper:
        position = "above_upper"
    elif close >= mid:
        position = "riding"
    elif close >= lower:
        position = "below_mid"
    else:
        position = "near_lower"
    return dict(
        riding_upper=riding,
        broke_upper=broke,
        position=position,
        bb_pct=bb_pct,
        last_close=close,
        upper=upper,
        mid=mid,
        lower=lower,
    )


def wr_signal(df: pd.DataFrame, threshold: float = -20.0) -> dict:
    """Legacy dict interface."""
    if df.empty or "WR" not in df.columns:
        return {}
    wr = df["WR"].dropna()
    if wr.empty:
        return {}
    last_wr = float(wr.iloc[-1])
    prev_wr = float(wr.iloc[-2]) if len(wr) > 1 else last_wr
    diff = last_wr - prev_wr
    trend = "rising" if diff > 1 else ("falling" if diff < -1 else "flat")
    return dict(
        in_momentum=(last_wr >= threshold),
        wr_value=last_wr,
        trend=trend,
        threshold=threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RICH SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_price_signals(
    price_df: pd.DataFrame,
    wr_thresh: float = -20.0,
    vol_lookback: int = 100,  # bars for bandwidth percentile
) -> PriceSignals:
    """
    Compute all price-derived signals in one pass.
    Returns a PriceSignals dataclass consumed by analytics + commentary.
    """
    ps = PriceSignals()
    if price_df is None or price_df.empty:
        return ps

    df = price_df.copy()
    if "BB_Upper" not in df.columns or "WR" not in df.columns:
        return ps

    last = df.iloc[-1]
    close = float(last["Close"])
    ps.last_close = close

    # ── Daily bias (resample 4H → daily 20-SMA) ───────────────────────────────
    try:
        daily = _resample_daily(df)
        if not daily.empty and len(daily) >= 20:
            sma20 = daily["Close"].rolling(20).mean()
            ps.daily_sma = (
                float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else None
            )
            if ps.daily_sma:
                ps.daily_bias_pct = (close - ps.daily_sma) / ps.daily_sma * 100
                if close > ps.daily_sma * 1.002:
                    ps.daily_bias = "BULLISH"
                elif close < ps.daily_sma * 0.998:
                    ps.daily_bias = "BEARISH"
                else:
                    ps.daily_bias = "NEUTRAL"
    except Exception:
        pass

    # ── BB signals ────────────────────────────────────────────────────────────
    if "BB_Upper" in df.columns:
        upper = float(last["BB_Upper"])
        lower = float(last["BB_Lower"])
        mid = float(last["BB_Mid"])
        bb_pct = float(last.get("BB_Pct", 0.5) or 0.5)

        ps.upper = upper
        ps.mid = mid
        ps.lower = lower
        ps.bb_pct = bb_pct

        if close >= upper:
            ps.bb_position = "above_upper"
        elif close >= mid:
            ps.bb_position = "riding"
        elif close >= lower:
            ps.bb_position = "below_mid"
        else:
            ps.bb_position = "near_lower"

        ps.position_state = _position_state(df)

        # Volatility: BB bandwidth percentile
        if "BB_Width" in df.columns:
            widths = df["BB_Width"].dropna().tail(vol_lookback)
            cur_width = float(last.get("BB_Width", widths.mean()))
            ps.bb_width_pct = cur_width * 100
            if len(widths) >= 20:
                pctl = float((widths < cur_width).mean() * 100)
                ps.bb_width_pctl = pctl
                if pctl <= 20:
                    ps.vol_state = "SQUEEZE"
                    ps.bb_squeezing = True
                elif pctl >= 80:
                    ps.vol_state = "EXPANDED"
                else:
                    ps.vol_state = "NORMAL"

    # ── Williams %R signals ───────────────────────────────────────────────────
    if "WR" in df.columns:
        wr_series = df["WR"].dropna()
        if not wr_series.empty:
            last_wr = float(wr_series.iloc[-1])
            prev_wr = float(wr_series.iloc[-2]) if len(wr_series) > 1 else last_wr
            ps.wr_value = last_wr
            ps.wr_in_momentum = last_wr >= wr_thresh
            diff = last_wr - prev_wr
            ps.wr_trend = "rising" if diff > 1 else ("falling" if diff < -1 else "flat")

            # Bars since WR crossed above -50 (entry momentum confirmation)
            bars_since = _bars_since_wr_cross(wr_series, level=-50.0)
            ps.wr_bars_since_cross50 = bars_since

            if not ps.wr_in_momentum:
                ps.wr_phase = "NONE"
            elif bars_since <= 3:
                ps.wr_phase = "FRESH"
            elif bars_since <= 10:
                ps.wr_phase = "DEVELOPING"
            else:
                ps.wr_phase = "LATE"

    # ── Combined entry gate ───────────────────────────────────────────────────
    ps.entry_valid = ps.wr_in_momentum and (ps.daily_bias in ("BULLISH", "NEUTRAL"))

    return ps


def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 4H bars to daily OHLCV."""
    try:
        idx = df.index
        # Strip timezone for resampling
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
        tmp = df.copy()
        tmp.index = pd.DatetimeIndex(idx)
        daily = (
            tmp.resample("1D")
            .agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
            )
            .dropna(subset=["Close"])
        )
        return daily[daily["Close"] > 0]
    except Exception:
        return pd.DataFrame()


def _position_state(df: pd.DataFrame) -> str:
    """
    Determine which phase of the BB position cycle we are in:
      Riding Upper Band → First Dip (partial exit) → Mid-Band Broken (full exit)
    """
    if len(df) < 3:
        return "UNKNOWN"
    closes = df["Close"].values
    uppers = df["BB_Upper"].values
    mids = df["BB_Mid"].values

    c0, u0, m0 = closes[-1], uppers[-1], mids[-1]
    c1, u1 = closes[-2], uppers[-2]

    if pd.isna(u0) or pd.isna(m0):
        return "UNKNOWN"

    # Currently below mid → mid-band broken = full exit zone
    if c0 < m0:
        return "MID_BAND_BROKEN"

    # Currently below upper but above mid (and previous was above upper)
    if c0 < u0 and c1 >= u1:
        return "FIRST_DIP"

    # Currently above or hugging upper band
    if c0 >= u0 or (c0 >= m0 and (df["Close"].tail(4) >= df["BB_Mid"].tail(4)).all()):
        return "RIDING_UPPER"

    # Between mid and upper, not a fresh dip
    return "CONSOLIDATING"


def _bars_since_wr_cross(wr_series: pd.Series, level: float = -50.0) -> int:
    """
    How many bars ago did WR last cross upward through `level`?
    Returns 99 if no recent cross found in last 50 bars.
    """
    arr = wr_series.dropna().values[-50:]  # look back max 50 bars
    for i in range(len(arr) - 1, 0, -1):
        # Cross up: previous below level, current above
        if arr[i] >= level and arr[i - 1] < level:
            return len(arr) - 1 - i
    return 99
