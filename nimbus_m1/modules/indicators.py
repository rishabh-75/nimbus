"""
modules/indicators.py
─────────────────────
Price-based signals for NIMBUS. Computes:
- Bollinger Bands (configurable period/std)
- Williams %R (configurable period)
- MFI — Money Flow Index (daily-resampled, volume-reliability gated)
- Daily bias (20-SMA from daily data)
- Volatility state (SQUEEZE / NORMAL / EXPANDED)
- Position state (Riding / First Dip / Mid-Band Broken)
- WR momentum phase (FRESH / DEVELOPING / LATE)
- PriceSignals dataclass — single object passed to analytics + commentary

Phase 1 addition: adv_cr (20-day avg daily turnover Rs Cr) in PriceSignals
Phase 2 addition: MFI signals (mfi_value, mfi_state, mfi_diverge, mfi_reliable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import logging

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

    # Phase 1: 20-day average daily turnover (Rs Cr)
    adv_cr: float = 0.0

    # Phase 2: MFI — Money Flow Index (daily resampled, volume-gated)
    # mfi_reliable=False means thin volume; signal shown but score unchanged
    mfi_value: Optional[float] = None  # 0-100, None if Volume unavailable
    mfi_state: str = "NEUTRAL"  # STRONG | RISING | NEUTRAL | FALLING | WEAK
    mfi_diverge: bool = False  # price near high + MFI trending down = distribution
    mfi_reliable: bool = False  # False when adv_cr < threshold or bars < 10


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


def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Money Flow Index — volume-weighted RSI.
    Requires High, Low, Close, Volume columns.
    Adds MFI_14 column. Silently skips if columns missing.
    NOTE: Call this on daily data,
    to avoid single-bar block-deal noise (tested in test_mfi.py).
    """
    req = {"High", "Low", "Close", "Volume"}
    if not req.issubset(df.columns):
        return df
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    pos_r = pos.rolling(period).sum()
    neg_r = neg.rolling(period).sum()
    mfr = pos_r / neg_r.replace(0, float("nan"))
    df[f"MFI_{period}"] = (100 - (100 / (1 + mfr))).round(2)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index — trend strength (0-100).
    ADX > 20 = trending, ADX < 20 = ranging.
    Used by Mode A (Financial Early Momentum) as a trend confirmation filter.
    """
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return df
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df[f"ADX_{period}"] = dx.rolling(period).mean()
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

_MFI_RELIABLE_ADV_CR = 5.0  # below this, MFI is too noisy to score


def compute_price_signals(
    price_df: pd.DataFrame,
    wr_thresh: float = -20.0,
    vol_lookback: int = 100,
) -> PriceSignals:
    """
    Compute all price-derived signals in one pass.
    Returns a PriceSignals dataclass consumed by analytics + commentary.

    MFI is computed on the DAILY resampled frame (reuses the resample already
    done for daily bias) — no extra network call.
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

    # ── Daily bias + MFI (both use the same daily resample) ──────────────
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

            # ── MFI on daily frame ────────────────────────────────────────
            # add Volume to daily resample if available in source df
            daily = add_mfi(daily, period=14)
            _compute_mfi_signals(daily, ps, close)

    # In compute_price_signals, replace the bare except:
    except Exception as _e:
        logging.getLogger(__name__).warning("MFI daily block failed: %s", _e)
        pass

    # ── BB signals ────────────────────────────────────────────────────────
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

    # ── Williams %R signals ───────────────────────────────────────────────
    if "WR" in df.columns:
        wr_series = df["WR"].dropna()
        if not wr_series.empty:
            last_wr = float(wr_series.iloc[-1])
            prev_wr = float(wr_series.iloc[-2]) if len(wr_series) > 1 else last_wr
            ps.wr_value = last_wr
            ps.wr_in_momentum = last_wr >= wr_thresh
            diff = last_wr - prev_wr
            ps.wr_trend = "rising" if diff > 1 else ("falling" if diff < -1 else "flat")

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

    # ── ADV — 20-day average daily turnover Rs Cr ─────────────────────────
    if {"Close", "Volume"}.issubset(df.columns):
        try:
            ps.adv_cr = round(
                (df["Close"] * pd.to_numeric(df["Volume"], errors="coerce"))
                .tail(20)
                .mean()
                / 1e7,
                2,
            )
        except Exception:
            ps.adv_cr = 0.0

    # ── Combined entry gate ───────────────────────────────────────────────
    ps.entry_valid = ps.wr_in_momentum and (ps.daily_bias in ("BULLISH", "NEUTRAL"))

    return ps


def _compute_mfi_signals(daily: pd.DataFrame, ps: PriceSignals, close: float) -> None:
    """
    Populate ps.mfi_* from a daily frame that already has MFI_14 column.
    Mutates ps in place. Called only from compute_price_signals.
    """
    mfi_col = "MFI_14"
    if mfi_col not in daily.columns:
        return

    mfi_series = daily[mfi_col].dropna()
    if len(mfi_series) < 3:
        return

    mfi_now = float(mfi_series.iloc[-1])
    ps.mfi_value = mfi_now

    # State classification
    if mfi_now > 70:
        ps.mfi_state = "STRONG"
    elif mfi_now > 55:
        ps.mfi_state = "RISING"
    elif mfi_now > 45:
        ps.mfi_state = "NEUTRAL"
    elif mfi_now > 30:
        ps.mfi_state = "FALLING"
    else:
        ps.mfi_state = "WEAK"

    # Reliability gate — thin volume makes MFI noisy; VSR already covers that
    ps.mfi_reliable = ps.adv_cr >= _MFI_RELIABLE_ADV_CR and len(mfi_series) >= 10

    # Single-bar spike guard (block deal / FII block buy on last daily bar)
    last_bar_jump = (
        len(mfi_series) >= 2 and abs(mfi_now - float(mfi_series.iloc[-2])) > 20
    )

    # Bearish divergence detection (only when reliable + no spike)
    if len(mfi_series) >= 6 and len(daily) >= 9 and not last_bar_jump:
        price_near_high = close >= float(daily["Close"].iloc[-9:-1].max()) * 0.98

        # Directional: MFI must be trending down consistently over 3+ steps
        mfi_trending_down = (
            len(mfi_series) >= 6
            and float(mfi_series.iloc[-1]) < float(mfi_series.iloc[-3])
            and float(mfi_series.iloc[-3]) < float(mfi_series.iloc[-6])
        )
        mfi_falling_from_high = (
            mfi_now < float(mfi_series.iloc[-6]) and mfi_trending_down
        )
        ps.mfi_diverge = price_near_high and mfi_falling_from_high


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    try:
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
        tmp = df.copy()
        tmp.index = pd.DatetimeIndex(idx)

        # Only aggregate columns that actually exist in df
        _candidates = {
            "Open": ("Open", "first"),
            "High": ("High", "max"),
            "Low": ("Low", "min"),
            "Close": ("Close", "last"),
            "Volume": ("Volume", "sum"),
        }
        agg_dict = {k: v for k, v in _candidates.items() if k in tmp.columns}

        daily = tmp.resample("1D").agg(**agg_dict).dropna(subset=["Close"])
        return daily[daily["Close"] > 0]
    except Exception as _e:
        logging.getLogger(__name__).warning("_resample_daily failed: %s", _e)
        return pd.DataFrame()


def _position_state(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "UNKNOWN"
    closes = df["Close"].values
    uppers = df["BB_Upper"].values
    mids = df["BB_Mid"].values

    c0, u0, m0 = closes[-1], uppers[-1], mids[-1]
    c1, u1 = closes[-2], uppers[-2]

    if pd.isna(u0) or pd.isna(m0):
        return "UNKNOWN"
    if c0 < m0:
        return "MID_BAND_BROKEN"
    if c0 < u0 and c1 >= u1:
        return "FIRST_DIP"
    if c0 >= u0 or (c0 >= m0 and (df["Close"].tail(4) >= df["BB_Mid"].tail(4)).all()):
        return "RIDING_UPPER"
    return "CONSOLIDATING"


def _bars_since_wr_cross(wr_series: pd.Series, level: float = -50.0) -> int:
    arr = wr_series.dropna().values[-50:]
    for i in range(len(arr) - 1, 0, -1):
        if arr[i] >= level and arr[i - 1] < level:
            return len(arr) - 1 - i
    return 99
