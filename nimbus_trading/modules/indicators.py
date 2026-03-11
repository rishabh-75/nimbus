"""
modules/indicators.py
─────────────────────
Bollinger Bands (1σ, 20-period) + Williams %R (50-period).
Pure pandas, no external dependencies beyond numpy.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


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
    df["BB_Pct"] = (df[col] - df["BB_Lower"]) / band  # 0=lower, 1=upper
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


def bb_signal(df: pd.DataFrame) -> dict:
    """Current Bollinger Band state. Returns {} if no BB columns present."""
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
    """Current Williams %R state. Returns {} if no WR column."""
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
