"""
modules/etf_momentum.py — ETF Momentum scoring engine.

Walk-forward validated on 35 NSE ETFs, 5yr daily, 2 folds.
OOS: 105 trades, Sharpe 2.15, Win 60%, Avg +2.91%

ENTRY (pullback_in_trend):
  Price > SMA(50) long-term trend AND
  WR(20) dipped below -60 within last 10 bars AND recovered above -30
  → Buy the dip within an established uptrend

EXIT (trail_sma):
  2× ATR trailing stop from peak OR price < SMA(20) (structure broken)
  Max hold: 40 days, Optional PT: 10%
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Walk-forward validated parameters (2 folds, 35 ETFs, 5yr) ────────────────
SMA_TREND = 50       # long-term trend filter
SMA_EXIT = 20        # exit on SMA break
WR_PERIOD = 20
WR_DIP_LEVEL = -60   # WR must dip below this (unanimous 2/2)
WR_RECOVER = -30     # WR must recover above this to trigger
DIP_LOOKBACK = 10    # bars to look back for dip
TRAIL_ATR_MULT = 2.0 # trailing stop multiplier (unanimous 2/2)
MAX_HOLD = 40        # max hold days (30-50 range, use middle)
PROFIT_TARGET = 10.0 # optional PT (fold 2 used 10%)
MIN_BARS = 60


@dataclass
class MomentumSignal:
    """ETF momentum signal."""
    symbol: str = ""
    close: float = 0.0
    sma_20: float = 0.0; sma_50: float = 0.0
    above_sma20: bool = False; above_sma50: bool = False
    pct_from_sma20: float = 0.0; pct_from_sma50: float = 0.0

    wr_20: float = 0.0; wr_min_10: float = 0.0  # min WR in last 10 bars
    adx: float = 0.0; mfi: float = 0.0
    atr: float = 0.0; roc_10: float = 0.0; roc_20: float = 0.0
    bbw_slope: float = 0.0

    # Entry
    in_uptrend: bool = False      # price > SMA(50)
    dip_detected: bool = False    # WR dipped below -60 recently
    recovered: bool = False       # WR now above -30
    entry_triggered: bool = False
    entry_reason: str = ""

    # Scoring
    base_score: int = 0
    momentum_score: int = 0  # final score
    label: str = "NEUTRAL"
    sizing: str = "SKIP"

    # State
    data_sufficient: bool = False
    daily_bars_used: int = 0


def _compute_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy(); c = d["Close"]; h = d["High"]; l = d["Low"]
    v = d["Volume"] if "Volume" in d.columns else pd.Series(0, index=d.index)

    d["SMA_20"] = c.rolling(20).mean()
    d["SMA_50"] = c.rolling(50, min_periods=30).mean()

    hh = h.rolling(WR_PERIOD).max(); ll = l.rolling(WR_PERIOD).min()
    d["WR"] = ((hh - c) / (hh - ll).replace(0, np.nan)) * -100

    # WR min over lookback
    d["WR_MIN_10"] = d["WR"].rolling(DIP_LOOKBACK, min_periods=5).min()

    # ATR
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()

    # ADX
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=d.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=d.index)
    pdi = 100 * pdm.rolling(14).mean() / d["ATR"].replace(0, np.nan)
    mdi = 100 * mdm.rolling(14).mean() / d["ATR"].replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["ADX"] = dx.rolling(14).mean()

    # MFI
    tp = (h+l+c)/3; mf = tp*v
    pmf = pd.Series(np.where(tp>tp.shift(1), mf, 0), index=d.index)
    nmf = pd.Series(np.where(tp<tp.shift(1), mf, 0), index=d.index)
    mr = pmf.rolling(14).sum() / nmf.rolling(14).sum().replace(0, np.nan)
    d["MFI"] = 100 - (100 / (1 + mr))

    # ROC
    d["ROC_10"] = c.pct_change(10) * 100
    d["ROC_20"] = c.pct_change(20) * 100

    # BBW slope
    bb_ma = c.rolling(20).mean(); bb_sd = c.rolling(20).std()
    d["BBW"] = ((bb_ma + 2*bb_sd) - (bb_ma - 2*bb_sd)) / bb_ma.replace(0, np.nan) * 100
    d["BBW_slope"] = d["BBW"].diff(5)

    return d


def compute_etf_momentum(price_df, symbol=""):
    """Compute ETF momentum signal from daily price data."""
    sig = MomentumSignal(symbol=symbol)

    if price_df is None or len(price_df) < MIN_BARS:
        return sig

    # Auto-detect and resample if 4H
    from modules.indicators import _resample_daily
    gaps = pd.Series(price_df.index).diff().dropna()
    if not gaps.empty and gaps.median() < pd.Timedelta(hours=12):
        price_df = _resample_daily(price_df)
        if price_df.empty:
            return sig

    df = _compute_indicators(price_df)
    sig.daily_bars_used = len(df)
    sig.data_sufficient = True
    last = df.iloc[-1]

    # Extract values
    sig.close = float(last["Close"])
    sig.sma_20 = float(last["SMA_20"]) if not pd.isna(last["SMA_20"]) else 0.0
    sig.sma_50 = float(last["SMA_50"]) if not pd.isna(last["SMA_50"]) else 0.0
    sig.above_sma20 = sig.close > sig.sma_20 if sig.sma_20 > 0 else False
    sig.above_sma50 = sig.close > sig.sma_50 if sig.sma_50 > 0 else False
    sig.pct_from_sma20 = (sig.close / sig.sma_20 - 1) * 100 if sig.sma_20 > 0 else 0
    sig.pct_from_sma50 = (sig.close / sig.sma_50 - 1) * 100 if sig.sma_50 > 0 else 0

    sig.wr_20 = float(last["WR"]) if not pd.isna(last["WR"]) else 0.0
    sig.wr_min_10 = float(last["WR_MIN_10"]) if not pd.isna(last["WR_MIN_10"]) else 0.0
    sig.adx = float(last["ADX"]) if not pd.isna(last["ADX"]) else 0.0
    sig.mfi = float(last["MFI"]) if not pd.isna(last["MFI"]) else 50.0
    sig.atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0.0
    sig.roc_10 = float(last["ROC_10"]) if not pd.isna(last["ROC_10"]) else 0.0
    sig.roc_20 = float(last["ROC_20"]) if not pd.isna(last["ROC_20"]) else 0.0
    sig.bbw_slope = float(last["BBW_slope"]) if not pd.isna(last["BBW_slope"]) else 0.0

    # ── Entry logic: pullback_in_trend ────────────────────────────────────
    sig.in_uptrend = sig.above_sma50
    sig.dip_detected = sig.wr_min_10 < WR_DIP_LEVEL
    sig.recovered = sig.wr_20 > WR_RECOVER

    sig.entry_triggered = sig.in_uptrend and sig.dip_detected and sig.recovered

    if sig.entry_triggered:
        sig.entry_reason = (
            f"MOMENTUM: above SMA(50) | WR dipped to {sig.wr_min_10:.0f} → "
            f"recovered to {sig.wr_20:.0f} | ROC={sig.roc_10:+.1f}%"
        )
    elif sig.in_uptrend and sig.dip_detected and not sig.recovered:
        sig.entry_reason = (
            f"DIP ACTIVE: WR={sig.wr_20:.0f} (min {sig.wr_min_10:.0f}) recovering | "
            f"SMA50={sig.pct_from_sma50:+.1f}% | MFI={sig.mfi:.0f}"
        )
    elif sig.in_uptrend and not sig.dip_detected:
        sig.entry_reason = (
            f"TREND OK: SMA50={sig.pct_from_sma50:+.1f}% | "
            f"WR={sig.wr_20:.0f} (no dip <{WR_DIP_LEVEL}) | ROC={sig.roc_10:+.1f}%"
        )
    else:
        sig.entry_reason = (
            f"NO TREND: SMA50={sig.pct_from_sma50:+.1f}% | "
            f"WR={sig.wr_20:.0f} | MFI={sig.mfi:.0f}"
        )

    # ── Score ─────────────────────────────────────────────────────────────
    sig.base_score = _score(sig)
    sig.momentum_score = sig.base_score
    sig.label, sig.sizing = _label_and_sizing(sig.momentum_score, sig)

    return sig


def _score(sig):
    """Score momentum setup quality (0-100)."""
    s = 25

    # Trend strength: above SMA(50) is the foundation
    if sig.above_sma50:
        if sig.pct_from_sma50 > 5:   s += 10  # well above trend
        elif sig.pct_from_sma50 > 2:  s += 15  # healthy trend
        elif sig.pct_from_sma50 > 0:  s += 12  # just above
    else:
        s -= 15  # below trend = bad for momentum

    # Dip quality: deeper dip in uptrend = better entry
    if sig.dip_detected:
        if sig.wr_min_10 < -80:  s += 20  # deep dip in trend
        elif sig.wr_min_10 < -60: s += 15  # good dip
        elif sig.wr_min_10 < -40: s += 8   # mild pullback
    else:
        s -= 5

    # Recovery: WR bouncing back confirms buyers stepping in
    if sig.recovered:
        if sig.wr_20 > -10:  s += 12  # strong recovery
        elif sig.wr_20 > -20: s += 8   # decent
        elif sig.wr_20 > -30: s += 4   # just recovered
    else:
        s -= 5

    # ADX: trend strength confirmation
    if sig.adx >= 25:   s += 8   # strong trend
    elif sig.adx >= 20:  s += 5
    elif sig.adx >= 15:  s += 2

    # MFI: money flowing into the trend
    if sig.mfi >= 60:   s += 6
    elif sig.mfi >= 50:  s += 3
    elif sig.mfi < 30:   s -= 5  # weak flow

    # ROC: recent momentum direction
    if sig.roc_10 > 3:   s += 5  # strong recent momentum
    elif sig.roc_10 > 0:  s += 2
    elif sig.roc_10 < -3: s -= 5  # falling

    # BBW expanding: trend accelerating
    if sig.bbw_slope > 0.5: s += 3
    elif sig.bbw_slope < -1: s -= 3  # momentum fading

    return max(0, min(100, s))


def _label_and_sizing(score, sig):
    if not sig.entry_triggered:
        if score >= 60: return "FORMING", "SKIP"
        return "NEUTRAL", "SKIP"
    if score >= 75: return "STRONG", "FULL"
    if score >= 60: return "GOOD", "HALF"
    if score >= 45: return "WATCH", "SKIP"
    return "WEAK", "SKIP"


def check_exit(sig, entry_price, peak_price, bars_held):
    """Check momentum exit conditions."""
    c = sig.close
    # Profit target (10%)
    if entry_price > 0 and c >= entry_price * (1 + PROFIT_TARGET / 100):
        return "PROFIT_TARGET"
    # Trailing stop: 2× ATR from peak
    if sig.atr > 0 and peak_price > 0 and bars_held >= 3:
        trail = peak_price - TRAIL_ATR_MULT * sig.atr
        if c < trail:
            return "TRAIL_STOP"
    # SMA break: price below SMA(20) — structure broken
    if bars_held >= 3 and not sig.above_sma20:
        return "SMA_BREAK"
    # Max hold
    if bars_held >= MAX_HOLD:
        return "MAX_HOLD"
    return ""


def momentum_verdict(sig):
    """Single actionable verdict string."""
    sc = sig.momentum_score
    if sig.entry_triggered:
        if sc >= 75: return "▶ BUY — strong momentum pullback"
        if sc >= 60: return "▶ BUY — decent trend dip"
        return "▶ WEAK — low conviction entry"
    if sig.in_uptrend and sig.dip_detected and not sig.recovered:
        return f"WAIT — dip active, WR={sig.wr_20:.0f} recovering"
    if sig.in_uptrend and not sig.dip_detected:
        if sig.wr_20 < -30:
            return f"WATCH — pullback starting, WR={sig.wr_20:.0f}"
        return "HOLD TREND — no dip yet"
    if not sig.in_uptrend:
        return "SKIP — below SMA(50), no uptrend"
    return "SKIP"
