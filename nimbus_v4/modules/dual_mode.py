"""
modules/dual_mode.py — Unified mean-reversion scoring engine for NIMBUS.

Sweep-validated on 30 NSE stocks, 3yr daily data, 60/40 IS/OOS split.
WF-validated: 1733 OOS trades, Sharpe 1.17, Win 73.4%, PF 1.96

ENTRY (all on DAILY bars):
  Core:    WR(30) < -30 AND close < SMA(20) AND MFI(14) > 30
  Primary: + drawdown > 5% from 50d high AND ≥2 consecutive red days AND vol > 0.5x avg
  Secondary: core only (more signals, lower conviction)

EXIT:
  BBW contraction (BBW slope < 0 AND above SMA) OR +5% profit target
  Max hold: 25 days

Score = base(0-100) + options_overlay(±10) + filing_overlay(±10)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from modules.indicators import add_williams_r, add_adx, _resample_daily

logger = logging.getLogger(__name__)

# ── Walk-forward validated parameters (68 stocks, 5yr, 5 folds) ────────────────────────────────────────────────
WR_PERIOD = 30; WR_THRESH = -30
SMA_PERIOD = 20; MFI_PERIOD = 14; MFI_MIN = 30
DD_MODERATE = -5.0    # drawdown from 50d high (%)
RED_STREAK_MIN = 2    # consecutive red days (WF: unanimous 5/5, was 3)
VOL_MIN_RATIO = 0.5   # volume vs 20d average
PROFIT_TARGET = 5.0   # exit at +5% (WF: unanimous 5/5, was 3%)
MAX_HOLD = 25          # max hold days (WF: 3/5 chose 25, was 30)
MIN_DAILY_BARS = 35

# ── Signal dataclass ──────────────────────────────────────────────────────────
@dataclass
class DualModeSignal:
    """Unified mean-reversion signal. Named DualModeSignal for backward compat."""
    symbol: str = ""
    close: float = 0.0; sma_20: float = 0.0
    above_sma: bool = False; pct_from_sma: float = 0.0
    input_interval: str = ""; daily_bars_used: int = 0; data_sufficient: bool = False

    # Indicators
    wr_30: float = -100.0; mfi: float = 50.0; mfi_slope: float = 0.0
    adx_14: float = 0.0; rsi: float = 50.0
    bbw: float = 0.0; bbw_slope: float = 0.0; bbw_pctl: float = 50.0
    vol_ratio: float = 1.0
    dd_from_high: float = 0.0   # % drawdown from 50d high (negative)
    red_streak: int = 0          # consecutive red candles
    bb_upper: float = 0.0; bb_lower: float = 0.0

    # Entry classification
    tier: str = "NONE"          # PRIMARY, SECONDARY, NONE
    entry_triggered: bool = False
    entry_reason: str = ""
    core_met: bool = False       # WR + SMA + MFI
    primary_met: bool = False    # core + DD + streak + vol
    secondary_met: bool = False  # core only

    # Scoring
    base_score: int = 0
    options_overlay: int = 0; options_detail: str = ""
    filing_overlay: int = 0; filing_detail: str = ""
    is_trap: bool = False
    dual_score: int = 0; dual_label: str = "NEUTRAL"; dual_sizing: str = "SKIP"

    # Exit state
    exit_signal: str = ""
    bbw_contracting: bool = False; above_sma_exit: bool = False

    # Backward compat aliases
    segment: str = "UNIFIED"; mode: str = "MR"
    wr_20: float = -100.0; wr_20_cross_bars: int = 99; adx_trending: bool = False
    mode_a_entry: bool = False; mode_b_entry: bool = False


# ── Indicator computation ─────────────────────────────────────────────────────

def _compute_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators needed for scoring. Operates on daily bars."""
    d = daily.copy()
    c = d["Close"]; h = d["High"]; l = d["Low"]
    v = d["Volume"] if "Volume" in d.columns else pd.Series(0, index=d.index)

    # SMA
    d["SMA"] = c.rolling(SMA_PERIOD).mean()

    # WR(30)
    hh = h.rolling(WR_PERIOD).max(); ll = l.rolling(WR_PERIOD).min()
    d["WR"] = ((hh - c) / (hh - ll).replace(0, np.nan)) * -100

    # ADX(14)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); d["ATR"] = atr
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=d.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=d.index)
    pdi = 100 * pdm.rolling(14).mean() / atr.replace(0, np.nan)
    mdi = 100 * mdm.rolling(14).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["ADX"] = dx.rolling(14).mean()

    # MFI(14)
    tp = (h + l + c) / 3; mf = tp * v
    pmf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=d.index)
    nmf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=d.index)
    mr = pmf.rolling(MFI_PERIOD).sum() / nmf.rolling(MFI_PERIOD).sum().replace(0, np.nan)
    d["MFI"] = 100 - (100 / (1 + mr))
    d["MFI_slope"] = d["MFI"].diff(3)

    # RSI(14)
    delta = c.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.rolling(14).mean(); al = loss.rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))

    # BB (period=20, std=2)
    bb_ma = c.rolling(20).mean(); bb_sd = c.rolling(20).std()
    d["BB_Upper"] = bb_ma + 2.0 * bb_sd
    d["BB_Lower"] = bb_ma - 2.0 * bb_sd
    d["BBW"] = ((d["BB_Upper"] - d["BB_Lower"]) / bb_ma.replace(0, np.nan) * 100)
    d["BBW_slope"] = d["BBW"].diff(5)
    d["BBW_pctl"] = d["BBW"].rolling(100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Volume ratio
    d["VOL_RATIO"] = v / v.rolling(20).mean().replace(0, np.nan)

    # Drawdown from 50d high (use min_periods for short histories)
    d["HIGH_50"] = h.rolling(50, min_periods=20).max()
    d["DD_FROM_HIGH"] = ((c - d["HIGH_50"]) / d["HIGH_50"]) * 100

    # Consecutive red days
    red = (c < c.shift(1)).astype(int)
    d["RED_STREAK"] = red.groupby((red != red.shift()).cumsum()).cumsum()

    return d


# ── Detect interval and resample ──────────────────────────────────────────────

def _detect_and_resample(df):
    if df is None or df.empty or len(df) < 10:
        return df, "UNKNOWN"
    gaps = pd.Series(df.index).diff().dropna()
    if gaps.empty:
        return df, "UNKNOWN"
    if gaps.median() < pd.Timedelta(hours=12):
        daily = _resample_daily(df)
        return (daily, "4H") if not daily.empty else (df, "4H_FAILED")
    return df, "1D"


# ── Main computation ──────────────────────────────────────────────────────────

def compute_dual_mode(price_df, symbol="", segment="", options_ctx=None, filing_variance=None):
    """Compute unified mean-reversion signal from price data."""
    sig = DualModeSignal(symbol=symbol, segment="UNIFIED", mode="MR")

    if price_df is None or len(price_df) < 20:
        return sig

    daily, interval = _detect_and_resample(price_df)
    sig.input_interval = interval
    if daily is None or len(daily) < MIN_DAILY_BARS:
        return sig

    sig.daily_bars_used = len(daily)
    sig.data_sufficient = True

    # Compute all indicators
    df = _compute_indicators(daily)
    last = df.iloc[-1]

    # Extract values
    sig.close = float(last["Close"])
    sig.sma_20 = float(last["SMA"]) if not pd.isna(last["SMA"]) else 0.0
    if sig.sma_20 > 0:
        sig.above_sma = sig.close > sig.sma_20
        sig.pct_from_sma = (sig.close / sig.sma_20 - 1) * 100

    sig.wr_30 = float(last["WR"]) if not pd.isna(last["WR"]) else -100.0
    sig.mfi = float(last["MFI"]) if not pd.isna(last["MFI"]) else 50.0
    sig.mfi_slope = float(last["MFI_slope"]) if not pd.isna(last["MFI_slope"]) else 0.0
    sig.adx_14 = float(last["ADX"]) if not pd.isna(last["ADX"]) else 0.0
    sig.adx_trending = sig.adx_14 >= 20
    sig.rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50.0
    sig.bbw = float(last["BBW"]) if not pd.isna(last["BBW"]) else 0.0
    sig.bbw_slope = float(last["BBW_slope"]) if not pd.isna(last["BBW_slope"]) else 0.0
    sig.bbw_pctl = float(last["BBW_pctl"]) if not pd.isna(last.get("BBW_pctl", np.nan)) else 50.0
    sig.vol_ratio = float(last["VOL_RATIO"]) if not pd.isna(last["VOL_RATIO"]) else 1.0
    sig.dd_from_high = float(last["DD_FROM_HIGH"]) if not pd.isna(last["DD_FROM_HIGH"]) else 0.0
    sig.red_streak = int(last["RED_STREAK"]) if not pd.isna(last["RED_STREAK"]) else 0
    sig.bb_upper = float(last["BB_Upper"]) if not pd.isna(last["BB_Upper"]) else 0.0
    sig.bb_lower = float(last["BB_Lower"]) if not pd.isna(last["BB_Lower"]) else 0.0

    # BBW exit state
    sig.bbw_contracting = sig.bbw_slope < 0
    sig.above_sma_exit = sig.above_sma

    # Backward compat
    sig.wr_20 = sig.wr_30; sig.wr_20_cross_bars = 99

    # ── Entry classification ──────────────────────────────────────────────
    # Core: WR(30) < -30, below SMA(20), MFI > 30
    sig.core_met = (
        sig.wr_30 < WR_THRESH
        and not sig.above_sma
        and sig.sma_20 > 0
        and sig.mfi >= MFI_MIN
    )

    # Primary: core + drawdown + red streak + volume
    sig.primary_met = (
        sig.core_met
        and sig.dd_from_high <= DD_MODERATE
        and sig.red_streak >= RED_STREAK_MIN
        and sig.vol_ratio >= VOL_MIN_RATIO
    )

    # Secondary: core only
    sig.secondary_met = sig.core_met and not sig.primary_met

    if sig.primary_met:
        sig.tier = "PRIMARY"
        sig.entry_triggered = True
        sig.entry_reason = (
            f"PRIMARY: WR(30)={sig.wr_30:.0f} | {sig.pct_from_sma:+.1f}% vs SMA | "
            f"MFI={sig.mfi:.0f} | DD={sig.dd_from_high:.1f}% | "
            f"{sig.red_streak}d red | Vol={sig.vol_ratio:.1f}x"
        )
    elif sig.secondary_met:
        sig.tier = "SECONDARY"
        sig.entry_triggered = True
        sig.entry_reason = (
            f"SECONDARY: WR(30)={sig.wr_30:.0f} | {sig.pct_from_sma:+.1f}% vs SMA | "
            f"MFI={sig.mfi:.0f}"
        )
    else:
        sig.tier = "NONE"
        sig.entry_triggered = False

    # Backward compat
    sig.mode_b_entry = sig.entry_triggered
    sig.mode_a_entry = False

    # ── Base score ────────────────────────────────────────────────────────
    sig.base_score = _score(sig)

    # ── Overlays ──────────────────────────────────────────────────────────
    sig.options_overlay, sig.options_detail = _options_overlay(options_ctx, sig)
    sig.filing_overlay, sig.filing_detail, sig.is_trap = _filing_overlay(filing_variance, sig.base_score)

    # ── Final score ───────────────────────────────────────────────────────
    final = max(0, min(100, sig.base_score + sig.options_overlay + sig.filing_overlay))
    if sig.is_trap:
        final = min(final, 30)
    sig.dual_score = final
    sig.dual_label, sig.dual_sizing = _label_and_sizing(final, sig)

    return sig


# ── Scoring function ──────────────────────────────────────────────────────────

def _score(sig):
    """Score the mean-reversion setup quality (0-100)."""
    s = 30  # base

    # WR depth (deeper oversold = stronger bounce)
    if sig.wr_30 < -70:   s += 25
    elif sig.wr_30 < -50: s += 20
    elif sig.wr_30 < -30: s += 15
    elif sig.wr_30 < -20: s += 5

    # Distance below SMA (larger pullback = more room for reversion)
    if not sig.above_sma:
        if sig.pct_from_sma < -5.0:   s += 15
        elif sig.pct_from_sma < -3.0: s += 12
        elif sig.pct_from_sma < -1.5: s += 8
        elif sig.pct_from_sma < -0.5: s += 4
    else:
        s -= 10  # above SMA = not a pullback

    # MFI (money flow)
    if sig.mfi >= 50:     s += 8   # strong flow into dip
    elif sig.mfi >= 30:   s += 3   # acceptable
    else:                 s -= 5   # weak = no accumulation

    # MFI rising (smart money buying the dip)
    if sig.mfi_slope > 3: s += 4

    # Drawdown from high (deeper = more capitulation)
    if sig.dd_from_high <= -10:  s += 8
    elif sig.dd_from_high <= -5: s += 4

    # Red streak (selling exhaustion)
    if sig.red_streak >= 4:   s += 6
    elif sig.red_streak >= 2: s += 3

    # Volume (institutional presence during dip)
    if sig.vol_ratio >= 1.5: s += 4  # volume surge
    elif sig.vol_ratio < 0.5: s -= 3  # dry volume

    # BBW state (squeeze before entry = coiled spring)
    if sig.bbw_pctl < 30:   s += 3
    elif sig.bbw_pctl > 70:  s -= 2  # already expanded

    return max(0, min(100, s))


# ── Options overlay (mean-reversion mode only) ────────────────────────────────

def _options_overlay(ctx, sig):
    if ctx is None: return 0, "No options data"
    ov = 0; det = []
    try:
        pcr = getattr(getattr(ctx, "walls", None), "pcr_oi", None)
        gex = getattr(getattr(ctx, "gex", None), "regime", None)
        sp = getattr(getattr(ctx, "walls", None), "support_pct", None)

        pcr_bearish = pcr is not None and pcr <= 0.7
        pcr_bullish = pcr is not None and pcr >= 1.3
        gex_neg = gex == "Negative"

        if pcr_bearish:
            ov -= 5; det.append(f"PCR {pcr:.2f} bearish")
            if gex_neg: ov -= 3; det.append("GEX neg amplifies")
        elif pcr_bullish:
            ov += 3; det.append(f"PCR {pcr:.2f} put support")
            if gex_neg: ov += 2; det.append("GEX neg + puts → sharp bounce")
        else:
            if gex_neg: det.append("GEX neg (volatile)")

        if sp is not None and abs(sp) < 2.0:
            ov += 2; det.append(f"Near support {sp:+.1f}%")
    except Exception: pass
    return max(-10, min(10, ov)), " | ".join(det) if det else "Neutral"


# ── Filing overlay ────────────────────────────────────────────────────────────

def _filing_overlay(fv, base_score):
    if fv is None: return 0, "No filing", False
    try:
        d = getattr(fv, "badge_color", "NONE")
        c = getattr(fv, "conviction", 0)
        if d == "BEARISH" and base_score >= 55:
            return -10, f"TRAP: bearish (conv {c}) vs high base", True
        if d == "BEARISH":
            return -5, f"Bearish filing (conv {c})", False
        if d == "BULLISH":
            ov = 7 if c >= 7 else (4 if c >= 5 else 2)
            return ov, f"Bullish filing (conv {c})", False
    except Exception: pass
    return 0, "Neutral", False


# ── Label and sizing ──────────────────────────────────────────────────────────

def _label_and_sizing(score, sig):
    if sig.is_trap: return "AVOID", "SKIP"
    if sig.tier == "PRIMARY":
        if score >= 75: return "STRONG", "FULL"
        if score >= 60: return "GOOD", "FULL"
        if score >= 45: return "WATCH", "HALF"
        return "AVOID", "SKIP"
    elif sig.tier == "SECONDARY":
        if score >= 75: return "STRONG", "HALF"
        if score >= 60: return "GOOD", "HALF"
        if score >= 45: return "WATCH", "SKIP"
        return "AVOID", "SKIP"
    else:
        if score >= 75: return "STRONG", "HALF"
        if score >= 60: return "GOOD", "SKIP"
        if score >= 45: return "WATCH", "SKIP"
        return "AVOID", "SKIP"


# ── Exit check ────────────────────────────────────────────────────────────────

def check_exit(sig, entry_price, bars_held):
    """Check exit conditions. Returns exit reason string or ''."""
    c = sig.close
    # Profit target (3%)
    if entry_price > 0 and c >= entry_price * (1 + PROFIT_TARGET / 100):
        return "PROFIT_TARGET"
    # BBW contraction + above SMA (momentum exhausting after bounce)
    if bars_held >= 5 and sig.bbw_contracting and sig.above_sma:
        return "BBW_CONTRACT"
    # Max hold
    if bars_held >= MAX_HOLD:
        return "MAX_HOLD"
    return ""


# ── Summary (for logging/display) ────────────────────────────────────────────

def signal_summary(sig):
    return {
        "tier": sig.tier, "mode": "Mean Reversion",
        "entry_triggered": sig.entry_triggered, "entry_reason": sig.entry_reason,
        "base_score": sig.base_score,
        "wr_30": sig.wr_30, "mfi": sig.mfi, "adx_14": sig.adx_14,
        "rsi": sig.rsi, "dd_from_high": sig.dd_from_high,
        "red_streak": sig.red_streak, "vol_ratio": sig.vol_ratio,
        "bbw_slope": sig.bbw_slope, "bbw_pctl": sig.bbw_pctl,
        "pct_from_sma": sig.pct_from_sma, "above_sma": sig.above_sma,
        "options_overlay": sig.options_overlay, "filing_overlay": sig.filing_overlay,
        "is_trap": sig.is_trap, "final_score": sig.dual_score,
        "label": sig.dual_label, "sizing": sig.dual_sizing,
        "close": sig.close, "sma_20": sig.sma_20,
    }

# Backward compat: old names still importable
_score_mode_a = lambda sig: _score(sig)
_score_mode_b = _score
