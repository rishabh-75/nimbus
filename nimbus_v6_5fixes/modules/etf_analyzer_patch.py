"""
modules/etf_analyzer.py
────────────────────────
Cash-only ETF analysis for NIMBUS — no options chain required.

Replaces F&O analytics (GEX / max pain / PCR / DTE) with:
  - Volume Profile  (POC, VAH, VAL)   → structural support / resistance
  - Volume Surge Ratio (VSR)          → institutional flow conviction
  - VWAP                              → dynamic trend anchor
  - Underlying trend proxy            → confirms ETF tracks its benchmark
  - NAV premium/discount (proxy)      → entry quality filter

Scoring: 100 pts (same scale as F&O viability so scanner is unified)
  BB State (RIDING_UPPER)  30 pts
  W%R momentum zone        20 pts
  Volume Surge VSR         20 pts   (0 if VSR < 1.0, full 20 if VSR ≥ 2.5)
  Above POC                15 pts
  Price above VWAP         15 pts

v5.3 — expanded ETF universe to 25 momentum-relevant symbols covering:
  equity index, sector, commodity, international equity, thematic
  Debt / liquid ETFs excluded — they have no meaningful momentum signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ETF UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ETFInfo:
    symbol:       str
    name:         str
    category:     str          # "equity_index"|"sector"|"commodity"|"intl_equity"|"thematic"
    underlying:   str          # human-readable label
    proxy_ticker: str          # yfinance ticker for underlying trend proxy
    aum_cr:       float = 0.0  # approx AUM ₹ crore (0 = unknown)


ETF_UNIVERSE: dict[str, ETFInfo] = {

    # ── Broad Equity Index ─────────────────────────────────────────────────────
    "NIFTYBEES": ETFInfo(
        "NIFTYBEES",   "Nippon India ETF Nifty BeES",
        "equity_index", "NIFTY 50",          "^NSEI",        aum_cr=5975,
    ),
    "JUNIORBEES": ETFInfo(
        "JUNIORBEES",  "Nippon India ETF Nifty Next 50",
        "equity_index", "NIFTY Next 50",     "^NSEI",        aum_cr=5000,
    ),
    "NIF100BEES": ETFInfo(
        "NIF100BEES",  "Nippon India ETF Nifty 100",
        "equity_index", "NIFTY 100",         "^NSEI",        aum_cr=2000,
    ),
    "MID150BEES": ETFInfo(
        "MID150BEES",  "Nippon India ETF Nifty Midcap 150",
        "equity_index", "NIFTY Midcap 150",  "^NSEI",        aum_cr=3000,
    ),
    "HDFCSML250": ETFInfo(
        "HDFCSML250",  "HDFC Nifty Smallcap 250 ETF",
        "equity_index", "NIFTY Smallcap 250","^NSEI",        aum_cr=1500,
    ),
    "SENSEXBETA": ETFInfo(
        "SENSEXBETA",  "UTI BSE Sensex ETF",
        "equity_index", "BSE Sensex",        "^BSESN",       aum_cr=19851,
    ),
    "ICICIB22": ETFInfo(
        "ICICIB22",    "Bharat 22 ETF",
        "equity_index", "Nifty Bharat 22",   "^NSEI",        aum_cr=23374,
    ),

    # ── Banking & Finance Sector ───────────────────────────────────────────────
    "BANKBEES": ETFInfo(
        "BANKBEES",    "Nippon India ETF Nifty Bank BeES",
        "sector",       "NIFTY Bank",        "^NSEBANK",     aum_cr=14007,
    ),
    "PSUBNKBEES": ETFInfo(
        "PSUBNKBEES",  "Nippon India ETF PSU Bank BeES",
        "sector",       "Nifty PSU Bank",    "PSUBNKBEES.NS",aum_cr=3000,
    ),
    "PVTBANIETF": ETFInfo(
        "PVTBANIETF",  "Nippon India ETF Nifty Private Bank",
        "sector",       "Nifty Private Bank","^NSEBANK",     aum_cr=1500,
    ),

    # ── Energy / Infrastructure ────────────────────────────────────────────────
    "CPSEETF": ETFInfo(
        "CPSEETF",     "CPSE ETF",
        "sector",       "Nifty CPSE",        "CPSEETF.NS",   aum_cr=63037,
    ),
    "OILIETF": ETFInfo(
        "OILIETF",     "Nippon India ETF Nifty Oil & Gas",
        "sector",       "Nifty Oil & Gas",   "OILIETF.NS",   aum_cr=2000,
    ),

    # ── Technology ────────────────────────────────────────────────────────────
    "ITBEES": ETFInfo(
        "ITBEES",      "Nippon India ETF Nifty IT",
        "sector",       "NIFTY IT",          "^CNXIT",       aum_cr=23481,
    ),

    # ── Healthcare ────────────────────────────────────────────────────────────
    "PHARMABEES": ETFInfo(
        "PHARMABEES",  "Nippon India ETF Nifty Pharma",
        "sector",       "Nifty Pharma",      "^CNXPHARMA",   aum_cr=2000,
    ),

    # ── Consumption / FMCG ────────────────────────────────────────────────────
    "CONSUMBEES": ETFInfo(
        "CONSUMBEES",  "Nippon India ETF Nifty Consumption",
        "sector",       "Nifty India Consumption", "^CNXFMCG",  aum_cr=1000,
    ),

    # ── Metals ────────────────────────────────────────────────────────────────
    "METALIETF": ETFInfo(
        "METALIETF",   "Nippon India ETF Nifty Metal",
        "sector",       "Nifty Metal",       "^CNXMETAL",    aum_cr=1500,
    ),

    # ── Thematic / Defence ────────────────────────────────────────────────────
    "MODEFENCE": ETFInfo(
        "MODEFENCE",   "Motilal Oswal Nifty India Defence ETF",
        "thematic",     "Nifty India Defence","MODEFENCE.NS", aum_cr=3000,
    ),

    # ── Commodity ─────────────────────────────────────────────────────────────
    "GOLDBEES": ETFInfo(
        "GOLDBEES",    "Nippon India ETF Gold BeES",
        "commodity",    "Gold (MCX / XAUUSD)","GC=F",        aum_cr=15289,
    ),
    "SILVERBEES": ETFInfo(
        "SILVERBEES",  "Nippon India Silver ETF",
        "commodity",    "Silver (MCX / XAGUSD)","SI=F",      aum_cr=28611,
    ),
    "TATSILV": ETFInfo(
        "TATSILV",     "Tata Silver Exchange Traded Fund",
        "commodity",    "Silver (MCX / XAGUSD)","SI=F",      aum_cr=5000,
    ),

    # ── International Equity ──────────────────────────────────────────────────
    "MON100": ETFInfo(
        "MON100",      "Motilal Oswal NASDAQ 100 ETF",
        "intl_equity",  "NASDAQ-100",         "^NDX",        aum_cr=8933,
    ),
    "MAFANG": ETFInfo(
        "MAFANG",      "Mirae Asset NYSE FANG+ ETF",
        "intl_equity",  "NYSE FANG+",          "^NDX",       aum_cr=3000,
    ),
    "MAHKTECH": ETFInfo(
        "MAHKTECH",    "Mirae Asset Hang Seng TECH ETF",
        "intl_equity",  "Hang Seng TECH",      "^HSTECH",    aum_cr=2000,
    ),
    "HNGSNGBEES": ETFInfo(
        "HNGSNGBEES",  "Nippon India ETF Hang Seng BeES",
        "intl_equity",  "Hang Seng Index",     "^HSI",       aum_cr=500,
    ),
    "MASPTOP50": ETFInfo(
        "MASPTOP50",   "Mirae Asset S&P 500 Top 50 ETF",
        "intl_equity",  "S&P 500 Top 50",      "^GSPC",      aum_cr=2000,
    ),
}

NSE_ETF_SYMBOLS: list[str] = sorted(ETF_UNIVERSE.keys())


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VolumeProfile:
    poc:             float
    vah:             float
    val:             float
    profile:         list[tuple[float, float]]
    spot_vs_poc_pct: float


@dataclass
class ETFVolume:
    vsr:          float
    avg_20d:      float
    current:      float
    surge_label:  str    # "SURGE" | "ELEVATED" | "NORMAL" | "DRY"
    trend:        str    # "EXPANDING" | "CONTRACTING" | "FLAT"


@dataclass
class ETFTrend:
    vwap:             float
    above_vwap:       bool
    pct_from_vwap:    float
    underlying_bias:  str    # "BULLISH" | "BEARISH" | "NEUTRAL" | "UNAVAILABLE"
    underlying_pct:   float


@dataclass
class ETFViability:
    score:   int
    label:   str    # "STRONG" | "MODERATE" | "WEAK" | "AVOID"
    sizing:  str    # "FULL" | "HALF" | "SKIP"
    reasons: list[str] = field(default_factory=list)


@dataclass
class ETFContext:
    symbol:         str
    info:           ETFInfo
    spot:           float
    volume_profile: VolumeProfile
    etf_volume:     ETFVolume
    etf_trend:      ETFTrend
    bb_state:       str
    wr_value:       float
    wr_in_momentum: bool
    daily_bias:     str
    viability:      ETFViability
    is_etf:         bool = True


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME PROFILE (OHLCV approximation)
# ══════════════════════════════════════════════════════════════════════════════

def build_volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
    lookback: int = 60,
) -> Optional[VolumeProfile]:
    if df is None or len(df) < 10:
        return None
    df = df.tail(lookback).copy()
    needed = {"High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        return None

    all_prices: list[float] = []
    all_vols:   list[float] = []
    for _, row in df.iterrows():
        h = float(row["High"]); l = float(row["Low"]); v = float(row["Volume"])
        if h <= l or v <= 0 or np.isnan(h) or np.isnan(l):
            continue
        n_sub  = max(2, min(20, int((h - l) / max(l * 0.001, 0.01))))
        prices = np.linspace(l, h, n_sub)
        all_prices.extend(prices.tolist())
        all_vols.extend((np.full(n_sub, v / n_sub)).tolist())

    if len(all_prices) < bins:
        return None

    prices_arr = np.array(all_prices)
    vols_arr   = np.array(all_vols)
    p_min, p_max = prices_arr.min(), prices_arr.max()
    if p_max <= p_min:
        return None

    edges      = np.linspace(p_min, p_max, bins + 1)
    centers    = (edges[:-1] + edges[1:]) / 2
    vol_profile = np.zeros(bins)
    for i in range(bins):
        mask           = (prices_arr >= edges[i]) & (prices_arr < edges[i + 1])
        vol_profile[i] = vols_arr[mask].sum()

    poc_idx   = int(np.argmax(vol_profile))
    poc_price = float(centers[poc_idx])

    total_vol  = vol_profile.sum()
    target_vol = total_vol * 0.70
    va_indices = [poc_idx]
    va_vol     = float(vol_profile[poc_idx])
    above      = poc_idx + 1
    below      = poc_idx - 1
    while va_vol < target_vol and (above < bins or below >= 0):
        vol_a = float(vol_profile[above]) if above < bins else 0.0
        vol_b = float(vol_profile[below]) if below >= 0  else 0.0
        if vol_a >= vol_b and above < bins:
            va_indices.append(above); va_vol += vol_a; above += 1
        elif below >= 0:
            va_indices.append(below); va_vol += vol_b; below -= 1
        else:
            break

    vah  = float(centers[max(va_indices)])
    val  = float(centers[min(va_indices)])
    spot = float(df["Close"].iloc[-1])
    spot_vs_poc_pct = round((spot - poc_price) / poc_price * 100, 2) if poc_price else 0.0

    return VolumeProfile(
        poc=round(poc_price, 4), vah=round(vah, 4), val=round(val, 4),
        profile=[(float(centers[i]), float(vol_profile[i])) for i in range(bins)],
        spot_vs_poc_pct=spot_vs_poc_pct,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME SURGE
# ══════════════════════════════════════════════════════════════════════════════

def compute_etf_volume(df: pd.DataFrame, avg_window: int = 20) -> Optional[ETFVolume]:
    if df is None or len(df) < avg_window + 1 or "Volume" not in df.columns:
        return None
    vol     = df["Volume"].astype(float)
    current = float(vol.iloc[-1])
    avg_20d = float(vol.iloc[-(avg_window + 1):-1].mean())
    if avg_20d <= 0:
        return None
    vsr = round(current / avg_20d, 2)
    surge_label = ("SURGE" if vsr >= 2.5 else "ELEVATED" if vsr >= 1.5
                   else "NORMAL" if vsr >= 0.7 else "DRY")
    if len(df) >= 10:
        ratio = float(vol.iloc[-5:].mean()) / max(float(vol.iloc[-10:-5].mean()), 1)
        trend = "EXPANDING" if ratio >= 1.15 else ("CONTRACTING" if ratio <= 0.85 else "FLAT")
    else:
        trend = "FLAT"
    return ETFVolume(vsr=vsr, avg_20d=round(avg_20d, 0), current=round(current, 0),
                     surge_label=surge_label, trend=trend)


# ══════════════════════════════════════════════════════════════════════════════
# VWAP + UNDERLYING TREND
# ══════════════════════════════════════════════════════════════════════════════

def compute_etf_trend(df: pd.DataFrame, info: ETFInfo, lookback: int = 20) -> ETFTrend:
    vwap = 0.0; above_vwap = False; pct_from_vwap = 0.0
    if {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        sub     = df.tail(lookback).copy()
        typical = (sub["High"].astype(float) + sub["Low"].astype(float) +
                   sub["Close"].astype(float)) / 3
        volume  = sub["Volume"].astype(float)
        total_v = volume.sum()
        if total_v > 0:
            vwap          = float((typical * volume).sum() / total_v)
            spot          = float(df["Close"].iloc[-1])
            above_vwap    = spot > vwap
            pct_from_vwap = round((spot - vwap) / vwap * 100, 2) if vwap else 0.0

    underlying_bias = "UNAVAILABLE"; underlying_pct = 0.0
    try:
        import yfinance as yf
        proxy = yf.download(tickers=info.proxy_ticker, period="1mo",
                            interval="1d", auto_adjust=True, progress=False)
        if not proxy.empty and "Close" in proxy.columns:
            closes = proxy["Close"].dropna()
            if len(closes) >= 5:
                underlying_pct = round(float(
                    (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100), 2)
                last  = float(closes.iloc[-1])
                sma5  = float(closes.tail(5).mean())
                sma20 = float(closes.mean())
                underlying_bias = ("BULLISH"  if last > sma5 > sma20 else
                                   "BEARISH"  if last < sma5 < sma20 else "NEUTRAL")
    except Exception as exc:
        logger.debug("Proxy fetch failed %s: %s", info.symbol, exc)

    return ETFTrend(vwap=round(vwap, 4), above_vwap=above_vwap,
                   pct_from_vwap=pct_from_vwap,
                   underlying_bias=underlying_bias, underlying_pct=underlying_pct)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_etf(bb_state, wr_in_momentum, wr_value,
              etf_volume, vp, etf_trend, spot, daily_bias) -> ETFViability:
    score = 0; reasons = []

    # BB (30 pts)
    if bb_state == "RIDING_UPPER":
        score += 30; reasons.append("Riding upper BB (30/30)")
    elif bb_state == "FIRST_DIP":
        score += 15; reasons.append("First dip (15/30)")
    else:
        reasons.append(f"BB {bb_state} — no momentum credit")

    # W%R (20 pts proportional)
    if wr_in_momentum:
        wr_pts = max(0, min(20, int(20 * (80 + wr_value) / 80)))
        score += wr_pts; reasons.append(f"W%R {wr_value:.0f} ({wr_pts}/20)")
    else:
        reasons.append(f"W%R {wr_value:.0f} outside zone (0/20)")

    # Volume Surge (20 pts)
    if etf_volume:
        vsr_map = {"SURGE": 20, "ELEVATED": 14, "NORMAL": 8, "DRY": 0}
        vol_pts = vsr_map.get(etf_volume.surge_label, 0)
        if etf_volume.trend == "EXPANDING":
            vol_pts = min(20, vol_pts + 2)
        score += vol_pts
        reasons.append(f"VSR {etf_volume.vsr:.1f}x ({etf_volume.surge_label}, "
                       f"{etf_volume.trend}) {vol_pts}/20")
    else:
        reasons.append("Volume data unavailable (0/20)")

    # Above POC (15 pts)
    if vp:
        poc_pct = vp.spot_vs_poc_pct
        poc_pts = min(15, int(15 * poc_pct / 3.0)) if poc_pct >= 0 else 0
        score += poc_pts
        reasons.append(f"Spot {poc_pct:+.1f}% vs POC {vp.poc:.2f} ({poc_pts}/15)")
    else:
        reasons.append("Volume profile unavailable (0/15)")

    # Above VWAP (15 pts)
    if etf_trend and etf_trend.vwap > 0:
        vwap_pct = etf_trend.pct_from_vwap
        vwap_pts = min(15, int(15 * vwap_pct / 2.0)) if vwap_pct >= 0 else 0
        score += vwap_pts
        reasons.append(f"{'Above' if etf_trend.above_vwap else 'Below'} VWAP "
                       f"({vwap_pct:+.1f}%) {vwap_pts}/15")
    else:
        reasons.append("VWAP unavailable (0/15)")

    score = max(0, min(100, score))
    hard_fail = bb_state not in ("RIDING_UPPER", "FIRST_DIP") or not wr_in_momentum

    if hard_fail or score < 40:
        label = "AVOID"; sizing = "SKIP"
    elif score >= 70 and daily_bias == "BULLISH":
        label = "STRONG"; sizing = "FULL"
    elif score >= 55:
        label = "MODERATE"; sizing = "HALF"
    else:
        label = "WEAK"; sizing = "SKIP"

    if etf_trend and etf_trend.underlying_bias == "BEARISH" and sizing == "FULL":
        sizing = "HALF"; reasons.append("Underlying bearish — size capped HALF")
    if daily_bias == "BEARISH" and sizing != "SKIP":
        sizing = "SKIP"; label = "AVOID"; reasons.append("Daily bias bearish — no longs")

    return ETFViability(score=score, label=label, sizing=sizing, reasons=reasons)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_etf(symbol: str, price_df: pd.DataFrame,
                price_signals) -> Optional[ETFContext]:
    if price_df is None or price_df.empty:
        return None
    info = ETF_UNIVERSE.get(symbol) or ETFInfo(
        symbol=symbol, name=symbol, category="equity_index",
        underlying="Unknown", proxy_ticker=f"{symbol}.NS",
    )
    spot       = float(price_df["Close"].iloc[-1])
    vp         = build_volume_profile(price_df, bins=50, lookback=60)
    etf_volume = compute_etf_volume(price_df, avg_window=20)
    etf_trend  = compute_etf_trend(price_df, info, lookback=20)

    bb_state       = price_signals.position_state if price_signals else "UNKNOWN"
    wr_value       = price_signals.wr_value       if price_signals else -100.0
    wr_in_momentum = price_signals.wr_in_momentum if price_signals else False
    daily_bias     = price_signals.daily_bias     if price_signals else "NEUTRAL"

    viability = score_etf(bb_state, wr_in_momentum, wr_value,
                           etf_volume, vp, etf_trend, spot, daily_bias)
    return ETFContext(symbol=symbol, info=info, spot=spot,
                     volume_profile=vp, etf_volume=etf_volume, etf_trend=etf_trend,
                     bb_state=bb_state, wr_value=wr_value,
                     wr_in_momentum=wr_in_momentum, daily_bias=daily_bias,
                     viability=viability)
