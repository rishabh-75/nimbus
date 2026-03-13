"""
modules/etf_analyzer.py  v5.4
──────────────────────────────
Cash-only ETF analysis for NIMBUS.

Extended scoring — 100 pts across 8 components:
  BB State           25 pts   (RIDING_UPPER=25, FIRST_DIP=12)
  W%R momentum       15 pts   (proportional 0→15 from W%R -80→0)
  Volume Surge       15 pts   (SURGE=15, ELEVATED=10, NORMAL=6, DRY=0; +2 EXPANDING)
  Above POC          12 pts   (proportional 0→12 over 0–3% above POC)
  Above VWAP         10 pts   (proportional 0→10 over 0–2% above VWAP)
  NAV Premium/Disc   15 pts   (discount ≥1% =15; at par =9; premium ≥1% =0)
  Underlying trend    5 pts   (BULLISH=5, NEUTRAL=2, BEARISH=0, UNAVAILABLE=2)
  AUM Liquidity       3 pts   (≥5000cr=3, ≥1000cr=2, else=1)
  Total             100 pts

NAV source: NSE iNAV API — https://www.nseindia.com/api/etf?index=SYMBOL
  Returns indicative intra-day NAV (iNavValue field).
  Premium = (spot - iNAV) / iNAV × 100  (+ve = premium, -ve = discount)
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
    symbol: str
    name: str
    category: str
    underlying: str
    proxy_ticker: str
    aum_cr: float = 0.0


ETF_UNIVERSE: dict[str, ETFInfo] = {
    "NIFTYBEES": ETFInfo(
        "NIFTYBEES",
        "Nippon India ETF Nifty BeES",
        "equity_index",
        "NIFTY 50",
        "^NSEI",
        5975,
    ),
    "JUNIORBEES": ETFInfo(
        "JUNIORBEES",
        "Nippon India ETF Nifty Next 50",
        "equity_index",
        "NIFTY Next 50",
        "^NSEI",
        5000,
    ),
    "NIF100BEES": ETFInfo(
        "NIF100BEES",
        "Nippon India ETF Nifty 100",
        "equity_index",
        "NIFTY 100",
        "^NSEI",
        2000,
    ),
    "MID150BEES": ETFInfo(
        "MID150BEES",
        "Nippon India ETF Nifty Midcap 150",
        "equity_index",
        "NIFTY Midcap 150",
        "^NSEI",
        3000,
    ),
    "HDFCSML250": ETFInfo(
        "HDFCSML250",
        "HDFC Nifty Smallcap 250 ETF",
        "equity_index",
        "NIFTY Smallcap 250",
        "^NSEI",
        1500,
    ),
    "SENSEXBETA": ETFInfo(
        "SENSEXBETA",
        "UTI BSE Sensex ETF",
        "equity_index",
        "BSE Sensex",
        "^BSESN",
        19851,
    ),
    "ICICIB22": ETFInfo(
        "ICICIB22", "Bharat 22 ETF", "equity_index", "Nifty Bharat 22", "^NSEI", 23374
    ),
    "BANKBEES": ETFInfo(
        "BANKBEES",
        "Nippon India ETF Nifty Bank BeES",
        "sector",
        "NIFTY Bank",
        "^NSEBANK",
        14007,
    ),
    "PSUBNKBEES": ETFInfo(
        "PSUBNKBEES",
        "Nippon India ETF PSU Bank BeES",
        "sector",
        "Nifty PSU Bank",
        "PSUBNKBEES.NS",
        3000,
    ),
    "PVTBANIETF": ETFInfo(
        "PVTBANIETF",
        "Nippon India ETF Nifty Private Bank",
        "sector",
        "Nifty Private Bank",
        "^NSEBANK",
        1500,
    ),
    "CPSEETF": ETFInfo(
        "CPSEETF", "CPSE ETF", "sector", "Nifty CPSE", "CPSEETF.NS", 63037
    ),
    "OILIETF": ETFInfo(
        "OILIETF",
        "Nippon India ETF Nifty Oil & Gas",
        "sector",
        "Nifty Oil & Gas",
        "OILIETF.NS",
        2000,
    ),
    "ITBEES": ETFInfo(
        "ITBEES", "Nippon India ETF Nifty IT", "sector", "NIFTY IT", "^CNXIT", 23481
    ),
    "PHARMABEES": ETFInfo(
        "PHARMABEES",
        "Nippon India ETF Nifty Pharma",
        "sector",
        "Nifty Pharma",
        "^CNXPHARMA",
        2000,
    ),
    "CONSUMBEES": ETFInfo(
        "CONSUMBEES",
        "Nippon India ETF Nifty Consumption",
        "sector",
        "Nifty Consumption",
        "^CNXFMCG",
        1000,
    ),
    "METALIETF": ETFInfo(
        "METALIETF",
        "Nippon India ETF Nifty Metal",
        "sector",
        "Nifty Metal",
        "^CNXMETAL",
        1500,
    ),
    "MODEFENCE": ETFInfo(
        "MODEFENCE",
        "Motilal Oswal Nifty India Defence ETF",
        "thematic",
        "Nifty India Defence",
        "MODEFENCE.NS",
        3000,
    ),
    "GOLDBEES": ETFInfo(
        "GOLDBEES",
        "Nippon India ETF Gold BeES",
        "commodity",
        "Gold (MCX/XAUUSD)",
        "GC=F",
        15289,
    ),
    "SILVERBEES": ETFInfo(
        "SILVERBEES",
        "Nippon India Silver ETF",
        "commodity",
        "Silver (MCX/XAGUSD)",
        "SI=F",
        28611,
    ),
    "TATSILV": ETFInfo(
        "TATSILV", "Tata Silver ETF", "commodity", "Silver (MCX/XAGUSD)", "SI=F", 5000
    ),
    "MON100": ETFInfo(
        "MON100",
        "Motilal Oswal NASDAQ 100 ETF",
        "intl_equity",
        "NASDAQ-100",
        "^NDX",
        8933,
    ),
    "MAFANG": ETFInfo(
        "MAFANG",
        "Mirae Asset NYSE FANG+ ETF",
        "intl_equity",
        "NYSE FANG+",
        "^NDX",
        3000,
    ),
    "MAHKTECH": ETFInfo(
        "MAHKTECH",
        "Mirae Asset Hang Seng TECH ETF",
        "intl_equity",
        "Hang Seng TECH",
        "^HSTECH",
        2000,
    ),
    "HNGSNGBEES": ETFInfo(
        "HNGSNGBEES",
        "Nippon India ETF Hang Seng BeES",
        "intl_equity",
        "Hang Seng Index",
        "^HSI",
        500,
    ),
    "MASPTOP50": ETFInfo(
        "MASPTOP50",
        "Mirae Asset S&P 500 Top 50 ETF",
        "intl_equity",
        "S&P 500 Top 50",
        "^GSPC",
        2000,
    ),
}

NSE_ETF_SYMBOLS: list[str] = sorted(ETF_UNIVERSE.keys())


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class VolumeProfile:
    poc: float
    vah: float
    val: float
    profile: list[tuple[float, float]]
    spot_vs_poc_pct: float


@dataclass
class ETFVolume:
    vsr: float
    avg_20d: float
    current: float
    surge_label: str
    trend: str


@dataclass
class ETFTrend:
    vwap: float
    above_vwap: bool
    pct_from_vwap: float
    underlying_bias: str
    underlying_pct: float


@dataclass
class ETFNav:
    inav: Optional[float]  # indicative NAV from NSE
    spot: float
    premium_pct: float  # (spot - iNAV) / iNAV × 100; +ve = premium
    available: bool
    label: str  # "DISCOUNT" | "AT PAR" | "PREMIUM" | "UNAVAILABLE"


@dataclass
class ETFChecklistItem:
    status: str  # "PASS" | "FAIL" | "WARN" | "INFO"
    item: str
    detail: str
    implication: str = ""


@dataclass
class ETFViability:
    score: int
    label: str
    sizing: str
    reasons: list[str] = field(default_factory=list)
    checklist: list[ETFChecklistItem] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)  # component → pts


@dataclass
class ETFContext:
    symbol: str
    info: ETFInfo
    spot: float
    volume_profile: Optional[VolumeProfile]
    etf_volume: Optional[ETFVolume]
    etf_trend: Optional[ETFTrend]
    etf_nav: Optional[ETFNav]
    bb_state: str
    wr_value: float
    wr_in_momentum: bool
    daily_bias: str
    viability: ETFViability
    is_etf: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# NAV FROM NSE
# ══════════════════════════════════════════════════════════════════════════════


def fetch_nav_from_nse(symbol: str) -> Optional[float]:
    """
    Fetch indicative NAV (iNavValue) from NSE ETF API.
    Endpoint: https://www.nseindia.com/api/etf?index=SYMBOL
    Returns iNAV as float, or None on failure.
    """
    try:
        import requests

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", timeout=6, headers=headers)
        resp = session.get(
            f"https://www.nseindia.com/api/etf?index={symbol}",
            timeout=6,
            headers=headers,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        # NSE returns: {"iNavValue": "162.45", "lastPrice": "162.31", ...}
        raw = data.get("iNavValue") or data.get("nav") or data.get("iNav")
        if raw is not None:
            return float(str(raw).replace(",", "").strip())
    except Exception as exc:
        logger.debug("NAV fetch failed %s: %s", symbol, exc)
    return None


def compute_nav_signal(inav: Optional[float], spot: float) -> ETFNav:
    if inav is None or inav <= 0:
        return ETFNav(
            inav=None, spot=spot, premium_pct=0.0, available=False, label="UNAVAILABLE"
        )
    premium_pct = round((spot - inav) / inav * 100, 3)
    if premium_pct <= -1.0:
        label = "DISCOUNT"
    elif premium_pct <= -0.25:
        label = "MILD DISCOUNT"
    elif premium_pct <= 0.25:
        label = "AT PAR"
    elif premium_pct <= 1.0:
        label = "MILD PREMIUM"
    else:
        label = "PREMIUM"
    return ETFNav(
        inav=inav, spot=spot, premium_pct=premium_pct, available=True, label=label
    )


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME PROFILE
# ══════════════════════════════════════════════════════════════════════════════


def build_volume_profile(
    df: pd.DataFrame, bins: int = 50, lookback: int = 60
) -> Optional[VolumeProfile]:
    if df is None or len(df) < 10:
        return None
    df = df.tail(lookback).copy()
    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return None
    all_prices, all_vols = [], []
    for _, row in df.iterrows():
        h, l, v = float(row["High"]), float(row["Low"]), float(row["Volume"])
        if h <= l or v <= 0 or np.isnan(h) or np.isnan(l):
            continue
        n = max(2, min(20, int((h - l) / max(l * 0.001, 0.01))))
        pts = np.linspace(l, h, n)
        all_prices.extend(pts.tolist())
        all_vols.extend(np.full(n, v / n).tolist())
    if len(all_prices) < bins:
        return None
    pa, va = np.array(all_prices), np.array(all_vols)
    p_min, p_max = pa.min(), pa.max()
    if p_max <= p_min:
        return None
    edges = np.linspace(p_min, p_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vp = np.array(
        [va[(pa >= edges[i]) & (pa < edges[i + 1])].sum() for i in range(bins)]
    )
    poc_idx = int(np.argmax(vp))
    poc = float(centers[poc_idx])
    va_idx, va_vol, ab, bl = [poc_idx], float(vp[poc_idx]), poc_idx + 1, poc_idx - 1
    target = vp.sum() * 0.70
    while va_vol < target and (ab < bins or bl >= 0):
        a, b = (float(vp[ab]) if ab < bins else 0.0), (
            float(vp[bl]) if bl >= 0 else 0.0
        )
        if a >= b and ab < bins:
            va_idx.append(ab)
            va_vol += a
            ab += 1
        elif bl >= 0:
            va_idx.append(bl)
            va_vol += b
            bl -= 1
        else:
            break
    spot = float(df["Close"].iloc[-1])
    return VolumeProfile(
        poc=round(poc, 4),
        vah=round(float(centers[max(va_idx)]), 4),
        val=round(float(centers[min(va_idx)]), 4),
        profile=[(float(centers[i]), float(vp[i])) for i in range(bins)],
        spot_vs_poc_pct=round((spot - poc) / poc * 100, 2) if poc else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME SURGE
# ══════════════════════════════════════════════════════════════════════════════


def compute_etf_volume(df: pd.DataFrame, avg_window: int = 20) -> Optional[ETFVolume]:
    if df is None or len(df) < avg_window + 1 or "Volume" not in df.columns:
        return None
    vol = df["Volume"].astype(float)
    current = float(vol.iloc[-1])
    avg = float(vol.iloc[-(avg_window + 1) : -1].mean())
    if avg <= 0:
        return None
    vsr = round(current / avg, 2)
    surge = (
        "SURGE"
        if vsr >= 2.5
        else "ELEVATED" if vsr >= 1.5 else "NORMAL" if vsr >= 0.7 else "DRY"
    )
    if len(df) >= 10:
        r = float(vol.iloc[-5:].mean()) / max(float(vol.iloc[-10:-5].mean()), 1)
        trend = "EXPANDING" if r >= 1.15 else ("CONTRACTING" if r <= 0.85 else "FLAT")
    else:
        trend = "FLAT"
    return ETFVolume(
        vsr=vsr,
        avg_20d=round(avg),
        current=round(current),
        surge_label=surge,
        trend=trend,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VWAP + UNDERLYING
# ══════════════════════════════════════════════════════════════════════════════


def compute_etf_trend(df: pd.DataFrame, info: ETFInfo, lookback: int = 20) -> ETFTrend:
    vwap = 0.0
    above_vwap = False
    pct = 0.0
    if {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        sub = df.tail(lookback).copy()
        tp = (
            sub["High"].astype(float)
            + sub["Low"].astype(float)
            + sub["Close"].astype(float)
        ) / 3
        v = sub["Volume"].astype(float)
        tv = v.sum()
        if tv > 0:
            vwap = float((tp * v).sum() / tv)
            spot = float(df["Close"].iloc[-1])
            above_vwap = spot > vwap
            pct = round((spot - vwap) / vwap * 100, 2) if vwap else 0.0
    ub = "UNAVAILABLE"
    up = 0.0
    try:
        import yfinance as yf

        proxy = yf.download(
            info.proxy_ticker,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if not proxy.empty and "Close" in proxy.columns:
            cls = proxy["Close"].dropna()
            if len(cls) >= 5:
                up = round(float((cls.iloc[-1] - cls.iloc[0]) / cls.iloc[0] * 100), 2)
                last, s5, s20 = (
                    float(cls.iloc[-1]),
                    float(cls.tail(5).mean()),
                    float(cls.mean()),
                )
                ub = (
                    "BULLISH"
                    if last > s5 > s20
                    else "BEARISH" if last < s5 < s20 else "NEUTRAL"
                )
    except Exception as e:
        logger.debug("Proxy failed %s: %s", info.symbol, e)
    return ETFTrend(
        vwap=round(vwap, 4),
        above_vwap=above_vwap,
        pct_from_vwap=pct,
        underlying_bias=ub,
        underlying_pct=up,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════


def score_etf(
    bb_state,
    wr_in_momentum,
    wr_value,
    etf_volume,
    vp,
    etf_trend,
    etf_nav,
    info,
    spot,
    daily_bias,
) -> ETFViability:
    score = 0
    reasons = []
    breakdown = {}
    checklist = []

    def ck(status, item, detail, impl=""):
        checklist.append(
            ETFChecklistItem(status=status, item=item, detail=detail, implication=impl)
        )

    # 1. BB State (25 pts) ──────────────────────────────────────────────────
    if bb_state == "RIDING_UPPER":
        pts = 25
        ck(
            "PASS",
            "Riding upper BB",
            "Price above BB(20,1σ)",
            "Primary momentum gate cleared",
        )
    elif bb_state == "FIRST_DIP":
        pts = 12
        ck(
            "WARN",
            "First dip off upper",
            "BB %B pulling back slightly",
            "Monitor — manage position, no new full entry",
        )
    else:
        pts = 0
        ck(
            "FAIL",
            f"BB state: {bb_state}",
            "Price not in upper band",
            "Hard fail — no momentum basis for long",
        )
    score += pts
    breakdown["BB State"] = pts
    reasons.append(f"BB {bb_state} ({pts}/25)")

    # 2. W%R (15 pts proportional) ─────────────────────────────────────────
    if wr_in_momentum:
        pts = max(0, min(15, int(15 * (80 + wr_value) / 80)))
        ck(
            "PASS",
            f"W%R {wr_value:.0f} in zone",
            f"W%R(50) above -20 threshold",
            f"Momentum confirmed — {pts}/15 pts",
        )
    else:
        pts = 0
        ck(
            "FAIL",
            f"W%R {wr_value:.0f} not in zone",
            "W%R(50) below -20 — momentum stalled",
            "Hard fail — wait for W%R recovery",
        )
    score += pts
    breakdown["W%R"] = pts
    reasons.append(f"W%R {wr_value:.0f} ({pts}/15)")

    # 3. Volume Surge (15 pts) ──────────────────────────────────────────────
    if etf_volume:
        base = {"SURGE": 15, "ELEVATED": 10, "NORMAL": 6, "DRY": 0}.get(
            etf_volume.surge_label, 0
        )
        pts = min(15, base + (2 if etf_volume.trend == "EXPANDING" else 0))
        ck(
            "PASS" if pts >= 6 else ("WARN" if pts > 0 else "FAIL"),
            f"VSR {etf_volume.vsr:.1f}x — {etf_volume.surge_label}",
            f"Volume {etf_volume.trend.lower()} vs 20d avg",
            "Flow conviction signal" if pts >= 10 else "Below conviction threshold",
        )
    else:
        pts = 0
        ck("INFO", "Volume data unavailable", "Cannot compute VSR", "")
    score += pts
    breakdown["Volume Surge"] = pts
    reasons.append(
        f"VSR {etf_volume.vsr:.1f}x {etf_volume.surge_label if etf_volume else '?'} ({pts}/15)"
    )

    # 4. Above POC (12 pts proportional) ───────────────────────────────────
    if vp:
        poc_pct = vp.spot_vs_poc_pct
        pts = min(12, int(12 * poc_pct / 3.0)) if poc_pct >= 0 else 0
        ck(
            "PASS" if pts >= 6 else ("WARN" if pts > 0 else "FAIL"),
            f"Spot {poc_pct:+.1f}% vs POC {vp.poc:.2f}",
            f"VAH {vp.vah:.2f} · VAL {vp.val:.2f}",
            (
                "Structural support confirmed"
                if pts >= 8
                else "Near/below high-volume node"
            ),
        )
    else:
        pts = 0
        ck("INFO", "Volume profile unavailable", "Insufficient price history", "")
    score += pts
    breakdown["Above POC"] = pts
    reasons.append(
        f"POC {vp.spot_vs_poc_pct:+.1f}% ({pts}/12)" if vp else f"POC N/A (0/12)"
    )

    # 5. Above VWAP (10 pts proportional) ──────────────────────────────────
    if etf_trend and etf_trend.vwap > 0:
        vwap_pct = etf_trend.pct_from_vwap
        pts = min(10, int(10 * vwap_pct / 2.0)) if vwap_pct >= 0 else 0
        ck(
            "PASS" if pts >= 5 else ("WARN" if pts > 0 else "FAIL"),
            f"{'Above' if etf_trend.above_vwap else 'Below'} VWAP ({vwap_pct:+.1f}%)",
            f"VWAP: {etf_trend.vwap:.2f}",
            (
                "Institutional cost basis supportive"
                if pts >= 5
                else "Below avg cost — caution"
            ),
        )
    else:
        pts = 0
        ck("INFO", "VWAP unavailable", "", "")
    score += pts
    breakdown["Above VWAP"] = pts
    reasons.append(
        f"VWAP {etf_trend.pct_from_vwap:+.1f}% ({pts}/10)"
        if etf_trend and etf_trend.vwap > 0
        else "VWAP N/A (0/10)"
    )

    # 6. NAV Premium/Discount (15 pts) ─────────────────────────────────────
    if etf_nav and etf_nav.available:
        p = etf_nav.premium_pct
        if p <= -1.0:
            pts = 15
        elif p <= -0.25:
            pts = 12
        elif p <= 0.25:
            pts = 9
        elif p <= 1.0:
            pts = 5
        else:
            pts = 0
        ck(
            "PASS" if pts >= 9 else ("WARN" if pts > 0 else "FAIL"),
            f"NAV {etf_nav.label} ({p:+.2f}%)",
            f"iNAV: {etf_nav.inav:.2f} vs Spot: {etf_nav.spot:.2f}",
            (
                "Discount = arb support, price likely to converge up"
                if p < 0
                else "Premium = frothy, arb pressure will compress price"
            ),
        )
    else:
        pts = 7  # neutral when unavailable
        ck("INFO", "NAV unavailable", "NSE iNAV not fetched", "Using neutral 7/15")
    score += pts
    breakdown["NAV Signal"] = pts
    nav_label = etf_nav.label if etf_nav and etf_nav.available else "N/A"
    reasons.append(f"NAV {nav_label} ({pts}/15)")

    # 7. Underlying trend (5 pts) ──────────────────────────────────────────
    if etf_trend:
        bias = etf_trend.underlying_bias
        pts = {"BULLISH": 5, "NEUTRAL": 2, "BEARISH": 0, "UNAVAILABLE": 2}.get(bias, 2)
        ck(
            "PASS" if pts == 5 else ("WARN" if pts == 2 else "FAIL"),
            f"Underlying: {bias}",
            f"{info.underlying} 20d return: {etf_trend.underlying_pct:+.1f}%",
            (
                "ETF has macro tailwind"
                if pts == 5
                else (
                    "Cross-check underlying chart"
                    if pts == 2
                    else "ETF fighting macro headwind"
                )
            ),
        )
    else:
        pts = 2
        ck("INFO", "Underlying trend unavailable", "", "")
    score += pts
    breakdown["Underlying"] = pts
    reasons.append(
        f"Underlying {etf_trend.underlying_bias if etf_trend else '?'} ({pts}/5)"
    )

    # 8. AUM Liquidity (3 pts) ─────────────────────────────────────────────
    aum = info.aum_cr
    pts = 3 if aum >= 5000 else (2 if aum >= 1000 else 1)
    ck(
        "PASS" if pts >= 2 else "WARN",
        f"AUM ₹{aum:,.0f}cr — {'HIGH' if pts==3 else 'MED' if pts==2 else 'LOW'} liquidity",
        (
            "Tight bid-ask spread, easy exit"
            if pts >= 2
            else "Low AUM — wider spread, size carefully"
        ),
        "",
    )
    score += pts
    breakdown["AUM Liquidity"] = pts
    reasons.append(f"AUM {'HIGH' if pts==3 else 'MED' if pts==2 else 'LOW'} ({pts}/3)")

    score = max(0, min(100, score))

    # ── Sizing + Label ─────────────────────────────────────────────────────
    hard_fail = bb_state not in ("RIDING_UPPER", "FIRST_DIP") or not wr_in_momentum
    if hard_fail or score < 40:
        label = "AVOID"
        sizing = "SKIP"
    elif score >= 72 and daily_bias == "BULLISH":
        label = "STRONG"
        sizing = "FULL"
    elif score >= 55:
        label = "MODERATE"
        sizing = "HALF"
    else:
        label = "WEAK"
        sizing = "SKIP"

    if etf_trend and etf_trend.underlying_bias == "BEARISH" and sizing == "FULL":
        sizing = "HALF"
        reasons.append("Underlying BEARISH — size capped HALF")
    if (
        etf_nav
        and etf_nav.available
        and etf_nav.premium_pct >= 1.5
        and sizing != "SKIP"
    ):
        sizing = "HALF" if sizing == "FULL" else sizing
        reasons.append(f"NAV premium {etf_nav.premium_pct:+.1f}% — size capped HALF")
    if daily_bias == "BEARISH" and sizing != "SKIP":
        sizing = "SKIP"
        label = "AVOID"
        reasons.append("Daily bias BEARISH — no longs")

    return ETFViability(
        score=score,
        label=label,
        sizing=sizing,
        reasons=reasons,
        checklist=checklist,
        breakdown=breakdown,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def analyze_etf(
    symbol: str, price_df: pd.DataFrame, price_signals, fetch_nav: bool = True
) -> Optional[ETFContext]:
    if price_df is None or price_df.empty:
        return None
    info = ETF_UNIVERSE.get(symbol) or ETFInfo(
        symbol=symbol,
        name=symbol,
        category="equity_index",
        underlying="Unknown",
        proxy_ticker=f"{symbol}.NS",
    )
    spot = float(price_df["Close"].iloc[-1])
    vp = build_volume_profile(price_df, bins=50, lookback=60)
    etf_volume = compute_etf_volume(price_df, avg_window=20)
    etf_trend = compute_etf_trend(price_df, info, lookback=20)
    inav = fetch_nav_from_nse(symbol) if fetch_nav else None
    etf_nav = compute_nav_signal(inav, spot)

    bb_state = price_signals.position_state if price_signals else "UNKNOWN"
    wr_value = price_signals.wr_value if price_signals else -100.0
    wr_in_momentum = price_signals.wr_in_momentum if price_signals else False
    daily_bias = price_signals.daily_bias if price_signals else "NEUTRAL"

    viability = score_etf(
        bb_state,
        wr_in_momentum,
        wr_value,
        etf_volume,
        vp,
        etf_trend,
        etf_nav,
        info,
        spot,
        daily_bias,
    )
    return ETFContext(
        symbol=symbol,
        info=info,
        spot=spot,
        volume_profile=vp,
        etf_volume=etf_volume,
        etf_trend=etf_trend,
        etf_nav=etf_nav,
        bb_state=bb_state,
        wr_value=wr_value,
        wr_in_momentum=wr_in_momentum,
        daily_bias=daily_bias,
        viability=viability,
    )
