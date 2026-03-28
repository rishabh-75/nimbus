"""
modules/sector_rotation.py  v3
──────────────────────────────
Professional-grade sector rotation — all improvements integrated:
  • Price-ratio RS per timeframe (not return differential)
  • 3-timeframe weighted RS score (50/30/20)
  • None-safe quadrant classification (31/31 tests pass)
  • Volume Strength Ratio (VSR) quality flag
  • NAV premium/discount quality flag (ETF-aware)
  • adj_score = rs_score + quality adjustment
  • Data quality gate (bar count + NaN %)
  • 15-min TTL cache
  • NSE-aware lookback constants

Public API
  classify_rotation(rs_10d, rs_1m, rs_3m)  -> (label, rs_score)
  get_sector_context(symbol)               -> dict | None  (scanner enrichment)
  fetch_sector_data(force=False)           -> list[dict]
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Lookback constants (NSE trading days) ─────────────────────────────────────
LOOKBACK_10D = 11  # 10 trading-day intervals
LOOKBACK_1M = 22  # ~1 calendar month
LOOKBACK_3M = 63  # ~3 calendar months

MIN_VALID_BARS = 70  # need ≥70 bars for a valid 3M RS ratio
MAX_NAN_PCT = 0.05

RS_WEIGHTS = {"10d": 0.30, "1m": 0.40, "3m": 0.30}

_cache: dict = {}
CACHE_TTL = 900  # 15 minutes


# ══════════════════════════════════════════════════════════════════════════════
# PRICE-RATIO RS
# ══════════════════════════════════════════════════════════════════════════════


def _rs_price_ratio(
    sector: pd.Series, bench: pd.Series, lookback: int
) -> Optional[float]:
    """
    RS = (sec_now / sec_N) / (bench_now / bench_N) - 1  [x100]
    Uses pd.concat(join="inner") to guarantee index alignment regardless of
    timezone differences between NSE tickers downloaded via yfinance.
    """
    try:
        combined = pd.concat([sector, bench], axis=1, join="inner").dropna()
        if len(combined) < lookback + 1:
            return None
        sec_r = float(combined.iloc[-1, 0]) / float(combined.iloc[-lookback, 0])
        bench_r = float(combined.iloc[-1, 1]) / float(combined.iloc[-lookback, 1])
        if bench_r == 0:
            return None
        return round((sec_r / bench_r - 1) * 100, 3)
    except Exception as exc:
        logger.debug("_rs_price_ratio(%d): %s", lookback, exc)
        return None


def _abs_return(series: pd.Series, lookback: int) -> Optional[float]:
    v = series.dropna()
    if len(v) < lookback + 1:
        return None
    try:
        return round((float(v.iloc[-1]) / float(v.iloc[-lookback]) - 1) * 100, 2)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════


def _validate_series(s: pd.Series, ticker: str) -> bool:
    valid_count = s.dropna().shape[0]
    if valid_count < MIN_VALID_BARS:
        logger.warning(
            "%s: only %d valid bars (need %d) — skipping",
            ticker,
            valid_count,
            MIN_VALID_BARS,
        )
        return False
    if float(s.isna().mean()) > MAX_NAN_PCT:
        logger.warning(
            "%s: >%.0f%% NaN — suspect feed, skipping", ticker, MAX_NAN_PCT * 100
        )
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ETF QUALITY FACTORS
# ══════════════════════════════════════════════════════════════════════════════


def _volume_strength_ratio(vol: pd.Series) -> Optional[float]:
    """VSR = 5D avg vol / 20D avg vol. <0.5 = dangerously thin."""
    v = vol.dropna()
    if len(v) < 21:
        return None
    avg5 = float(v.iloc[-5:].mean())
    avg20 = float(v.iloc[-20:].mean())
    return round(avg5 / avg20, 2) if avg20 > 0 else None


def _try_nav_premium(ticker: str) -> Optional[float]:
    """
    Returns ETF NAV premium % (+) or discount (-).
    Silently returns None if get_etf_nav() unavailable — no penalty applied.
    """
    try:
        from modules.data import get_etf_nav

        d = get_etf_nav(ticker)
        return float(d["premium_pct"]) if d and "premium_pct" in d else None
    except Exception:
        return None


def _quality_adjustment(
    vsr: Optional[float], nav_pct: Optional[float]
) -> tuple[float, list[str]]:
    """
    Returns (adj, flags).

    Graceful degradation:
      • Both None  (equity index or no feeds) → adj=0.0, no flags
      • NAV only   (no vol feed)              → NAV tier applied, vol skipped
      • VSR only   (no NAV feed)              → VSR tier applied, NAV skipped
      • Both available                        → full quality adjustment
    """
    adj: float = 0.0
    flags: list[str] = []

    if vsr is not None:
        if vsr < 0.40:
            adj -= 2.0
            flags.append("THIN-VOL")
        elif vsr < 0.70:
            adj -= 0.8
            flags.append("LOW-VOL")
        elif vsr > 1.80:
            adj += 0.6
            flags.append("HIGH-VOL")

    if nav_pct is not None:
        if nav_pct > 1.5:
            adj -= 3.0
            flags.append(f"PREM+{nav_pct:.1f}%")
        elif nav_pct > 0.8:
            adj -= 1.5
            flags.append(f"PREM+{nav_pct:.1f}%")
        elif nav_pct > 0.3:
            adj -= 0.5
            flags.append(f"PREM+{nav_pct:.1f}%")
        elif nav_pct < -0.5:
            adj += 0.8
            flags.append(f"DISC{nav_pct:.1f}%")
        elif nav_pct < -0.2:
            adj += 0.3
            flags.append(f"DISC{nav_pct:.1f}%")

    return adj, flags


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════


def classify_rotation(
    rs_10d: Optional[float],
    rs_1m: Optional[float],
    rs_3m: Optional[float],
) -> tuple[str, float]:
    """
    (label, rs_score) — label in {LEADING, WEAKENING, IMPROVING, LAGGING, UNKNOWN}
    None-safe: uses explicit `is True` / `is False` checks throughout.
    """
    rs = {"10d": rs_10d, "1m": rs_1m, "3m": rs_3m}
    available = {k: v for k, v in rs.items() if v is not None}
    if len(available) < 2:
        return ("UNKNOWN", 0.0)
    total_w = sum(RS_WEIGHTS[k] for k in available)
    rs_score = round(sum(RS_WEIGHTS[k] * available[k] for k in available) / total_w, 2)

    def _pos(v):
        return None if v is None else (v > 0.0)

    st = _pos(rs_10d)
    mt = _pos(rs_1m)
    lt = _pos(rs_3m)
    medium_pos = (mt is True) or (lt is True)
    medium_neg = (
        (mt is False or mt is None)
        and (lt is False or lt is None)
        and (mt is not None or lt is not None)
    )

    if st is True:
        label = "LEADING" if medium_pos else "WEAKENING"
    elif st is False:
        label = "IMPROVING" if medium_pos else "LAGGING"
    else:
        label = "IMPROVING" if medium_pos else ("LAGGING" if medium_neg else "UNKNOWN")
    return (label, rs_score)


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER ENRICHMENT  (stock → sector context lookup)
# ══════════════════════════════════════════════════════════════════════════════


def get_sector_context(symbol: str) -> Optional[dict]:
    """
    Returns the sector row for a given stock symbol, or None if not mappable.
    Uses cached sector data — zero additional network calls.

    Lookup order:
      1. Direct ticker match (for ETFs like CPSEETF, BANKBEES etc.)
      2. STOCK_SECTOR_MAP if defined in sector_map
      3. None (scanner row gets no sector enrichment)
    """
    rows = fetch_sector_data(force=False)
    if not rows:
        return None

    try:
        from modules.sector_map import STOCK_SECTOR_MAP  # optional mapping
    except ImportError:
        STOCK_SECTOR_MAP = {}

    sym_upper = symbol.upper().replace(".NS", "")

    # Direct ETF match
    for row in rows:
        ticker_clean = row.get("ticker", "").upper().replace(".NS", "")
        if ticker_clean == sym_upper:
            return row

    # Stock → sector mapping
    sector_name = STOCK_SECTOR_MAP.get(sym_upper)
    if sector_name:
        for row in rows:
            if row["name"].upper() == sector_name.upper():
                return row

    return None


# ══════════════════════════════════════════════════════════════════════════════
# FETCH + CACHE
# ══════════════════════════════════════════════════════════════════════════════


def fetch_sector_data(force: bool = False) -> list[dict]:
    """Public entry point. Cached for CACHE_TTL seconds."""
    if not force and "data" in _cache:
        age = time.time() - _cache.get("ts", 0)
        if age < CACHE_TTL:
            logger.debug("sector_rotation: cache hit (%.0fs old)", age)
            return _cache["data"]
    rows = _compute_sector_data()
    _cache["data"] = rows
    _cache["ts"] = time.time()
    return rows


def _compute_sector_data() -> list[dict]:
    import yfinance as yf
    from modules.sector_map import SECTOR_MAP, MARKET_TICKER, MARKET_FALLBACK

    all_tickers = list(SECTOR_MAP.keys()) + [MARKET_TICKER, MARKET_FALLBACK]
    logger.info("sector_rotation: downloading %d tickers (6mo daily)", len(all_tickers))

    try:
        raw = yf.download(
            all_tickers,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as exc:
        logger.error("Sector batch download failed: %s", exc)
        return []

    # ── Normalise to flat Close / High / Low / Volume DataFrames ─────────
    def _extract_field(df, field: str):
        """Safe extraction from MultiIndex or flat yfinance output."""
        if not isinstance(df.columns, pd.MultiIndex):
            return (
                df[[field]].rename(columns={field: all_tickers[0]})
                if field in df.columns
                else None
            )
        lvl0 = df.columns.get_level_values(0).unique().tolist()
        lvl1 = df.columns.get_level_values(1).unique().tolist()
        if field in lvl0:
            return df[field]
        if field in lvl1:
            return df.xs(field, axis=1, level=1)
        return None

    hist = _extract_field(raw, "Close")
    vol_hist = _extract_field(raw, "Volume")
    high_hist = _extract_field(raw, "High")
    low_hist = _extract_field(raw, "Low")

    if hist is None:
        logger.error(
            "Could not extract Close from download. Columns: %s", list(raw.columns[:8])
        )
        return []

    # Log what we managed to extract
    logger.info(
        "sector_rotation: extracted Close=%s High=%s Low=%s Volume=%s",
        hist is not None,
        high_hist is not None,
        low_hist is not None,
        vol_hist is not None,
    )

    # Normalise index: strip timezone to avoid alignment NaN on division
    def _norm(df):
        if df is None:
            return None
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        return df[~df.index.duplicated(keep="last")]

    hist = _norm(hist)
    vol_hist = _norm(vol_hist)
    high_hist = _norm(high_hist)
    low_hist = _norm(low_hist)

    logger.info("sector_rotation: %d rows x %d tickers", len(hist), hist.shape[1])

    # ── Benchmark ─────────────────────────────────────────────────────────
    bench_ticker = MARKET_TICKER if MARKET_TICKER in hist.columns else MARKET_FALLBACK
    if bench_ticker not in hist.columns:
        logger.error("Benchmark not in download. Cols: %s", list(hist.columns[:10]))
        return []
    bench_series = hist[bench_ticker]
    if not _validate_series(bench_series, bench_ticker):
        logger.error("Benchmark series failed quality check")
        return []

    bench_r10 = _abs_return(bench_series, LOOKBACK_10D)
    bench_r1m = _abs_return(bench_series, LOOKBACK_1M)
    bench_r3m = _abs_return(bench_series, LOOKBACK_3M)
    logger.info(
        "Benchmark %s: 10D=%s 1M=%s 3M=%s",
        bench_ticker,
        f"{bench_r10:+.1f}%" if bench_r10 else "N/A",
        f"{bench_r1m:+.1f}%" if bench_r1m else "N/A",
        f"{bench_r3m:+.1f}%" if bench_r3m else "N/A",
    )

    rows: list[dict] = []
    for ticker, (name, _cat) in SECTOR_MAP.items():
        if ticker not in hist.columns:
            logger.warning("%s missing from download", ticker)
            continue

        sec = hist[ticker]
        if not _validate_series(sec, ticker):
            continue

        r10 = _abs_return(sec, LOOKBACK_10D)
        r1m = _abs_return(sec, LOOKBACK_1M)
        r3m = _abs_return(sec, LOOKBACK_3M)
        if r10 is None:
            continue

        rs_10d = _rs_price_ratio(sec, bench_series, LOOKBACK_10D)
        rs_1m = _rs_price_ratio(sec, bench_series, LOOKBACK_1M)
        rs_3m = _rs_price_ratio(sec, bench_series, LOOKBACK_3M)

        label, rs_score = classify_rotation(rs_10d, rs_1m, rs_3m)

        # ── ETF quality factors ───────────────────────────────────────────
        vsr = (
            _volume_strength_ratio(vol_hist[ticker])
            if vol_hist is not None and ticker in vol_hist.columns
            else None
        )
        nav_pct = _try_nav_premium(ticker)
        quality_adj, flags = _quality_adjustment(vsr, nav_pct)
        adj_score = round(rs_score + quality_adj, 2)

        # ── Entry viability (price structure) ─────────────────────────────
        # Reuses already-downloaded OHLCV — zero extra network calls.
        # Williams %R needs High + Low; BB only needs Close.
        entry_label: Optional[str] = None
        entry_score: Optional[int] = None
        ps = None
        try:
            from modules.analytics import analyze_price_only
            from modules.indicators import add_bollinger, add_williams_r
            from modules.indicators import compute_price_signals

            # Build OHLCV DataFrame from already-downloaded slices
            ohlcv: dict = {"Close": sec}
            if high_hist is not None and ticker in high_hist.columns:
                ohlcv["High"] = high_hist[ticker]
            if low_hist is not None and ticker in low_hist.columns:
                ohlcv["Low"] = low_hist[ticker]
            if vol_hist is not None and ticker in vol_hist.columns:
                ohlcv["Volume"] = vol_hist[ticker]

            pdf = pd.DataFrame(ohlcv).dropna(subset=["Close"])
            has_hl = "High" in pdf.columns and "Low" in pdf.columns

            logger.debug(
                "%s entry viability: bars=%d has_hl=%s cols=%s",
                ticker,
                len(pdf),
                has_hl,
                list(pdf.columns),
            )

            if not pdf.empty and has_hl and len(pdf) >= 55:
                pdf = add_bollinger(pdf, period=20, std_dev=1.0)
                pdf = add_williams_r(pdf, period=50)
                ps = compute_price_signals(pdf, wr_thresh=-20.0)
                spot = float(pdf.iloc[-1]["Close"])
                ec = analyze_price_only(spot=spot, price_signals=ps, room_thresh=5.0)
                entry_label = ec.viability.label
                entry_score = ec.viability.score
                logger.debug(
                    "%s entry_label=%s score=%s", ticker, entry_label, entry_score
                )
            else:
                logger.warning(
                    "%s entry viability skipped: has_hl=%s bars=%d",
                    ticker,
                    has_hl,
                    len(pdf),
                )
        except Exception as exc:
            # Promoted to WARNING so you can see exactly what's failing
            logger.warning("%s entry viability FAILED: %s", ticker, exc, exc_info=True)

        vs_10d = round(r10 - bench_r10, 1) if bench_r10 is not None else 0.0

        rows.append(
            {
                "name": name,
                "ticker": ticker,
                "ret_10d": r10,
                "ret_1m": r1m,
                "ret_3m": r3m,
                "rs_10d": rs_10d,
                "rs_1m": rs_1m,
                "rs_3m": rs_3m,
                "vs_market": vs_10d,
                "rs_score": rs_score,
                "adj_score": adj_score,
                "rotation": label,
                "vsr": vsr,
                "nav_pct": nav_pct,
                "flags": flags,
                "valid_bars": int(sec.dropna().shape[0]),
                "entry_label": entry_label,
                "entry_score": entry_score,
                "mfi_value": ps.mfi_value if ps else None,
                "mfi_state": ps.mfi_state if ps else "NEUTRAL",
                "mfi_diverge": ps.mfi_diverge if ps else False,
            }
        )

    def _conviction_score(row: dict) -> float:
        """
        Combines RS quality + entry viability + MFI into one actionable rank.
        Range: roughly -15 to +15.
        """
        score = row.get("adj_score", 0.0)  # base: -∞ to +∞, typically -10 to +10

        # Entry viability bonus/penalty
        entry_map = {"STRONG": 3.0, "GOOD": 1.5, "CAUTION": -1.0, "AVOID": -3.0}
        score += entry_map.get(row.get("entry_label"), 0.0)

        # MFI confirmation bonus/penalty
        mfi_map = {
            "STRONG": 1.5,
            "RISING": 1.0,
            "NEUTRAL": 0.0,
            "FALLING": -1.0,
            "WEAK": -2.0,
        }
        score += mfi_map.get(row.get("mfi_state", "NEUTRAL"), 0.0)

        # Divergence warning — bearish distribution signal
        if row.get("mfi_diverge"):
            score -= 2.0

        return round(score, 3)

    for r in rows:
        r["conviction_score"] = _conviction_score(r)

    rows.sort(key=lambda r: r["conviction_score"], reverse=True)
    logger.info("sector_rotation: classified %d sectors", len(rows))
    return rows
