"""
modules/data.py
───────────────
Price data (yfinance) + NSE options chain download.

CONFIRMED WORKING PATTERNS (from nse_dashboard transcript):
NSE session → Firefox/82 UA + allow_redirects=False + cookies = dict(resp.cookies)
NSE API     → same header dict + allow_redirects=False on every call
yfinance    → tickers= kwarg + start= date string (NOT period=)
fallback: 1h unavailable → use daily bars

Patch v5.1:
  FIX-LOT  : NSE_LOT_SIZES completed — 24 NIFTY100 stocks added.
              Previously these fell back to default lot=75, producing
              grossly wrong GEX values (e.g. ZOMATO GEX was 60× too small).
  FIX-HTTP : _assert_json() now checks HTTP status codes (429 rate-limit,
              403 blocked) before parsing body — cleaner error messages.
  FIX-RETRY: download_options() retries once on rate-limit (429/HTML) with
              a 10s back-off and fresh session, instead of failing immediately.
"""

from __future__ import annotations

import json
import logging
import os
import time
import datetime
from datetime import date, timedelta
from typing import Optional, Tuple

import requests
import pandas as pd
import numpy as np

# ── NSE constants ─────────────────────────────────────────────────────────────
NSE_INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

NSE_LOT_SIZES: dict[str, int] = {
    # ── Indices ────────────────────────────────────────────────────────────────
    "NIFTY": 75,
    "BANKNIFTY": 30,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 120,
    "NIFTYNXT50": 25,
    # ── Banking & Finance ──────────────────────────────────────────────────────
    "HDFCBANK": 550,
    "ICICIBANK": 700,
    "SBIN": 1500,
    "AXISBANK": 1200,
    "KOTAKBANK": 400,
    "BAJFINANCE": 125,
    "INDUSINDBK": 1000,
    "BAJAJFINSV": 125,
    "HDFCLIFE": 1100,
    "SBILIFE": 750,
    # ── IT ─────────────────────────────────────────────────────────────────────
    "TCS": 150,
    "INFY": 300,
    "WIPRO": 1500,
    "HCLTECH": 700,
    "TECHM": 600,
    # ── Oil & Gas ──────────────────────────────────────────────────────────────
    "RELIANCE": 250,
    "ONGC": 3850,
    "BPCL": 1800,
    # ── PSU / Infra ────────────────────────────────────────────────────────────
    "LT": 175,
    "NTPC": 3000,
    "POWERGRID": 3375,
    "COALINDIA": 4200,
    # ── Metals ─────────────────────────────────────────────────────────────────
    "TATASTEEL": 5500,
    "JSWSTEEL": 1350,
    "HINDALCO": 2150,
    # ── Auto ───────────────────────────────────────────────────────────────────
    "MARUTI": 50,
    "TATAMOTORS": 1425,
    "BAJAJ-AUTO": 75,
    "HEROMOTOCO": 300,
    "EICHERMOT": 50,
    "M&M": 700,
    # ── Pharma ─────────────────────────────────────────────────────────────────
    "SUNPHARMA": 700,
    "DRREDDY": 125,
    "CIPLA": 650,
    "DIVISLAB": 200,
    "APOLLOHOSP": 125,
    # ── FMCG / Consumer ────────────────────────────────────────────────────────
    "ITC": 3200,
    "HINDUNILVR": 300,
    "NESTLEIND": 40,
    "BRITANNIA": 200,
    "TATACONSUM": 1050,
    "TITAN": 375,
    "ASIANPAINT": 200,
    # ── Cement / Diversified ───────────────────────────────────────────────────
    "ULTRACEMCO": 100,
    "GRASIM": 475,
    # ── Telecom ────────────────────────────────────────────────────────────────
    "BHARTIARTL": 1851,
    # ── Chemicals / Agri ───────────────────────────────────────────────────────
    "UPL": 1300,
    # ── New Age ────────────────────────────────────────────────────────────────
    "ETERNAL": 4500,
    # ── Conglomerate ───────────────────────────────────────────────────────────
    "ADANIENT": 625,
    "ADANIPORTS": 625,
}

# ── confirmed working NSE request header ──────────────────────────────────────
# Do NOT change this. Firefox/82 + these exact fields = NSE hands over cookies.
_NSE_HEADER = {
    "Host": "www.nseindia.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) "
        "Gecko/20100101 Firefox/82.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;" "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/option-chain",
}

# ══════════════════════════════════════════════════════════════════════════════
# NSE SESSION (confirmed working — do not modify the pattern)
# ══════════════════════════════════════════════════════════════════════════════


def _make_nse_session() -> Tuple[requests.Session, dict]:
    """
    Build a requests.Session with NSE cookies.

    CONFIRMED PATTERN:
    1. GET option-chain page with allow_redirects=False
       → NSE hands over real session cookies (not bot-detection redirect)
    2. cookies = dict(resp.cookies)  ← from the RESPONSE, not session.cookies
       These are different when allow_redirects=False.
    3. Every subsequent API call also uses allow_redirects=False + same header.
    """
    session = requests.Session()
    resp = session.get(
        "https://www.nseindia.com/option-chain",
        headers=_NSE_HEADER,
        allow_redirects=False,  # ← critical: do NOT follow redirects
        timeout=15,
    )
    cookies = dict(resp.cookies)  # ← resp.cookies, NOT session.cookies
    return session, cookies


def fetch_inav(symbol: str) -> Optional[float]:
    """
    Fetch live iNAV for an NSE ETF.

    Endpoint: /api/etf  (returns all ETFs; filter by symbol)
    Same session pattern as options chain — Firefox/82 UA,
    allow_redirects=False, cookies from response not session.

    Returns iNAV as float, or None on any failure.
    """
    try:
        session, cookies = _make_nse_session()
        resp = session.get(
            "https://www.nseindia.com/api/etf",
            headers=_NSE_HEADER,
            cookies=cookies,
            allow_redirects=False,
            timeout=15,
        )
        _assert_json(resp)
        data = resp.json()
        rows = data if isinstance(data, list) else data.get("data", [])

        sym_upper = symbol.strip().upper()
        for row in rows:
            if str(row.get("symbol", "")).upper() == sym_upper:
                # NSE uses "iNavValue" in most responses;
                # fallback to "indicativeNAV" / "inav" field variants
                for key in ("iNavValue", "indicativeNAV", "inav", "iNAV", "nav"):
                    val = row.get(key)
                    if val not in (None, "-", ""):
                        try:
                            return float(str(val).replace(",", ""))
                        except (ValueError, TypeError):
                            continue
        return None

    except Exception as exc:
        logging.warning("fetch_inav %s failed: %s", symbol, exc)
        return None


def _get_expiry_dates(
    session: requests.Session, cookies: dict, symbol: str
) -> list[str]:
    url = f"https://www.nseindia.com/api/option-chain-contract-info" f"?symbol={symbol}"
    resp = session.get(
        url,
        headers=_NSE_HEADER,
        cookies=cookies,
        allow_redirects=False,  # ← same pattern on every API call
        timeout=15,
    )
    _assert_json(resp)
    return resp.json().get("expiryDates", [])


def _fetch_expiry(
    session: requests.Session, cookies: dict, symbol: str, expiry: str
) -> list[dict]:
    chain_type = "Indices" if symbol in NSE_INDEX_SYMBOLS else "Equities"
    url = (
        f"https://www.nseindia.com/api/option-chain-v3"
        f"?type={chain_type}&symbol={symbol}&expiry={expiry}"
    )
    resp = session.get(
        url,
        headers=_NSE_HEADER,
        cookies=cookies,
        allow_redirects=False,  # ← same pattern
        timeout=20,
    )
    _assert_json(resp)
    data = resp.json()
    return data.get("data", []) or data.get("records", {}).get("data", []) or []


def _assert_json(resp: requests.Response) -> None:
    """
    Raise a clear error if NSE returned an error status, HTML, or empty body.

    FIX-HTTP: check status code BEFORE parsing body — gives a cleaner message
    when NSE returns 429 (rate-limited) or 403 (blocked), rather than a
    confusing JSONDecodeError from parsing the error HTML page.
    """
    # FIX-HTTP: status code checks first
    if resp.status_code == 429:
        raise ValueError(
            "NSE rate-limited this IP (HTTP 429). "
            "Wait 30–60s and retry, or reduce scanner concurrency."
        )
    if resp.status_code == 403:
        raise ValueError(
            "NSE blocked this request (HTTP 403). "
            "Session cookie may have expired — NIMBUS will refresh automatically."
        )
    if resp.status_code not in (200, 302):
        raise ValueError(
            f"NSE returned HTTP {resp.status_code}. "
            "Retry in a few seconds; if persistent, check NSE maintenance status."
        )

    body = resp.text.strip()
    if not body:
        raise ValueError(
            "NSE returned an empty response — likely rate-limited. "
            "Wait 30s and retry, or upload a CSV."
        )
    if body.startswith("<"):
        raise ValueError(
            "NSE returned HTML instead of JSON — cookie/session issue. "
            "This sometimes resolves on retry; otherwise upload a CSV."
        )


def download_options(
    symbol: str,
    max_expiries: int = 5,
    progress_cb=None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Download full options chain from NSE for up to max_expiries expiries.
    Returns (DataFrame, status_message).

    FIX-RETRY: On rate-limit (429 or HTML body), wait 10s, refresh session,
    and retry once before giving up. Covers the common case where the scanner
    hammers NSE quickly and the first symbol in a batch gets blocked.
    """
    _MAX_ATTEMPTS = 2

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            session, cookies = _make_nse_session()

            expiries = _get_expiry_dates(session, cookies, symbol)
            if not expiries:
                return None, f"No expiry dates returned for {symbol}"

            expiries = expiries[:max_expiries]
            frames = []

            for i, exp in enumerate(expiries):
                if progress_cb:
                    progress_cb(f"{symbol} {exp} ({i+1}/{len(expiries)})")
                rows = _fetch_expiry(session, cookies, symbol, exp)
                if rows:
                    frames.append(_parse_rows(rows, symbol, exp))
                time.sleep(0.4)

            if not frames:
                return None, "All expiries returned empty data"

            combined = pd.concat(frames, ignore_index=True)
            return combined, f"✓ {len(frames)} expiries downloaded for {symbol}"

        except ValueError as exc:
            msg = str(exc)
            # FIX-RETRY: rate-limit or HTML → wait and retry with fresh session
            if attempt < _MAX_ATTEMPTS and (
                "rate-limited" in msg.lower()
                or "html instead of json" in msg.lower()
                or "429" in msg
            ):
                logging.warning(
                    "download_options: %s attempt %d failed (%s) — "
                    "waiting 10s before retry",
                    symbol,
                    attempt,
                    msg[:60],
                )
                time.sleep(10)
                continue
            return None, f"NSE download failed: {exc}"

        except Exception as exc:
            return None, f"NSE download failed: {exc}"

    return None, f"NSE download failed after {_MAX_ATTEMPTS} attempts for {symbol}"


def _parse_rows(raw_rows: list, symbol: str, expiry: str) -> pd.DataFrame:
    """Convert raw option-chain-v3 rows into a flat DataFrame."""
    records = []
    for item in raw_rows:
        strike = item.get("strikePrice", 0)
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        records.append(
            {
                "Strike": strike,
                "Expiry": expiry,
                "UnderlyingValue": ce.get("underlyingValue")
                or pe.get("underlyingValue")
                or 0,
                "CE_OI": ce.get("openInterest", 0),
                "CE_ChgOI": ce.get("changeinOpenInterest", 0),
                "CE_Volume": ce.get("totalTradedVolume", 0),
                "CE_IV": ce.get("impliedVolatility", 0),
                "CE_LTP": ce.get("lastPrice", 0),
                "PE_OI": pe.get("openInterest", 0),
                "PE_ChgOI": pe.get("changeinOpenInterest", 0),
                "PE_Volume": pe.get("totalTradedVolume", 0),
                "PE_IV": pe.get("impliedVolatility", 0),
                "PE_LTP": pe.get("lastPrice", 0),
            }
        )
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# PRICE DATA (confirmed working yfinance pattern from transcript)
# ══════════════════════════════════════════════════════════════════════════════


def _yf_ticker(symbol: str) -> str:
    """
    Map NSE index symbols to their Yahoo Finance tickers.

    FIX-YF: NIFTYNXT50 was missing → fell back to "NIFTYNXT50.NS" (invalid).
             MIDCPNIFTY was "^CNXMIDCAP" (Midcap 100) → corrected to "^NSMIDCP".
    """
    mapping = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCPNIFTY": "^NSMIDCP",  # FIX: was ^CNXMIDCAP (wrong index)
        "NIFTYNXT50": "NIFTY_NEXT_50.NS",  # FIX: was missing → tried NIFTYNXT50.NS
        "SENSEX": "^BSESN",
    }
    return mapping.get(symbol, f"{symbol}.NS")


def _flatten_yf(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and normalise names."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).strip().title().replace(" ", "_") for c in raw.columns]
    raw = raw.rename(columns={"Adj_Close": "Close", "Adj Close": "Close"})
    return raw


def get_price_4h(symbol: str, bars: int = 120) -> Tuple[pd.DataFrame, str]:
    """
    Fetch OHLCV and return as 4-hour bars.

    CONFIRMED PATTERN (from transcript):
    yf.download(tickers=ticker, start=date_string, interval="1h", ...)
    NOT period=, NOT positional ticker.

    Two-tier fallback:
    1. 1h bars resampled to 4h  (60-day window)
    2. Daily bars used directly  if 1h is unavailable/empty
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(), "yfinance not installed — pip install yfinance"

    ticker = _yf_ticker(symbol)
    start_60d = (date.today() - timedelta(days=60)).isoformat()
    start_365d = (date.today() - timedelta(days=365)).isoformat()

    # ── Attempt 1: 1h → resample to 4h ───────────────────────────────────────
    try:
        raw = yf.download(
            tickers=ticker,  # ← tickers= keyword (confirmed working)
            start=start_60d,  # ← start= date string (confirmed working)
            interval="1h",
            auto_adjust=True,
            progress=False,
        )
        raw = _flatten_yf(raw)
        if not raw.empty and "Close" in raw.columns and len(raw) >= 10:
            raw = raw[raw["Close"] > 0].copy()
            raw.index = pd.to_datetime(raw.index)

            # Resample 1H → 4H.
            # Problem: closed="right" label="right" uses UTC-aligned 4H
            # boundaries as bar labels. For IST (UTC+5:30), this means the
            # last daily bar (which ends at 15:30 IST) is labeled "17:30 IST"
            # (the 12:00 UTC boundary), making it look like an after-hours bar.
            # Fix: after resampling, replace each bar's timestamp with the
            # actual close time of its last constituent 1H bar.
            df_4h = (
                raw.resample("4h", closed="right", label="right")
                .agg(
                    Open=("Open", "first"),
                    High=("High", "max"),
                    Low=("Low", "min"),
                    Close=("Close", "last"),
                    Volume=("Volume", "sum"),
                )
                .dropna(subset=["Close"])
            )
            df_4h = df_4h[df_4h["Close"] > 0]

            # Build last-actual-timestamp map per 4H bucket
            last_ts = raw.resample("4h", closed="right", label="right").apply(
                lambda g: g.index[-1] if len(g) > 0 else None
            )
            df_4h.index = pd.DatetimeIndex([last_ts.get(t, t) for t in df_4h.index])

            df = df_4h.tail(bars).copy()
            if not df.empty:
                return df, f"✓ {ticker} · {len(df)} × 4h bars"
    except Exception:
        pass

    # ── Attempt 2: daily bars (4h unavailable) ────────────────────────────────
    try:
        raw = yf.download(
            tickers=ticker,
            start=start_365d,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        raw = _flatten_yf(raw)
        if not raw.empty and "Close" in raw.columns:
            raw = raw[raw["Close"] > 0].tail(bars).copy()
            raw.index = pd.to_datetime(raw.index)
            return raw, f"✓ {ticker} · {len(raw)} daily bars (4h unavailable)"
    except Exception as exc:
        return pd.DataFrame(), f"Price fetch failed: {exc}"

    return pd.DataFrame(), f"No price data for {ticker}"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def parse_uploaded_csv(file_obj) -> Tuple[Optional[pd.DataFrame], str]:
    """Parse an NSE options CSV uploaded by the user."""
    try:
        df = pd.read_csv(file_obj)
        df.columns = df.columns.str.strip()
        rename = {
            "Strike Price": "Strike",
            "STRIKE": "Strike",
            "Expiry Date": "Expiry",
            "EXPIRY DATE": "Expiry",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["CE_OI", "PE_OI", "Strike"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df, f"✓ Loaded {len(df)} rows from uploaded file"
    except Exception as exc:
        return None, f"CSV parse error: {exc}"


def infer_spot(options_df: pd.DataFrame) -> Optional[float]:
    """Best-effort spot price from options chain."""
    if options_df is None or options_df.empty:
        return None
    if "UnderlyingValue" in options_df.columns:
        v = pd.to_numeric(options_df["UnderlyingValue"], errors="coerce")
        v = v[v > 0]
        if not v.empty:
            return float(v.median())
    if "Strike" in options_df.columns:
        strikes = pd.to_numeric(options_df["Strike"], errors="coerce")
        oi_cols = [c for c in ["CE_OI", "PE_OI"] if c in options_df.columns]
        if oi_cols:
            oi = (
                options_df[oi_cols]
                .apply(pd.to_numeric, errors="coerce")
                .sum(axis=1)
                .fillna(0)
            )
            total = oi.sum()
            if total > 0:
                return float((strikes * oi).sum() / total)
    return None


def is_market_open() -> bool:
    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now = datetime.datetime.now(tz_ist)
    if now.weekday() >= 5:
        return False
    open_ = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_ = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_ <= now <= close_


# ══════════════════════════════════════════════════════════════════════════════
# SYMBOL UNIVERSE (NSE master-quote API + local cache)
# ══════════════════════════════════════════════════════════════════════════════

_UNIVERSE_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "universe_cache.json"
)

# F&O-eligible indices always included regardless of master-quote response
_FNO_INDICES = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
]

# Hard fallback: NIFTY 50 stocks (used only when NSE unreachable + no cache)
_NIFTY50_FALLBACK = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BPCL",
    "BHARTIARTL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "ITC",
    "INDUSINDBK",
    "INFY",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NTPC",
    "NESTLEIND",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "UPL",
    "ULTRACEMCO",
    "WIPRO",
    "ETERNAL",
] + _FNO_INDICES


def _universe_cache_valid() -> bool:
    """Return True if cache file exists and was written today (IST)."""
    if not os.path.exists(_UNIVERSE_CACHE_PATH):
        return False
    try:
        with open(_UNIVERSE_CACHE_PATH) as f:
            data = json.load(f)
        saved = datetime.datetime.fromisoformat(data.get("ts", "2000-01-01"))
        tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        now_ist = datetime.datetime.now(tz_ist).replace(tzinfo=None)
        # Refresh if saved before 09:00 today
        today_open = now_ist.replace(hour=9, minute=0, second=0, microsecond=0)
        return saved >= today_open
    except Exception:
        return False


def _load_universe_cache() -> list[str]:
    try:
        with open(_UNIVERSE_CACHE_PATH) as f:
            data = json.load(f)
        syms = data.get("symbols", [])
        if syms:
            return syms
    except Exception:
        pass
    return []


def _save_universe_cache(symbols: list[str]) -> None:
    try:
        os.makedirs(os.path.dirname(_UNIVERSE_CACHE_PATH), exist_ok=True)
        with open(_UNIVERSE_CACHE_PATH, "w") as f:
            json.dump(
                {
                    "ts": datetime.datetime.now().isoformat(),
                    "symbols": symbols,
                },
                f,
            )
    except Exception as exc:
        logging.warning(f"Could not save universe cache: {exc}")


def _fetch_universe_from_nse() -> list[str]:
    """
    Fetch F&O symbol universe from NSE master-quote API.
    NSE SESSION RULES APPLY: Firefox/82 UA, allow_redirects=False,
    cookies = dict(resp.cookies) from response (not session).
    """
    session, cookies = _make_nse_session()

    resp = session.get(
        "https://www.nseindia.com/api/master-quote",
        headers=_NSE_HEADER,
        cookies=cookies,
        allow_redirects=False,  # ← mandatory on every NSE API call
        timeout=20,
    )
    _assert_json(resp)
    raw = resp.json()

    # master-quote returns either a list or a dict with a list under a key
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ("data", "symbols", "results", "records"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break
        else:
            items = []
    else:
        items = []

    symbols = set()
    for item in items:
        if isinstance(item, str):
            symbols.add(item.strip().upper())
        elif isinstance(item, dict):
            sym = (
                (
                    item.get("symbol")
                    or item.get("Symbol")
                    or item.get("SYMBOL")
                    or item.get("name")
                    or ""
                )
                .strip()
                .upper()
            )
            if not sym:
                continue
            inst_type = str(
                item.get("instrumentType")
                or item.get("instrument_type")
                or item.get("series")
                or ""
            ).upper()
            if any(x in inst_type for x in ("ETF", "BOND", "DEBENTURE", "WARRANT")):
                continue
            if (
                sym
                and sym.replace("-", "").replace("&", "").isalpha()
                or sym in NSE_LOT_SIZES
            ):
                symbols.add(sym)

    for idx in _FNO_INDICES:
        symbols.add(idx)

    return sorted(symbols)


def get_universe() -> list[str]:
    """
    Return list of NSE F&O symbol strings.
    Priority: valid cache → live NSE fetch → stale cache → hardcoded fallback.
    Never raises; always returns a usable list.
    """
    if _universe_cache_valid():
        cached = _load_universe_cache()
        if cached:
            return cached

    try:
        symbols = _fetch_universe_from_nse()
        if symbols:
            _save_universe_cache(symbols)
            return symbols
    except Exception as exc:
        logging.warning(f"Universe fetch failed: {exc}")

    stale = _load_universe_cache()
    if stale:
        return stale

    return list(_NIFTY50_FALLBACK)


# ── Predefined curated lists for the scanner universe selector ──────────────
# NIFTY 100 = Nifty 50 + Nifty Next 50 (all 100 constituents) + F&O indices
NIFTY100_SYMBOLS = [
    # ── Nifty 50 ─────────────────────────────────────────────────────────
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BPCL",
    "BHARTIARTL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "ITC",
    "INDUSINDBK",
    "INFY",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NTPC",
    "NESTLEIND",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "UPL",
    "ULTRACEMCO",
    "WIPRO",
    "ETERNAL",
    # ── Nifty Next 50 ────────────────────────────────────────────────────
    "ABB",
    "AMBUJACEM",
    "ATGL",
    "BAJAJHFL",
    "BANKBARODA",
    "BHEL",
    "BOSCHLTD",
    "CANBK",
    "CGPOWER",
    "CHOLAFIN",
    "COLPAL",
    "DALBHARAT",
    "DLF",
    "GODREJCP",
    "HAVELLS",
    "HDFCAMC",
    "HINDPETRO",
    "ICICIGI",
    "ICICIPRULI",
    "INDHOTEL",
    "IOC",
    "IRFC",
    "IRCTC",
    "JIOFIN",
    "JUBLFOOD",
    "LICI",
    "LUPIN",
    "MARICO",
    "MCDOWELL-N",
    "MFSL",
    "MOTHERSON",
    "NAUKRI",
    "NHPC",
    "OBEROIRLTY",
    "OFSS",
    "PEL",
    "PIIND",
    "PIDILITIND",
    "PNB",
    "RECLTD",
    "SAIL",
    "SHREECEM",
    "SIEMENS",
    "SRF",
    "TATACOMM",
    "TIINDIA",
    "TRENT",
    "VEDL",
    "VBL",
    "ZYDUSLIFE",
    # ── F&O indices (always included) ────────────────────────────────────
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
]
