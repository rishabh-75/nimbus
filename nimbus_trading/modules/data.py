"""
modules/data.py
───────────────
Price data (yfinance) + NSE options chain download.

CONFIRMED WORKING PATTERNS (from nse_dashboard transcript):
  NSE session  → Firefox/82 UA + allow_redirects=False + cookies = dict(resp.cookies)
  NSE API      → same header dict + allow_redirects=False on every call
  yfinance     → tickers= kwarg + start= date string (NOT period=)
                 fallback: 1h unavailable → use daily bars
"""

from __future__ import annotations

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
    "NIFTY": 75,
    "BANKNIFTY": 30,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 120,
    "NIFTYNXT50": 25,
    "RELIANCE": 250,
    "TCS": 150,
    "INFY": 300,
    "HDFCBANK": 550,
    "ICICIBANK": 700,
    "SBIN": 1500,
    "AXISBANK": 1200,
    "KOTAKBANK": 400,
    "BAJFINANCE": 125,
    "WIPRO": 1500,
    "LT": 175,
    "MARUTI": 50,
    "ADANIENT": 625,
    "ADANIPORTS": 625,
    "TATAMOTORS": 1425,
    "TATASTEEL": 5500,
    "JSWSTEEL": 1350,
    "HINDALCO": 2150,
    "SUNPHARMA": 700,
    "DRREDDY": 125,
    "CIPLA": 650,
    "NTPC": 3000,
    "POWERGRID": 3375,
    "ONGC": 3850,
    "BPCL": 1800,
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
# NSE SESSION  (confirmed working — do not modify the pattern)
# ══════════════════════════════════════════════════════════════════════════════


def _make_nse_session() -> Tuple[requests.Session, dict]:
    """
    Build a requests.Session with NSE cookies.

    CONFIRMED PATTERN:
      1. GET option-chain page with allow_redirects=False
         → NSE hands over real session cookies (not bot-detection redirect)
      2. cookies = dict(resp.cookies)   ← from the RESPONSE, not session.cookies
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
    """Raise a clear error if NSE returned HTML or empty body."""
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
    """
    try:
        session, cookies = _make_nse_session()

        expiries = _get_expiry_dates(session, cookies, symbol)
        if not expiries:
            return None, f"No expiry dates returned for {symbol}"

        expiries = expiries[:max_expiries]
        frames = []

        for i, exp in enumerate(expiries):
            if progress_cb:
                progress_cb(f"{symbol} {exp}  ({i+1}/{len(expiries)})")
            rows = _fetch_expiry(session, cookies, symbol, exp)
            if rows:
                frames.append(_parse_rows(rows, symbol, exp))
            time.sleep(0.4)

        if not frames:
            return None, "All expiries returned empty data"

        combined = pd.concat(frames, ignore_index=True)
        return combined, f"✓ {len(frames)} expiries downloaded for {symbol}"

    except Exception as exc:
        return None, f"NSE download failed: {exc}"


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
# PRICE DATA  (confirmed working yfinance pattern from transcript)
# ══════════════════════════════════════════════════════════════════════════════


def _yf_ticker(symbol: str) -> str:
    mapping = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCPNIFTY": "^CNXMIDCAP",
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
      1. 1h bars resampled to 4h (60-day window)
      2. Daily bars used directly if 1h is unavailable/empty
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
            df = (
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
            df = df[df["Close"] > 0].tail(bars).copy()
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
# SYMBOL UNIVERSE  (NSE master-quote API + local cache)
# ══════════════════════════════════════════════════════════════════════════════

import json
import os
import logging

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
    "ZOMATO",
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
        # try common key names
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
            # Extract symbol; prefer 'symbol' key, fallback to others
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
            # Filter: keep equities and F&O-eligible; exclude ETFs / bonds
            inst_type = str(
                item.get("instrumentType")
                or item.get("instrument_type")
                or item.get("series")
                or ""
            ).upper()
            # Skip clearly non-equity instruments
            if any(x in inst_type for x in ("ETF", "BOND", "DEBENTURE", "WARRANT")):
                continue
            # Only include if it looks like a plain equity ticker (no spaces, etc.)
            if (
                sym
                and sym.replace("-", "").replace("&", "").isalpha()
                or sym in NSE_LOT_SIZES
            ):
                symbols.add(sym)

    # Always include the F&O indices
    for idx in _FNO_INDICES:
        symbols.add(idx)

    return sorted(symbols)


def get_universe() -> list[str]:
    """
    Return list of NSE F&O symbol strings.
    Priority: valid cache → live NSE fetch → stale cache → hardcoded fallback.
    Never raises; always returns a usable list.
    """
    # 1. Valid cache (written today after 09:00 IST)
    if _universe_cache_valid():
        cached = _load_universe_cache()
        if cached:
            return cached

    # 2. Try live fetch
    try:
        symbols = _fetch_universe_from_nse()
        if symbols:
            _save_universe_cache(symbols)
            return symbols
    except Exception as exc:
        logging.warning(f"Universe fetch failed: {exc}")

    # 3. Stale cache is better than hardcoded
    stale = _load_universe_cache()
    if stale:
        return stale

    # 4. Absolute fallback
    return list(_NIFTY50_FALLBACK)


# Predefined curated lists for the scanner universe selector
NIFTY100_SYMBOLS = [
    "NIFTY",
    "BANKNIFTY",
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "SBIN",
    "AXISBANK",
    "LT",
    "BAJFINANCE",
    "KOTAKBANK",
    "HCLTECH",
    "WIPRO",
    "MARUTI",
    "ONGC",
    "NTPC",
    "POWERGRID",
    "SUNPHARMA",
    "TATAMOTORS",
    "ADANIENT",
    "ADANIPORTS",
    "TATASTEEL",
    "JSWSTEEL",
    "HINDALCO",
    "BHARTIARTL",
    "TITAN",
    "ASIANPAINT",
    "BAJAJ-AUTO",
    "HEROMOTOCO",
    "DRREDDY",
    "CIPLA",
    "DIVISLAB",
    "NESTLEIND",
    "ULTRACEMCO",
    "GRASIM",
    "TECHM",
    "EICHERMOT",
    "INDUSINDBK",
    "BAJAJFINSV",
    "BPCL",
    "ITC",
    "COALINDIA",
    "M&M",
    "UPL",
    "TATACONSUM",
    "APOLLOHOSP",
    "HDFCLIFE",
    "SBILIFE",
    "BRITANNIA",
    "ZOMATO",
    "FINNIFTY",
    "MIDCPNIFTY",
]
