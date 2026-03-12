"""
modules/signal_engine.py
────────────────────────
Enhanced viability scoring: wraps analytics.analyze() and applies
bonus/penalty adjustments from new intelligence signals.

Does NOT modify analytics._viability() — adds on top of it.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Optional

import pandas as pd

from modules.analytics import OptionsContext, analyze
from modules.indicators import PriceSignals

logger = logging.getLogger(__name__)

_EVENT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "event_calendar.json"
)


# ══════════════════════════════════════════════════════════════════════════════
# EVENT CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

def load_event_calendar() -> list[dict]:
    try:
        if not os.path.exists(_EVENT_PATH):
            return []
        with open(_EVENT_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def save_event_calendar(events: list[dict]):
    try:
        os.makedirs(os.path.dirname(_EVENT_PATH), exist_ok=True)
        with open(_EVENT_PATH, "w") as f:
            json.dump(events, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Could not save event calendar: %s", exc)


def check_event_risk(symbol: str, dte_days: int = 7) -> dict:
    """Check if any event affects this symbol within dte_days."""
    today = datetime.date.today()
    for event in load_event_calendar():
        syms = event.get("symbols", [])
        if symbol in syms or "ALL" in syms:
            try:
                event_date = datetime.date.fromisoformat(event["date"])
                days_away = (event_date - today).days
                if 0 <= days_away <= dte_days:
                    return {
                        "has_event": True,
                        "event": event["event"],
                        "days_away": days_away,
                        "impact": event.get("impact", "MEDIUM"),
                    }
            except Exception:
                continue
    return {"has_event": False, "event": "", "days_away": 99, "impact": "NONE"}


# ══════════════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════════════

def _get_return(ticker: str, days: int = 10) -> Optional[float]:
    """Fetch N-day return for a ticker. Returns None on failure."""
    try:
        import yfinance as yf
        from datetime import timedelta
        end = datetime.date.today()
        start = end - timedelta(days=days + 5)  # buffer for weekends
        df = yf.download(tickers=ticker, start=start.isoformat(),
                         interval="1d", auto_adjust=True, progress=False)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 2:
            return None
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        ret = (float(closes.iloc[-1]) - float(closes.iloc[-days])) / float(closes.iloc[-days])
        return ret
    except Exception:
        return None


def compute_rs(symbol: str, lookback_days: int = 10) -> dict:
    """
    Compute relative strength vs market (NIFTY 500) and sector.
    Returns dict with rs_vs_market, rs_vs_sector, rs_signal.
    """
    from modules.sector_map import SECTOR_INDEX, MARKET_TICKER, MARKET_FALLBACK

    sector_ticker = SECTOR_INDEX.get(symbol, SECTOR_INDEX["DEFAULT"])
    yf_symbol = f"{symbol}.NS" if not symbol.startswith("^") else symbol

    stock_ret = _get_return(yf_symbol, lookback_days)
    market_ret = _get_return(MARKET_TICKER, lookback_days)
    if market_ret is None:
        market_ret = _get_return(MARKET_FALLBACK, lookback_days)
    sector_ret = _get_return(sector_ticker, lookback_days)

    rs_mkt = (stock_ret / market_ret) if (stock_ret is not None and market_ret and market_ret != 0) else None
    rs_sec = (stock_ret / sector_ret) if (stock_ret is not None and sector_ret and sector_ret != 0) else None

    # Signal
    if rs_mkt is not None and rs_sec is not None:
        if rs_sec > 1.1 and rs_mkt > 1.1:
            signal = "LEADING"
        elif rs_sec < 0.9 and rs_mkt < 0.9:
            signal = "LAGGING"
        else:
            signal = "NEUTRAL"
    else:
        signal = "N/A"

    return {
        "rs_vs_market": round(rs_mkt, 2) if rs_mkt is not None else None,
        "rs_vs_sector": round(rs_sec, 2) if rs_sec is not None else None,
        "rs_signal": signal,
    }


# ══════════════════════════════════════════════════════════════════════════════
# IVR (IV Rank)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ivr(options_df: Optional[pd.DataFrame], spot: float) -> dict:
    """
    Compute IV Rank from options chain ATM IV.
    Returns dict with current_iv, ivr, ivr_state.
    Without 52-week history, returns N/A.
    """
    if options_df is None or options_df.empty or spot <= 0:
        return {"current_iv": None, "ivr": None, "ivr_state": "N/A"}

    try:
        # Find ATM strike
        df = options_df.copy()
        df["_dist"] = (df["Strike"] - spot).abs()
        atm = df.loc[df["_dist"].idxmin()]
        current_iv = float(atm.get("CE_IV", 0))
        if current_iv <= 0:
            return {"current_iv": None, "ivr": None, "ivr_state": "N/A"}

        # Without 52-week history, we estimate from the chain spread
        iv_vals = pd.to_numeric(df["CE_IV"], errors="coerce").dropna()
        iv_vals = iv_vals[iv_vals > 0]
        if len(iv_vals) < 5:
            return {"current_iv": current_iv, "ivr": None, "ivr_state": "N/A"}

        iv_low = float(iv_vals.quantile(0.05))
        iv_high = float(iv_vals.quantile(0.95))
        if iv_high <= iv_low:
            return {"current_iv": current_iv, "ivr": None, "ivr_state": "N/A"}

        ivr = (current_iv - iv_low) / (iv_high - iv_low) * 100
        ivr = max(0, min(100, ivr))

        if ivr < 30:
            state = "CHEAP"
        elif ivr > 70:
            state = "RICH"
        else:
            state = "FAIR"

        return {"current_iv": round(current_iv, 1), "ivr": round(ivr, 0), "ivr_state": state}
    except Exception:
        return {"current_iv": None, "ivr": None, "ivr_state": "N/A"}


# ══════════════════════════════════════════════════════════════════════════════
# FII/DII FLOW
# ══════════════════════════════════════════════════════════════════════════════

_fii_dii_cache = {"data": None, "ts": None}


def fetch_fii_dii() -> dict:
    """
    Fetch FII/DII cash market flow from NSE.
    Returns dict with fii_net, dii_net, fii_buy, fii_sell, dii_buy, dii_sell, signal.
    """
    # Return cache if fresh (< 5 min)
    import time
    if _fii_dii_cache["data"] and _fii_dii_cache["ts"] and time.time() - _fii_dii_cache["ts"] < 300:
        return _fii_dii_cache["data"]

    try:
        from modules.data import _make_nse_session, _NSE_HEADER, _assert_json
        session, cookies = _make_nse_session()
        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=_NSE_HEADER, cookies=cookies,
            allow_redirects=False, timeout=15,
        )
        _assert_json(resp)
        raw = resp.json()

        # Parse — NSE returns array of category objects
        result = {
            "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
            "dii_buy": 0, "dii_sell": 0, "dii_net": 0,
            "signal": "N/A", "date": "",
        }

        for item in (raw if isinstance(raw, list) else [raw]):
            cat = str(item.get("category", "")).upper()
            if "FII" in cat or "FPI" in cat:
                result["fii_buy"] = float(item.get("buyValue", 0))
                result["fii_sell"] = float(item.get("sellValue", 0))
                result["fii_net"] = float(item.get("netValue", 0))
                result["date"] = item.get("date", "")
            elif "DII" in cat:
                result["dii_buy"] = float(item.get("buyValue", 0))
                result["dii_sell"] = float(item.get("sellValue", 0))
                result["dii_net"] = float(item.get("netValue", 0))

        # Signal
        fii_pos = result["fii_net"] > 0
        dii_pos = result["dii_net"] > 0
        if fii_pos and dii_pos:
            result["signal"] = "INSTITUTIONAL TAILWIND"
        elif not fii_pos and dii_pos:
            result["signal"] = "DII SUPPORT / FII EXIT"
        elif not fii_pos and not dii_pos:
            result["signal"] = "INSTITUTIONAL HEADWIND"
        else:
            result["signal"] = "FII INFLOW / DII EXIT"

        _fii_dii_cache["data"] = result
        _fii_dii_cache["ts"] = time.time()
        return result
    except Exception as exc:
        logger.warning("FII/DII fetch failed: %s", exc)
        return {
            "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
            "dii_buy": 0, "dii_sell": 0, "dii_net": 0,
            "signal": "N/A", "date": "",
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED VIABILITY (bonus/penalty on top of base)
# ══════════════════════════════════════════════════════════════════════════════

def compute_bonuses(
    symbol: str,
    ctx: Optional[OptionsContext],
    ps: Optional[PriceSignals],
    options_df: Optional[pd.DataFrame] = None,
    spot: float = 0.0,
) -> dict:
    """
    Compute bonus/penalty adjustments to base viability score.
    Returns dict with individual adjustments and total.
    Does NOT call analyze() — works on top of ctx.viability.score.
    """
    bonuses = {}
    total = 0

    # IVR
    ivr_data = compute_ivr(options_df, spot)
    ivr = ivr_data.get("ivr")
    if ivr is not None:
        if ivr < 30:
            bonuses["ivr"] = +5
            total += 5
        elif ivr > 70:
            bonuses["ivr"] = -5
            total -= 5

    # Relative Strength (skip during scan for performance)
    # RS is computed separately and passed in

    # Event risk
    ev = check_event_risk(symbol)
    if ev["has_event"] and ev["days_away"] <= 2 and ev["impact"] == "HIGH":
        bonuses["event_cap"] = True  # cap score at 50

    # FII/DII
    try:
        flow = fetch_fii_dii()
        sig = flow.get("signal", "")
        if sig == "INSTITUTIONAL TAILWIND":
            bonuses["fii_dii"] = +5
            total += 5
        elif sig == "INSTITUTIONAL HEADWIND":
            bonuses["fii_dii"] = -5
            total -= 5
    except Exception:
        pass

    bonuses["total"] = total
    bonuses["ivr_data"] = ivr_data
    bonuses["event"] = ev
    return bonuses


def enhanced_score(base_score: int, bonuses: dict) -> int:
    """Apply bonuses to base score and clamp to 0-100."""
    score = base_score + bonuses.get("total", 0)
    if bonuses.get("event_cap"):
        score = min(score, 50)
    return max(0, min(100, score))
