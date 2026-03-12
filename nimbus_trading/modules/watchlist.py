"""
modules/watchlist.py
────────────────────
Persistent watchlist for NIMBUS.
Storage: data/watchlist.json
Schema per entry: symbol, added_at, notes, entry_price, stop_price,
                  target_price, tags
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_WATCHLIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "watchlist.json"
)


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════


def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(_WATCHLIST_PATH), exist_ok=True)


def load_watchlist() -> list[dict]:
    """Load watchlist from disk. Returns [] on any error."""
    try:
        if not os.path.exists(_WATCHLIST_PATH):
            return []
        with open(_WATCHLIST_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"Could not load watchlist: {exc}")
        return []


def save_watchlist(entries: list[dict]) -> None:
    """Write watchlist to disk atomically."""
    try:
        _ensure_dir()
        tmp = _WATCHLIST_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2, default=str)
        os.replace(tmp, _WATCHLIST_PATH)
    except Exception as exc:
        logger.error(f"Could not save watchlist: {exc}")


def add_entry(
    entries: list[dict],
    symbol: str,
    entry_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    target_price: Optional[float] = None,
    notes: str = "",
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """
    Add a new entry. If symbol already exists, update the existing entry
    rather than creating a duplicate. Saves to disk. Returns updated list.
    """
    symbol = symbol.strip().upper()
    # Check for duplicate
    for e in entries:
        if e["symbol"] == symbol:
            # Update fields if provided
            if entry_price is not None:
                e["entry_price"] = entry_price
            if stop_price is not None:
                e["stop_price"] = stop_price
            if target_price is not None:
                e["target_price"] = target_price
            if notes:
                e["notes"] = notes
            if tags is not None:
                e["tags"] = tags
            save_watchlist(entries)
            return entries

    new_entry = {
        "symbol": symbol,
        "added_at": datetime.datetime.now().isoformat(),
        "notes": notes,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "tags": tags or [],
    }
    entries.append(new_entry)
    save_watchlist(entries)
    return entries


def remove_entry(entries: list[dict], symbol: str) -> list[dict]:
    """Remove symbol from watchlist. Saves to disk."""
    symbol = symbol.strip().upper()
    entries = [e for e in entries if e["symbol"] != symbol]
    save_watchlist(entries)
    return entries


def update_entry(entries: list[dict], symbol: str, **kwargs) -> list[dict]:
    """Update fields on an existing entry. Saves to disk."""
    symbol = symbol.strip().upper()
    for e in entries:
        if e["symbol"] == symbol:
            for k, v in kwargs.items():
                e[k] = v
            break
    save_watchlist(entries)
    return entries


def get_entry(entries: list[dict], symbol: str) -> Optional[dict]:
    """Return the watchlist entry for a symbol, or None."""
    symbol = symbol.strip().upper()
    for e in entries:
        if e["symbol"] == symbol:
            return e
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LIVE REFRESH
# ══════════════════════════════════════════════════════════════════════════════


def refresh_watchlist(
    entries: list[dict],
    progress_cb=None,
) -> list[dict]:
    """
    Re-run analyze_symbol() for every entry in the watchlist.
    Returns the entries list, each enriched with a "live" sub-dict.
    Gracefully skips symbols that fail.
    """
    from modules.scanner import analyze_symbol  # local import avoids circularity

    total = len(entries)
    for i, entry in enumerate(entries):
        sym = entry["symbol"]
        if progress_cb:
            try:
                progress_cb(i + 1, total, sym)
            except Exception:
                pass
        try:
            snap = analyze_symbol(sym)
            entry["live"] = snap or {}
            entry["live_ts"] = datetime.datetime.now().isoformat()
        except Exception as exc:
            logger.warning(f"[watchlist refresh] {sym}: {exc}")
            entry["live"] = {}
            entry["live_ts"] = datetime.datetime.now().isoformat()
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# ALERT DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def detect_alerts(entries: list[dict]) -> list[dict]:
    """
    Return a list of alert dicts for watchlist symbols that need attention.
    Alert dict: {symbol, alert_type, message, severity}
    severity: "warn" | "error"
    """
    alerts = []
    for e in entries:
        live = e.get("live", {})
        if not live:
            continue
        sym = e["symbol"]
        state = live.get("position_state", "")
        score = live.get("viability_score", 100)

        if state == "MID_BAND_BROKEN":
            alerts.append(
                {
                    "symbol": sym,
                    "alert_type": "EXIT_SIGNAL",
                    "message": f"{sym} — Mid-band broken. FULL EXIT signal. Close position.",
                    "severity": "error",
                }
            )
        elif state == "FIRST_DIP":
            alerts.append(
                {
                    "symbol": sym,
                    "alert_type": "PARTIAL_EXIT",
                    "message": f"{sym} — First dip below upper band. Scale out 50%.",
                    "severity": "warn",
                }
            )

        if score < 40:
            alerts.append(
                {
                    "symbol": sym,
                    "alert_type": "LOW_VIABILITY",
                    "message": f"{sym} viability dropped to {score}/100. Review position.",
                    "severity": "warn",
                }
            )

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# P&L CALCULATION
# ══════════════════════════════════════════════════════════════════════════════


def calc_pnl(entry: dict) -> Optional[float]:
    """
    Compute unrealised % P&L for a watchlist entry.
    Returns None if entry_price or last_price is missing.
    """
    ep = entry.get("entry_price")
    live = entry.get("live", {})
    lp = live.get("last_price")
    if ep and lp and ep > 0:
        return round((lp - ep) / ep * 100, 2)
    return None
