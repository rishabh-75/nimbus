"""
ui/watchlist_db.py
──────────────────
SQLite-backed watchlist for NIMBUS Qt.

Same public function signatures as modules/watchlist.py:
    load_watchlist, save_watchlist, add_entry, remove_entry,
    update_entry, get_entry

Auto-migrates from data/watchlist.json → data/watchlist.db on first run.

Schema:
    CREATE TABLE watchlist (
        symbol       TEXT PRIMARY KEY,
        added_at     TEXT,
        notes        TEXT DEFAULT '',
        entry_price  REAL,
        stop_price   REAL,
        target_price REAL,
        tags         TEXT DEFAULT '[]'
    )

Tags stored as JSON string (TEXT column).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_DB_PATH  = os.path.join(_DATA_DIR, "watchlist.db")
_JSON_PATH = os.path.join(_DATA_DIR, "watchlist.json")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS watchlist (
    symbol       TEXT PRIMARY KEY,
    added_at     TEXT,
    notes        TEXT DEFAULT '',
    entry_price  REAL,
    stop_price   REAL,
    target_price REAL,
    tags         TEXT DEFAULT '[]'
)
"""


# ══════════════════════════════════════════════════════════════════════════════
# INIT + MIGRATION
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_db() -> sqlite3.Connection:
    """Ensure data/ dir exists, DB created, and return connection."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def _auto_migrate():
    """
    If watchlist.json exists and watchlist.db does not (or is empty),
    migrate entries from JSON into SQLite on first run.
    """
    if not os.path.exists(_JSON_PATH):
        return

    conn = _ensure_db()
    cursor = conn.execute("SELECT COUNT(*) FROM watchlist")
    count = cursor.fetchone()[0]

    if count > 0:
        # DB already has data — skip migration
        conn.close()
        return

    try:
        with open(_JSON_PATH) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            entries = []

        migrated = 0
        for e in entries:
            sym = e.get("symbol", "").strip().upper()
            if not sym:
                continue
            tags = json.dumps(e.get("tags", []))
            conn.execute(
                """INSERT OR IGNORE INTO watchlist
                   (symbol, added_at, notes, entry_price, stop_price, target_price, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    sym,
                    e.get("added_at", datetime.datetime.now().isoformat()),
                    e.get("notes", ""),
                    e.get("entry_price"),
                    e.get("stop_price"),
                    e.get("target_price"),
                    tags,
                ),
            )
            migrated += 1

        conn.commit()
        logger.info("Migrated %d entries from watchlist.json → watchlist.db", migrated)
    except Exception as exc:
        logger.warning("Watchlist JSON migration failed: %s", exc)
    finally:
        conn.close()


def init_watchlist():
    """Called once at startup to ensure DB exists and run migration."""
    _ensure_db().close()
    _auto_migrate()
    logger.info("Watchlist DB ready: %s", _DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# ROW CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to the same dict format as modules/watchlist.py."""
    d = dict(row)
    # Parse tags JSON back to list
    try:
        d["tags"] = json.loads(d.get("tags", "[]"))
    except (json.JSONDecodeError, TypeError):
        d["tags"] = []
    return d


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API (matches modules/watchlist.py signatures)
# ══════════════════════════════════════════════════════════════════════════════

def load_watchlist() -> list[dict]:
    """Load all watchlist entries from SQLite. Returns [] on any error."""
    try:
        conn = _ensure_db()
        cursor = conn.execute(
            "SELECT * FROM watchlist ORDER BY added_at DESC"
        )
        entries = [_row_to_dict(row) for row in cursor.fetchall()]
        conn.close()
        return entries
    except Exception as exc:
        logger.warning("Could not load watchlist: %s", exc)
        return []


def save_watchlist(entries: list[dict]) -> None:
    """
    Replace entire watchlist with the given entries.
    (For compatibility with modules/watchlist.py which overwrites the file.)
    """
    try:
        conn = _ensure_db()
        conn.execute("DELETE FROM watchlist")
        for e in entries:
            sym = e.get("symbol", "").strip().upper()
            if not sym:
                continue
            tags = json.dumps(e.get("tags", []))
            conn.execute(
                """INSERT OR REPLACE INTO watchlist
                   (symbol, added_at, notes, entry_price, stop_price, target_price, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    sym,
                    e.get("added_at", datetime.datetime.now().isoformat()),
                    e.get("notes", ""),
                    e.get("entry_price"),
                    e.get("stop_price"),
                    e.get("target_price"),
                    tags,
                ),
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.error("Could not save watchlist: %s", exc)


def add_entry(
    entries: list[dict],
    symbol: str,
    entry_price: Optional[float] = None,
    stop_price:  Optional[float] = None,
    target_price: Optional[float] = None,
    notes: str = "",
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """
    Add or update an entry. Saves to SQLite. Returns updated list.
    Signature matches modules/watchlist.add_entry().
    """
    symbol = symbol.strip().upper()

    # Check for existing entry in the list
    for e in entries:
        if e["symbol"] == symbol:
            if entry_price  is not None: e["entry_price"]  = entry_price
            if stop_price   is not None: e["stop_price"]   = stop_price
            if target_price is not None: e["target_price"] = target_price
            if notes:                    e["notes"]         = notes
            if tags is not None:         e["tags"]          = tags
            save_watchlist(entries)
            return entries

    new_entry = {
        "symbol":       symbol,
        "added_at":     datetime.datetime.now().isoformat(),
        "notes":        notes,
        "entry_price":  entry_price,
        "stop_price":   stop_price,
        "target_price": target_price,
        "tags":         tags or [],
    }
    entries.append(new_entry)
    save_watchlist(entries)
    return entries


def remove_entry(entries: list[dict], symbol: str) -> list[dict]:
    """Remove symbol from watchlist. Saves to SQLite."""
    symbol = symbol.strip().upper()
    entries = [e for e in entries if e["symbol"] != symbol]
    save_watchlist(entries)
    return entries


def update_entry(entries: list[dict], symbol: str, **kwargs) -> list[dict]:
    """Update fields on an existing entry. Saves to SQLite."""
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
