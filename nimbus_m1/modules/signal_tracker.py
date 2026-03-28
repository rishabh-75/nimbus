"""
modules/signal_tracker.py
──────────────────────────
Forward validation system for NIMBUS dual-mode signals.

Auto-logs every dual-mode signal when entry_triggered=True.
Resolves outcomes after 10 trading days by fetching the actual price.
Monitors live performance vs backtest expectations and flags regime drift.

DB: data/signal_tracker.db
Tables:
  signals     — every logged signal with full state snapshot
  outcomes    — resolved P&L after hold period

Workflow:
  1. DataManager calls log_signal() on every entry trigger  (auto)
  2. run_resolve.py fetches close prices for mature signals  (daily cron / manual)
  3. run_resolve.py --export produces a report for analysis  (weekly)
  4. Drift alerts surface in the dashboard status bar        (auto)
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
_DB_PATH = os.path.join(_DATA_DIR, "signal_tracker.db")

# ── Backtest expectations (from live NSE param sweep) ─────────────────────────
# Used for drift detection: if rolling 20-signal performance drops below
# the lower bound, something has changed and the system should alert.
EXPECTED = {
    "PRIMARY": {
        "win_rate": 73.4,
        "avg_return": 1.26,
        "sharpe": 1.17,
        "min_win_rate": 55.0,
        "min_avg_return": 0.0,
    },
    "SECONDARY": {
        "win_rate": 73.7,
        "avg_return": 1.32,
        "sharpe": 1.33,
        "min_win_rate": 50.0,
        "min_avg_return": 0.0,
    },
}

from modules.dual_mode import MAX_HOLD as HOLD_DAYS  # keep alias for compat

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    signal_date     TEXT NOT NULL,
    signal_ts       TEXT NOT NULL,
    mode            TEXT NOT NULL,
    segment         TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    sma_20          REAL,
    pct_from_sma    REAL,
    wr_value        REAL,
    adx_14          REAL,
    base_score      INTEGER,
    options_overlay INTEGER DEFAULT 0,
    filing_overlay  INTEGER DEFAULT 0,
    is_trap         INTEGER DEFAULT 0,
    dual_score      INTEGER,
    dual_label      TEXT,
    dual_sizing     TEXT,
    entry_reason    TEXT,
    input_interval  TEXT,
    daily_bars      INTEGER,
    -- Outcome fields (filled by resolver)
    resolved        INTEGER DEFAULT 0,
    resolve_date    TEXT,
    exit_price      REAL,
    pnl_pct         REAL,
    peak_price      REAL,
    max_drawdown    REAL,
    exit_reason     TEXT,
    UNIQUE(symbol, signal_date)
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_unresolved
ON signals(resolved, signal_date)
"""


# ══════════════════════════════════════════════════════════════════════════════
# DB INIT
# ══════════════════════════════════════════════════════════════════════════════


def _ensure_db() -> sqlite3.Connection:
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_SIGNALS)
    conn.execute(_CREATE_INDEX)
    conn.commit()
    return conn


def init_tracker():
    """Called once at startup."""
    _ensure_db().close()
    logger.info("Signal tracker DB ready: %s", _DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOGGING (called from DataManager)
# ══════════════════════════════════════════════════════════════════════════════


def log_signal(dm_sig) -> bool:
    """
    Log a dual-mode entry signal to the tracker DB.

    Called by DataManager when dm_sig.entry_triggered is True.
    Deduplicates by (symbol, signal_date) — one entry per symbol per day.

    Returns True if logged (new), False if duplicate or error.
    """
    if not getattr(dm_sig, "entry_triggered", False):
        return False

    try:
        conn = _ensure_db()
        today = datetime.date.today().isoformat()
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        conn.execute(
            """INSERT OR IGNORE INTO signals
               (symbol, signal_date, signal_ts, mode, segment,
                entry_price, sma_20, pct_from_sma,
                wr_value, adx_14,
                base_score, options_overlay, filing_overlay, is_trap,
                dual_score, dual_label, dual_sizing,
                entry_reason, input_interval, daily_bars)
               VALUES (?,?,?,?,?, ?,?,?, ?,?, ?,?,?,?, ?,?,?, ?,?,?)""",
            (
                dm_sig.symbol,
                today,
                ts,
                dm_sig.tier,
                "UNIFIED",
                dm_sig.close,
                dm_sig.sma_20,
                dm_sig.pct_from_sma,
                dm_sig.wr_30,
                dm_sig.adx_14,
                dm_sig.base_score,
                dm_sig.options_overlay,
                dm_sig.filing_overlay,
                1 if dm_sig.is_trap else 0,
                dm_sig.dual_score,
                dm_sig.dual_label,
                dm_sig.dual_sizing,
                dm_sig.entry_reason,
                dm_sig.input_interval,
                dm_sig.daily_bars_used,
            ),
        )
        inserted = conn.total_changes
        conn.commit()
        conn.close()

        if inserted:
            logger.info(
                "Signal logged: %s %s mode=%s score=%d",
                dm_sig.symbol, today, dm_sig.tier, dm_sig.dual_score,
            )
            return True
        return False

    except Exception as exc:
        logger.warning("Signal log failed: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOME RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════


def get_unresolved(max_age_days: int = 30) -> list[dict]:
    """Get signals that need outcome resolution."""
    try:
        conn = _ensure_db()
        cutoff = (
            datetime.date.today() - datetime.timedelta(days=max_age_days)
        ).isoformat()
        cursor = conn.execute(
            """SELECT * FROM signals
               WHERE resolved = 0 AND signal_date >= ?
               ORDER BY signal_date ASC""",
            (cutoff,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        logger.warning("get_unresolved failed: %s", exc)
        return []


def resolve_signal(
    signal_id: int,
    exit_price: float,
    peak_price: float = 0.0,
    exit_reason: str = "HOLD_COMPLETE",
) -> bool:
    """
    Mark a signal as resolved with its outcome.

    Called by run_resolve.py after fetching the close price
    HOLD_DAYS trading days after signal_date.
    """
    try:
        conn = _ensure_db()
        row = conn.execute(
            "SELECT entry_price FROM signals WHERE id = ?", (signal_id,)
        ).fetchone()
        if not row:
            conn.close()
            return False

        entry_price = row["entry_price"]
        pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0.0
        max_dd = (
            (min(exit_price, peak_price) / entry_price - 1) * 100
            if peak_price > 0 and entry_price > 0
            else 0.0
        )

        conn.execute(
            """UPDATE signals SET
                 resolved = 1,
                 resolve_date = ?,
                 exit_price = ?,
                 pnl_pct = ?,
                 peak_price = ?,
                 max_drawdown = ?,
                 exit_reason = ?
               WHERE id = ?""",
            (
                datetime.date.today().isoformat(),
                exit_price,
                round(pnl_pct, 4),
                peak_price,
                round(max_dd, 4),
                exit_reason,
                signal_id,
            ),
        )
        conn.commit()
        conn.close()

        logger.info(
            "Signal %d resolved: P&L=%+.2f%% exit_reason=%s",
            signal_id, pnl_pct, exit_reason,
        )
        return True

    except Exception as exc:
        logger.warning("resolve_signal failed: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACKING
# ══════════════════════════════════════════════════════════════════════════════


def get_performance(mode: str = None, last_n: int = None) -> dict:
    """
    Compute live performance stats from resolved signals.

    Args:
        mode: "A" or "B" or None for all
        last_n: only consider last N resolved signals (rolling window)

    Returns dict with win_rate, avg_pnl, n_signals, sharpe, etc.
    """
    try:
        conn = _ensure_db()
        query = "SELECT * FROM signals WHERE resolved = 1"
        params = []
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        query += " ORDER BY signal_date DESC"
        if last_n:
            query += " LIMIT ?"
            params.append(last_n)

        cursor = conn.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if not rows:
            return {"n_signals": 0, "status": "NO_DATA"}

        pnls = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]
        if not pnls:
            return {"n_signals": len(rows), "status": "NO_OUTCOMES"}

        import numpy as np
        pnls = np.array(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        avg = float(pnls.mean())
        std = float(pnls.std()) if len(pnls) > 1 else 0.0

        return {
            "n_signals": len(pnls),
            "win_rate": round(len(wins) / len(pnls) * 100, 1),
            "avg_pnl": round(avg, 3),
            "med_pnl": round(float(np.median(pnls)), 3),
            "total_pnl": round(float(pnls.sum()), 2),
            "avg_win": round(float(wins.mean()), 3) if len(wins) > 0 else 0,
            "avg_loss": round(float(losses.mean()), 3) if len(losses) > 0 else 0,
            "sharpe": round(avg / std * 7.1, 2) if std > 0 else 0,
            "profit_factor": round(
                float(wins.sum()) / abs(float(losses.sum())), 2
            ) if len(losses) > 0 and losses.sum() != 0 else 999,
            "status": "OK",
        }

    except Exception as exc:
        logger.warning("get_performance failed: %s", exc)
        return {"n_signals": 0, "status": f"ERROR: {exc}"}


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def check_drift(window: int = 20) -> dict:
    """
    Check if live performance has drifted from backtest expectations.

    Looks at the last `window` resolved signals per mode.
    Returns drift status per mode.

    Drift levels:
      OK        — within expectations
      WARNING   — win rate below expected but above threshold
      ALERT     — win rate or avg return below minimum threshold
      NO_DATA   — insufficient resolved signals
    """
    result = {}

    for mode_key in ("PRIMARY", "SECONDARY"):
        perf = get_performance(mode=mode_key, last_n=window)
        exp = EXPECTED[mode_key]

        if perf["n_signals"] < 10:
            result[mode_key] = {
                "status": "NO_DATA",
                "n": perf["n_signals"],
                "message": f"Need {10 - perf['n_signals']} more signals for Tier {mode_key}",
            }
            continue

        wr = perf["win_rate"]
        avg = perf["avg_pnl"]

        if wr < exp["min_win_rate"] or avg < exp["min_avg_return"]:
            status = "ALERT"
            message = (
                f"Tier {mode_key} DRIFT ALERT: "
                f"win={wr:.1f}% (expect ≥{exp['min_win_rate']:.0f}%), "
                f"avg={avg:+.2f}% (expect ≥{exp['min_avg_return']:.1f}%) "
                f"over last {perf['n_signals']} signals"
            )
        elif wr < exp["win_rate"] * 0.85:
            status = "WARNING"
            message = (
                f"Tier {mode_key} below expectation: "
                f"win={wr:.1f}% vs expected {exp['win_rate']:.0f}% "
                f"({perf['n_signals']} signals)"
            )
        else:
            status = "OK"
            message = (
                f"Tier {mode_key} on track: "
                f"win={wr:.1f}% avg={avg:+.2f}% "
                f"({perf['n_signals']} signals)"
            )

        result[mode_key] = {
            "status": status,
            "n": perf["n_signals"],
            "win_rate": wr,
            "avg_pnl": avg,
            "expected_wr": exp["win_rate"],
            "message": message,
            **perf,
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT (for upload to Claude)
# ══════════════════════════════════════════════════════════════════════════════


def export_report() -> str:
    """
    Generate a text report of all tracked signals and outcomes.
    Designed to be uploaded to Claude for validation.
    """
    lines = []
    lines.append("NIMBUS FORWARD VALIDATION REPORT")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Summary
    for mode_key in ("PRIMARY", "SECONDARY"):
        mode_name = "Primary" if mode_key == "PRIMARY" else "Secondary"
        perf = get_performance(mode=mode_key)
        exp = EXPECTED[mode_key]

        lines.append(f"═══ TIER {mode_key}: {mode_name} ═══")
        if perf["n_signals"] == 0:
            lines.append("  No resolved signals yet.")
        else:
            lines.append(f"  Resolved signals: {perf['n_signals']}")
            lines.append(f"  Win rate:   {perf['win_rate']:5.1f}%  (expected: {exp['win_rate']:.0f}%)")
            lines.append(f"  Avg P&L:    {perf['avg_pnl']:+.3f}%  (expected: {exp['avg_return']:+.2f}%)")
            lines.append(f"  Median P&L: {perf['med_pnl']:+.3f}%")
            lines.append(f"  Profit fac: {perf['profit_factor']:.2f}")
            lines.append(f"  Sharpe:     {perf['sharpe']:.2f}  (expected: {exp['sharpe']:.2f})")
        lines.append("")

    # Drift
    drift = check_drift()
    lines.append("═══ DRIFT STATUS ═══")
    for mode_key, d in drift.items():
        lines.append(f"  Mode {mode_key}: {d['status']} — {d['message']}")
    lines.append("")

    # Signal log
    try:
        conn = _ensure_db()
        cursor = conn.execute(
            "SELECT * FROM signals ORDER BY signal_date DESC LIMIT 100"
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
    except Exception:
        rows = []

    lines.append("═══ SIGNAL LOG (last 100) ═══")
    lines.append(
        f"{'Date':12s} {'Symbol':10s} {'Mode':4s} {'Score':5s} {'Label':8s} "
        f"{'Size':5s} {'Entry':>8s} {'Exit':>8s} {'P&L':>7s} {'Reason':15s}"
    )
    lines.append("-" * 95)

    for r in rows:
        pnl = f"{r['pnl_pct']:+.2f}%" if r["pnl_pct"] is not None else "—"
        exit_p = f"{r['exit_price']:.2f}" if r["exit_price"] else "—"
        reason = r.get("exit_reason", "") or ("pending" if not r["resolved"] else "")
        lines.append(
            f"{r['signal_date']:12s} {r['symbol']:10s} {r['mode']:4s} "
            f"{r['dual_score']:5d} {r['dual_label']:8s} {r['dual_sizing']:5s} "
            f"{r['entry_price']:8.2f} {exit_p:>8s} {pnl:>7s} {reason:15s}"
        )

    lines.append("")
    lines.append("── END OF REPORT ──")
    return "\n".join(lines)
