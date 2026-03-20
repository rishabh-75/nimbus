"""
tests/test_signal_tracker.py
─────────────────────────────
Tests for forward validation signal tracker.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import sqlite3

_PASS = 0; _FAIL = 0; _ERRORS = []
def _run(name, fn):
    global _PASS, _FAIL
    try: fn(); _PASS += 1; print(f"  ✓ {name}")
    except Exception as e: _FAIL += 1; _ERRORS.append((name, str(e))); print(f"  ✗ {name}: {e}")


def _clean_db():
    """Remove test DB before each group."""
    from modules.signal_tracker import _DB_PATH
    if os.path.exists(_DB_PATH):
        os.remove(_DB_PATH)


def _make_signal(symbol="TATASTEEL", mode="MR", entry_triggered=True, score=72):
    """Build a mock DualModeSignal for testing."""
    from types import SimpleNamespace
    return SimpleNamespace(
        symbol=symbol, mode=mode, tier="SECONDARY", segment="UNIFIED",
        close=1500.0, sma_20=1520.0, pct_from_sma=-1.3,
        wr_20=-45.0, wr_30=-42.0, adx_14=18.0,
        mfi=45.0, mfi_slope=2.0, rsi=35.0,
        dd_from_high=-6.0, red_streak=3, vol_ratio=1.2,
        bbw_slope=-0.5, bbw_pctl=40.0,
        base_score=score, options_overlay=3, filing_overlay=0,
        is_trap=False, dual_score=score + 3, dual_label="GOOD",
        dual_sizing="HALF", entry_triggered=entry_triggered,
        entry_reason="SECONDARY: WR(30)=-42 | -1.3% vs SMA | MFI=45",
        input_interval="1D", daily_bars_used=120,
    )


def test_init():
    print("\n── TRACKER INIT ──")
    from modules.signal_tracker import init_tracker, _DB_PATH

    def t_creates_db():
        _clean_db()
        init_tracker()
        assert os.path.exists(_DB_PATH), "DB file not created"
        conn = sqlite3.connect(_DB_PATH)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "signals" in tables
    _run("creates_db_with_signals_table", t_creates_db)

    def t_idempotent():
        init_tracker()  # should not crash on second call
        init_tracker()
    _run("init_idempotent", t_idempotent)


def test_logging():
    print("\n── SIGNAL LOGGING ──")
    from modules.signal_tracker import log_signal, _ensure_db

    def t_logs_entry():
        _clean_db()
        sig = _make_signal()
        result = log_signal(sig)
        assert result is True, "Should return True for new signal"
        conn = _ensure_db()
        row = conn.execute("SELECT * FROM signals WHERE symbol='TATASTEEL'").fetchone()
        conn.close()
        assert row is not None
        assert row["mode"] == "SECONDARY"
        assert row["entry_price"] == 1500.0
        assert row["dual_score"] == 75  # 72 + 3 options
    _run("logs_entry_signal", t_logs_entry)

    def t_deduplicates():
        sig = _make_signal()
        r1 = log_signal(sig)
        r2 = log_signal(sig)  # same symbol + same day = duplicate
        assert r2 is False, "Duplicate should return False"
    _run("deduplicates_same_day", t_deduplicates)

    def t_skips_no_entry():
        _clean_db()
        sig = _make_signal(entry_triggered=False)
        result = log_signal(sig)
        assert result is False
    _run("skips_non_entry_signal", t_skips_no_entry)

    def t_logs_different_symbols():
        _clean_db()
        log_signal(_make_signal(symbol="TATASTEEL"))
        log_signal(_make_signal(symbol="HDFCBANK"))
        conn = _ensure_db()
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        conn.close()
        assert count == 2
    _run("logs_different_symbols", t_logs_different_symbols)


def test_resolution():
    print("\n── OUTCOME RESOLUTION ──")
    from modules.signal_tracker import (
        log_signal, resolve_signal, get_unresolved, _ensure_db,
    )

    def t_resolve():
        _clean_db()
        log_signal(_make_signal(symbol="SBIN"))
        unresolved = get_unresolved()
        assert len(unresolved) == 1
        sid = unresolved[0]["id"]

        result = resolve_signal(sid, exit_price=1560.0, peak_price=1580.0)
        assert result is True

        conn = _ensure_db()
        row = conn.execute("SELECT * FROM signals WHERE id=?", (sid,)).fetchone()
        conn.close()
        assert row["resolved"] == 1
        assert row["exit_price"] == 1560.0
        assert abs(row["pnl_pct"] - 4.0) < 0.1  # (1560-1500)/1500 = 4%
    _run("resolves_with_pnl", t_resolve)

    def t_unresolved_excludes_resolved():
        unresolved = get_unresolved()
        assert len(unresolved) == 0, "Resolved signals should not appear"
    _run("unresolved_excludes_resolved", t_unresolved_excludes_resolved)


def test_performance():
    print("\n── PERFORMANCE TRACKING ──")
    from modules.signal_tracker import (
        log_signal, resolve_signal, get_performance, get_unresolved, _ensure_db,
    )

    def t_performance_stats():
        _clean_db()
        # Log and resolve several signals
        for i, (sym, exit_p) in enumerate([
            ("SYM01", 1530),  # +2%
            ("SYM02", 1545),  # +3%
            ("SYM03", 1470),  # -2%
            ("SYM04", 1560),  # +4%
            ("SYM05", 1440),  # -4%
        ]):
            sig = _make_signal(symbol=sym, score=70)
            log_signal(sig)
            # Manually set signal_date in past so it looks mature
            conn = _ensure_db()
            past_date = (datetime.date.today() - datetime.timedelta(days=20+i)).isoformat()
            conn.execute(
                "UPDATE signals SET signal_date=? WHERE symbol=?",
                (past_date, sym),
            )
            conn.commit()
            conn.close()
            sid = get_unresolved()[0]["id"] if get_unresolved() else None
            if sid:
                resolve_signal(sid, exit_price=exit_p, peak_price=max(exit_p, 1500))

        perf = get_performance(mode="SECONDARY")
        assert perf["n_signals"] == 5
        assert perf["win_rate"] == 60.0  # 3/5
        assert perf["avg_pnl"] > 0  # net positive
    _run("computes_performance_stats", t_performance_stats)


def test_drift():
    print("\n── DRIFT DETECTION ──")
    from modules.signal_tracker import check_drift

    def t_drift_no_data():
        _clean_db()
        drift = check_drift()
        for mode_key in ("PRIMARY", "SECONDARY"):
            assert drift[mode_key]["status"] == "NO_DATA"
    _run("no_data_returns_no_data", t_drift_no_data)


def test_export():
    print("\n── EXPORT ──")
    from modules.signal_tracker import export_report

    def t_export_format():
        _clean_db()
        report = export_report()
        assert "NIMBUS FORWARD VALIDATION REPORT" in report
        assert "PRIMARY" in report
        assert "SECONDARY" in report
        assert "DRIFT STATUS" in report
        assert "SIGNAL LOG" in report
    _run("export_has_all_sections", t_export_format)


def test_source_wiring():
    print("\n── SOURCE WIRING ──")

    def t_data_manager_logs():
        with open(os.path.join(os.path.dirname(__file__), "..", "ui", "data_manager.py")) as f:
            src = f.read()
        assert "log_signal" in src, "log_signal not called in data_manager"
        assert "signal_tracker" in src
    _run("data_manager_calls_log_signal", t_data_manager_logs)

    def t_main_window_inits():
        with open(os.path.join(os.path.dirname(__file__), "..", "ui", "main_window.py")) as f:
            src = f.read()
        assert "init_tracker" in src, "init_tracker not in main_window"
    _run("main_window_inits_tracker", t_main_window_inits)


if __name__ == "__main__":
    print("=" * 60)
    print("NIMBUS Signal Tracker Test Suite")
    print("=" * 60)

    test_init()
    test_logging()
    test_resolution()
    test_performance()
    test_drift()
    test_export()
    test_source_wiring()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    if _ERRORS:
        for n, e in _ERRORS:
            print(f"  {n}: {e}")
    sys.exit(1 if _FAIL > 0 else 0)
