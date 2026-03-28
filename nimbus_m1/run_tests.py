#!/usr/bin/env python3
"""
run_tests.py — Standalone test runner (no pytest required).
Uses Python's built-in unittest. Run from project root:

    python3 run_tests.py
"""

import sys
import os
import traceback
import datetime
from unittest.mock import patch
from types import SimpleNamespace

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# MINI TEST FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0
_ERRORS = []


def _run(name, fn):
    global _PASS, _FAIL
    try:
        fn()
        _PASS += 1
        print(f"  ✓ {name}")
    except AssertionError as e:
        _FAIL += 1
        _ERRORS.append((name, str(e)))
        print(f"  ✗ {name}: {e}")
    except Exception as e:
        _FAIL += 1
        _ERRORS.append((name, traceback.format_exc()))
        print(f"  ✗ {name}: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _make_ohlcv(n=100, base=1000.0, trend=0.002, vol=1e6, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.normal(0, 0.01)))
    c = np.array(closes)
    return pd.DataFrame({
        "Open": c * (1 + rng.normal(0, 0.005, n)),
        "High": c * (1 + rng.uniform(0.001, 0.015, n)),
        "Low": c * (1 - rng.uniform(0.001, 0.015, n)),
        "Close": c,
        "Volume": vol * (1 + rng.uniform(-0.3, 0.5, n)),
    }, index=idx)


def _expiry_str(days):
    return (datetime.date.today() + datetime.timedelta(days=days)).strftime("%d-%b-%Y")


def _make_opts(spot=1000, res=1100, sup=900, dte=10, ce_oi=50000, pe_oi=50000):
    strikes = sorted({res - 200, res - 100, res, sup, sup + 100, spot})
    exp = _expiry_str(dte)
    rows = []
    for s in strikes:
        rows.append({
            "Strike": float(s), "Expiry": exp,
            "CE_OI": float(ce_oi if s == res else 5000),
            "CE_IV": 20.0, "CE_LTP": max(1.0, (spot - s) * 0.1 + 10),
            "PE_OI": float(pe_oi if s == sup else 5000),
            "PE_IV": 22.0, "PE_LTP": max(1.0, (s - spot) * 0.1 + 10),
            "UnderlyingValue": spot,
        })
    return pd.DataFrame(rows)


def _ps(**kw):
    defaults = dict(
        wr_value=-12.0, wr_in_momentum=True, wr_phase="FRESH",
        wr_bars_since_cross50=2, bb_position="above_upper",
        position_state="RIDING_UPPER", vol_state="SQUEEZE",
        daily_bias="BULLISH", daily_bias_pct=3.5, daily_sma=980.0,
        bb_pct=0.85, bb_width_pctl=12.0, adv_cr=15.0,
        mfi_value=None, mfi_state="NEUTRAL", mfi_diverge=False,
        mfi_reliable=True, last_close=1000.0, upper=1010.0,
        mid=990.0, lower=970.0, entry_valid=True, bb_squeezing=True,
        bb_width_pct=2.0, wr_trend="rising",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ══════════════════════════════════════════════════════════════════════════════


def test_mfi_signals():
    """MFI divergence detection + state classification."""
    print("\n── MFI SIGNALS ──")
    from modules.indicators import PriceSignals, _compute_mfi_signals

    def _daily(mfi_vals, close_vals):
        n = len(mfi_vals)
        return pd.DataFrame({
            "Close": close_vals,
            "High": [c * 1.005 for c in close_vals],
            "Low": [c * 0.995 for c in close_vals],
            "Volume": [5e7] * n,
            "MFI_14": mfi_vals,
        }, index=pd.date_range("2024-06-01", periods=n, freq="1D"))

    # Bearish divergence: price near high, MFI falling 3-step
    def t_div_detected():
        mfi = [75, 73, 71, 68, 65, 62, 58, 55, 52, 50]
        close = [2830, 2840, 2850, 2855, 2860, 2865, 2870, 2860, 2855, 2850]
        ps = PriceSignals(adv_cr=25.0)
        _compute_mfi_signals(_daily(mfi, close), ps, close[-1])
        assert ps.mfi_diverge is True, f"Expected diverge=True, got {ps.mfi_diverge}"
    _run("bearish_divergence_detected (RELIANCE Jan 2024 pattern)", t_div_detected)

    def t_no_div_rising():
        mfi = [45, 48, 52, 55, 58, 62, 65, 68, 72, 75]
        close = [1500 + i * 10 for i in range(10)]
        ps = PriceSignals(adv_cr=20.0)
        _compute_mfi_signals(_daily(mfi, close), ps, close[-1])
        assert ps.mfi_diverge is False
    _run("no_divergence_when_mfi_rising", t_no_div_rising)

    def t_spike_guard():
        mfi = [65, 64, 63, 62, 61, 60, 59, 58, 85, 55]
        close = [5000] * 10
        ps = PriceSignals(adv_cr=20.0)
        _compute_mfi_signals(_daily(mfi, close), ps, 5000)
        assert ps.mfi_diverge is False, "Spike guard should block"
    _run("spike_guard_blocks_false_divergence", t_spike_guard)

    def t_unreliable_low_adv():
        mfi = [75, 73, 71, 68, 65, 62, 58, 55, 52, 50]
        ps = PriceSignals(adv_cr=2.0)
        _compute_mfi_signals(_daily(mfi, [100] * 10), ps, 100)
        assert ps.mfi_reliable is False
    _run("unreliable_when_adv_below_5cr", t_unreliable_low_adv)

    def t_state_thresholds():
        for mfi_now, expected in [(75, "STRONG"), (60, "RISING"), (50, "NEUTRAL"),
                                   (35, "FALLING"), (20, "WEAK")]:
            ps = PriceSignals(adv_cr=10.0)
            d = _daily([50]*9 + [mfi_now], [1000]*10)
            _compute_mfi_signals(d, ps, 1000)
            assert ps.mfi_state == expected, f"MFI {mfi_now}: expected {expected}, got {ps.mfi_state}"
    _run("mfi_state_classification_thresholds", t_state_thresholds)


def test_williams_r():
    """Williams %R phase classification."""
    print("\n── WILLIAMS %R PHASE ──")
    from modules.indicators import compute_price_signals, add_bollinger, add_williams_r

    def t_fresh():
        df = _make_ohlcv(100, trend=0.005)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)
        df.iloc[-1, df.columns.get_loc("WR")] = -10.0
        df.iloc[-2, df.columns.get_loc("WR")] = -15.0
        df.iloc[-3, df.columns.get_loc("WR")] = -45.0
        df.iloc[-4, df.columns.get_loc("WR")] = -55.0
        ps = compute_price_signals(df)
        assert ps.wr_phase == "FRESH", f"Expected FRESH, got {ps.wr_phase}"
    _run("fresh_phase (≤3 bars since -50 cross)", t_fresh)

    def t_developing():
        df = _make_ohlcv(100, trend=0.005)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)
        for i in range(8):
            df.iloc[-(i+1), df.columns.get_loc("WR")] = -15.0
        df.iloc[-9, df.columns.get_loc("WR")] = -55.0
        ps = compute_price_signals(df)
        assert ps.wr_phase == "DEVELOPING", f"Expected DEVELOPING, got {ps.wr_phase}"
    _run("developing_phase (4-10 bars)", t_developing)

    def t_late():
        df = _make_ohlcv(100, trend=0.005)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)
        for i in range(15):
            df.iloc[-(i+1), df.columns.get_loc("WR")] = -10.0
        df.iloc[-16, df.columns.get_loc("WR")] = -55.0
        ps = compute_price_signals(df)
        assert ps.wr_phase == "LATE", f"Expected LATE, got {ps.wr_phase}"
    _run("late_phase (>10 bars)", t_late)

    def t_none():
        df = _make_ohlcv(100, trend=-0.003)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)
        df.iloc[-1, df.columns.get_loc("WR")] = -60.0
        ps = compute_price_signals(df)
        assert ps.wr_phase == "NONE"
    _run("none_phase (WR below threshold)", t_none)


def test_daily_resample():
    """Daily resample handles missing columns + preserves volume."""
    print("\n── DAILY RESAMPLE ──")
    from modules.indicators import _resample_daily

    def t_no_open():
        idx = pd.date_range("2024-01-01", periods=50, freq="4h")
        df = pd.DataFrame({
            "High": np.random.uniform(100, 110, 50),
            "Low": np.random.uniform(90, 100, 50),
            "Close": np.random.uniform(95, 105, 50),
            "Volume": np.random.uniform(1e6, 5e6, 50),
        }, index=idx)
        daily = _resample_daily(df)
        assert not daily.empty, "Should not fail without Open column"
        assert "Close" in daily.columns
    _run("resample_without_open_column (sector pipeline)", t_no_open)

    def t_vol_summed():
        idx = pd.date_range("2024-01-01", periods=8, freq="4h")
        df = pd.DataFrame({
            "Open": [100]*8, "High": [105]*8, "Low": [95]*8,
            "Close": [100]*8, "Volume": [1e6]*8,
        }, index=idx)
        daily = _resample_daily(df)
        assert daily["Volume"].iloc[0] > 1e6, "Volume should be summed"
    _run("volume_summed_not_averaged", t_vol_summed)


def test_sector_rotation():
    """Sector RS and rotation classification."""
    print("\n── SECTOR ROTATION ──")
    from modules.sector_rotation import classify_rotation, _rs_price_ratio

    def t_rs_positive():
        sec = pd.Series([100.0]*10 + [110.0], index=pd.date_range("2024-01-01", periods=11))
        bench = pd.Series([100.0]*10 + [105.0], index=pd.date_range("2024-01-01", periods=11))
        rs = _rs_price_ratio(sec, bench, 10)
        assert rs is not None and rs > 0, f"RS should be positive, got {rs}"
    _run("rs_positive_outperformance", t_rs_positive)

    def t_rs_negative():
        sec = pd.Series([100.0]*10 + [102.0], index=pd.date_range("2024-01-01", periods=11))
        bench = pd.Series([100.0]*10 + [108.0], index=pd.date_range("2024-01-01", periods=11))
        rs = _rs_price_ratio(sec, bench, 10)
        assert rs is not None and rs < 0
    _run("rs_negative_underperformance", t_rs_negative)

    def t_all_quadrants():
        l, _ = classify_rotation(3.0, 2.0, 1.0)
        assert l == "LEADING"
        l, _ = classify_rotation(-3.0, -2.0, -1.0)
        assert l == "LAGGING"
        # WEAKENING: short-term positive but medium negative (fading leadership)
        l, _ = classify_rotation(2.0, -3.0, -2.0)
        assert l == "WEAKENING", f"Expected WEAKENING, got {l}"
        # IMPROVING: short-term negative but medium positive
        l, _ = classify_rotation(-2.0, 3.0, 2.0)
        assert l == "IMPROVING", f"Expected IMPROVING, got {l}"
    _run("all_four_quadrants_reachable", t_all_quadrants)

    def t_unknown_insufficient():
        l, _ = classify_rotation(5.0, None, None)
        assert l == "UNKNOWN"
        l, _ = classify_rotation(None, None, None)
        assert l == "UNKNOWN"
    _run("unknown_with_insufficient_data", t_unknown_insufficient)


def test_setup_classifier():
    """Setup classifier historical precedent tests."""
    print("\n── SETUP CLASSIFIER ──")
    from modules.setup_classifier import (
        classify_setup_v3, SetupType, OptionsSignalState, MomentumState,
    )

    def t_trap():
        st, _ = classify_setup_v3(
            viability_score=68, filing_variance=-15,
            filing_direction="BEARISH", filing_conviction=8,
            filing_category="RESULT",
        )
        assert st == SetupType.TRAP, f"Expected TRAP, got {st}"
    _run("trap_bearish_filing_high_score (DHFL pattern)", t_trap)

    def t_confirmed():
        opts = OptionsSignalState(
            gex_regime="Negative", pcr=1.4, pcr_trending="RISING", delta_bias="LONG",
        )
        mom = MomentumState(
            position_state="RIDING_UPPER", wr_in_momentum=True, wr_phase="DEVELOPING",
        )
        st, _ = classify_setup_v3(
            viability_score=72, filing_variance=12,
            filing_direction="BULLISH", filing_conviction=7,
            filing_category="RESULT", opts=opts, mom=mom,
        )
        assert st == SetupType.CONFIRMED
    _run("confirmed_full_alignment (TATAMOTORS Aug 2023)", t_confirmed)

    def t_pre_breakout():
        opts = OptionsSignalState(
            gex_regime="Negative", pcr=1.5, pcr_trending="RISING", delta_bias="LONG",
        )
        mom = MomentumState(position_state="CONSOLIDATING", vol_state="SQUEEZE")
        st, _ = classify_setup_v3(
            viability_score=55, filing_variance=10,
            filing_direction="BULLISH", filing_conviction=7,
            filing_category="CORP_ACTION", opts=opts, mom=mom,
        )
        assert st == SetupType.PRE_BREAKOUT
    _run("pre_breakout_before_technical_confirmation", t_pre_breakout)

    def t_neutral():
        st, _ = classify_setup_v3(viability_score=52)
        assert st == SetupType.NEUTRAL
    _run("neutral_no_signals", t_neutral)

    def t_options_only():
        opts = OptionsSignalState(
            gex_regime="Negative", pcr=1.5, pcr_trending="RISING",
            delta_bias="LONG", iv_skew="CALL_CHEAP",
        )
        st, _ = classify_setup_v3(viability_score=55, filing_variance=None, opts=opts)
        assert st == SetupType.OPTIONS_ONLY
    _run("options_only_strong_no_filing", t_options_only)


def test_viability_edge_cases():
    """Viability scoring hard overrides."""
    print("\n── VIABILITY EDGE CASES ──")
    from modules.analytics import analyze, analyze_price_only

    def t_bearish_avoid():
        df = _make_opts(spot=1000, dte=10, pe_oi=60000, ce_oi=20000)
        ctx = analyze(df, 1000, price_signals=_ps(daily_bias="BEARISH"))
        assert ctx.viability.label == "AVOID"
        assert ctx.viability.sizing == "SKIP"
    _run("bearish_bias_forces_AVOID", t_bearish_avoid)

    def t_mid_band_exit():
        df = _make_opts(spot=1000, dte=10, pe_oi=60000, ce_oi=20000)
        ctx = analyze(df, 1000, price_signals=_ps(position_state="MID_BAND_BROKEN"))
        assert ctx.viability.sizing == "SKIP"
        assert ctx.viability.label == "AVOID"
    _run("mid_band_broken_forces_EXIT", t_mid_band_exit)

    def t_price_only_cap():
        ctx = analyze_price_only(
            spot=1000, price_signals=_ps(wr_phase="FRESH", vol_state="SQUEEZE"),
        )
        assert ctx.viability.score <= 70, f"Score {ctx.viability.score} exceeds 70 cap"
    _run("analyze_price_only_capped_at_70", t_price_only_cap)

    def t_score_bounds():
        df = _make_opts(spot=1000, dte=10)
        ctx1 = analyze(df, 1000, price_signals=_ps(
            daily_bias="BULLISH", vol_state="SQUEEZE", wr_phase="FRESH",
            mfi_value=80, mfi_state="STRONG", mfi_reliable=True,
        ))
        assert 0 <= ctx1.viability.score <= 100
        ctx2 = analyze(df, 1000, price_signals=_ps(
            daily_bias="BEARISH", position_state="MID_BAND_BROKEN",
            wr_in_momentum=False, wr_phase="NONE", wr_value=-80,
            mfi_value=15, mfi_state="WEAK", mfi_reliable=True,
        ))
        assert 0 <= ctx2.viability.score <= 100
    _run("score_always_0_to_100", t_score_bounds)


def test_data_quality_gates():
    """Sector data quality validation."""
    print("\n── DATA QUALITY GATES ──")
    from modules.sector_rotation import _validate_series

    def t_reject_few():
        s = pd.Series([100.0]*50, index=pd.date_range("2024-01-01", periods=50))
        assert _validate_series(s, "TEST") is False
    _run("reject_fewer_than_70_bars", t_reject_few)

    def t_accept_enough():
        s = pd.Series([100.0]*80, index=pd.date_range("2024-01-01", periods=80))
        assert _validate_series(s, "TEST") is True
    _run("accept_80_bars", t_accept_enough)

    def t_reject_nan():
        vals = [100.0]*90 + [float("nan")]*10
        s = pd.Series(vals, index=pd.date_range("2024-01-01", periods=100))
        assert _validate_series(s, "TEST") is False
    _run("reject_10pct_nan", t_reject_nan)


def test_position_state():
    """BB position state classification."""
    print("\n── POSITION STATE ──")
    from modules.indicators import _position_state

    def t_mid_broken():
        df = pd.DataFrame({
            "Close": [100]*9 + [103],
            "BB_Upper": [110]*10,
            "BB_Mid": [105]*10,
        })
        assert _position_state(df) == "MID_BAND_BROKEN"
    _run("mid_band_broken", t_mid_broken)

    def t_first_dip():
        df = pd.DataFrame({
            "Close": [110, 112, 115, 118, 114],
            "BB_Upper": [108, 110, 112, 115, 116],
            "BB_Mid": [100, 102, 104, 106, 108],
        })
        assert _position_state(df) == "FIRST_DIP"
    _run("first_dip_detection", t_first_dip)


def test_freshness_state():
    """DataFreshnessState includes CACHED."""
    print("\n── FRESHNESS STATE ──")

    def t_cached_exists():
        try:
            from ui.workers import DataFreshnessState
        except ImportError:
            # Verify directly from source file instead
            with open(os.path.join(os.path.dirname(__file__), "ui", "workers.py")) as f:
                src = f.read()
            assert 'CACHED = "CACHED"' in src, "CACHED state not found in workers.py"
            print("    (verified via source — PyQt6 not available)")
            return
        assert hasattr(DataFreshnessState, "CACHED")
        assert DataFreshnessState.CACHED.value == "CACHED"
    _run("CACHED_state_exists", t_cached_exists)


def test_kite_removal():
    """KiteTicker code removed from main_window."""
    print("\n── KITE REMOVAL ──")

    def t_no_kite_in_main():
        with open(os.path.join(os.path.dirname(__file__), "ui", "main_window.py")) as f:
            src = f.read()
        assert "KiteSessionManager" not in src, "KiteSessionManager still in main_window"
        assert "TickerWorker" not in src, "TickerWorker still in main_window"
        assert "TickAggregator" not in src, "TickAggregator still in main_window"
        assert "_on_kite_session_ready" not in src, "Kite handler still in main_window"
        assert "_on_tick_received" not in src, "Tick handler still in main_window"
    _run("no_kite_imports_or_methods_in_main_window", t_no_kite_in_main)


def test_session_guard():
    """Session guard enforcement in data_manager."""
    print("\n── SESSION GUARD ──")

    def t_guard_in_refresh():
        with open(os.path.join(os.path.dirname(__file__), "ui", "data_manager.py")) as f:
            src = f.read()
        assert "is_market_open()" in src, "is_market_open() not called in data_manager"
        assert "CACHED" in src, "CACHED state not referenced in data_manager"
        assert "market closed" in src.lower(), "No market-closed guard logic found"
    _run("session_guard_in_data_manager_refresh", t_guard_in_refresh)

    def t_guard_in_scanner():
        with open(os.path.join(os.path.dirname(__file__), "ui", "workers.py")) as f:
            src = f.read()
        assert "is_market_open" in src, "Session guard not in ScanWorker"
    _run("session_guard_in_scan_worker", t_guard_in_scanner)

    def t_is_market_open_logic():
        """Verify is_market_open() correctly implements NSE hours."""
        from modules.data import is_market_open
        # This calls real datetime so just verify it returns a bool
        result = is_market_open()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    _run("is_market_open_returns_bool", t_is_market_open_logic)


def test_conviction_ui():
    """Conviction score displayed in Market Context tab."""
    print("\n── CONVICTION SCORE UI ──")

    def t_conv_column():
        with open(os.path.join(os.path.dirname(__file__), "ui", "market_context_tab.py")) as f:
            src = f.read()
        assert '"Conv"' in src, "Conv column header not found in market_context_tab"
        assert "conviction_score" in src, "conviction_score not referenced in UI"
    _run("conviction_column_in_sector_table", t_conv_column)

    def t_etf_panel_fix():
        with open(os.path.join(os.path.dirname(__file__), "ui", "components.py")) as f:
            src = f.read()
        # LevelRow name width should be 90 (widened from 72)
        assert "setFixedWidth(90)" in src, "LevelRow name width not widened to 90"
    _run("etf_level_row_widened", t_etf_panel_fix)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NIMBUS Signal Chain Test Suite")
    print("=" * 60)

    test_mfi_signals()
    test_williams_r()
    test_daily_resample()
    test_sector_rotation()
    test_setup_classifier()
    test_viability_edge_cases()
    test_data_quality_gates()
    test_position_state()
    test_freshness_state()
    test_kite_removal()
    test_session_guard()
    test_conviction_ui()

    print("\n" + "=" * 60)
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)

    if _ERRORS:
        print("\nFAILURES:")
        for name, err in _ERRORS:
            print(f"\n  {name}:")
            for line in err.split("\n")[:5]:
                print(f"    {line}")

    sys.exit(1 if _FAIL > 0 else 0)
