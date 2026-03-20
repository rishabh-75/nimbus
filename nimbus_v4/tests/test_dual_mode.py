"""
tests/test_dual_mode.py — Tests for unified mean-reversion scoring engine.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd

_PASS = 0; _FAIL = 0; _ERRORS = []
def _run(name, fn):
    global _PASS, _FAIL
    try: fn(); _PASS += 1; print(f"  ✓ {name}")
    except Exception as e: _FAIL += 1; _ERRORS.append((name, str(e))); print(f"  ✗ {name}: {e}")

def _make_daily(n=120, trend=-0.002, vol=0.012, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    closes = 1000 * np.cumprod(1 + rng.normal(trend, vol, n))
    return pd.DataFrame({
        "Open": closes*0.999, "High": closes*1.006,
        "Low": closes*0.994, "Close": closes,
        "Volume": 5e6*np.ones(n),
    }, index=idx)

def _make_4h(n=120, trend=-0.002, vol=0.012, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2025-01-01", periods=n//2)
    times = []
    for d in idx:
        times.append(d.replace(hour=9, minute=15))
        times.append(d.replace(hour=13, minute=15))
    idx_4h = pd.DatetimeIndex(times[:n])
    closes = 1000 * np.cumprod(1 + rng.normal(trend, vol, len(idx_4h)))
    return pd.DataFrame({
        "Open": closes*0.999, "High": closes*1.006,
        "Low": closes*0.994, "Close": closes,
        "Volume": 5e6*np.ones(len(idx_4h)),
    }, index=idx_4h)

def test_signal_dataclass():
    print("\n── SIGNAL DATACLASS ──")
    from modules.dual_mode import DualModeSignal
    def t(): sig = DualModeSignal(); assert sig.tier == "NONE"; assert sig.mode == "MR"
    _run("default_values", t)
    def t2():
        sig = DualModeSignal(tier="PRIMARY", dual_score=80)
        assert sig.tier == "PRIMARY"; assert sig.dual_score == 80
    _run("custom_values", t2)

def test_compute_basic():
    print("\n── COMPUTE BASIC ──")
    from modules.dual_mode import compute_dual_mode
    def t_bearish():
        df = _make_daily(120, trend=-0.003)
        sig = compute_dual_mode(df, symbol="TEST")
        assert sig.data_sufficient
        assert sig.wr_30 < 0
        assert sig.mfi > 0
        assert sig.dual_score > 0
    _run("bearish_trend_produces_signal", t_bearish)

    def t_bullish():
        df = _make_daily(120, trend=0.003)
        sig = compute_dual_mode(df, symbol="TEST")
        assert sig.data_sufficient
        assert sig.above_sma  # above SMA in uptrend
    _run("bullish_trend_above_sma", t_bullish)

    def t_insufficient():
        df = _make_daily(15)
        sig = compute_dual_mode(df, symbol="TEST")
        assert not sig.data_sufficient
    _run("insufficient_data", t_insufficient)

def test_entry_tiers():
    print("\n── ENTRY TIERS ──")
    from modules.dual_mode import compute_dual_mode
    def t_core():
        # Deep pullback: WR should be oversold, below SMA, MFI exists
        df = _make_daily(120, trend=-0.004, seed=100)
        sig = compute_dual_mode(df, symbol="TEST")
        assert sig.data_sufficient
        # Check core conditions are evaluated
        assert isinstance(sig.core_met, bool)
        assert isinstance(sig.primary_met, bool)
        assert isinstance(sig.secondary_met, bool)
    _run("core_conditions_evaluated", t_core)

    def t_tier_labels():
        from modules.dual_mode import DualModeSignal, _label_and_sizing
        sig = DualModeSignal(tier="PRIMARY", is_trap=False)
        l, s = _label_and_sizing(80, sig)
        assert s == "FULL", f"PRIMARY+80 should be FULL, got {s}"
        sig2 = DualModeSignal(tier="SECONDARY", is_trap=False)
        l2, s2 = _label_and_sizing(80, sig2)
        assert s2 == "HALF", f"SECONDARY+80 should be HALF, got {s2}"
    _run("tier_sizing_primary_full_secondary_half", t_tier_labels)

    def t_trap():
        from modules.dual_mode import DualModeSignal, _label_and_sizing
        sig = DualModeSignal(tier="PRIMARY", is_trap=True)
        l, s = _label_and_sizing(80, sig)
        assert l == "AVOID" and s == "SKIP"
    _run("trap_overrides_all", t_trap)

def test_scoring():
    print("\n── SCORING ──")
    from modules.dual_mode import DualModeSignal, _score
    def t_deep_oversold():
        sig = DualModeSignal(wr_30=-80, above_sma=False, pct_from_sma=-6.0,
                             mfi=55, dd_from_high=-12, red_streak=6, vol_ratio=1.8, bbw_pctl=20)
        s = _score(sig)
        assert s >= 80, f"Perfect setup should be ≥80, got {s}"
    _run("deep_oversold_high_score", t_deep_oversold)

    def t_shallow():
        sig = DualModeSignal(wr_30=-20, above_sma=True, pct_from_sma=1.0,
                             mfi=25, dd_from_high=-1, red_streak=0, vol_ratio=0.3, bbw_pctl=80)
        s = _score(sig)
        assert s < 30, f"No setup should be <30, got {s}"
    _run("no_setup_low_score", t_shallow)

    def t_mfi_matters():
        sig_strong = DualModeSignal(wr_30=-50, above_sma=False, pct_from_sma=-3.0,
                                    mfi=60, dd_from_high=-5, red_streak=3)
        sig_weak = DualModeSignal(wr_30=-50, above_sma=False, pct_from_sma=-3.0,
                                  mfi=20, dd_from_high=-5, red_streak=3)
        assert _score(sig_strong) > _score(sig_weak), "MFI strong should beat MFI weak"
    _run("mfi_strong_beats_weak", t_mfi_matters)

    def t_dd_matters():
        sig_deep = DualModeSignal(wr_30=-40, above_sma=False, pct_from_sma=-2.0,
                                  mfi=40, dd_from_high=-12)
        sig_shallow = DualModeSignal(wr_30=-40, above_sma=False, pct_from_sma=-2.0,
                                     mfi=40, dd_from_high=-2)
        assert _score(sig_deep) > _score(sig_shallow), "Deep DD should beat shallow"
    _run("deep_dd_beats_shallow", t_dd_matters)

    def t_streak_matters():
        sig_ext = DualModeSignal(wr_30=-40, above_sma=False, pct_from_sma=-2.0,
                                 mfi=40, red_streak=6)
        sig_none = DualModeSignal(wr_30=-40, above_sma=False, pct_from_sma=-2.0,
                                  mfi=40, red_streak=0)
        assert _score(sig_ext) > _score(sig_none)
    _run("red_streak_adds_score", t_streak_matters)

    def t_bounded():
        sig_max = DualModeSignal(wr_30=-99, above_sma=False, pct_from_sma=-15.0,
                                 mfi=90, dd_from_high=-20, red_streak=10, vol_ratio=3.0, bbw_pctl=5)
        sig_min = DualModeSignal(wr_30=0, above_sma=True, pct_from_sma=10.0,
                                 mfi=5, dd_from_high=0, red_streak=0, vol_ratio=0.1, bbw_pctl=99)
        assert 0 <= _score(sig_max) <= 100
        assert 0 <= _score(sig_min) <= 100
    _run("score_bounded_0_100", t_bounded)

def test_options_overlay():
    print("\n── OPTIONS OVERLAY ──")
    from types import SimpleNamespace
    from modules.dual_mode import DualModeSignal, _options_overlay
    def _ctx(pcr=1.0, gex="Neutral", sup=None):
        return SimpleNamespace(
            walls=SimpleNamespace(pcr_oi=pcr, support_pct=sup),
            gex=SimpleNamespace(regime=gex, net_gex=0),
        )
    sig = DualModeSignal()
    def t_bear():
        ov, _ = _options_overlay(_ctx(pcr=0.5, gex="Negative"), sig)
        assert ov <= -7, f"Bearish PCR+neg GEX should be ≤-7, got {ov}"
    _run("bearish_pcr_neg_gex_compound", t_bear)

    def t_bull():
        ov, _ = _options_overlay(_ctx(pcr=1.5, gex="Negative"), sig)
        assert ov >= 4, f"Bullish PCR+neg GEX should be ≥4, got {ov}"
    _run("bullish_pcr_neg_gex_bounce", t_bull)

    def t_none():
        ov, _ = _options_overlay(None, sig)
        assert ov == 0
    _run("no_options_zero", t_none)

    def t_bounded():
        ov1, _ = _options_overlay(_ctx(pcr=0.1, gex="Negative"), sig)
        ov2, _ = _options_overlay(_ctx(pcr=2.0, gex="Negative"), sig)
        assert -10 <= ov1 <= 10 and -10 <= ov2 <= 10
    _run("overlay_bounded", t_bounded)

def test_filing_overlay():
    print("\n── FILING OVERLAY ──")
    from types import SimpleNamespace
    from modules.dual_mode import _filing_overlay
    def t_trap():
        fv = SimpleNamespace(badge_color="BEARISH", conviction=8, variance=0)
        ov, _, trap = _filing_overlay(fv, 70)
        assert trap is True
    _run("trap_detection", t_trap)
    def t_bullish():
        fv = SimpleNamespace(badge_color="BULLISH", conviction=8, variance=0)
        ov, _, trap = _filing_overlay(fv, 50)
        assert ov == 7 and not trap
    _run("bullish_high_conviction", t_bullish)

def test_exit():
    print("\n── EXIT LOGIC ──")
    from modules.dual_mode import DualModeSignal, check_exit
    def t_profit():
        sig = DualModeSignal(close=1050); reason = check_exit(sig, 1000, 5)
        assert reason == "PROFIT_TARGET"
    _run("profit_target_5pct", t_profit)
    def t_bbw():
        sig = DualModeSignal(close=1010, bbw_contracting=True, above_sma=True)
        reason = check_exit(sig, 1000, 10)
        assert reason == "BBW_CONTRACT"
    _run("bbw_contraction_exit", t_bbw)
    def t_max():
        sig = DualModeSignal(close=990); reason = check_exit(sig, 1000, 25)
        assert reason == "MAX_HOLD"
    _run("max_hold_25d", t_max)
    def t_hold():
        sig = DualModeSignal(close=1030); reason = check_exit(sig, 1000, 3)
        assert reason == "", "Should hold early"
    _run("no_early_exit", t_hold)

def test_4h_resample():
    print("\n── 4H RESAMPLE ──")
    from modules.dual_mode import compute_dual_mode
    def t_4h():
        df = _make_4h(120, trend=-0.003)
        sig = compute_dual_mode(df, symbol="TEST")
        assert sig.input_interval == "4H"
        assert sig.data_sufficient
    _run("4h_detected_and_resampled", t_4h)
    def t_daily():
        df = _make_daily(120, trend=-0.003)
        sig = compute_dual_mode(df, symbol="TEST")
        assert sig.input_interval == "1D"
    _run("daily_passthrough", t_daily)

def test_ui_wiring():
    print("\n── UI WIRING ──")
    def t_dm():
        with open(os.path.join(os.path.dirname(__file__), "..", "ui", "data_manager.py")) as f:
            src = f.read()
        assert "log_signal" in src
        assert "compute_dual_mode" in src
    _run("data_manager_calls_log_signal", t_dm)

    def t_dash():
        with open(os.path.join(os.path.dirname(__file__), "..", "ui", "dashboard_tab.py")) as f:
            src = f.read()
        assert "on_dual_mode_updated" in src
        assert "PRIMARY" in src
        assert "SECONDARY" in src
    _run("dashboard_uses_tiers", t_dash)

    def t_scanner():
        with open(os.path.join(os.path.dirname(__file__), "..", "ui", "scanner_tab.py")) as f:
            src = f.read()
        assert "dm_tier" in src
        assert "dm_mfi" in src
    _run("scanner_has_new_columns", t_scanner)

if __name__ == "__main__":
    print("=" * 60)
    print("NIMBUS Unified Mean-Reversion Test Suite")
    print("=" * 60)
    test_signal_dataclass()
    test_compute_basic()
    test_entry_tiers()
    test_scoring()
    test_options_overlay()
    test_filing_overlay()
    test_exit()
    test_4h_resample()
    test_ui_wiring()
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    if _ERRORS:
        for n, e in _ERRORS: print(f"  {n}: {e}")
    sys.exit(1 if _FAIL > 0 else 0)
