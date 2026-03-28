#!/usr/bin/env python3
"""
tests/test_validation.py
─────────────────────────
Real-world scenario validation for the dual-mode scoring system.

Tests realistic combinations of price state + options state + filing state
and verifies the final score, label, sizing, and verdict are coherent.

Each test case is a named market scenario with expected behavior.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace
from modules.dual_mode import (
    DualModeSignal, _score_mode_a, _score_mode_b,
    _options_overlay, _filing_overlay, _label_and_sizing,
)

_PASS = 0; _FAIL = 0; _ERRORS = []
def _run(name, fn):
    global _PASS, _FAIL
    try: fn(); _PASS += 1; print(f"  ✓ {name}")
    except Exception as e: _FAIL += 1; _ERRORS.append((name, str(e))); print(f"  ✗ {name}: {e}")


def _make_ctx(pcr=None, gex_regime="Neutral", net_gex=0, res_pct=None, sup_pct=None):
    """Build mock OptionsContext."""
    if pcr is None and gex_regime == "Neutral" and net_gex == 0:
        return None
    return SimpleNamespace(
        walls=SimpleNamespace(pcr_oi=pcr, resistance_pct=res_pct, support_pct=sup_pct, pcr_sentiment=""),
        gex=SimpleNamespace(regime=gex_regime, net_gex=net_gex),
    )


def _full_score(mode, segment, wr_20=-50, wr_30=-50, wr_cross=99,
                adx=20, above_sma=False, pct_sma=-2.0,
                pcr=None, gex="Neutral", net_gex=0, res_pct=None, sup_pct=None,
                filing_dir=None, filing_conv=0, base_override=None):
    """Compute full dual-mode score for a scenario."""
    sig = DualModeSignal(
        mode=mode, segment=segment,
        wr_20=wr_20, wr_30=wr_30, wr_20_cross_bars=wr_cross,
        adx_14=adx, adx_trending=(adx >= 20),
        above_sma=above_sma, pct_from_sma=pct_sma,
        close=1000.0, sma_20=1000 / (1 + pct_sma/100),
    )

    base = base_override if base_override else (
        _score_mode_a(sig) if mode == "A" else _score_mode_b(sig)
    )
    sig.base_score = base

    ctx = _make_ctx(pcr, gex, net_gex, res_pct, sup_pct)
    opt_ov, opt_det = _options_overlay(ctx, sig)
    sig.options_overlay = opt_ov
    sig.options_detail = opt_det

    fv = None
    if filing_dir:
        fv = SimpleNamespace(badge_color=filing_dir, conviction=filing_conv, variance=0)
    fil_ov, fil_det, is_trap = _filing_overlay(fv, base)
    sig.filing_overlay = fil_ov
    sig.filing_detail = fil_det
    sig.is_trap = is_trap

    final = max(0, min(100, base + opt_ov + fil_ov))
    if is_trap:
        final = min(final, 30)
    label, sizing = _label_and_sizing(final, sig)

    return {
        "base": base, "opt": opt_ov, "fil": fil_ov, "final": final,
        "label": label, "sizing": sizing, "trap": is_trap,
        "opt_detail": opt_det, "fil_detail": fil_det,
    }


# ═════════════════════════════════════════════════════════════════
# MODE B: MEAN REVERSION SCENARIOS
# ═════════════════════════════════════════════════════════════════

def test_mode_b_scenarios():
    print("\n── MODE B: MEAN REVERSION SCENARIOS ──")

    def t_ideal_setup():
        """Deep oversold, below SMA, ranging, bullish options."""
        r = _full_score("B", "CYCLICAL", wr_30=-75, adx=12,
                        pct_sma=-4.0, pcr=1.5, gex="Positive", net_gex=500)
        assert r["base"] >= 80, f"Deep oversold + ranging base should be ≥80, got {r['base']}"
        assert r["opt"] > 0, f"Bullish PCR should give positive overlay, got {r['opt']}"
        assert r["final"] >= 80, f"Ideal setup should score ≥80, got {r['final']}"
        assert r["label"] == "STRONG"
        assert r["sizing"] == "FULL"
    _run("ideal_mean_reversion (WR=-75, ADX=12, PCR=1.5)", t_ideal_setup)

    def t_divislab_like():
        """DIVISLAB: WR=-89, ADX=33, PCR=0.67 bearish, GEX negative."""
        r = _full_score("B", "CYCLICAL", wr_30=-89, adx=33,
                        pct_sma=-4.4, pcr=0.673, gex="Negative", net_gex=-5726,
                        res_pct=8.4, sup_pct=-6.6)
        assert r["base"] == 80, f"Base expected 80, got {r['base']}"
        assert r["opt"] < -5, f"Bearish PCR+GEX should give ≤-5 overlay, got {r['opt']}"
        assert r["final"] < 75, f"Options penalty should pull below 75, got {r['final']}"
        assert r["label"] in ("GOOD", "STRONG"), f"Should be GOOD/STRONG, got {r['label']}"
        assert "bearish" in r["opt_detail"].lower()
        assert "amplif" in r["opt_detail"].lower()
    _run("DIVISLAB-like (WR=-89, bearish PCR, neg GEX)", t_divislab_like)

    def t_shallow_pullback():
        """Barely below SMA, WR not deep enough."""
        r = _full_score("B", "CYCLICAL", wr_30=-25, adx=18,
                        pct_sma=-0.3)
        assert r["base"] <= 50, f"Shallow pullback should score ≤50, got {r['base']}"
        assert r["label"] in ("WATCH", "AVOID")
    _run("shallow_pullback (WR=-25, -0.3% SMA)", t_shallow_pullback)

    def t_above_sma_rejected():
        """Above SMA — NOT a mean reversion setup."""
        r = _full_score("B", "CYCLICAL", wr_30=-50, adx=15,
                        above_sma=True, pct_sma=2.0)
        assert r["base"] < 60, f"Above SMA should penalize, got {r['base']}"
    _run("above_sma_rejected (not a pullback)", t_above_sma_rejected)

    def t_strong_trend_caution():
        """Deep oversold but strong trend (ADX=40) — risky for reversion."""
        r = _full_score("B", "CYCLICAL", wr_30=-70, adx=40, pct_sma=-5.0)
        r_range = _full_score("B", "CYCLICAL", wr_30=-70, adx=12, pct_sma=-5.0)
        assert r["base"] < r_range["base"], (
            f"ADX=40 ({r['base']}) should score less than ADX=12 ({r_range['base']})"
        )
    _run("trending_penalized_vs_ranging (ADX=40 vs ADX=12)", t_strong_trend_caution)

    def t_bearish_options_compound():
        """PCR bearish + GEX negative = maximum penalty for Mode B."""
        r = _full_score("B", "CYCLICAL", wr_30=-60, adx=20, pct_sma=-3.0,
                        pcr=0.5, gex="Negative", net_gex=-8000)
        assert r["opt"] <= -8, f"Compound bearish should give ≤-8, got {r['opt']}"
    _run("compound_bearish_max_penalty (PCR=0.5, GEX neg)", t_bearish_options_compound)

    def t_bullish_options_boost():
        """PCR bullish + GEX negative = sharp bounce expected."""
        r = _full_score("B", "CYCLICAL", wr_30=-60, adx=15, pct_sma=-3.0,
                        pcr=1.5, gex="Negative", net_gex=-3000)
        assert r["opt"] >= 4, f"Bullish PCR + neg GEX bounce should give ≥4, got {r['opt']}"
    _run("bullish_pcr_neg_gex_bounce (PCR=1.5, GEX neg)", t_bullish_options_boost)

    def t_near_support():
        """Price near OI support — extra cushion for Mode B entry."""
        r = _full_score("B", "CYCLICAL", wr_30=-40, adx=18, pct_sma=-1.5,
                        pcr=1.0, gex="Neutral", sup_pct=-1.5)
        r_no_sup = _full_score("B", "CYCLICAL", wr_30=-40, adx=18, pct_sma=-1.5,
                               pcr=1.0, gex="Neutral", sup_pct=-8.0)
        assert r["opt"] > r_no_sup["opt"], (
            f"Near support ({r['opt']}) should beat far support ({r_no_sup['opt']})"
        )
    _run("near_oi_support_boost (sup -1.5% vs -8.0%)", t_near_support)

    def t_no_options():
        """No options data — overlay should be 0."""
        r = _full_score("B", "CYCLICAL", wr_30=-60, adx=15, pct_sma=-3.0)
        assert r["opt"] == 0, f"No options should give 0 overlay, got {r['opt']}"
    _run("no_options_zero_overlay", t_no_options)


# ═════════════════════════════════════════════════════════════════
# MODE A: EARLY MOMENTUM SCENARIOS
# ═════════════════════════════════════════════════════════════════

def test_mode_a_scenarios():
    print("\n── MODE A: EARLY MOMENTUM SCENARIOS ──")

    def t_ideal_fresh_cross():
        """Fresh WR cross + strong ADX + bullish options."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-40, wr_cross=1,
                        adx=30, above_sma=True, pct_sma=2.0,
                        pcr=1.4, gex="Negative", net_gex=-2000, res_pct=7.0)
        assert r["base"] >= 80, f"Fresh cross + strong ADX base ≥80, got {r['base']}"
        assert r["opt"] > 5, f"Bullish options should give >5, got {r['opt']}"
        assert r["label"] == "STRONG"
        assert r["sizing"] == "HALF"  # Mode A always HALF
    _run("ideal_momentum (fresh cross, ADX=30, bullish opts)", t_ideal_fresh_cross)

    def t_no_cross_high_adx():
        """ADX is strong but WR hasn't crossed — no trigger yet."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-80, wr_cross=99,
                        adx=35, above_sma=True, pct_sma=1.0)
        assert r["base"] < 60, f"No cross should cap base, got {r['base']}"
    _run("no_cross_waiting (WR=-80, ADX=35, no cross)", t_no_cross_high_adx)

    def t_cross_but_no_trend():
        """Fresh cross but ADX < 20 — no trend confirmation."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-45, wr_cross=2,
                        adx=12, above_sma=True, pct_sma=1.0)
        assert r["base"] < 70, f"Cross without trend should be moderate, got {r['base']}"
    _run("cross_no_trend (cross 2d ago, ADX=12)", t_cross_but_no_trend)

    def t_pinning_kills_momentum():
        """GEX positive = dealers pinning = momentum suppressed."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-42, wr_cross=1,
                        adx=25, above_sma=True, pct_sma=1.5,
                        pcr=0.6, gex="Positive", net_gex=5000, res_pct=1.0)
        assert r["opt"] < -5, f"Bearish PCR + Pinning + tight room should penalize, got {r['opt']}"
    _run("gex_pinning_kills_momentum (GEX pos, bearish PCR, tight room)", t_pinning_kills_momentum)

    def t_sizing_always_half():
        """Mode A sizing never exceeds HALF regardless of score."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-35, wr_cross=0,
                        adx=40, above_sma=True, pct_sma=3.0,
                        pcr=1.5, gex="Negative", net_gex=-5000, res_pct=10.0)
        assert r["sizing"] == "HALF", f"Mode A should be HALF, got {r['sizing']}"
    _run("mode_a_sizing_capped_half (score={})".format(
        _full_score("A", "INSTITUTIONAL", wr_20=-35, wr_cross=0,
                    adx=40, above_sma=True, pct_sma=3.0,
                    pcr=1.5, gex="Negative", net_gex=-5000, res_pct=10.0)["final"]
    ), t_sizing_always_half)

    def t_deep_below_sma():
        """Institutional stock deeply below SMA — macro broken."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-90, wr_cross=99,
                        adx=50, above_sma=False, pct_sma=-6.0)
        assert r["base"] <= 45, f"Deep below SMA should penalize Mode A, got {r['base']}"
    _run("below_sma_penalized_mode_a (pct=-6%)", t_deep_below_sma)


# ═════════════════════════════════════════════════════════════════
# FILING OVERLAY SCENARIOS
# ═════════════════════════════════════════════════════════════════

def test_filing_scenarios():
    print("\n── FILING OVERLAY SCENARIOS ──")

    def t_trap_detection():
        """High base + bearish filing = TRAP. Score capped at 30."""
        r = _full_score("B", "CYCLICAL", wr_30=-70, adx=15, pct_sma=-4.0,
                        filing_dir="BEARISH", filing_conv=8)
        assert r["trap"] is True, "Should detect TRAP"
        assert r["final"] <= 30, f"TRAP should cap at 30, got {r['final']}"
        assert r["label"] == "AVOID"
        assert r["sizing"] == "SKIP"
    _run("trap_detected (base=80, bearish filing)", t_trap_detection)

    def t_bearish_filing_low_base():
        """Low base + bearish filing = not TRAP (already avoiding)."""
        r = _full_score("B", "CYCLICAL", wr_30=-25, adx=30, pct_sma=-0.3,
                        above_sma=True,
                        filing_dir="BEARISH", filing_conv=6)
        assert r["trap"] is False, f"Low base shouldn't trigger TRAP"
        assert r["fil"] == -5
    _run("bearish_filing_low_base (not trap)", t_bearish_filing_low_base)

    def t_bullish_filing_boost():
        """Bullish filing with high conviction adds to score."""
        r = _full_score("B", "CYCLICAL", wr_30=-50, adx=18, pct_sma=-2.0,
                        filing_dir="BULLISH", filing_conv=8)
        assert r["fil"] == 7, f"High conviction bullish should give +7, got {r['fil']}"
    _run("bullish_filing_high_conviction (+7)", t_bullish_filing_boost)

    def t_no_filing():
        """No filing data — overlay = 0."""
        r = _full_score("B", "CYCLICAL", wr_30=-50, adx=18, pct_sma=-2.0)
        assert r["fil"] == 0
    _run("no_filing_zero_overlay", t_no_filing)


# ═════════════════════════════════════════════════════════════════
# COMPOSITE SCORING EDGE CASES
# ═════════════════════════════════════════════════════════════════

def test_composite_edges():
    print("\n── COMPOSITE EDGE CASES ──")

    def t_score_bounded():
        """Final score always 0-100 regardless of overlay extremes."""
        # Max everything bullish
        r = _full_score("B", "CYCLICAL", wr_30=-90, adx=10, pct_sma=-5.0,
                        pcr=1.8, gex="Negative", net_gex=-10000, sup_pct=-0.5,
                        filing_dir="BULLISH", filing_conv=9)
        assert 0 <= r["final"] <= 100, f"Score {r['final']} out of bounds"

        # Max everything bearish
        r2 = _full_score("A", "INSTITUTIONAL", wr_20=-95, wr_cross=99,
                         adx=5, above_sma=False, pct_sma=-8.0,
                         pcr=0.3, gex="Positive", net_gex=10000, res_pct=0.5,
                         filing_dir="BEARISH", filing_conv=9)
        assert 0 <= r2["final"] <= 100, f"Score {r2['final']} out of bounds"
    _run("scores_bounded_0_100", t_score_bounded)

    def t_options_vs_no_options():
        """Bearish options should strictly lower score vs no options."""
        r_with = _full_score("B", "CYCLICAL", wr_30=-60, adx=20, pct_sma=-3.0,
                             pcr=0.5, gex="Negative", net_gex=-5000)
        r_without = _full_score("B", "CYCLICAL", wr_30=-60, adx=20, pct_sma=-3.0)
        assert r_with["final"] < r_without["final"], (
            f"Bearish opts ({r_with['final']}) should < no opts ({r_without['final']})"
        )
    _run("bearish_opts_lower_than_no_opts", t_options_vs_no_options)

    def t_mode_a_vs_mode_b_same_data():
        """Same bearish data: Mode B should score higher than Mode A."""
        r_b = _full_score("B", "CYCLICAL", wr_30=-70, adx=15,
                          pct_sma=-4.0, above_sma=False)
        r_a = _full_score("A", "INSTITUTIONAL", wr_20=-70, wr_cross=99,
                          adx=15, pct_sma=-4.0, above_sma=False)
        assert r_b["base"] > r_a["base"], (
            f"Mode B ({r_b['base']}) should beat Mode A ({r_a['base']}) in oversold"
        )
    _run("mode_b_beats_mode_a_in_oversold", t_mode_a_vs_mode_b_same_data)

    def t_trap_overrides_everything():
        """TRAP caps score at 30 even with perfect price setup."""
        r = _full_score("B", "CYCLICAL", wr_30=-90, adx=10, pct_sma=-5.0,
                        pcr=1.5, gex="Negative", net_gex=-3000, sup_pct=-1.0,
                        filing_dir="BEARISH", filing_conv=8)
        assert r["final"] <= 30, f"TRAP should cap at 30 even with bullish opts, got {r['final']}"
        assert r["sizing"] == "SKIP"
    _run("trap_overrides_bullish_options", t_trap_overrides_everything)


# ═════════════════════════════════════════════════════════════════
# REAL-WORLD MARKET SCENARIOS (from screenshot data)
# ═════════════════════════════════════════════════════════════════

def test_real_scenarios():
    print("\n── REAL-WORLD SCENARIOS ──")

    def t_nifty_march_2025():
        """NIFTY March 2025: WR(20)=-95, ADX=63, below SMA -5.7%, INSTITUTIONAL."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-95, wr_cross=99,
                        adx=63, above_sma=False, pct_sma=-5.7)
        assert r["base"] < 50, f"No WR cross = low base, got {r['base']}"
        assert r["label"] in ("AVOID", "WATCH")
        print(f"    NIFTY: base={r['base']} opt={r['opt']} → {r['final']} {r['label']}")
    _run("nifty_march_2025 (no cross, deep oversold)", t_nifty_march_2025)

    def t_divislab_march_2025():
        """DIVISLAB: WR(30)=-89, ADX=33, PCR=0.67, GEX=-5726M."""
        r = _full_score("B", "CYCLICAL", wr_30=-89, adx=33,
                        pct_sma=-4.4, pcr=0.673, gex="Negative", net_gex=-5726,
                        res_pct=8.4, sup_pct=-6.6)
        assert 65 <= r["final"] <= 78, f"Expected 65-78, got {r['final']}"
        assert r["opt"] <= -5
        print(f"    DIVISLAB: base={r['base']} opt={r['opt']} → {r['final']} {r['label']}")
    _run("divislab_march_2025 (deep oversold, bearish opts)", t_divislab_march_2025)

    def t_maruti_oversold():
        """MARUTI: WR=-94, ADX=57, below SMA -10%, CYCLICAL."""
        r = _full_score("B", "CYCLICAL", wr_30=-94, adx=57,
                        pct_sma=-10.0)
        assert r["base"] >= 70, f"Deep oversold below SMA should score well, got {r['base']}"
        # ADX=57 is very high — caution
        print(f"    MARUTI: base={r['base']} → {r['final']} {r['label']}")
    _run("maruti_deep_oversold (WR=-94, ADX=57)", t_maruti_oversold)

    def t_hdfcbank_waiting():
        """HDFCBANK: institutional, waiting for WR cross, ADX=25."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-60, wr_cross=99,
                        adx=25, above_sma=False, pct_sma=-2.0,
                        pcr=1.1, gex="Neutral")
        assert r["label"] in ("AVOID", "WATCH"), f"No cross = wait, got {r['label']}"
        print(f"    HDFCBANK: base={r['base']} → {r['final']} {r['label']}")
    _run("hdfcbank_waiting (no WR cross)", t_hdfcbank_waiting)

    def t_hdfcbank_cross_fires():
        """HDFCBANK: WR just crossed -50, ADX=25 — ENTRY."""
        r = _full_score("A", "INSTITUTIONAL", wr_20=-42, wr_cross=1,
                        adx=25, above_sma=False, pct_sma=-2.0,
                        pcr=1.1, gex="Neutral")
        assert r["base"] >= 70, f"Fresh cross + moderate ADX should be ≥70, got {r['base']}"
        assert r["sizing"] == "HALF"
        print(f"    HDFCBANK cross: base={r['base']} → {r['final']} {r['label']}")
    _run("hdfcbank_cross_fires (WR cross 1d ago)", t_hdfcbank_cross_fires)


# ═════════════════════════════════════════════════════════════════
# OPTIONS OVERLAY TRUTH TABLE
# ═════════════════════════════════════════════════════════════════

def test_options_truth_table():
    print("\n── OPTIONS OVERLAY TRUTH TABLE ──")
    sig_b = DualModeSignal(mode="B")
    sig_a = DualModeSignal(mode="A")

    cases = [
        # (label, pcr, gex, net_gex, res, sup, mode, expected_sign, expected_min_abs)
        ("B: bearish PCR only",      0.5, "Neutral", 0,     None, None, "B", -1, 3),
        ("B: bearish PCR + neg GEX", 0.5, "Negative", -5000, None, None, "B", -1, 7),
        ("B: bullish PCR only",      1.5, "Neutral", 0,     None, None, "B", +1, 2),
        ("B: bullish PCR + neg GEX", 1.5, "Negative", -3000, None, None, "B", +1, 4),
        ("B: neutral + near support",1.0, "Neutral", 0,     None, -1.0, "B", +1, 1),
        ("B: neutral no support",    1.0, "Neutral", 0,     None, -8.0, "B",  0, 0),
        ("A: bullish PCR + neg GEX", 1.5, "Negative", -3000, 7.0, None, "A", +1, 8),
        ("A: bearish PCR + pos GEX", 0.5, "Positive", 5000,  1.0, None, "A", -1, 8),
        ("A: bull PCR + room",       1.4, "Neutral", 0,      8.0, None, "A", +1, 5),
    ]

    for label, pcr, gex, ngex, res, sup, mode, exp_sign, exp_min_abs in cases:
        def _test(l=label, p=pcr, g=gex, ng=ngex, r=res, s=sup, m=mode, es=exp_sign, ema=exp_min_abs):
            ctx = _make_ctx(p, g, ng, r, s)
            sig = DualModeSignal(mode=m)
            ov, det = _options_overlay(ctx, sig)
            if es > 0:
                assert ov > 0, f"{l}: expected positive, got {ov} ({det})"
            elif es < 0:
                assert ov < 0, f"{l}: expected negative, got {ov} ({det})"
            assert abs(ov) >= ema, f"{l}: expected |ov|≥{ema}, got {abs(ov)} ({det})"
        _run(label, _test)


# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NIMBUS Validation Test Suite")
    print("=" * 60)

    test_mode_b_scenarios()
    test_mode_a_scenarios()
    test_filing_scenarios()
    test_composite_edges()
    test_real_scenarios()
    test_options_truth_table()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    if _ERRORS:
        print("\nFAILURES:")
        for n, e in _ERRORS:
            print(f"  {n}: {e}")
    sys.exit(1 if _FAIL > 0 else 0)
