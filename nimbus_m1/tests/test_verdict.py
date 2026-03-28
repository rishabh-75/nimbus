"""
tests/test_verdict.py — Exhaustive tests for scanner verdict logic.
Tests every branch of _compute_verdict against real-world scenarios.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PASS = 0; _FAIL = 0; _ERRORS = []
def _run(name, fn):
    global _PASS, _FAIL
    try: fn(); _PASS += 1; print(f"  ✓ {name}")
    except Exception as e: _FAIL += 1; _ERRORS.append((name, str(e))); print(f"  ✗ {name}: {e}")

def _row(**kwargs):
    """Build a scanner result dict with defaults."""
    base = {
        "dm_tier": "NONE", "dm_score": 50, "dm_sizing": "SKIP",
        "dm_mfi": 40, "dm_wr": -40, "dm_dd": -5.0,
        "dm_streak": 2, "dm_entry": False, "dm_pct_sma": -3.0,
    }
    base.update(kwargs)
    return base

from modules.scanner import _compute_verdict


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY TIER
# ══════════════════════════════════════════════════════════════════════════════

def test_primary_tier():
    print("\n── PRIMARY TIER ──")

    def t_primary_strong():
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=85, dm_sizing="FULL"))
        assert "BUY FULL" in v and "strong" in v, f"Got: {v}"
    _run("primary_score_85_strong", t_primary_strong)

    def t_primary_75():
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=75, dm_sizing="FULL"))
        assert "BUY FULL" in v and "strong" in v, f"Got: {v}"
    _run("primary_score_75_strong_boundary", t_primary_75)

    def t_primary_good():
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=65, dm_sizing="FULL"))
        assert "BUY FULL" in v and "good" in v, f"Got: {v}"
    _run("primary_score_65_good", t_primary_good)

    def t_primary_60():
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=60, dm_sizing="FULL"))
        assert "BUY FULL" in v, f"Got: {v}"
    _run("primary_score_60_still_buy", t_primary_60)

    def t_primary_low():
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=45, dm_sizing="FULL"))
        assert "BUY FULL" in v, f"Primary always says BUY FULL regardless of score. Got: {v}"
    _run("primary_score_45_still_buy", t_primary_low)


# ══════════════════════════════════════════════════════════════════════════════
# SECONDARY TIER
# ══════════════════════════════════════════════════════════════════════════════

def test_secondary_tier():
    print("\n── SECONDARY TIER ──")

    def t_sec_strong():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=87, dm_sizing="HALF"))
        assert "BUY HALF" in v and "strong" in v, f"Got: {v}"
    _run("secondary_87_strong", t_sec_strong)

    def t_sec_75():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=75, dm_sizing="HALF"))
        assert "BUY HALF" in v and "strong" in v, f"Got: {v}"
    _run("secondary_75_boundary_strong", t_sec_75)

    def t_sec_decent():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=68, dm_sizing="HALF"))
        assert "BUY HALF" in v and "decent" in v, f"Got: {v}"
    _run("secondary_68_decent", t_sec_decent)

    def t_sec_60():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=60, dm_sizing="HALF"))
        assert "BUY HALF" in v, f"Got: {v}"
    _run("secondary_60_boundary_decent", t_sec_60)

    def t_sec_marginal():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=50, dm_sizing="HALF"))
        assert "HALF" in v and "marginal" in v, f"Got: {v}"
    _run("secondary_50_marginal", t_sec_marginal)

    def t_sec_45():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=45, dm_sizing="HALF"))
        assert "HALF" in v and "marginal" in v, f"Got: {v}"
    _run("secondary_45_marginal", t_sec_45)


# ══════════════════════════════════════════════════════════════════════════════
# TRAP / VERY LOW SCORE
# ══════════════════════════════════════════════════════════════════════════════

def test_trap():
    print("\n── TRAP / LOW SCORE ──")

    def t_trap_30():
        v = _compute_verdict(_row(dm_score=30))
        assert "AVOID" in v, f"Score 30 should AVOID. Got: {v}"
    _run("score_30_avoid", t_trap_30)

    def t_trap_20():
        v = _compute_verdict(_row(dm_score=20))
        assert "AVOID" in v, f"Got: {v}"
    _run("score_20_avoid", t_trap_20)

    def t_trap_0():
        v = _compute_verdict(_row(dm_score=0))
        assert "AVOID" in v, f"Got: {v}"
    _run("score_0_avoid", t_trap_0)

    def t_trap_overrides_primary():
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=25))
        assert "AVOID" in v, f"TRAP should override PRIMARY. Got: {v}"
    _run("trap_overrides_primary", t_trap_overrides_primary)

    def t_trap_overrides_secondary():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=28))
        assert "AVOID" in v, f"TRAP should override SECONDARY. Got: {v}"
    _run("trap_overrides_secondary", t_trap_overrides_secondary)

    def t_score_31_not_trap():
        v = _compute_verdict(_row(dm_score=31, dm_mfi=20, dm_wr=-20))
        assert "AVOID" not in v, f"Score 31 should NOT be AVOID. Got: {v}"
    _run("score_31_not_trapped", t_score_31_not_trap)


# ══════════════════════════════════════════════════════════════════════════════
# NO ENTRY — MFI WEAK
# ══════════════════════════════════════════════════════════════════════════════

def test_mfi_weak():
    print("\n── NO ENTRY: MFI WEAK ──")

    def t_capitulation():
        """Deep WR + deep DD + weak MFI → WATCH (best watchlist candidates)"""
        v = _compute_verdict(_row(dm_mfi=18, dm_wr=-97, dm_dd=-26.0, dm_score=66))
        assert "WATCH" in v and "capitulation" in v, f"Got: {v}"
    _run("capitulation_watch_bpcl", t_capitulation)

    def t_cap_boundary_wr():
        v = _compute_verdict(_row(dm_mfi=15, dm_wr=-51, dm_dd=-11.0, dm_score=60))
        assert "WATCH" in v and "capitulation" in v, f"Got: {v}"
    _run("capitulation_boundary_wr51_dd11", t_cap_boundary_wr)

    def t_cap_wr50_not_deep():
        """WR=-50 is NOT < -50, so this should be WAIT not WATCH"""
        v = _compute_verdict(_row(dm_mfi=15, dm_wr=-50, dm_dd=-12.0, dm_score=60))
        assert "WAIT" in v, f"WR=-50 not deep enough for capitulation. Got: {v}"
    _run("wr50_not_deep_enough", t_cap_wr50_not_deep)

    def t_wait_oversold():
        """WR < -30 but not deep + weak MFI → WAIT"""
        v = _compute_verdict(_row(dm_mfi=25, dm_wr=-42, dm_dd=-5.0, dm_score=68))
        assert "WAIT" in v and "MFI=25" in v, f"Got: {v}"
    _run("wait_oversold_mfi25", t_wait_oversold)

    def t_wait_axisbank():
        """AXISBANK: WR=-90, MFI=25 → WAIT"""
        v = _compute_verdict(_row(dm_mfi=25, dm_wr=-90, dm_dd=-7.8, dm_score=68))
        assert "WAIT" in v or "WATCH" in v, f"Got: {v}"
    _run("wait_axisbank_mfi25", t_wait_axisbank)

    def t_skip_not_oversold_mfi_weak():
        """WR > -30 + MFI weak → SKIP"""
        v = _compute_verdict(_row(dm_mfi=20, dm_wr=-15, dm_score=40))
        assert "SKIP" in v and "MFI" in v, f"Got: {v}"
    _run("skip_not_oversold_mfi_weak", t_skip_not_oversold_mfi_weak)

    def t_mfi_29_weak():
        """MFI=29 is < 30, should be filtered"""
        v = _compute_verdict(_row(dm_mfi=29, dm_wr=-45, dm_score=65))
        assert "WAIT" in v or "WATCH" in v, f"MFI=29 should be weak. Got: {v}"
    _run("mfi_29_still_weak", t_mfi_29_weak)

    def t_mfi_30_not_weak():
        """MFI=30 is >= 30, should NOT be filtered as weak"""
        v = _compute_verdict(_row(dm_mfi=30, dm_wr=-45, dm_score=65))
        assert "WAIT" not in v and "WATCH" not in v, f"MFI=30 should NOT be weak. Got: {v}"
    _run("mfi_30_not_weak", t_mfi_30_not_weak)


# ══════════════════════════════════════════════════════════════════════════════
# NO ENTRY — WR NOT OVERSOLD
# ══════════════════════════════════════════════════════════════════════════════

def test_wr_not_oversold():
    print("\n── NO ENTRY: WR NOT OVERSOLD ──")

    def t_wr_above():
        v = _compute_verdict(_row(dm_mfi=50, dm_wr=-15, dm_score=45))
        assert "SKIP" in v and "not oversold" in v, f"Got: {v}"
    _run("wr_minus15_skip", t_wr_above)

    def t_wr_minus30_boundary():
        """WR=-30 is >= -30, NOT oversold"""
        v = _compute_verdict(_row(dm_mfi=45, dm_wr=-30, dm_score=50))
        assert "SKIP" in v and "not oversold" in v, f"WR=-30 is NOT < -30. Got: {v}"
    _run("wr_minus30_boundary_skip", t_wr_minus30_boundary)

    def t_wr_minus31_oversold():
        """WR=-31 IS < -30, should NOT skip for WR reason"""
        v = _compute_verdict(_row(dm_mfi=45, dm_wr=-31, dm_score=65))
        assert "not oversold" not in v, f"WR=-31 should be oversold. Got: {v}"
    _run("wr_minus31_oversold", t_wr_minus31_oversold)

    def t_wr_zero():
        v = _compute_verdict(_row(dm_mfi=50, dm_wr=0, dm_score=40))
        assert "SKIP" in v, f"Got: {v}"
    _run("wr_zero_skip", t_wr_zero)

    def t_wr_none():
        """WR=None (no data) should default to 0 and SKIP"""
        v = _compute_verdict(_row(dm_mfi=50, dm_wr=None, dm_score=45))
        assert "SKIP" in v, f"Got: {v}"
    _run("wr_none_skip", t_wr_none)


# ══════════════════════════════════════════════════════════════════════════════
# FORMING — core met, no entry
# ══════════════════════════════════════════════════════════════════════════════

def test_forming():
    print("\n── FORMING ──")

    def t_forming():
        """MFI>=30, WR<-30, above SMA (so tier=NONE but core partially met)"""
        v = _compute_verdict(_row(dm_tier="NONE", dm_mfi=40, dm_wr=-45, dm_score=65))
        assert "FORMING" in v, f"Got: {v}"
    _run("forming_score65", t_forming)

    def t_forming_60():
        v = _compute_verdict(_row(dm_tier="NONE", dm_mfi=35, dm_wr=-35, dm_score=60))
        assert "FORMING" in v, f"Got: {v}"
    _run("forming_score60_boundary", t_forming_60)

    def t_low_score_skip():
        """Same conditions but score < 60 → SKIP"""
        v = _compute_verdict(_row(dm_tier="NONE", dm_mfi=35, dm_wr=-35, dm_score=55))
        assert "SKIP" in v and "too low" in v, f"Got: {v}"
    _run("score_55_skip_too_low", t_low_score_skip)

    def t_score_40():
        v = _compute_verdict(_row(dm_tier="NONE", dm_mfi=35, dm_wr=-35, dm_score=40))
        assert "SKIP" in v, f"Got: {v}"
    _run("score_40_skip", t_score_40)


# ══════════════════════════════════════════════════════════════════════════════
# REAL-WORLD SCANNER SCENARIOS (from your screenshots)
# ══════════════════════════════════════════════════════════════════════════════

def test_real_world():
    print("\n── REAL-WORLD SCENARIOS ──")

    def t_adaniports():
        """ADANIPORTS: score=87, SEC, HALF, WR=-92, MFI=45, DD=-13.6%"""
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=87, dm_sizing="HALF",
                                  dm_wr=-92, dm_mfi=45, dm_dd=-13.6, dm_streak=0))
        assert "BUY HALF" in v and "strong" in v, f"Got: {v}"
    _run("adaniports_87_sec_strong", t_adaniports)

    def t_bajfinance():
        """BAJFINANCE: score=78, SEC, HALF"""
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=78, dm_sizing="HALF",
                                  dm_wr=-95, dm_mfi=32, dm_dd=-20.6))
        assert "BUY HALF" in v and "strong" in v, f"Got: {v}"
    _run("bajfinance_78_sec", t_bajfinance)

    def t_divislab():
        """DIVISLAB: score=70, SEC, HALF"""
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=70, dm_sizing="HALF",
                                  dm_wr=-73, dm_mfi=35, dm_dd=-7.1))
        assert "BUY HALF" in v and "decent" in v, f"Got: {v}"
    _run("divislab_70_sec_decent", t_divislab)

    def t_axisbank():
        """AXISBANK: score=68, NONE (MFI=25), WR=-90"""
        v = _compute_verdict(_row(dm_tier="NONE", dm_score=68, dm_sizing="SKIP",
                                  dm_wr=-90, dm_mfi=25, dm_dd=-7.8))
        assert "WAIT" in v or "WATCH" in v, f"Got: {v}"
        assert "MFI" in v, f"Should mention MFI. Got: {v}"
    _run("axisbank_68_wait_mfi", t_axisbank)

    def t_bpcl():
        """BPCL: score=66, NONE (MFI=18), WR=-97, DD=-26.3%"""
        v = _compute_verdict(_row(dm_tier="NONE", dm_score=66, dm_sizing="SKIP",
                                  dm_wr=-97, dm_mfi=18, dm_dd=-26.3))
        assert "WATCH" in v and "capitulation" in v, f"Got: {v}"
    _run("bpcl_66_capitulation_watch", t_bpcl)

    def t_cipla():
        """CIPLA: score=61, NONE (MFI=18), WR=-81"""
        v = _compute_verdict(_row(dm_tier="NONE", dm_score=61, dm_sizing="SKIP",
                                  dm_wr=-81, dm_mfi=18, dm_dd=-9.1))
        # WR=-81 < -50 and DD=-9.1 is NOT < -10, so not capitulation → WAIT
        assert "WAIT" in v, f"Got: {v}"
    _run("cipla_61_wait_not_capitulation", t_cipla)

    def t_lt():
        """LT: score=66, NONE (MFI=24), WR=-96, DD=-22.6%"""
        v = _compute_verdict(_row(dm_tier="NONE", dm_score=66, dm_sizing="SKIP",
                                  dm_wr=-96, dm_mfi=24, dm_dd=-22.6))
        assert "WATCH" in v and "capitulation" in v, f"Got: {v}"
    _run("lt_66_capitulation_watch", t_lt)

    def t_asianpaint():
        """ASIANPAINT: score=72, MFI=30 (boundary), WR=-86"""
        v = _compute_verdict(_row(dm_tier="NONE", dm_score=72, dm_sizing="SKIP",
                                  dm_wr=-86, dm_mfi=30, dm_dd=-20.8))
        # MFI=30 is NOT < 30, so doesn't hit the weak MFI path
        # WR=-86 < -30, MFI>=30, tier=NONE → FORMING
        assert "FORMING" in v or "BUY" in v, f"MFI=30 should not be weak. Got: {v}"
    _run("asianpaint_72_mfi30_not_weak", t_asianpaint)

    def t_bhartiartl():
        """BHARTIARTL: score=73, SEC, MFI=45"""
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=73, dm_sizing="HALF",
                                  dm_wr=-74, dm_mfi=45, dm_dd=-10.2))
        assert "BUY HALF" in v and "decent" in v, f"Got: {v}"
    _run("bhartiartl_73_sec_decent", t_bhartiartl)


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n── EDGE CASES ──")

    def t_perfect_primary():
        """Maximum everything — PRIMARY, score 100"""
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=100, dm_sizing="FULL",
                                  dm_wr=-99, dm_mfi=90, dm_dd=-25, dm_streak=8))
        assert "BUY FULL" in v and "strong" in v, f"Got: {v}"
    _run("perfect_primary_100", t_perfect_primary)

    def t_worst_case():
        """Everything bad — score 0"""
        v = _compute_verdict(_row(dm_score=0, dm_mfi=5, dm_wr=-5, dm_dd=-0.5))
        assert "AVOID" in v, f"Got: {v}"
    _run("worst_case_score0", t_worst_case)

    def t_missing_fields():
        """Minimal dict — should not crash"""
        v = _compute_verdict({"dm_score": 50})
        assert isinstance(v, str) and len(v) > 0, f"Got: {v}"
    _run("missing_fields_no_crash", t_missing_fields)

    def t_empty_dict():
        """Empty dict — should not crash, defaults handle it"""
        v = _compute_verdict({})
        assert isinstance(v, str), f"Got: {v}"
    _run("empty_dict_no_crash", t_empty_dict)

    def t_none_wr():
        """WR=None should not crash"""
        v = _compute_verdict(_row(dm_wr=None, dm_mfi=40, dm_score=50))
        assert isinstance(v, str), f"Got: {v}"
    _run("none_wr_no_crash", t_none_wr)

    def t_score_boundary_30_31():
        """Score 30 → AVOID, Score 31 → not AVOID"""
        v30 = _compute_verdict(_row(dm_score=30))
        v31 = _compute_verdict(_row(dm_score=31, dm_mfi=20, dm_wr=-20))
        assert "AVOID" in v30, f"30 should AVOID. Got: {v30}"
        assert "AVOID" not in v31, f"31 should NOT AVOID. Got: {v31}"
    _run("score_boundary_30_31", t_score_boundary_30_31)

    def t_secondary_59_marginal():
        """SECONDARY score 59 (below 60) → marginal, not decent"""
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=59))
        assert "marginal" in v, f"Got: {v}"
    _run("secondary_59_marginal_not_decent", t_secondary_59_marginal)

    def t_primary_beats_mfi_filter():
        """PRIMARY tier should BUY even if MFI is technically low (tier already passed)"""
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=70, dm_mfi=31))
        assert "BUY FULL" in v, f"PRIMARY should always BUY. Got: {v}"
    _run("primary_bypasses_mfi_filter", t_primary_beats_mfi_filter)


# ══════════════════════════════════════════════════════════════════════════════
# VERDICT CONSISTENCY — same inputs always produce same verdict category
# ══════════════════════════════════════════════════════════════════════════════

def test_consistency():
    print("\n── CONSISTENCY ──")

    def t_higher_score_never_worse():
        """For same tier, higher score should never produce a worse verdict"""
        v60 = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=60))
        v75 = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=75))
        v85 = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=85))
        # All should say BUY HALF, but higher should be "strong"
        assert "BUY HALF" in v60 and "BUY HALF" in v75 and "BUY HALF" in v85
        assert "strong" in v85 and "strong" in v75
    _run("higher_sec_score_not_worse", t_higher_score_never_worse)

    def t_primary_always_full():
        """PRIMARY tier always says FULL regardless of score (above TRAP)"""
        for s in [40, 50, 60, 75, 90]:
            v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=s))
            assert "FULL" in v, f"PRIMARY score {s} should say FULL. Got: {v}"
    _run("primary_always_full", t_primary_always_full)

    def t_secondary_always_half():
        """SECONDARY tier always says HALF (above TRAP)"""
        for s in [40, 50, 60, 75, 90]:
            v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=s))
            assert "HALF" in v, f"SECONDARY score {s} should say HALF. Got: {v}"
    _run("secondary_always_half", t_secondary_always_half)

    def t_none_tier_never_buy():
        """NONE tier should never say BUY"""
        for mfi in [20, 35, 50, 70]:
            for wr in [-90, -40, -20, 0]:
                v = _compute_verdict(_row(dm_tier="NONE", dm_mfi=mfi, dm_wr=wr, dm_score=65))
                assert "BUY" not in v, f"NONE tier should never BUY (mfi={mfi},wr={wr}). Got: {v}"
    _run("none_tier_never_says_buy", t_none_tier_never_buy)

    def t_capitulation_vs_wait():
        """Capitulation (deep WR + deep DD) is always WATCH, never just WAIT"""
        v = _compute_verdict(_row(dm_mfi=10, dm_wr=-80, dm_dd=-15, dm_score=60))
        assert "WATCH" in v and "capitulation" in v, f"Got: {v}"
        # Same but DD shallow → WAIT not WATCH
        v2 = _compute_verdict(_row(dm_mfi=10, dm_wr=-80, dm_dd=-8, dm_score=60))
        assert "WAIT" in v2, f"Shallow DD should be WAIT. Got: {v2}"
    _run("capitulation_vs_wait_dd_threshold", t_capitulation_vs_wait)


# ══════════════════════════════════════════════════════════════════════════════
# COLOR MAPPING (verify scanner UI will color correctly)
# ══════════════════════════════════════════════════════════════════════════════

def test_color_keywords():
    print("\n── COLOR KEYWORD PRESENCE ──")

    def t_buy_full_has_keyword():
        v = _compute_verdict(_row(dm_tier="PRIMARY", dm_score=80))
        assert "BUY FULL" in v, f"Got: {v}"
    _run("buy_full_keyword_for_green", t_buy_full_has_keyword)

    def t_buy_half_has_keyword():
        v = _compute_verdict(_row(dm_tier="SECONDARY", dm_score=70))
        assert "BUY HALF" in v, f"Got: {v}"
    _run("buy_half_keyword_for_gold", t_buy_half_has_keyword)

    def t_watch_has_keyword():
        v = _compute_verdict(_row(dm_mfi=15, dm_wr=-60, dm_dd=-12, dm_score=60))
        assert "WATCH" in v, f"Got: {v}"
    _run("watch_keyword_for_blue", t_watch_has_keyword)

    def t_avoid_has_keyword():
        v = _compute_verdict(_row(dm_score=20))
        assert "AVOID" in v, f"Got: {v}"
    _run("avoid_keyword_for_red", t_avoid_has_keyword)

    def t_wait_has_keyword():
        v = _compute_verdict(_row(dm_mfi=25, dm_wr=-40, dm_score=60))
        assert "WAIT" in v, f"Got: {v}"
    _run("wait_keyword_for_gold", t_wait_has_keyword)

    def t_forming_has_keyword():
        v = _compute_verdict(_row(dm_tier="NONE", dm_mfi=40, dm_wr=-40, dm_score=65))
        assert "FORMING" in v, f"Got: {v}"
    _run("forming_keyword_for_gold", t_forming_has_keyword)


if __name__ == "__main__":
    print("=" * 60)
    print("NIMBUS Verdict Test Suite")
    print("=" * 60)
    test_primary_tier()
    test_secondary_tier()
    test_trap()
    test_mfi_weak()
    test_wr_not_oversold()
    test_forming()
    test_real_world()
    test_edge_cases()
    test_consistency()
    test_color_keywords()
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    if _ERRORS:
        for n, e in _ERRORS: print(f"  {n}: {e}")
    import sys; sys.exit(1 if _FAIL > 0 else 0)
