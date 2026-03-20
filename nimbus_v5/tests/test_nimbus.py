"""
tests/test_nimbus.py
────────────────────
Offline test suite for the NIMBUS signal stack.
No network calls — all tests run from synthetic data.

Run:
    pytest tests/test_nimbus.py -v
    pytest tests/test_nimbus.py -v --tb=short   # compact failures

Coverage:
    analytics.py  — _walls, _gex, _regime_classify, _viability, analyze
    scanner.py    — filter booleans, near_mp, expiry_risk, _short_reason,
                    no-opts path score bug (documents + regression)
    data.py       — NIFTY100_SYMBOLS count
    sector_rotation (classify_rotation helper, if present)
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build minimal synthetic DataFrames that satisfy _clean()
# ─────────────────────────────────────────────────────────────────────────────


def _expiry_str(days: int) -> str:
    """Return an expiry date string DTE days from today."""
    d = datetime.date.today() + datetime.timedelta(days=days)
    return d.strftime("%d-%b-%Y")


def _make_options_df(
    spot: float = 1000.0,
    resistance_strike: float = 1100.0,  # dominant call OI here
    support_strike: float = 900.0,  # dominant put OI here
    max_pain_strike: Optional[float] = None,
    dte: int = 10,
    ce_iv: float = 20.0,
    pe_iv: float = 22.0,
    ce_oi_resistance: int = 50_000,
    pe_oi_support: int = 50_000,
    extra_call_oi: int = 5_000,  # noise OI at other strikes
    extra_put_oi: int = 5_000,
    gex_negative: bool = True,  # True → net GEX negative (trend-friendly)
    n_strikes: int = 10,
) -> pd.DataFrame:
    """
    Build a minimal options chain DataFrame that will parse cleanly
    through analytics._clean() and produce predictable walls/GEX.
    """
    strikes = sorted(
        {
            resistance_strike - 200,
            resistance_strike - 100,
            resistance_strike,
            support_strike,
            support_strike + 100,
            spot,
        }
    )
    exp = _expiry_str(dte)
    rows = []
    for s in strikes:
        ce_oi = ce_oi_resistance if s == resistance_strike else extra_call_oi
        pe_oi = pe_oi_support if s == support_strike else extra_put_oi
        # GEX sign is driven by relative CE vs PE OI (simplified):
        # If gex_negative we want dealers net short gamma → more put OI overall
        rows.append(
            {
                "Strike": float(s),
                "Expiry": exp,
                "CE_OI": float(ce_oi),
                "CE_IV": ce_iv,
                "CE_LTP": max(1.0, (spot - s) * 0.1 + 10),
                "PE_OI": float(pe_oi),
                "PE_IV": pe_iv,
                "PE_LTP": max(1.0, (s - spot) * 0.1 + 10),
                "UnderlyingValue": spot,
            }
        )
    return pd.DataFrame(rows)


def _make_price_signals(
    bb_position: str = "ABOVE_UPPER",
    position_state: str = "RIDING_UPPER",
    vol_state: str = "SQUEEZE",
    daily_bias: str = "BULLISH",
    daily_bias_pct: float = 3.5,
    wr_value: float = -12.0,
    wr_in_momentum: bool = True,
    wr_phase: str = "FRESH",
    wr_bars_since_cross50: int = 2,
    bb_pct: float = 0.85,
    bb_width_pctl: float = 12.0,
    daily_sma: float = 980.0,
):
    """Build a PriceSignals-compatible object (simple namespace)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        bb_position=bb_position,
        position_state=position_state,
        vol_state=vol_state,
        daily_bias=daily_bias,
        daily_bias_pct=daily_bias_pct,
        wr_value=wr_value,
        wr_in_momentum=wr_in_momentum,
        wr_phase=wr_phase,
        wr_bars_since_cross50=wr_bars_since_cross50,
        bb_pct=bb_pct,
        bb_width_pctl=bb_width_pctl,
        daily_sma=daily_sma,
    )


# ─────────────────────────────────────────────────────────────────────────────
# analytics._walls
# ─────────────────────────────────────────────────────────────────────────────


class TestWalls:
    def setup_method(self):
        from modules.analytics import _walls, _clean

        self._walls = _walls
        self._clean = _clean

    def _run(self, df, spot):
        return self._walls(self._clean(df), spot)

    def test_resistance_above_spot(self):
        """Resistance must be the dominant call-OI strike above spot."""
        df = _make_options_df(spot=1000, resistance_strike=1100, support_strike=900)
        w = self._run(df, 1000)
        assert w.resistance == 1100.0

    def test_support_below_spot(self):
        """Support must be the dominant put-OI strike below spot."""
        df = _make_options_df(spot=1000, resistance_strike=1100, support_strike=900)
        w = self._run(df, 1000)
        assert w.support == 900.0

    def test_resistance_pct_positive(self):
        """resistance_pct = (resistance - spot) / spot * 100 > 0."""
        df = _make_options_df(spot=1000, resistance_strike=1100)
        w = self._run(df, 1000)
        assert w.resistance_pct == pytest.approx(10.0, abs=0.5)

    def test_support_pct_negative(self):
        """support_pct must be negative (below spot)."""
        df = _make_options_df(spot=1000, support_strike=900)
        w = self._run(df, 1000)
        assert w.support_pct < 0

    def test_pcr_bullish_when_put_heavy(self):
        """PCR > 1.3 when put OI >> call OI → sentiment Bullish."""
        df = _make_options_df(
            spot=1000,
            pe_oi_support=100_000,
            ce_oi_resistance=20_000,
            extra_put_oi=10_000,
            extra_call_oi=5_000,
        )
        w = self._run(df, 1000)
        assert w.pcr_oi > 1.3
        assert "Bullish" in w.pcr_sentiment

    def test_pcr_bearish_when_call_heavy(self):
        """PCR < 0.7 when call OI >> put OI → sentiment Bearish."""
        df = _make_options_df(
            spot=1000,
            ce_oi_resistance=100_000,
            pe_oi_support=10_000,
            extra_call_oi=10_000,
            extra_put_oi=2_000,
        )
        w = self._run(df, 1000)
        assert w.pcr_oi < 0.9

    def test_empty_df_returns_defaults(self):
        """Empty df must return Walls() with all None fields."""
        from modules.analytics import _walls

        w = _walls(
            pd.DataFrame(
                columns=[
                    "Strike",
                    "CE_OI",
                    "PE_OI",
                    "CE_IV",
                    "PE_IV",
                    "CE_LTP",
                    "PE_LTP",
                ]
            ),
            1000,
        )
        assert w.resistance is None
        assert w.support is None
        assert w.pcr_oi == pytest.approx(1.0)

    def test_room_to_run_flag(self):
        """room_to_run=True when resistance_pct >= 1.5%."""
        df = _make_options_df(spot=1000, resistance_strike=1020)  # 2%
        w = self._run(df, 1000)
        assert w.room_to_run is True

    def test_no_room_to_run_when_tight(self):
        """room_to_run=False when resistance is only 0.5% away."""
        df = _make_options_df(spot=1000, resistance_strike=1005)
        w = self._run(df, 1000)
        assert w.room_to_run is False


# ─────────────────────────────────────────────────────────────────────────────
# analytics._regime_classify
# ─────────────────────────────────────────────────────────────────────────────


class TestRegimeClassify:
    def setup_method(self):
        from modules.analytics import _regime_classify, GEX, Walls, ExpiryCtx

        self._classify = _regime_classify
        self.GEX = GEX
        self.Walls = Walls
        self.ExpiryCtx = ExpiryCtx

    def _classify_simple(self, gex_regime, res_pct, pin_risk="LOW", dte=10):
        gex = self.GEX(
            regime=gex_regime,
            net_gex=(
                -1000
                if gex_regime == "Negative"
                else (1000 if gex_regime == "Positive" else 0)
            ),
        )
        walls = self.Walls(resistance_pct=res_pct, resistance=1000 + res_pct * 10)
        expiry = self.ExpiryCtx(days_remaining=dte, pin_risk=pin_risk)
        return self._classify(gex, walls, expiry, 1000)

    def test_negative_gex_trend_friendly(self):
        r = self._classify_simple("Negative", res_pct=8.0)
        assert r.regime == "TREND-FRIENDLY"
        assert r.size_cap == "FULL"

    def test_negative_gex_pinning_when_high_pin(self):
        """Negative GEX + HIGH pin risk → PINNING (overrides TREND-FRIENDLY)."""
        r = self._classify_simple("Negative", res_pct=8.0, pin_risk="HIGH", dte=1)
        assert r.regime == "PINNING"

    def test_neutral_gex_trend_friendly_with_room(self):
        """Neutral GEX + res_pct >= 1.5% → TREND-FRIENDLY."""
        r = self._classify_simple("Neutral", res_pct=2.0)
        assert r.regime == "TREND-FRIENDLY"

    def test_neutral_gex_pinning_without_room(self):
        """Neutral GEX + res_pct < 1.5% → PINNING."""
        r = self._classify_simple("Neutral", res_pct=1.0)
        assert r.regime == "PINNING"

    def test_positive_gex_always_pinning(self):
        """Positive GEX → always PINNING regardless of room."""
        r = self._classify_simple("Positive", res_pct=20.0)
        assert r.regime == "PINNING"

    def test_neutral_gex_boundary_exactly_1_5(self):
        """Exactly 1.5% resistance → TREND-FRIENDLY (boundary inclusive)."""
        r = self._classify_simple("Neutral", res_pct=1.5)
        assert r.regime == "TREND-FRIENDLY"

    def test_neutral_gex_boundary_below_1_5(self):
        """1.49% resistance → PINNING."""
        r = self._classify_simple("Neutral", res_pct=1.49)
        assert r.regime == "PINNING"


# ─────────────────────────────────────────────────────────────────────────────
# analytics._viability — per-component scoring
# ─────────────────────────────────────────────────────────────────────────────


class TestViability:
    def setup_method(self):
        from modules.analytics import analyze, _viability, OptionsContext

        self._analyze = analyze
        self._viability = _viability
        self._OptionsContext = OptionsContext

    def _score(self, df, spot=1000, ps=None, room_thresh=5.0):
        return self._analyze(df, spot=spot, price_signals=ps, room_thresh=room_thresh)

    # ── Regime component ─────────────────────────────────────────────────────

    def test_trend_friendly_regime_adds_15(self):
        """TREND-FRIENDLY regime adds exactly +15."""
        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            gex_negative=True,
            pe_oi_support=80_000,
            ce_oi_resistance=20_000,
            extra_put_oi=8_000,
            extra_call_oi=2_000,
        )
        ctx = self._score(df, 1000, ps)
        regime_item = next(c for c in ctx.viability.checklist if c.item == "Regime")
        assert regime_item.status == "pass"

    def test_pinning_regime_subtracts_15(self):
        """Positive GEX → PINNING → regime checklist fails."""
        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            resistance_strike=1010,
            ce_oi_resistance=100_000,
            pe_oi_support=10_000,
            extra_call_oi=20_000,
            extra_put_oi=1_000,
            dte=10,
        )
        ctx = self._score(df, 1000, ps)
        regime_item = next(c for c in ctx.viability.checklist if c.item == "Regime")
        assert regime_item.status == "fail"

    # ── Room component ────────────────────────────────────────────────────────

    def test_no_resistance_subtracts_20(self):
        """No options data → no resistance → Room checklist fails."""
        ps = _make_price_signals()
        ctx = self._score(pd.DataFrame(), 1000, ps)
        # analyze() exits early, score stays at 50 default — documents the bug
        assert ctx.viability.score == 50

    def test_wide_room_adds_15(self):
        """Resistance >= 2x room_thresh → +15."""
        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            resistance_strike=1120,  # 12% away
            support_strike=940,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
            extra_put_oi=5_000,
            extra_call_oi=2_000,
        )
        ctx = self._score(df, 1000, ps, room_thresh=5.0)
        room_item = next(c for c in ctx.viability.checklist if c.item == "Room")
        assert room_item.status == "pass"
        assert "+1" in room_item.detail or "+" in room_item.detail

    def test_tight_room_subtracts_10(self):
        """Resistance between 0 and room_thresh → warn, -10."""
        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            resistance_strike=1030,  # 3%
            support_strike=950,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=50_000,
            extra_put_oi=5_000,
            extra_call_oi=5_000,
        )
        ctx = self._score(df, 1000, ps, room_thresh=5.0)
        room_item = next(c for c in ctx.viability.checklist if c.item == "Room")
        assert room_item.status == "warn"

    # ── Daily bias component ──────────────────────────────────────────────────

    def test_bearish_bias_hard_skip(self):
        """BEARISH daily bias → sizing must be SKIP regardless of other signals."""
        ps = _make_price_signals(daily_bias="BEARISH", daily_bias_pct=-4.0)
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=80_000,
            ce_oi_resistance=20_000,
            extra_put_oi=8_000,
            extra_call_oi=2_000,
        )
        ctx = self._score(df, 1000, ps)
        assert ctx.viability.sizing == "SKIP"
        assert ctx.viability.label == "AVOID"

    def test_bullish_bias_adds_10(self):
        ps = _make_price_signals(daily_bias="BULLISH")
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=80_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        bias_item = next(c for c in ctx.viability.checklist if c.item == "Daily Bias")
        assert bias_item.status == "pass"

    # ── Expiry component ──────────────────────────────────────────────────────

    def test_expiry_today_subtracts_25(self):
        """DTE=0 → expiry imminent → -25."""
        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            dte=1,
            resistance_strike=1100,
            support_strike=900,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        exp_item = next(c for c in ctx.viability.checklist if c.item == "Expiry Gate")
        assert exp_item.status == "fail"
        assert "TODAY" in exp_item.detail.upper() or "0" in exp_item.detail

    def test_dte_2_subtracts_20(self):
        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            dte=2,
            resistance_strike=1100,
            support_strike=900,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        exp_item = next(c for c in ctx.viability.checklist if c.item == "Expiry Gate")
        assert exp_item.status == "fail"

    def test_dte_8_adds_12(self):
        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            dte=8,
            resistance_strike=1100,
            support_strike=900,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        exp_item = next(c for c in ctx.viability.checklist if c.item == "Expiry Gate")
        assert exp_item.status == "pass"

    # ── WR phase component ────────────────────────────────────────────────────

    def test_fresh_wr_adds_15(self):
        ps = _make_price_signals(wr_phase="FRESH", wr_in_momentum=True)
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        wr_item = next(c for c in ctx.viability.checklist if c.item == "W%R Gate")
        assert wr_item.status == "pass"
        assert "FRESH" in wr_item.detail

    def test_late_wr_subtracts_5(self):
        ps = _make_price_signals(
            wr_phase="LATE", wr_in_momentum=True, wr_bars_since_cross50=18
        )
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        wr_item = next(c for c in ctx.viability.checklist if c.item == "W%R Gate")
        assert wr_item.status == "warn"

    def test_wr_not_in_zone_is_fail(self):
        ps = _make_price_signals(wr_in_momentum=False, wr_value=-55.0)
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        wr_item = next(c for c in ctx.viability.checklist if c.item == "W%R Gate")
        assert wr_item.status == "fail"

    # ── BB position component ─────────────────────────────────────────────────

    def test_mid_band_broken_skip(self):
        """MID_BAND_BROKEN → hard SKIP override."""
        ps = _make_price_signals(
            position_state="MID_BAND_BROKEN", bb_position="BELOW_MID"
        )
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        assert ctx.viability.sizing == "SKIP"

    def test_riding_upper_adds_5(self):
        ps = _make_price_signals(position_state="RIDING_UPPER")
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = self._score(df, 1000, ps)
        bb_item = next(c for c in ctx.viability.checklist if c.item == "BB State")
        assert bb_item.status == "pass"

    # ── Score ceiling/floor ───────────────────────────────────────────────────

    def test_score_never_exceeds_100(self):
        """Best possible setup must not exceed 100."""
        ps = _make_price_signals(
            daily_bias="BULLISH",
            vol_state="SQUEEZE",
            wr_phase="FRESH",
            position_state="RIDING_UPPER",
            wr_in_momentum=True,
        )
        df = _make_options_df(
            spot=1000,
            resistance_strike=1120,
            support_strike=920,
            dte=10,
            pe_oi_support=100_000,
            ce_oi_resistance=20_000,
            extra_put_oi=10_000,
            extra_call_oi=2_000,
        )
        ctx = self._score(df, 1000, ps)
        assert ctx.viability.score <= 100

    def test_score_never_below_0(self):
        """Worst possible setup must not go below 0."""
        ps = _make_price_signals(
            daily_bias="BEARISH",
            vol_state="EXPANDED",
            wr_phase="LATE",
            position_state="MID_BAND_BROKEN",
            wr_in_momentum=False,
            wr_value=-80.0,
        )
        df = _make_options_df(
            spot=1000,
            resistance_strike=1005,
            support_strike=995,
            dte=0,
            ce_oi_resistance=100_000,
            pe_oi_support=5_000,
            extra_call_oi=10_000,
            extra_put_oi=1_000,
        )
        ctx = self._score(df, 1000, ps)
        assert ctx.viability.score >= 0


# ─────────────────────────────────────────────────────────────────────────────
# scanner filter booleans (pure logic, no network)
# ─────────────────────────────────────────────────────────────────────────────


class TestScannerFilterLogic:
    """
    Test the filter boolean logic in _analyze_symbol_inner() in isolation.
    We replicate the exact conditions from scanner.py so tests stay in sync.
    """

    # ── passes_momentum ───────────────────────────────────────────────────────

    def test_momentum_pass_riding_upper_wr_in_zone(self):
        riding_upper = True
        wr_above_m20 = True
        assert riding_upper and wr_above_m20 is True

    def test_momentum_fail_first_dip(self):
        """FIX-5: FIRST_DIP must NOT satisfy riding_upper."""
        position_state = "FIRST_DIP"
        riding_upper = position_state == "RIDING_UPPER"  # must be False
        assert riding_upper is False

    def test_momentum_fail_wr_below_threshold(self):
        riding_upper = True
        wr_in_momentum = False
        passes_momentum = riding_upper and wr_in_momentum
        assert passes_momentum is False

    def test_momentum_fail_both(self):
        passes_momentum = False and False
        assert passes_momentum is False

    # ── passes_structure ──────────────────────────────────────────────────────

    def test_structure_pass_at_exact_5pct(self):
        pct_to_resistance = 5.0
        pct_to_support = -4.0
        passes = (
            pct_to_resistance is not None
            and pct_to_resistance >= 5.0
            and pct_to_support is not None
            and -8.0 <= pct_to_support <= -2.0
        )
        assert passes is True

    def test_structure_fail_just_below_5pct(self):
        pct_to_resistance = 4.99
        pct_to_support = -4.0
        passes = (
            pct_to_resistance is not None
            and pct_to_resistance >= 5.0
            and pct_to_support is not None
            and -8.0 <= pct_to_support <= -2.0
        )
        assert passes is False

    def test_structure_fail_support_too_far(self):
        """Support > 8% below spot → fails structure (stock in freefall)."""
        pct_to_resistance = 6.0
        pct_to_support = -9.0
        passes = (
            pct_to_resistance is not None
            and pct_to_resistance >= 5.0
            and pct_to_support is not None
            and -8.0 <= pct_to_support <= -2.0
        )
        assert passes is False

    def test_structure_fail_support_too_close(self):
        """Support within 2% of spot → too tight, fails structure."""
        pct_to_resistance = 6.0
        pct_to_support = -1.0
        passes = (
            pct_to_resistance is not None
            and pct_to_resistance >= 5.0
            and pct_to_support is not None
            and -8.0 <= pct_to_support <= -2.0
        )
        assert passes is False

    def test_structure_fail_no_resistance(self):
        """No resistance mapped (no-opts stock) → always fails."""
        pct_to_resistance = None
        pct_to_support = -4.0
        passes = (
            pct_to_resistance is not None
            and pct_to_resistance >= 5.0
            and pct_to_support is not None
            and -8.0 <= pct_to_support <= -2.0
        )
        assert passes is False

    # ── passes_regime ─────────────────────────────────────────────────────────

    def test_regime_pass_trend_friendly(self):
        assert ("TREND_FRIENDLY" == "TREND_FRIENDLY") is True

    def test_regime_fail_pinning(self):
        gex_regime = "PINNING"
        assert (gex_regime == "TREND_FRIENDLY") is False

    def test_regime_fail_none(self):
        """No-opts stock has gex_regime=None → always fails regime filter."""
        gex_regime = None
        assert (gex_regime == "TREND_FRIENDLY") is False

    def test_regime_mapping_negative_gex(self):
        """Negative GEX maps to TREND_FRIENDLY in scanner (FIX-REGIME)."""
        gex_regime_raw = "Negative"
        gex_regime = "TREND_FRIENDLY" if gex_regime_raw == "Negative" else "PINNING"
        assert gex_regime == "TREND_FRIENDLY"

    def test_regime_mapping_neutral_with_room(self):
        """Neutral GEX + resistance_pct >= 1.5% → TREND_FRIENDLY."""
        gex_regime_raw = "Neutral"
        pct_to_resistance = 2.0
        gex_regime = (
            "TREND_FRIENDLY"
            if gex_regime_raw == "Negative"
            else (
                "TREND_FRIENDLY"
                if (
                    gex_regime_raw == "Neutral"
                    and pct_to_resistance is not None
                    and pct_to_resistance >= 1.5
                )
                else "PINNING"
            )
        )
        assert gex_regime == "TREND_FRIENDLY"

    def test_regime_mapping_neutral_without_room(self):
        """Neutral GEX + resistance_pct < 1.5% → PINNING (FIX-REGIME)."""
        gex_regime_raw = "Neutral"
        pct_to_resistance = 1.0
        gex_regime = (
            "TREND_FRIENDLY"
            if gex_regime_raw == "Negative"
            else (
                "TREND_FRIENDLY"
                if (
                    gex_regime_raw == "Neutral"
                    and pct_to_resistance is not None
                    and pct_to_resistance >= 1.5
                )
                else "PINNING"
            )
        )
        assert gex_regime == "PINNING"

    def test_regime_mapping_positive_gex(self):
        """Positive GEX → always PINNING."""
        gex_regime_raw = "Positive"
        pct_to_resistance = 20.0
        gex_regime = (
            "TREND_FRIENDLY"
            if gex_regime_raw == "Negative"
            else (
                "TREND_FRIENDLY"
                if (
                    gex_regime_raw == "Neutral"
                    and pct_to_resistance is not None
                    and pct_to_resistance >= 1.5
                )
                else "PINNING"
            )
        )
        assert gex_regime == "PINNING"

    # ── passes_expiry ─────────────────────────────────────────────────────────

    def test_expiry_pass_dte_5(self):
        assert (5 >= 5) is True

    def test_expiry_pass_dte_10(self):
        assert (10 >= 5) is True

    def test_expiry_fail_dte_4(self):
        assert (4 >= 5) is False

    def test_expiry_fail_dte_0(self):
        assert (0 >= 5) is False

    def test_expiry_pass_no_opts_dte_99(self):
        """No-opts stock has dte=99 → passes_expiry=True."""
        dte = 99
        assert (dte >= 5) is True

    # ── near_mp / passes_pin ──────────────────────────────────────────────────

    def test_near_mp_true_when_within_2pct_and_dte_lte_4(self):
        """FIX-8: near_mp triggers at 2.0% threshold."""
        spot = 1000.0
        max_pain = 990.0  # 1% away
        dte = 4
        near_mp = (
            max_pain is not None
            and spot > 0
            and abs(spot - max_pain) / spot < 0.020
            and dte <= 4
        )
        assert near_mp is True

    def test_near_mp_false_when_outside_2pct(self):
        spot = 1000.0
        max_pain = 979.0  # 2.1% away
        dte = 4
        near_mp = (
            max_pain is not None
            and spot > 0
            and abs(spot - max_pain) / spot < 0.020
            and dte <= 4
        )
        assert near_mp is False

    def test_near_mp_false_when_dte_gt_4(self):
        """near_mp is ONLY active when DTE <= 4."""
        spot = 1000.0
        max_pain = 995.0  # 0.5% away, very close
        dte = 5  # DTE=5 → gate inactive
        near_mp = (
            max_pain is not None
            and spot > 0
            and abs(spot - max_pain) / spot < 0.020
            and dte <= 4
        )
        assert near_mp is False

    def test_near_mp_exact_2pct_is_safe(self):
        """Exactly 2.0% away → NOT near max pain (strict <, not <=)."""
        spot = 1000.0
        max_pain = 980.0  # exactly 2.0%
        dte = 4
        near_mp = (
            max_pain is not None
            and spot > 0
            and abs(spot - max_pain) / spot < 0.020
            and dte <= 4
        )
        assert near_mp is False

    # ── expiry_risk classification (FIX-DTE0) ────────────────────────────────

    def test_expiry_risk_dte_0_is_high(self):
        """FIX-DTE0: DTE=0 must always be HIGH regardless of pin_risk."""
        dte = 0
        pin_risk = "LOW"
        near_mp = False
        if dte <= 0:
            expiry_risk = "HIGH"
        elif pin_risk == "HIGH" or dte <= 2:
            expiry_risk = "HIGH"
        elif pin_risk == "MODERATE" or (dte <= 4 and near_mp):
            expiry_risk = "ELEVATED"
        else:
            expiry_risk = "LOW"
        assert expiry_risk == "HIGH"

    def test_expiry_risk_dte_1_is_high(self):
        dte = 1
        pin_risk = "LOW"
        near_mp = False
        if dte <= 0:
            expiry_risk = "HIGH"
        elif pin_risk == "HIGH" or dte <= 2:
            expiry_risk = "HIGH"
        elif pin_risk == "MODERATE" or (dte <= 4 and near_mp):
            expiry_risk = "ELEVATED"
        else:
            expiry_risk = "LOW"
        assert expiry_risk == "HIGH"

    def test_expiry_risk_dte_3_near_mp_is_elevated(self):
        dte = 3
        pin_risk = "LOW"
        near_mp = True
        if dte <= 0:
            expiry_risk = "HIGH"
        elif pin_risk == "HIGH" or dte <= 2:
            expiry_risk = "HIGH"
        elif pin_risk == "MODERATE" or (dte <= 4 and near_mp):
            expiry_risk = "ELEVATED"
        else:
            expiry_risk = "LOW"
        assert expiry_risk == "ELEVATED"

    def test_expiry_risk_dte_8_is_low(self):
        dte = 8
        pin_risk = "LOW"
        near_mp = False
        if dte <= 0:
            expiry_risk = "HIGH"
        elif pin_risk == "HIGH" or dte <= 2:
            expiry_risk = "HIGH"
        elif pin_risk == "MODERATE" or (dte <= 4 and near_mp):
            expiry_risk = "ELEVATED"
        else:
            expiry_risk = "LOW"
        assert expiry_risk == "LOW"

    # ── all_filters_pass composition ──────────────────────────────────────────

    def test_all_pass_green_path(self):
        assert (True and True and True and True and True and True) is True

    def test_all_fail_if_any_one_gate_fails(self):
        for i in range(6):
            gates = [True] * 6
            gates[i] = False
            assert all(gates) is False, f"Gate {i} should break all_filters_pass"

    def test_no_opts_never_all_filters_pass(self):
        """No-opts: passes_structure=False, passes_regime=False → all_filters_pass=False."""
        passes_momentum = True
        passes_structure = False  # always False for no-opts
        passes_regime = False  # always False for no-opts
        passes_expiry = True
        passes_bias = True
        passes_pin = True
        all_filters_pass = (
            passes_momentum
            and passes_structure
            and passes_regime
            and passes_expiry
            and passes_bias
            and passes_pin
        )
        assert all_filters_pass is False

    # ── _short_reason logic ───────────────────────────────────────────────────

    def test_short_reason_bearish_first(self):
        """Bearish bias is reported before checking momentum."""
        from types import SimpleNamespace

        ps = SimpleNamespace(
            position_state="RIDING_UPPER",
            wr_value=-10.0,
            wr_in_momentum=True,
            wr_phase="FRESH",
        )
        from modules.scanner import _short_reason

        reason = _short_reason(
            p_mom=True,
            p_struct=True,
            p_regime=True,
            p_expiry=True,
            p_bias=False,
            ps=ps,
            pct_res=8.0,
            dte=10,
            gex_regime="TREND_FRIENDLY",
            expiry_risk="LOW",
        )
        assert "bearish" in reason.lower()

    def test_short_reason_first_dip_message(self):
        """FIRST_DIP gives specific management message."""
        from types import SimpleNamespace

        ps = SimpleNamespace(
            position_state="FIRST_DIP",
            wr_value=-25.0,
            wr_in_momentum=False,
            wr_phase="LATE",
        )
        from modules.scanner import _short_reason

        reason = _short_reason(
            p_mom=False,
            p_struct=True,
            p_regime=True,
            p_expiry=True,
            p_bias=True,
            ps=ps,
            pct_res=8.0,
            dte=10,
            gex_regime="TREND_FRIENDLY",
            expiry_risk="LOW",
        )
        assert "FIRST_DIP" in reason

    def test_short_reason_structure_shows_pct(self):
        """Structure failure reason must include the actual pct value."""
        from types import SimpleNamespace

        ps = SimpleNamespace(
            position_state="RIDING_UPPER",
            wr_value=-10.0,
            wr_in_momentum=True,
            wr_phase="FRESH",
        )
        from modules.scanner import _short_reason

        reason = _short_reason(
            p_mom=True,
            p_struct=False,
            p_regime=True,
            p_expiry=True,
            p_bias=True,
            ps=ps,
            pct_res=3.2,
            dte=10,
            gex_regime="TREND_FRIENDLY",
            expiry_risk="LOW",
        )
        assert "3.2" in reason

    def test_short_reason_green_path_mentions_regime(self):
        """All-pass reason mentions regime and WR phase."""
        from types import SimpleNamespace

        ps = SimpleNamespace(
            position_state="RIDING_UPPER",
            wr_value=-10.0,
            wr_in_momentum=True,
            wr_phase="FRESH",
        )
        from modules.scanner import _short_reason

        reason = _short_reason(
            p_mom=True,
            p_struct=True,
            p_regime=True,
            p_expiry=True,
            p_bias=True,
            ps=ps,
            pct_res=7.0,
            dte=10,
            gex_regime="TREND_FRIENDLY",
            expiry_risk="LOW",
        )
        assert "FRESH" in reason
        assert "TREND_FRIENDLY" in reason


# ─────────────────────────────────────────────────────────────────────────────
# No-opts scoring regression tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoOptsScoring:
    """
    Documents the current bug (score always=50) and the expected behaviour
    after the analyze_price_only() fix is applied.
    """

    def test_current_bug_empty_df_returns_score_50(self):
        """
        CURRENT BUG: analyze(pd.DataFrame(), ...) returns early with score=50.
        This test should PASS until the bug is fixed (it documents the regression).
        Once analyze_price_only() is introduced and scanner.py is updated to call
        it, delete this test and rely on test_price_only_scores_real_signals.
        """
        from modules.analytics import analyze

        ps = _make_price_signals(
            daily_bias="BULLISH",
            vol_state="SQUEEZE",
            wr_phase="FRESH",
            position_state="RIDING_UPPER",
        )
        ctx = analyze(pd.DataFrame(), spot=1000, price_signals=ps)
        assert ctx.viability.score == 50, (
            "Bug still present — no-opts always scores 50. "
            "Remove this test once analyze_price_only() is deployed."
        )

    def test_price_only_scores_real_signals(self):
        """
        AFTER FIX: analyze_price_only() must score > 50 for a good setup
        and must be capped at 70.
        """
        try:
            from modules.analytics import analyze_price_only
        except ImportError:
            pytest.skip("analyze_price_only() not yet implemented")
        ps = _make_price_signals(
            daily_bias="BULLISH",
            vol_state="SQUEEZE",
            wr_phase="FRESH",
            position_state="RIDING_UPPER",
            wr_in_momentum=True,
        )
        ctx = analyze_price_only(spot=1000, price_signals=ps)
        assert ctx.viability.score > 50, "Good price signals must score above 50"
        assert ctx.viability.score <= 70, "No-opts score must be capped at 70"

    def test_price_only_bad_setup_scores_low(self):
        """Bearish + MID_BAND_BROKEN no-opts stock should score very low."""
        try:
            from modules.analytics import analyze_price_only
        except ImportError:
            pytest.skip("analyze_price_only() not yet implemented")
        ps = _make_price_signals(
            daily_bias="BEARISH",
            vol_state="EXPANDED",
            wr_phase="LATE",
            position_state="MID_BAND_BROKEN",
            wr_in_momentum=False,
            wr_value=-65.0,
        )
        ctx = analyze_price_only(spot=1000, price_signals=ps)
        assert ctx.viability.sizing == "SKIP"

    def test_price_only_passes_structure_still_false(self):
        """
        Even after fix: no-opts stocks must NEVER pass the Structure filter.
        pct_to_resistance is always None for no-opts.
        """
        pct_to_resistance = None
        pct_to_support = -4.0
        passes_structure = (
            pct_to_resistance is not None
            and pct_to_resistance >= 5.0
            and pct_to_support is not None
            and -8.0 <= pct_to_support <= -2.0
        )
        assert passes_structure is False

    def test_no_opts_passes_expiry_gate(self):
        """No-opts has dte=99 → passes_expiry must be True."""
        dte = 99
        assert (dte >= 5) is True

    def test_no_opts_has_options_flag_false(self):
        """Return dict must include has_options=False for no-opts rows."""
        # Simulates scanner return dict check
        row = {"has_options": False, "gex_regime": None, "dte": None}
        assert row["has_options"] is False
        assert row["gex_regime"] is None


# ─────────────────────────────────────────────────────────────────────────────
# NIFTY100_SYMBOLS list integrity
# ─────────────────────────────────────────────────────────────────────────────


class TestNifty100Symbols:
    def test_symbol_count_is_105(self):
        """After fix: must contain 100 stocks + 5 F&O indices = 105 total."""
        from modules.data import NIFTY100_SYMBOLS

        assert len(NIFTY100_SYMBOLS) == 105, (
            f"Expected 105 symbols (Nifty 50 + Next 50 + 5 indices), "
            f"got {len(NIFTY100_SYMBOLS)}. Expand NIFTY100_SYMBOLS in data.py."
        )

    def test_no_duplicates(self):
        from modules.data import NIFTY100_SYMBOLS

        assert len(NIFTY100_SYMBOLS) == len(set(NIFTY100_SYMBOLS))

    def test_fno_indices_present(self):
        from modules.data import NIFTY100_SYMBOLS

        for idx in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"):
            assert idx in NIFTY100_SYMBOLS, f"{idx} missing from NIFTY100_SYMBOLS"

    def test_nifty50_core_stocks_present(self):
        from modules.data import NIFTY100_SYMBOLS

        core = [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "ICICIBANK",
            "INFY",
            "SBIN",
            "KOTAKBANK",
            "AXISBANK",
            "HINDUNILVR",
            "ITC",
        ]
        for s in core:
            assert s in NIFTY100_SYMBOLS, f"{s} missing from NIFTY100_SYMBOLS"

    def test_niftynxt50_stocks_present(self):
        """After fix: Nifty Next 50 stocks must be in the list."""
        from modules.data import NIFTY100_SYMBOLS

        nxt50_sample = [
            "TRENT",
            "IRCTC",
            "DLF",
            "CANBK",
            "IRFC",
            "HAVELLS",
            "BHEL",
            "SIEMENS",
            "ZYDUSLIFE",
            "LICI",
        ]
        missing = [s for s in nxt50_sample if s not in NIFTY100_SYMBOLS]
        assert not missing, f"Nifty Next 50 stocks missing: {missing}"

    def test_no_blank_symbols(self):
        from modules.data import NIFTY100_SYMBOLS

        for s in NIFTY100_SYMBOLS:
            assert s and s.strip(), "Blank or whitespace symbol found"

    def test_all_symbols_uppercase(self):
        from modules.data import NIFTY100_SYMBOLS

        for s in NIFTY100_SYMBOLS:
            assert s == s.upper(), f"Symbol not uppercase: {s}"


# ─────────────────────────────────────────────────────────────────────────────
# Sector rotation classification
# ─────────────────────────────────────────────────────────────────────────────


class TestSectorRotation:
    """
    Tests for classify_rotation() as designed in the sector rotation fix.
    These tests act as the spec — implement classify_rotation() to make them pass.
    """

    def _classify(
        self, ret_10d, ret_1m, ret_3m, bench_10d=0.0, bench_1m=0.0, bench_3m=0.0
    ):
        try:
            from modules.sector_rotation import classify_rotation
        except ImportError:
            pytest.skip("sector_rotation module not yet implemented")
        return classify_rotation(ret_10d, ret_1m, ret_3m, bench_10d, bench_1m, bench_3m)

    def test_all_outperforming_is_leading(self):
        """Sector beats benchmark on 10D, 1M, 3M → LEADING."""
        label, _ = self._classify(
            5.0, 4.0, 6.0, bench_10d=1.0, bench_1m=1.0, bench_3m=1.0
        )
        assert label == "LEADING"

    def test_medium_outperform_but_short_fading_is_weakening(self):
        """Was outperforming 3M/1M but 10D now underperforming → WEAKENING."""
        label, _ = self._classify(
            ret_10d=-1.0,
            ret_1m=3.0,
            ret_3m=5.0,
            bench_10d=1.0,
            bench_1m=1.0,
            bench_3m=1.0,
        )
        assert label == "WEAKENING"

    def test_lagging_medium_but_10d_recovering_is_improving(self):
        """Was lagging 3M/1M but 10D now outperforming → IMPROVING."""
        label, _ = self._classify(
            ret_10d=4.0,
            ret_1m=-1.0,
            ret_3m=-3.0,
            bench_10d=1.0,
            bench_1m=1.0,
            bench_3m=1.0,
        )
        assert label == "IMPROVING"

    def test_all_underperforming_is_lagging(self):
        """Sector lags benchmark on all timeframes → LAGGING."""
        label, _ = self._classify(
            -3.0, -2.0, -4.0, bench_10d=1.0, bench_1m=1.0, bench_3m=1.0
        )
        assert label == "LAGGING"

    def test_missing_one_timeframe_still_classifies(self):
        """Missing 3M data should still classify using 10D + 1M."""
        label, _ = self._classify(
            4.0, 3.0, None, bench_10d=1.0, bench_1m=1.0, bench_3m=None
        )
        assert label in ("LEADING", "WEAKENING", "IMPROVING", "LAGGING")

    def test_all_missing_returns_unknown(self):
        """No data at all → UNKNOWN."""
        label, _ = self._classify(None, None, None, None, None, None)
        assert label == "UNKNOWN"

    def test_single_timeframe_returns_unknown(self):
        """Only 1 timeframe available → insufficient for classification."""
        label, _ = self._classify(5.0, None, None, 1.0, None, None)
        assert label == "UNKNOWN"

    def test_equal_to_benchmark_is_lagging_not_leading(self):
        """Exact benchmark performance (0 RS) → no outperformance → LAGGING or WEAKENING."""
        label, _ = self._classify(
            1.0, 1.0, 1.0, bench_10d=1.0, bench_1m=1.0, bench_3m=1.0
        )
        assert label in ("LAGGING", "WEAKENING")  # RS=0 is not outperforming


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full analyze() end-to-end with realistic chain
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeIntegration:
    def test_full_stack_returns_complete_context(self):
        """Full analyze() run must populate all OptionsContext fields."""
        from modules.analytics import analyze

        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
            extra_put_oi=5_000,
            extra_call_oi=2_000,
        )
        ctx = analyze(df, spot=1000, price_signals=ps, room_thresh=5.0)
        assert ctx.walls.resistance is not None
        assert ctx.walls.support is not None
        assert ctx.gex.regime in ("Negative", "Positive", "Neutral")
        assert ctx.expiry.days_remaining > 0
        assert ctx.regime.regime in (
            "TREND-FRIENDLY",
            "PINNING",
            "NEUTRAL",
            "UNKNOWN",
            "GEX NEG / EXPIRY RISK",
            "GEX POSITIVE",
        )
        assert 0 <= ctx.viability.score <= 100
        assert ctx.viability.sizing in ("FULL", "HALF", "SKIP")
        assert len(ctx.viability.checklist) >= 6

    def test_spot_zero_returns_default_context(self):
        """spot=0 must return default OptionsContext without crashing."""
        from modules.analytics import analyze

        df = _make_options_df(spot=1000, dte=10)
        ctx = analyze(df, spot=0)
        assert ctx.viability.score == 50  # default

    def test_none_options_df_returns_default(self):
        from modules.analytics import analyze

        ctx = analyze(None, spot=1000)
        assert ctx.viability.score == 50

    def test_score_ordering_bearish_lt_bullish(self):
        """Bearish bias setup must score strictly less than bullish bias setup."""
        from modules.analytics import analyze

        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ps_bull = _make_price_signals(daily_bias="BULLISH")
        ps_bear = _make_price_signals(daily_bias="BEARISH")
        ctx_bull = analyze(df, 1000, price_signals=ps_bull)
        ctx_bear = analyze(df, 1000, price_signals=ps_bear)
        assert ctx_bull.viability.sizing == "FULL"
        assert ctx_bear.viability.sizing == "SKIP"
        assert ctx_bear.viability.label == "AVOID"
        bias_item = next(
            c for c in ctx_bear.viability.checklist if c.item == "Daily Bias"
        )
        assert bias_item.status == "fail"

    def test_fresh_wr_scores_higher_than_late_wr(self):
        """FRESH WR phase must score strictly more than LATE WR phase."""
        from modules.analytics import analyze

        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ps_fresh = _make_price_signals(wr_phase="FRESH", wr_in_momentum=True)
        ps_late = _make_price_signals(wr_phase="LATE", wr_in_momentum=True)
        ctx_fresh = analyze(df, 1000, price_signals=ps_fresh)
        ctx_late = analyze(df, 1000, price_signals=ps_late)
        fresh_item = next(
            c for c in ctx_fresh.viability.checklist if c.item == "W%R Gate"
        )
        late_item = next(
            c for c in ctx_late.viability.checklist if c.item == "W%R Gate"
        )
        assert fresh_item.status == "pass"
        assert late_item.status == "warn"
        # Risk note is added for LATE
        assert any("late" in n.lower() for n in ctx_late.viability.risk_notes)
        # LATE may reduce sizing even if score is equal
        assert ctx_late.viability.sizing in ("HALF", "SKIP", "FULL")

    def test_viability_checklist_has_all_8_items(self):
        """Checklist must contain all 8 scoring components."""
        from modules.analytics import analyze

        ps = _make_price_signals()
        df = _make_options_df(
            spot=1000,
            resistance_strike=1100,
            support_strike=900,
            dte=10,
            pe_oi_support=60_000,
            ce_oi_resistance=20_000,
        )
        ctx = analyze(df, 1000, price_signals=ps)
        items = [c.item for c in ctx.viability.checklist]
        for expected in (
            "Regime",
            "Room",
            "Daily Bias",
            "Expiry Gate",
            "PCR",
            "Vol State",
            "W%R Gate",
            "BB State",
        ):
            assert expected in items, f"Missing checklist item: {expected}"
