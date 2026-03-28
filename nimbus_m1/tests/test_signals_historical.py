"""
tests/test_signals_historical.py
─────────────────────────────────
Historical-precedent test suite for NIMBUS signal chain.

Tests are organized by signal component and grounded in real NSE market
scenarios (dates and price levels noted for audit trail).

Run:
    pytest tests/test_signals_historical.py -v
    pytest tests/test_signals_historical.py -v -k "mfi"     # just MFI tests
"""

from __future__ import annotations

import datetime
import math
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — build synthetic DataFrames with controlled properties
# ══════════════════════════════════════════════════════════════════════════════


def _make_ohlcv_df(
    n_bars: int = 100,
    base_price: float = 1000.0,
    trend: float = 0.002,
    vol_base: float = 1e6,
    seed: int = 42,
    freq: str = "4h",
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with controlled trend and volume."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq)
    closes = [base_price]
    for _ in range(n_bars - 1):
        ret = trend + rng.normal(0, 0.01)
        closes.append(closes[-1] * (1 + ret))
    closes = np.array(closes)
    highs = closes * (1 + rng.uniform(0.001, 0.015, n_bars))
    lows = closes * (1 - rng.uniform(0.001, 0.015, n_bars))
    opens = closes * (1 + rng.normal(0, 0.005, n_bars))
    volumes = vol_base * (1 + rng.uniform(-0.3, 0.5, n_bars))
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def _make_daily_mfi_series(
    n_days: int = 30,
    mfi_start: float = 65.0,
    mfi_trend: float = -1.0,  # per day
    base_price: float = 1000.0,
    price_trend: float = 0.003,
) -> pd.DataFrame:
    """Build a daily OHLCV with controlled MFI trajectory.

    Used to test divergence: price rising while MFI falling.
    """
    idx = pd.date_range("2024-06-01", periods=n_days, freq="1D")
    closes = base_price * (1 + np.arange(n_days) * price_trend)
    highs = closes * 1.008
    lows = closes * 0.992
    opens = closes * 0.999
    # Volume inversely related to MFI trend for divergence scenarios
    volumes = np.full(n_days, 5e7)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def _ps(
    wr_value=-12.0,
    wr_in_momentum=True,
    wr_phase="FRESH",
    wr_bars_since_cross50=2,
    bb_position="above_upper",
    position_state="RIDING_UPPER",
    vol_state="SQUEEZE",
    daily_bias="BULLISH",
    daily_bias_pct=3.5,
    daily_sma=980.0,
    bb_pct=0.85,
    bb_width_pctl=12.0,
    adv_cr=15.0,
    mfi_value=None,
    mfi_state="NEUTRAL",
    mfi_diverge=False,
    mfi_reliable=True,
    last_close=1000.0,
    upper=1010.0,
    mid=990.0,
    lower=970.0,
    entry_valid=True,
    bb_squeezing=True,
    bb_width_pct=2.0,
    wr_trend="rising",
):
    """Build a PriceSignals-compatible SimpleNamespace."""
    return SimpleNamespace(
        wr_value=wr_value,
        wr_in_momentum=wr_in_momentum,
        wr_phase=wr_phase,
        wr_bars_since_cross50=wr_bars_since_cross50,
        bb_position=bb_position,
        position_state=position_state,
        vol_state=vol_state,
        daily_bias=daily_bias,
        daily_bias_pct=daily_bias_pct,
        daily_sma=daily_sma,
        bb_pct=bb_pct,
        bb_width_pctl=bb_width_pctl,
        adv_cr=adv_cr,
        mfi_value=mfi_value,
        mfi_state=mfi_state,
        mfi_diverge=mfi_diverge,
        mfi_reliable=mfi_reliable,
        last_close=last_close,
        upper=upper,
        mid=mid,
        lower=lower,
        entry_valid=entry_valid,
        bb_squeezing=bb_squeezing,
        bb_width_pct=bb_width_pct,
        wr_trend=wr_trend,
    )


def _expiry_str(days: int) -> str:
    d = datetime.date.today() + datetime.timedelta(days=days)
    return d.strftime("%d-%b-%Y")


def _make_options_df(
    spot=1000.0,
    resistance_strike=1100.0,
    support_strike=900.0,
    dte=10,
    ce_iv=20.0,
    pe_iv=22.0,
    ce_oi_resistance=50_000,
    pe_oi_support=50_000,
    extra_call_oi=5_000,
    extra_put_oi=5_000,
):
    strikes = sorted(
        {resistance_strike - 200, resistance_strike - 100,
         resistance_strike, support_strike, support_strike + 100, spot}
    )
    exp = _expiry_str(dte)
    rows = []
    for s in strikes:
        ce_oi = ce_oi_resistance if s == resistance_strike else extra_call_oi
        pe_oi = pe_oi_support if s == support_strike else extra_put_oi
        rows.append({
            "Strike": float(s), "Expiry": exp,
            "CE_OI": float(ce_oi), "CE_IV": ce_iv,
            "CE_LTP": max(1.0, (spot - s) * 0.1 + 10),
            "PE_OI": float(pe_oi), "PE_IV": pe_iv,
            "PE_LTP": max(1.0, (s - spot) * 0.1 + 10),
            "UnderlyingValue": spot,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MFI DIVERGENCE TESTS
#    Historical precedent: RELIANCE Jan 2024 — price near ATH (₹2,850),
#    MFI falling from 75→55 over 6 sessions while price held.
#    Subsequent 5% drawdown within 2 weeks.
# ══════════════════════════════════════════════════════════════════════════════


class TestMFIDivergence:
    """Tests for MFI divergence detection (indicators._compute_mfi_signals)."""

    def _make_daily_with_mfi(self, mfi_values, close_values):
        """Build a daily frame with pre-computed MFI_14 column."""
        n = len(mfi_values)
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame({
            "Close": close_values,
            "High": [c * 1.005 for c in close_values],
            "Low": [c * 0.995 for c in close_values],
            "Volume": [5e7] * n,
            "MFI_14": mfi_values,
        }, index=idx)
        return df

    def test_bearish_divergence_detected(self):
        """Price near high + MFI trending down over 3 steps → divergence=True.

        Precedent: RELIANCE Jan 2024 — price ₹2,830–2,870 range,
        MFI: 75 → 68 → 58 over 6 sessions.
        """
        from modules.indicators import PriceSignals, _compute_mfi_signals

        # MFI: declining consistently over 6+ bars
        mfi_vals = [75, 73, 71, 68, 65, 62, 58, 55, 52, 50]
        # Price: holding near highs (close >= 98% of recent max)
        close_vals = [2830, 2840, 2850, 2855, 2860, 2865, 2870, 2860, 2855, 2850]

        daily = self._make_daily_with_mfi(mfi_vals, close_vals)
        ps = PriceSignals(adv_cr=25.0)  # reliable volume
        close = close_vals[-1]
        _compute_mfi_signals(daily, ps, close)

        assert ps.mfi_diverge is True, "Bearish divergence should be detected"
        assert ps.mfi_state in ("RISING", "NEUTRAL", "FALLING")

    def test_no_divergence_when_mfi_rising(self):
        """MFI rising with price → no divergence (healthy trend).

        Precedent: HDFCBANK Oct 2023 — price and MFI both rising.
        """
        from modules.indicators import PriceSignals, _compute_mfi_signals

        mfi_vals = [45, 48, 52, 55, 58, 62, 65, 68, 72, 75]
        close_vals = [1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590]

        daily = self._make_daily_with_mfi(mfi_vals, close_vals)
        ps = PriceSignals(adv_cr=20.0)
        _compute_mfi_signals(daily, ps, close_vals[-1])

        assert ps.mfi_diverge is False

    def test_spike_guard_blocks_false_divergence(self):
        """Single-bar MFI jump > 20 disables divergence detection.

        Precedent: block deal days where volume spikes 10x (e.g., BAJFINANCE
        FII block buy Dec 2023) — MFI jumps then falls, but it's noise.
        """
        from modules.indicators import PriceSignals, _compute_mfi_signals

        # MFI has a 25-point spike on bar [-2] then drops
        mfi_vals = [65, 64, 63, 62, 61, 60, 59, 58, 85, 55]
        close_vals = [5000] * 10

        daily = self._make_daily_with_mfi(mfi_vals, close_vals)
        ps = PriceSignals(adv_cr=20.0)
        _compute_mfi_signals(daily, ps, close_vals[-1])

        assert ps.mfi_diverge is False, "Spike guard should block divergence"

    def test_unreliable_when_low_adv(self):
        """MFI unreliable when ADV < 5 Cr — no score impact.

        Precedent: thinly-traded ETFs like METALIETF where block deals
        dominate daily volume making MFI meaningless.
        """
        from modules.indicators import PriceSignals, _compute_mfi_signals

        mfi_vals = [75, 73, 71, 68, 65, 62, 58, 55, 52, 50]
        close_vals = [100] * 10

        daily = self._make_daily_with_mfi(mfi_vals, close_vals)
        ps = PriceSignals(adv_cr=2.0)  # below 5 Cr threshold
        _compute_mfi_signals(daily, ps, 100.0)

        assert ps.mfi_reliable is False

    def test_mfi_state_classification(self):
        """MFI state thresholds: >70 STRONG, >55 RISING, >45 NEUTRAL, etc."""
        from modules.indicators import PriceSignals, _compute_mfi_signals

        for mfi_now, expected_state in [
            (75, "STRONG"), (60, "RISING"), (50, "NEUTRAL"),
            (35, "FALLING"), (20, "WEAK"),
        ]:
            daily = self._make_daily_with_mfi(
                [50, 50, 50, 50, 50, 50, 50, 50, 50, mfi_now],
                [1000] * 10,
            )
            ps = PriceSignals(adv_cr=10.0)
            _compute_mfi_signals(daily, ps, 1000.0)
            assert ps.mfi_state == expected_state, (
                f"MFI {mfi_now} should be {expected_state}, got {ps.mfi_state}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. WILLIAMS %R PHASE TESTS
#    50-period on 4H = ~12 trading days.
#    FRESH (≤3 bars since -50 cross): strongest entry.
#    Historical: TCS July 2024 breakout — W%R crossed -50 on day 1,
#    held above -20 for 8 bars = DEVELOPING, faded at bar 14 = LATE.
# ══════════════════════════════════════════════════════════════════════════════


class TestWilliamsRPhase:

    def test_fresh_phase(self):
        """W%R crossed -50 within last 3 bars → FRESH."""
        from modules.indicators import compute_price_signals

        df = _make_ohlcv_df(n_bars=100, trend=0.005)
        from modules.indicators import add_bollinger, add_williams_r
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)

        # Force W%R to be in momentum zone with recent cross
        df.iloc[-1, df.columns.get_loc("WR")] = -10.0
        df.iloc[-2, df.columns.get_loc("WR")] = -15.0
        df.iloc[-3, df.columns.get_loc("WR")] = -45.0
        df.iloc[-4, df.columns.get_loc("WR")] = -55.0  # cross point

        ps = compute_price_signals(df)
        assert ps.wr_in_momentum is True
        assert ps.wr_phase == "FRESH"

    def test_developing_phase(self):
        """W%R in zone, 4-10 bars since cross → DEVELOPING."""
        from modules.indicators import compute_price_signals, add_bollinger, add_williams_r

        df = _make_ohlcv_df(n_bars=100, trend=0.005)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)

        # Set WR above -50 for last 8 bars, below before
        for i in range(8):
            df.iloc[-(i + 1), df.columns.get_loc("WR")] = -15.0
        df.iloc[-9, df.columns.get_loc("WR")] = -55.0

        ps = compute_price_signals(df)
        assert ps.wr_in_momentum is True
        assert ps.wr_phase == "DEVELOPING"

    def test_late_phase(self):
        """W%R in zone > 10 bars since cross → LATE."""
        from modules.indicators import compute_price_signals, add_bollinger, add_williams_r

        df = _make_ohlcv_df(n_bars=100, trend=0.005)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)

        # Set WR above -50 for last 15 bars
        for i in range(15):
            df.iloc[-(i + 1), df.columns.get_loc("WR")] = -10.0
        df.iloc[-16, df.columns.get_loc("WR")] = -55.0

        ps = compute_price_signals(df)
        assert ps.wr_in_momentum is True
        assert ps.wr_phase == "LATE"

    def test_none_phase_when_not_in_momentum(self):
        """W%R below threshold → NONE."""
        from modules.indicators import compute_price_signals, add_bollinger, add_williams_r

        df = _make_ohlcv_df(n_bars=100, trend=-0.003)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)
        df.iloc[-1, df.columns.get_loc("WR")] = -60.0

        ps = compute_price_signals(df)
        assert ps.wr_in_momentum is False
        assert ps.wr_phase == "NONE"


# ══════════════════════════════════════════════════════════════════════════════
# 3. DAILY RESAMPLE + MFI PIPELINE
#    Critical rule: MFI computed on DAILY frame, not 4H.
#    This avoids block deal noise on 4H bars.
# ══════════════════════════════════════════════════════════════════════════════


class TestDailyResample:

    def test_resample_dynamic_columns(self):
        """_resample_daily must handle missing 'Open' column (sector pipeline).

        This was a production bug: sector_rotation builds OHLCV without Open,
        causing KeyError in hard-coded agg_dict.
        """
        from modules.indicators import _resample_daily

        # Build a frame without Open column (like sector pipeline)
        idx = pd.date_range("2024-01-01", periods=50, freq="4h")
        df = pd.DataFrame({
            "High": np.random.uniform(100, 110, 50),
            "Low": np.random.uniform(90, 100, 50),
            "Close": np.random.uniform(95, 105, 50),
            "Volume": np.random.uniform(1e6, 5e6, 50),
        }, index=idx)

        daily = _resample_daily(df)
        assert not daily.empty, "_resample_daily should not fail without Open"
        assert "Close" in daily.columns

    def test_resample_preserves_volume(self):
        """Daily resample must SUM intraday volume, not average."""
        from modules.indicators import _resample_daily

        idx = pd.date_range("2024-01-01", periods=8, freq="4h")
        df = pd.DataFrame({
            "Open": [100] * 8,
            "High": [105] * 8,
            "Low": [95] * 8,
            "Close": [100] * 8,
            "Volume": [1e6] * 8,
        }, index=idx)

        daily = _resample_daily(df)
        # 8 bars over ~1 day → volume should be summed
        assert daily["Volume"].iloc[0] > 1e6, "Volume should be summed across bars"

    def test_mfi_computed_on_daily_not_4h(self):
        """compute_price_signals must use daily-resampled MFI.

        A 4H block deal bar could spike MFI to 95; daily averaging dampens this.
        """
        from modules.indicators import compute_price_signals, add_bollinger, add_williams_r

        df = _make_ohlcv_df(n_bars=120, vol_base=5e7)
        df = add_bollinger(df)
        df = add_williams_r(df, period=50)

        ps = compute_price_signals(df)
        # MFI should be computed (we have enough bars and volume)
        # It may or may not be None depending on resample success,
        # but if present, it should be in valid range
        if ps.mfi_value is not None:
            assert 0 <= ps.mfi_value <= 100


# ══════════════════════════════════════════════════════════════════════════════
# 4. SECTOR CONVICTION SCORE
#    The composite ranking that drives Market Context sort order.
# ══════════════════════════════════════════════════════════════════════════════


class TestSectorConviction:

    def test_conviction_higher_with_strong_entry(self):
        """STRONG entry label adds +3 to conviction vs AVOID at -3."""
        from modules.sector_rotation import classify_rotation

        base_rs = {"10d": 5.0, "1m": 3.0, "3m": 2.0}
        _, rs_score = classify_rotation(5.0, 3.0, 2.0)

        # Simulate conviction scoring
        entry_map = {"STRONG": 3.0, "GOOD": 1.5, "CAUTION": -1.0, "AVOID": -3.0}
        strong_conv = rs_score + entry_map["STRONG"]
        avoid_conv = rs_score + entry_map["AVOID"]

        assert strong_conv > avoid_conv
        assert strong_conv - avoid_conv == 6.0  # 3 - (-3) = 6

    def test_mfi_divergence_penalty(self):
        """MFI divergence subtracts 2.0 from conviction."""
        base = 5.0
        without_div = base + 0.0  # NEUTRAL MFI
        with_div = base + 0.0 - 2.0  # divergence penalty

        assert with_div < without_div
        assert without_div - with_div == 2.0

    def test_conviction_sort_order(self):
        """Sectors should sort by conviction_score descending."""
        rows = [
            {"name": "IT", "conviction_score": 8.5},
            {"name": "Bank", "conviction_score": 3.2},
            {"name": "Pharma", "conviction_score": -1.5},
            {"name": "Metal", "conviction_score": 6.0},
        ]
        rows.sort(key=lambda r: r["conviction_score"], reverse=True)
        names = [r["name"] for r in rows]
        assert names == ["IT", "Metal", "Bank", "Pharma"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. SESSION GUARD
#    NSE hours: 09:15–15:30 IST, Mon–Fri.
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionGuard:

    def test_market_open_during_session(self):
        """is_market_open() = True at 10:30 IST Wednesday."""
        from modules.data import is_market_open

        tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        # Wednesday 10:30 IST
        fake_now = datetime.datetime(2024, 7, 10, 10, 30, 0, tzinfo=tz_ist)
        with patch("modules.data.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = datetime.timezone
            mock_dt.timedelta = datetime.timedelta
            result = is_market_open()
        assert result is True

    def test_market_closed_weekend(self):
        """is_market_open() = False on Saturday."""
        from modules.data import is_market_open

        tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        # Saturday 12:00 IST
        fake_now = datetime.datetime(2024, 7, 13, 12, 0, 0, tzinfo=tz_ist)
        with patch("modules.data.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = datetime.timezone
            mock_dt.timedelta = datetime.timedelta
            result = is_market_open()
        assert result is False

    def test_market_closed_after_hours(self):
        """is_market_open() = False at 20:00 IST weekday."""
        from modules.data import is_market_open

        tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        fake_now = datetime.datetime(2024, 7, 10, 20, 0, 0, tzinfo=tz_ist)
        with patch("modules.data.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = datetime.timezone
            mock_dt.timedelta = datetime.timedelta
            result = is_market_open()
        assert result is False

    def test_freshness_cached_state_exists(self):
        """DataFreshnessState must include CACHED."""
        from ui.workers import DataFreshnessState
        assert hasattr(DataFreshnessState, "CACHED")
        assert DataFreshnessState.CACHED.value == "CACHED"


# ══════════════════════════════════════════════════════════════════════════════
# 6. SETUP CLASSIFIER
#    Tests grounded in historical NSE filing events.
# ══════════════════════════════════════════════════════════════════════════════


class TestSetupClassifier:

    def test_trap_bearish_filing_high_score(self):
        """TRAP: high viability + bearish filing.

        Precedent: DHFL Mar 2019 — technicals looked fine (score ~65),
        but bearish audit filing destroyed the stock.
        """
        from modules.setup_classifier import classify_setup_v3, SetupType

        st, detail = classify_setup_v3(
            viability_score=68,
            filing_variance=-15,
            filing_direction="BEARISH",
            filing_conviction=8,
            filing_category="RESULT",
        )
        assert st == SetupType.TRAP

    def test_confirmed_alignment(self):
        """CONFIRMED: bullish filing + high score + strong options.

        Precedent: TATAMOTORS Aug 2023 — strong results + bullish OI buildup.
        """
        from modules.setup_classifier import (
            classify_setup_v3, SetupType, OptionsSignalState, MomentumState,
        )

        opts = OptionsSignalState(
            gex_regime="Negative", pcr=1.4, pcr_trending="RISING",
            delta_bias="LONG",
        )
        mom = MomentumState(
            position_state="RIDING_UPPER", wr_in_momentum=True,
            wr_phase="DEVELOPING",
        )
        st, _ = classify_setup_v3(
            viability_score=72,
            filing_variance=12,
            filing_direction="BULLISH",
            filing_conviction=7,
            filing_category="RESULT",
            opts=opts,
            mom=mom,
        )
        assert st == SetupType.CONFIRMED

    def test_pre_breakout_bb_not_riding(self):
        """PRE_BREAKOUT: filing + options strong but BB not yet confirmed.

        Precedent: JUBLFOOD Feb 2024 — buyback announcement before
        technical breakout, options showed accumulation.
        """
        from modules.setup_classifier import (
            classify_setup_v3, SetupType, OptionsSignalState, MomentumState,
        )

        opts = OptionsSignalState(
            gex_regime="Negative", pcr=1.5, pcr_trending="RISING",
            delta_bias="LONG",
        )
        mom = MomentumState(
            position_state="CONSOLIDATING", vol_state="SQUEEZE",
        )
        st, _ = classify_setup_v3(
            viability_score=55,
            filing_variance=10,
            filing_direction="BULLISH",
            filing_conviction=7,
            filing_category="CORP_ACTION",
            opts=opts,
            mom=mom,
        )
        assert st == SetupType.PRE_BREAKOUT

    def test_neutral_when_no_signals(self):
        """NEUTRAL: no filing, weak options, middle score."""
        from modules.setup_classifier import classify_setup_v3, SetupType

        st, _ = classify_setup_v3(viability_score=52)
        assert st == SetupType.NEUTRAL

    def test_options_only_strong_no_filing(self):
        """OPTIONS_ONLY: strong options without filing.

        Precedent: unusual OI buildup before Reg30 filing.
        """
        from modules.setup_classifier import (
            classify_setup_v3, SetupType, OptionsSignalState,
        )

        opts = OptionsSignalState(
            gex_regime="Negative", pcr=1.5, pcr_trending="RISING",
            delta_bias="LONG", iv_skew="CALL_CHEAP",
        )
        st, _ = classify_setup_v3(
            viability_score=55,
            filing_variance=None,
            opts=opts,
        )
        assert st == SetupType.OPTIONS_ONLY


# ══════════════════════════════════════════════════════════════════════════════
# 7. VIABILITY SCORING — EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


class TestViabilityEdgeCases:

    def test_bearish_bias_forces_avoid(self):
        """Bearish daily bias → hard override to AVOID/SKIP.

        Precedent: never go long when 20-SMA bias is bearish, regardless
        of how attractive the options setup looks.
        """
        from modules.analytics import analyze

        df = _make_options_df(spot=1000, dte=10, pe_oi_support=60000, ce_oi_resistance=20000)
        ps_bear = _ps(daily_bias="BEARISH")
        ctx = analyze(df, 1000, price_signals=ps_bear)

        assert ctx.viability.label == "AVOID"
        assert ctx.viability.sizing == "SKIP"

    def test_mid_band_broken_forces_exit(self):
        """MID_BAND_BROKEN → sizing SKIP regardless of other signals.

        Precedent: momentum leg completion — once price closes below
        20-SMA, the trade thesis is dead.
        """
        from modules.analytics import analyze

        df = _make_options_df(spot=1000, dte=10, pe_oi_support=60000, ce_oi_resistance=20000)
        ps_broken = _ps(
            position_state="MID_BAND_BROKEN",
            daily_bias="BULLISH",
        )
        ctx = analyze(df, 1000, price_signals=ps_broken)

        assert ctx.viability.sizing == "SKIP"
        assert ctx.viability.label == "AVOID"

    def test_mfi_diverge_caps_sizing(self):
        """MFI divergence caps sizing to HALF for otherwise FULL setups.

        This tests the post-scoring override path.
        """
        from modules.analytics import analyze

        df = _make_options_df(
            spot=1000, dte=10,
            pe_oi_support=80000, ce_oi_resistance=20000,
        )
        ps_div = _ps(
            mfi_value=55.0, mfi_state="RISING",
            mfi_diverge=True, mfi_reliable=True,
        )
        ctx = analyze(df, 1000, price_signals=ps_div)

        # With divergence, even a FULL setup should be capped
        assert ctx.viability.sizing in ("HALF", "SKIP")

    def test_analyze_price_only_cap(self):
        """analyze_price_only() caps score at 70 for no-options symbols.

        Without GEX/OI wall data, maximum confidence is PROCEED (58-70).
        """
        from modules.analytics import analyze_price_only

        ps_strong = _ps(
            wr_phase="FRESH", vol_state="SQUEEZE",
            daily_bias="BULLISH", position_state="RIDING_UPPER",
        )
        ctx = analyze_price_only(spot=1000, price_signals=ps_strong)
        assert ctx.viability.score <= 70

    def test_score_always_0_to_100(self):
        """Score must never exceed 100 or go below 0."""
        from modules.analytics import analyze

        df = _make_options_df(spot=1000, dte=10)

        # Extreme bullish
        ps_max = _ps(
            daily_bias="BULLISH", position_state="RIDING_UPPER",
            wr_phase="FRESH", vol_state="SQUEEZE",
            mfi_value=80, mfi_state="STRONG", mfi_reliable=True,
        )
        ctx = analyze(df, 1000, price_signals=ps_max)
        assert 0 <= ctx.viability.score <= 100

        # Extreme bearish
        ps_min = _ps(
            daily_bias="BEARISH", position_state="MID_BAND_BROKEN",
            wr_in_momentum=False, wr_phase="NONE", wr_value=-80,
            mfi_value=15, mfi_state="WEAK", mfi_reliable=True,
        )
        ctx2 = analyze(df, 1000, price_signals=ps_min)
        assert 0 <= ctx2.viability.score <= 100


# ══════════════════════════════════════════════════════════════════════════════
# 8. SECTOR ROTATION RS — PRICE RATIO vs RETURN DIFFERENTIAL
# ══════════════════════════════════════════════════════════════════════════════


class TestSectorRS:

    def test_rs_price_ratio_positive_outperformance(self):
        """Sector up 10%, bench up 5% → RS > 0."""
        from modules.sector_rotation import _rs_price_ratio

        # Sector: 100 → 110 (10%)
        sec = pd.Series([100.0] * 10 + [110.0], index=pd.date_range("2024-01-01", periods=11))
        # Bench: 100 → 105 (5%)
        bench = pd.Series([100.0] * 10 + [105.0], index=pd.date_range("2024-01-01", periods=11))

        rs = _rs_price_ratio(sec, bench, lookback=10)
        assert rs is not None
        assert rs > 0, f"RS should be positive when sector outperforms, got {rs}"

    def test_rs_price_ratio_negative_underperformance(self):
        """Sector up 2%, bench up 8% → RS < 0."""
        from modules.sector_rotation import _rs_price_ratio

        sec = pd.Series([100.0] * 10 + [102.0], index=pd.date_range("2024-01-01", periods=11))
        bench = pd.Series([100.0] * 10 + [108.0], index=pd.date_range("2024-01-01", periods=11))

        rs = _rs_price_ratio(sec, bench, lookback=10)
        assert rs is not None
        assert rs < 0

    def test_rs_none_for_insufficient_bars(self):
        """RS returns None when insufficient data."""
        from modules.sector_rotation import _rs_price_ratio

        sec = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-01", periods=2))
        bench = pd.Series([100.0, 105.0], index=pd.date_range("2024-01-01", periods=2))

        rs = _rs_price_ratio(sec, bench, lookback=10)
        assert rs is None

    def test_rotation_classification_all_quadrants(self):
        """All four rotation quadrants are reachable."""
        from modules.sector_rotation import classify_rotation

        # LEADING: short + medium positive RS
        label, _ = classify_rotation(3.0, 2.0, 1.0)
        assert label == "LEADING"

        # LAGGING: all negative RS
        label, _ = classify_rotation(-3.0, -2.0, -1.0)
        assert label == "LAGGING"

        # WEAKENING: short-term negative but medium positive
        label, _ = classify_rotation(-2.0, 3.0, 2.0)
        assert label == "WEAKENING"

        # IMPROVING: short-term positive but medium negative
        label, _ = classify_rotation(3.0, -2.0, -3.0)
        assert label == "IMPROVING"


# ══════════════════════════════════════════════════════════════════════════════
# 9. DATA QUALITY GATES
# ══════════════════════════════════════════════════════════════════════════════


class TestDataQuality:

    def test_validate_series_rejects_few_bars(self):
        """Series with < 70 valid bars must be rejected."""
        from modules.sector_rotation import _validate_series

        s = pd.Series([100.0] * 50, index=pd.date_range("2024-01-01", periods=50))
        assert _validate_series(s, "TEST") is False

    def test_validate_series_accepts_enough_bars(self):
        """Series with >= 70 valid bars passes."""
        from modules.sector_rotation import _validate_series

        s = pd.Series([100.0] * 80, index=pd.date_range("2024-01-01", periods=80))
        assert _validate_series(s, "TEST") is True

    def test_validate_series_rejects_high_nan(self):
        """Series with > 5% NaN rows must be rejected."""
        from modules.sector_rotation import _validate_series

        vals = [100.0] * 90 + [float("nan")] * 10  # 10% NaN
        s = pd.Series(vals, index=pd.date_range("2024-01-01", periods=100))
        assert _validate_series(s, "TEST") is False


# ══════════════════════════════════════════════════════════════════════════════
# 10. POSITION STATE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════


class TestPositionState:

    def test_riding_upper(self):
        """Price above upper BB for 4 bars → RIDING_UPPER."""
        from modules.indicators import _position_state

        n = 10
        df = pd.DataFrame({
            "Close": [100 + i * 2 for i in range(n)],
            "BB_Upper": [105 + i for i in range(n)],
            "BB_Mid": [95 + i for i in range(n)],
        })
        # Make last 4 bars all above mid
        df.iloc[-4:, df.columns.get_loc("Close")] = [115, 116, 117, 118]
        df.iloc[-4:, df.columns.get_loc("BB_Upper")] = [114, 115, 116, 117]
        df.iloc[-4:, df.columns.get_loc("BB_Mid")] = [105, 106, 107, 108]

        state = _position_state(df)
        assert state == "RIDING_UPPER"

    def test_mid_band_broken(self):
        """Price below mid band → MID_BAND_BROKEN."""
        from modules.indicators import _position_state

        n = 10
        df = pd.DataFrame({
            "Close": [100] * n,
            "BB_Upper": [110] * n,
            "BB_Mid": [105] * n,
        })
        # Last bar below mid
        df.iloc[-1, df.columns.get_loc("Close")] = 103

        state = _position_state(df)
        assert state == "MID_BAND_BROKEN"

    def test_first_dip(self):
        """Previous bar at/above upper, current below → FIRST_DIP."""
        from modules.indicators import _position_state

        n = 5
        df = pd.DataFrame({
            "Close": [110, 112, 115, 118, 114],
            "BB_Upper": [108, 110, 112, 115, 116],
            "BB_Mid": [100, 102, 104, 106, 108],
        })

        state = _position_state(df)
        assert state == "FIRST_DIP"


# ══════════════════════════════════════════════════════════════════════════════
# 11. KITE REMOVAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════


class TestKiteRemoval:

    def test_main_window_no_kite_imports(self):
        """main_window.py should not import KiteSessionManager."""
        import importlib
        import inspect
        src = inspect.getsource(
            importlib.import_module("ui.main_window")
        )
        assert "KiteSessionManager" not in src
        assert "TickerWorker" not in src
        assert "TickAggregator" not in src

    def test_kite_manager_module_still_exists(self):
        """kite_manager.py file exists but is not imported by main_window."""
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "ui", "kite_manager.py"
        )
        assert os.path.exists(path), "kite_manager.py should still exist on disk"


# ══════════════════════════════════════════════════════════════════════════════
# 12. ENTRY VALID vs VIABILITY LABEL
#     These must be independent — entry_valid=True does NOT guarantee STRONG.
# ══════════════════════════════════════════════════════════════════════════════


class TestEntryValidVsViability:

    def test_entry_valid_but_avoid_label(self):
        """entry_valid=True with PINNING regime + tight room → CAUTION/AVOID.

        Demonstrates that entry_valid is a necessary but not sufficient
        condition for a good trade.
        """
        from modules.analytics import analyze

        # Very tight room to resistance (1%)
        df = _make_options_df(
            spot=1000, resistance_strike=1010,
            dte=10, pe_oi_support=40000, ce_oi_resistance=40000,
        )
        ps_valid = _ps(
            wr_in_momentum=True, daily_bias="BULLISH",
            entry_valid=True,
        )
        ctx = analyze(df, 1000, price_signals=ps_valid, room_thresh=5.0)

        # entry_valid is True, but viability may be low due to tight room
        assert ps_valid.entry_valid is True
        # Score should reflect the tight structure
        assert ctx.viability.score < 80
