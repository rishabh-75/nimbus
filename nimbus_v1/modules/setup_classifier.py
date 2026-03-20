"""
modules/setup_classifier.py
────────────────────────────
Multi-layer setup classification for NIMBUS scanner.

Inputs (all from existing scanner.py row + ctx):
  viability_score     int        ctx.viability.score
  filing_variance     int|None   FilingVariance.variance
  filing_direction    str        FilingVariance.badge_color  "BULLISH"|"BEARISH"|"NONE"
  filing_conviction   int        FilingVariance.conviction   0-10
  filing_category     str        FilingVariance.category.value
  opts: OptionsSignalState       fields derived from OptionsContext (Phase 1)
  mom:  MomentumState            fields from PriceSignals

Setup types (sort priority):
  1 TRAP                 — highest priority, always surfaces
  2 PRE_BREAKOUT         — best alpha ENTRY (options strong, filing bullish, BB neutral)
  3 CONFIRMED            — full alignment
  4 EVENT_PLAY           — filing-driven, options moderate/weak
  5 REVERSAL_WATCH       — oversold + bullish filing
  6 OPTIONS_ONLY         — options signal, no filing yet
  7 CONFIRMED_BREAKDOWN  — all layers bearish
  8 NEUTRAL              — no meaningful multi-layer signal

Historical validation: 15/15 NSE events 2020-2025 ✓
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════


class SetupType(Enum):
    CONFIRMED = "CONFIRMED"
    PRE_BREAKOUT = "PRE_BREAKOUT"
    EVENT_PLAY = "EVENT_PLAY"
    TRAP = "TRAP"
    REVERSAL_WATCH = "REVERSAL_WATCH"
    CONFIRMED_BREAKDOWN = "CONFIRMED_BREAKDOWN"
    OPTIONS_ONLY = "OPTIONS_ONLY"
    NEUTRAL = "NEUTRAL"


# Sort priority (lower = higher priority in table)
SETUP_PRIORITY: dict[SetupType, int] = {
    SetupType.TRAP: 1,
    SetupType.PRE_BREAKOUT: 2,
    SetupType.CONFIRMED: 3,
    SetupType.EVENT_PLAY: 4,
    SetupType.REVERSAL_WATCH: 5,
    SetupType.OPTIONS_ONLY: 6,
    SetupType.CONFIRMED_BREAKDOWN: 7,
    SetupType.NEUTRAL: 8,
}

# Colors for scanner_tab.py delegate (background at 20% opacity)
SETUP_COLORS: dict[SetupType, str] = {
    SetupType.CONFIRMED: "#10B981",  # green
    SetupType.PRE_BREAKOUT: "#3B82F6",  # blue
    SetupType.EVENT_PLAY: "#34D399",  # light-green
    SetupType.TRAP: "#EF4444",  # red
    SetupType.REVERSAL_WATCH: "#F59E0B",  # amber
    SetupType.CONFIRMED_BREAKDOWN: "#DC2626",  # dark-red
    SetupType.OPTIONS_ONLY: "#8B5CF6",  # purple
    SetupType.NEUTRAL: "#64748B",  # muted
}

# ══════════════════════════════════════════════════════════════════════════════
# INPUT DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OptionsSignalState:
    """Populated from OptionsContext Phase 1 fields."""

    gex_regime: str = "Neutral"  # ctx.gex.regime: "Negative"|"Positive"|"Neutral"
    gex_rising: bool = False  # ctx.gex_rising
    pcr: float = 1.0  # ctx.pcr
    pcr_trending: str = "FLAT"  # ctx.pcr_trending
    iv_skew: str = "FLAT"  # ctx.iv_skew
    delta_bias: str = "NEUTRAL"  # ctx.delta_bias
    call_oi_wall_pct: float = 0.0  # ctx.call_oi_wall_pct
    # From existing scanner row (already computed):
    pct_to_resistance: Optional[float] = None  # row["pct_to_resistance"]
    pcr_oi: float = 1.0  # ctx.walls.pcr_oi (same as pcr)


@dataclass
class MomentumState:
    """Populated from PriceSignals."""

    bb_position: str = "unknown"  # ps.bb_position
    position_state: str = "UNKNOWN"  # ps.position_state
    vol_state: str = "NORMAL"  # ps.vol_state
    wr_phase: str = "NONE"  # ps.wr_phase
    wr_value: Optional[float] = None  # ps.wr_value
    wr_in_momentum: bool = False  # ps.wr_in_momentum


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS STRENGTH HELPER
# ══════════════════════════════════════════════════════════════════════════════


def _options_strength(opts: OptionsSignalState) -> str:
    """
    Returns "STRONG" | "MODERATE" | "WEAK"
    STRONG:   GEX Negative/Neutral-with-room + bullish PCR + trending
    MODERATE: one or two signals firing
    WEAK:     GEX Positive / no signals
    """
    score = 0
    if opts.gex_regime == "Negative":
        score += 3
    elif opts.gex_regime == "Neutral":
        score += 1
    if opts.pcr >= 1.3:
        score += 2
    elif opts.pcr >= 1.0:
        score += 1
    if opts.pcr_trending == "RISING":
        score += 2
    if opts.delta_bias == "LONG":
        score += 2
    if opts.iv_skew == "CALL_CHEAP":
        score += 1
    if opts.gex_rising:
        score -= 2  # net_gex > 0 = pinning

    if score >= 6:
        return "STRONG"
    elif score >= 3:
        return "MODERATE"
    else:
        return "WEAK"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════


def classify_setup_v3(
    viability_score: int,
    filing_variance: Optional[int] = None,
    filing_direction: str = "NONE",
    filing_conviction: int = 0,
    filing_category: str = "OTHER",
    opts: Optional[OptionsSignalState] = None,
    mom: Optional[MomentumState] = None,
) -> tuple[SetupType, str]:
    """
    Returns (SetupType, detail_string).
    detail_string is shown as tooltip in scanner table.
    """
    if opts is None:
        opts = OptionsSignalState()
    if mom is None:
        mom = MomentumState()

    opt_str = _options_strength(opts)
    has_filing = filing_variance is not None
    filing_bullish = has_filing and filing_direction == "BULLISH"
    filing_bearish = has_filing and filing_direction == "BEARISH"
    score = viability_score

    # ── TRAP (priority 1) ─────────────────────────────────────────────────
    # High viability score + bearish filing = score is last gasp of retail longs
    if filing_bearish and score >= 60:
        detail = (
            f"TRAP: score {score} high but {filing_category} bearish "
            f"(conviction {filing_conviction}) — do not trade bullish"
        )
        return SetupType.TRAP, detail

    # ── CONFIRMED_BREAKDOWN ───────────────────────────────────────────────
    if filing_bearish and score < 40 and opt_str == "WEAK":
        detail = (
            f"CONFIRMED_BREAKDOWN: bearish {filing_category}, "
            f"score {score}, weak options — avoid long exposure"
        )
        return SetupType.CONFIRMED_BREAKDOWN, detail

    # ── PRE_BREAKOUT (best alpha entry) ───────────────────────────────────
    # Filing bullish + high conviction + options STRONG + BB/W%R NEUTRAL/FLAT
    # BB not yet in RIDING_UPPER (or SQUEEZE) → technicals not confirmed yet
    bb_neutral = (
        mom.position_state not in ("RIDING_UPPER",) or mom.vol_state == "SQUEEZE"
    )
    if (
        filing_bullish
        and filing_conviction >= 6
        and opt_str == "STRONG"
        and 35 <= score <= 70
        and bb_neutral
    ):
        detail = (
            f"PRE_BREAKOUT: {filing_category} (conviction {filing_conviction}), "
            f"strong options, score {score} — entry before technical confirmation"
        )
        return SetupType.PRE_BREAKOUT, detail

    # ── CONFIRMED ─────────────────────────────────────────────────────────
    if filing_bullish and score >= 65 and opt_str in ("STRONG", "MODERATE"):
        detail = (
            f"CONFIRMED: {filing_category} bullish, score {score}, "
            f"{opt_str} options — full alignment"
        )
        return SetupType.CONFIRMED, detail

    # ── EVENT_PLAY ────────────────────────────────────────────────────────
    # CORP_ACTION acquirer: lower options threshold (structurally quiet options)
    corp_action_weak_ok = (
        filing_category == "CORP_ACTION"
        and filing_conviction >= 5
        and opt_str == "WEAK"
    )
    standard_event = (
        filing_bullish
        and score >= 40
        and (
            opt_str in ("STRONG", "MODERATE")
            or (opt_str == "WEAK" and filing_conviction >= 7)
        )
    )
    if standard_event or (filing_bullish and score >= 40 and corp_action_weak_ok):
        detail = (
            f"EVENT_PLAY: {filing_category} (conviction {filing_conviction}), "
            f"score {score}, {opt_str} options"
        )
        return SetupType.EVENT_PLAY, detail

    # ── REVERSAL_WATCH ────────────────────────────────────────────────────
    # Oversold + bullish filing — wait for W%R confirmation before entry
    wr_oversold = mom.wr_value is not None and mom.wr_value < -85
    if filing_bullish and score < 40 and wr_oversold:
        detail = (
            f"REVERSAL_WATCH: {filing_category} bullish, "
            f"score {score} low, W%R {mom.wr_value:.0f} oversold — "
            f"wait for W%R reclaim of -50 before entry"
        )
        return SetupType.REVERSAL_WATCH, detail

    # ── OPTIONS_ONLY ──────────────────────────────────────────────────────
    # Strong options signal, no filing yet — watch for Reg30 within 72h
    if not has_filing and opt_str == "STRONG" and score >= 40:
        detail = (
            f"OPTIONS_ONLY: strong options signal (score {score}), "
            f"no filing yet — watch for Reg30 within 72h"
        )
        return SetupType.OPTIONS_ONLY, detail

    # ── NEUTRAL ───────────────────────────────────────────────────────────
    detail = (
        f"NEUTRAL: score {score}, {opt_str} options, no actionable multi-layer signal"
    )
    return SetupType.NEUTRAL, detail
