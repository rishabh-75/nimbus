"""
modules/commentary.py
─────────────────────
Workflow narrative: GEX → Walls → Expiry → WR → Verdict.
Now aware of PriceSignals (daily bias, vol state, WR phase, position state).
"""

from __future__ import annotations

import json
import requests
from typing import Optional

import pandas as pd

from modules.analytics import OptionsContext
from modules.indicators import PriceSignals, bb_signal, wr_signal


_SYSTEM = """\
You are a senior quant analyst on a systematic trading desk.
Strategy: 4hr Bollinger Bands (1σ, 20-period) + Williams %R(50, threshold -20).
Weekly momentum swing trader on NSE (NIFTY / stocks).

Output ONLY a JSON object with these keys:
{
  "gex_line":    "single line: GEX observation → implication → sizing action",
  "wall_line":   "single line: resistance/support levels → room assessment → target/stop",
  "expiry_line": "single line: DTE + pin risk → expiry implication → action",
  "wr_line":     "single line: WR value + phase → momentum state → entry/exit signal",
  "bias_line":   "single line: daily bias + vol state + position state → combined read",
  "verdict":     "2 sentences max. Decisive. State exactly: entry/hold/exit + position size + key level",
  "sizing":      "FULL" or "HALF" or "SKIP"
}

Rules:
- Be specific: use exact numbers, not vague phrases
- → separates logical steps
- Verdict must state the action (enter/hold/exit), size, and the key level to watch
- No markdown, no preamble. JSON only."""


def get_commentary(
    ctx: OptionsContext,
    ps: Optional[PriceSignals] = None,
    symbol: str = "NIFTY",
    price_df: Optional[pd.DataFrame] = None,
) -> dict:
    bb = bb_signal(price_df) if price_df is not None and not price_df.empty else {}
    wr = wr_signal(price_df) if price_df is not None and not price_df.empty else {}

    # ── MFI line — built once, injected into whichever path wins ─────────
    if ps and ps.mfi_value is not None:
        mv = ps.mfi_value
        if not ps.mfi_reliable:
            mfi_line = f"MFI {mv:.0f} — thin volume, unreliable"
        elif ps.mfi_diverge:
            mfi_line = (
                f"MFI {mv:.0f} ⚠ BEARISH DIVERGENCE — "
                f"price near high, money leaving → SIZE ↓ HALF"
            )
        else:
            desc = {
                "STRONG": "strong buying pressure — volume confirming move",
                "RISING": (
                    "money flowing in — volume + price aligned"
                    if ps.wr_in_momentum
                    else "money rising but W%R not yet in zone"
                ),
                "NEUTRAL": "neutral money flow",
                "FALLING": (
                    "money softening — W%R falling too, watch closely"
                    if ps.wr_trend == "falling"
                    else "money flow softening — price still holding"
                ),
                "WEAK": "heavy selling pressure — distribution confirmed",
            }.get(ps.mfi_state, "")
            mfi_line = f"MFI {mv:.0f} {ps.mfi_state} — {desc}"
    else:
        mfi_line = "MFI — volume unavailable"

    # ── Try API path first ────────────────────────────────────────────────
    try:
        result = _call_api(_build_context(ctx, ps, bb, wr, symbol))
        if result and all(k in result for k in ("gex_line", "wall_line", "verdict")):
            result["mfi_line"] = mfi_line  # ← inject here
            return result
    except Exception:
        pass

    # ── Rule-based fallback ───────────────────────────────────────────────
    result = _rule_based(ctx, ps, bb, wr)
    result["mfi_line"] = mfi_line  # ← inject here too
    return result


def _build_context(
    ctx: OptionsContext, ps: Optional[PriceSignals], bb: dict, wr: dict, symbol: str
) -> str:
    spot = ctx.gex.spot or 0
    lines = [
        f"Symbol: {symbol}",
        f"Spot: {spot:,.0f}",
        f"GEX Regime: {ctx.gex.regime}  ({ctx.gex.net_gex:+,.0f}M)",
        f"Market Regime: {ctx.regime.regime} — {ctx.regime.reason}",
        f"OI Resistance: {ctx.walls.resistance} ({ctx.walls.resistance_pct:+.1f}%)",
        f"OI Support: {ctx.walls.support} ({ctx.walls.support_pct:+.1f}%)",
        f"Max Pain: {ctx.walls.max_pain}",
        f"PCR: {ctx.walls.pcr_oi:.2f} — {ctx.walls.pcr_sentiment}",
        f"Days to Expiry: {ctx.expiry.days_remaining}  (Pin Risk: {ctx.expiry.pin_risk})",
        f"Spot vs Max Pain: {ctx.expiry.spot_vs_maxpain:+.1f}%",
        f"GEX HVL: {ctx.gex.hvl}",
        f"Trade Viability: {ctx.viability.score}/100 ({ctx.viability.label}) → SIZE: {ctx.viability.sizing}",
        f"Risk notes: {'; '.join(ctx.viability.risk_notes) or 'none'}",
    ]
    if ps:
        lines += [
            f"Daily Bias: {ps.daily_bias} ({ps.daily_bias_pct:+.1f}% vs 20-SMA)",
            f"Vol State: {ps.vol_state} ({ps.bb_width_pctl:.0f}th pctl)",
            f"BB Position State: {ps.position_state}",
            f"Williams %R: {ps.wr_value:.1f}" if ps.wr_value else "Williams %R: N/A",
            f"WR in momentum zone: {ps.wr_in_momentum}",
            f"WR phase: {ps.wr_phase} ({ps.wr_bars_since_cross50} bars since -50 cross)",
        ]
    return "\n".join(lines)


def _call_api(context_block: str) -> Optional[dict]:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 700,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": context_block}],
        },
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    if resp.status_code != 200:
        return None
    text = "".join(b.get("text", "") for b in resp.json().get("content", [])).strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


def _rule_based(
    ctx: OptionsContext, ps: Optional[PriceSignals], bb: dict, wr: dict
) -> dict:
    g, w, e, v, r = ctx.gex, ctx.walls, ctx.expiry, ctx.viability, ctx.regime

    # ── GEX line ──────────────────────────────────────────────────────────────
    if g.regime == "Negative":
        gex_line = (
            f"GEX Negative ({g.net_gex:,.0f}M) → Dealers amplify moves "
            f"→ BB momentum MORE reliable → Full size"
        )
    elif g.regime == "Positive":
        gex_line = (
            f"GEX Positive ({g.net_gex:,.0f}M) → Pins toward {w.max_pain:,.0f} "
            f"→ Momentum faces headwinds → Size down 50%"
        )
    else:
        gex_line = (
            "GEX Neutral → No strong dealer bias "
            "→ Trade on walls + price action → Standard sizing"
        )

    # ── Wall line ─────────────────────────────────────────────────────────────
    if w.resistance and w.resistance_pct >= 2.0:
        wall_line = (
            f"Resistance {w.resistance:,.0f} ({w.resistance_pct:+.1f}%) | "
            f"Support {w.support:,.0f} ({w.support_pct:+.1f}%) "
            f"→ Adequate structural room → Target resistance, stop below support"
        )
    elif w.resistance:
        wall_line = (
            f"Resistance {w.resistance:,.0f} ({w.resistance_pct:+.1f}%) — CRAMPED "
            f"→ Wall likely caps the move → Reduce size or wait for break"
        )
    else:
        wall_line = "No options data → Cannot assess walls → Upload options chain"

    # ── Expiry line ───────────────────────────────────────────────────────────
    dte = e.days_remaining
    mp = f"{w.max_pain:,.0f}" if w.max_pain else "—"
    if dte <= 1:
        exp_line = (
            f"Expiry TODAY — INTRADAY ONLY → No new swing entries "
            f"→ Close any open positions or roll forward"
        )
    elif e.pin_risk == "HIGH":
        exp_line = (
            f"Expiry {dte}d → HIGH pin risk → "
            f"Market compressing toward {mp} → Skip or wait post-expiry"
        )
    elif e.pin_risk == "MODERATE":
        exp_line = (
            f"Expiry {dte}d → Moderate pin risk → "
            f"Watch Thu/Fri compression near {mp} → Take profit before last session"
        )
    else:
        exp_line = (
            f"Expiry {dte}d → LOW pin risk → "
            f"Plenty of time → No expiry adjustment needed"
        )

    # ── WR + phase line ───────────────────────────────────────────────────────
    wr_val = (ps.wr_value if ps else None) or wr.get("wr_value")
    wr_zone = (ps.wr_in_momentum if ps else None) or wr.get("in_momentum", False)
    wr_phase = ps.wr_phase if ps else "NONE"
    bars = ps.wr_bars_since_cross50 if ps else 99

    if wr_val is not None:
        if not wr_zone:
            wr_line = (
                f"W%R {wr_val:.1f} → Below -20 → NOT in momentum zone "
                f"→ Do NOT enter until W%R reclaims -20"
            )
        elif wr_phase == "FRESH":
            wr_line = (
                f"W%R {wr_val:.1f} → FRESH momentum ({bars}b since -50 cross) "
                f"→ Early in the move — best entry window → Proceed"
            )
        elif wr_phase == "DEVELOPING":
            wr_line = (
                f"W%R {wr_val:.1f} → DEVELOPING ({bars}b since cross) "
                f"→ Mid-phase momentum — still valid → Monitor for exhaustion"
            )
        else:
            wr_line = (
                f"W%R {wr_val:.1f} → LATE phase ({bars}b since -50 cross) "
                f"→ Risk of mean reversion → Tighten stops, no new entries"
            )
    else:
        wr_line = "W%R: Load price data to confirm momentum zone"

    # ── Daily bias + vol + position state line ────────────────────────────────
    bias_str = ps.daily_bias if ps else "?"
    vs_str = ps.vol_state if ps else "?"
    pos_str = ps.position_state.replace("_", " ").title() if ps else "?"
    bias_pct = f"{ps.daily_bias_pct:+.1f}%" if ps else ""

    if ps:
        bias_line = (
            f"Daily {bias_str} ({bias_pct} vs SMA) · Vol {vs_str} "
            f"({ps.bb_width_pctl:.0f}th pctl) · State: {pos_str}"
        )
    else:
        bias_line = "Load price data for daily bias + vol state"

    # ── Verdict ───────────────────────────────────────────────────────────────
    sizing = v.sizing
    score = v.score
    res_str = f"{w.resistance:,.0f}" if w.resistance else "—"
    sup_str = f"{w.support:,.0f}" if w.support else "—"
    mp_str = f"{w.max_pain:,.0f}" if w.max_pain else "—"
    wr_str = f"{wr_val:.1f}" if wr_val is not None else "?"

    # Position state gates — these override entry-focused verdicts
    if ps and ps.position_state == "MID_BAND_BROKEN":
        verdict = (
            f"EXIT signal. Price closed below 20-SMA — momentum leg complete. "
            f"Close all positions. Re-enter only after W%R resets below -50 and bounces."
        )
    elif ps and ps.position_state == "FIRST_DIP":
        verdict = (
            f"Partial exit. First close below upper band at {ps.last_close:,.0f}. "
            f"Scale out 50% now. Move stop to entry. Hold remainder to mid-band {ps.mid:,.1f}."
        )

    # Daily bearish — no longs
    elif ps and ps.daily_bias == "BEARISH":
        verdict = (
            f"No new longs. Daily price is BELOW 20-SMA ({ps.daily_bias_pct:+.1f}%). "
            f"Wait for daily bias to turn bullish before any fresh entries."
        )

    # WR gate — not in momentum zone
    elif wr_val is not None and not wr_zone:
        if score < 40 or sizing == "SKIP":
            verdict = (
                f"No entry. W%R {wr_str} not in momentum zone AND conditions unfavorable. "
                f"Wait for W%R > -20. Check again next session."
            )
        elif g.regime == "Positive":
            verdict = (
                f"Wait. W%R {wr_str} needs > -20. GEX Positive also pins toward {mp_str}. "
                f"Two reasons to stay out. Alert at W%R = -20."
            )
        else:
            verdict = (
                f"Setup building but W%R {wr_str} not yet at -20. "
                f"Levels: target {res_str}, floor {sup_str}. Alert at W%R = -20. No entry yet."
            )

    # SKIP / AVOID
    elif sizing == "SKIP" or score < 40:
        notes = v.risk_notes[:2] if v.risk_notes else []
        note_str = " | ".join(notes)
        verdict = (
            f"Skip. {note_str + '. ' if note_str else ''}"
            f"Conditions do not support a momentum trade. Wait for next cycle."
        )

    # HALF size
    elif sizing == "HALF":
        reasons = []
        if g.regime == "Positive":
            reasons.append(f"GEX Positive (→ {mp_str})")
        if e.pin_risk != "LOW":
            reasons.append(f"pin risk ({dte}d)")
        if ps and wr_phase == "LATE":
            reasons.append("late W%R phase")
        why = " + ".join(reasons) if reasons else "mixed conditions"
        verdict = (
            f"Half size only ({why}). W%R in zone, target {res_str}, stop below {sup_str}. "
            f"Take partial at midpoint, trail the rest."
        )

    # FULL size
    else:
        phase_note = f"Fresh W%R momentum ({bars}b). " if wr_phase == "FRESH" else ""
        squeeze_note = (
            "Vol squeeze breakout in play. " if ps and ps.vol_state == "SQUEEZE" else ""
        )
        verdict = (
            f"Full size. {phase_note}{squeeze_note}"
            f"Regime {r.regime}, adequate room to {res_str}. "
            f"Enter on 4h BB confirmation. Hard stop below {sup_str}."
        )

    return dict(
        gex_line=gex_line,
        wall_line=wall_line,
        expiry_line=exp_line,
        wr_line=wr_line,
        bias_line=bias_line,
        verdict=verdict,
        sizing=sizing,
    )
