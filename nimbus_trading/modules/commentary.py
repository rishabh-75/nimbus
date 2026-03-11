"""
modules/commentary.py
─────────────────────
Generates the workflow narrative: GEX → Walls → Expiry → WR → Verdict.
Rule-based by default; Claude API enriches when available.
"""

from __future__ import annotations

import json
import requests
from typing import Optional

import pandas as pd

from modules.analytics import OptionsContext
from modules.indicators import bb_signal, wr_signal


_SYSTEM = """You are a senior quant analyst at a systematic trading desk.
A trader uses: 4hr Bollinger Bands (1σ, 20-period) + Williams %%R (50-period, -20 threshold).
Your job: produce a pre-trade context analysis in this EXACT format.

Each line: [LABEL] → [observation] → [trade implication] → [action]
Keep each arrow-separated segment under 15 words.
Use specific numbers, not vague language.

Required keys in your JSON response:
{
  "gex_line":    "...",
  "wall_line":   "...",
  "expiry_line": "...",
  "wr_line":     "...",
  "verdict":     "1-2 sentences. Decisive. State position sizing explicitly.",
  "sizing":      "FULL" | "HALF" | "SKIP"
}

Rules:
- Be direct. No hedging. Make a call.
- Use → as separator between logical steps
- VERDICT must state exactly what to do and why
- Respond ONLY with the JSON object. No preamble, no markdown fences."""


def get_commentary(
    ctx: OptionsContext, price_df: Optional[pd.DataFrame] = None, symbol: str = "NIFTY"
) -> dict:
    """
    Generate workflow commentary.
    Tries Claude API first; falls back to rule-based.
    """
    bb = bb_signal(price_df) if price_df is not None and not price_df.empty else {}
    wr = wr_signal(price_df) if price_df is not None and not price_df.empty else {}

    # Try Claude API
    try:
        result = _call_api(_build_context(ctx, bb, wr, symbol))
        if result and all(k in result for k in ("gex_line", "wall_line", "verdict")):
            return result
    except Exception:
        pass

    # Fallback: deterministic rule-based
    return _rule_based(ctx, bb, wr)


def _build_context(ctx: OptionsContext, bb: dict, wr: dict, symbol: str) -> str:
    spot = ctx.gex.spot or 0
    lines = [
        f"Symbol: {symbol}",
        f"Spot: {spot:,.0f}",
        f"GEX Regime: {ctx.gex.regime}  (Net GEX: {ctx.gex.net_gex:,.0f}M)",
        f"OI Resistance: {ctx.walls.resistance} ({ctx.walls.resistance_pct:+.1f}% from spot)",
        f"OI Support: {ctx.walls.support} ({ctx.walls.support_pct:+.1f}% from spot)",
        f"Max Pain: {ctx.walls.max_pain}",
        f"PCR: {ctx.walls.pcr_oi:.2f} — {ctx.walls.pcr_sentiment}",
        f"Days to Expiry: {ctx.expiry.days_remaining}  (Pin Risk: {ctx.expiry.pin_risk})",
        f"Spot vs Max Pain: {ctx.expiry.spot_vs_maxpain:+.1f}%",
        f"GEX HVL: {ctx.gex.hvl}",
        f"Trade Viability: {ctx.viability.score}/100 ({ctx.viability.label})",
    ]
    if bb:
        lines += [
            f"BB Position: {bb.get('position')}",
            f"BB %B: {bb.get('bb_pct', 0):.2f}",
            f"Riding Upper Band: {bb.get('riding_upper')}",
        ]
    if wr:
        lines += [
            f"Williams %R: {wr.get('wr_value', 0):.1f}",
            f"In Momentum Zone (>= -20): {wr.get('in_momentum')}",
            f"WR Trend: {wr.get('trend')}",
        ]
    return "\n".join(lines)


def _call_api(context_block: str) -> Optional[dict]:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 600,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": context_block}],
        },
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    if resp.status_code != 200:
        return None
    text = "".join(b.get("text", "") for b in resp.json().get("content", [])).strip()
    # Strip JSON fences if present
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


def _rule_based(ctx: OptionsContext, bb: dict, wr: dict) -> dict:
    g, w, e, v = ctx.gex, ctx.walls, ctx.expiry, ctx.viability

    # GEX line
    if g.regime == "Negative":
        gex_line = (
            f"GEX Negative ({g.net_gex:,.0f}M) → Dealers amplify directional moves "
            f"→ BB momentum signals MORE reliable → Size normally"
        )
    elif g.regime == "Positive":
        gex_line = (
            f"GEX Positive ({g.net_gex:,.0f}M) → Market pins toward max pain {w.max_pain:,.0f} "
            f"→ Momentum trades face headwinds → Size down 50%"
        )
    else:
        gex_line = (
            "GEX Neutral → No strong dealer bias "
            "→ Trade on OI walls and price action alone → Standard sizing"
        )

    # Wall line
    if w.resistance and w.resistance_pct >= 1.5:
        wall_line = (
            f"Resistance {w.resistance:,.0f} ({w.resistance_pct:+.1f}%) | "
            f"Support {w.support:,.0f} ({w.support_pct:+.1f}%) → "
            f"Adequate room → Target resistance, hard stop below support"
        )
    elif w.resistance:
        wall_line = (
            f"Resistance {w.resistance:,.0f} ({w.resistance_pct:+.1f}%) — TOO CLOSE "
            f"→ Wall will likely cap the move → Reduce size or wait for wall to break"
        )
    else:
        wall_line = (
            "No options data → Cannot assess walls → Load or upload options chain"
        )

    # Expiry line
    dte = e.days_remaining
    mp = f"{w.max_pain:,.0f}" if w.max_pain else "—"
    if e.pin_risk == "HIGH":
        exp_line = (
            f"Expiry in {dte}d → HIGH pin risk → "
            f"Market likely to compress toward {mp} → Skip or wait until after expiry"
        )
    elif e.pin_risk == "MODERATE":
        exp_line = (
            f"Expiry in {dte}d → Moderate pin risk → "
            f"Watch for compression Thu/Fri → Take profit before last session"
        )
    else:
        exp_line = (
            f"Expiry in {dte}d → Low pin risk → "
            f"Plenty of time for trade to develop → No expiry adjustment needed"
        )

    # WR line
    wr_val = wr.get("wr_value")
    if wr_val is not None:
        zone = wr.get("in_momentum")
        trend = wr.get("trend", "").capitalize()
        if zone:
            wr_line = (
                f"Williams %R {wr_val:.1f} → Above -20 threshold → "
                f"Confirmed momentum zone — {trend} → Entry valid"
            )
        else:
            wr_line = (
                f"Williams %R {wr_val:.1f} → Below -20 → "
                f"NOT in momentum zone → Do NOT enter until %R reclaims -20"
            )
    else:
        wr_line = "Williams %R: Load price data to confirm momentum zone"

    # Verdict — integrates ALL three workflow steps + WR momentum check
    score = v.score
    sizing = v.sizing
    res_str = f"{w.resistance:,.0f}" if w.resistance else "—"
    sup_str = f"{w.support:,.0f}" if w.support else "—"
    mp_str = f"{w.max_pain:,.0f}" if w.max_pain else "—"

    wr_val = wr.get("wr_value")
    wr_in_zone = wr.get("in_momentum", False)  # WR >= -20
    wr_str = f"{wr_val:.1f}" if wr_val is not None else "?"

    # WR is the ENTRY GATE — if not in momentum zone, don't enter regardless of score
    if wr_val is not None and not wr_in_zone:
        if sizing == "SKIP" or score < 40:
            verdict = (
                f"No entry. W%R at {wr_str} — not in momentum zone. "
                f"Wait for W%R to reclaim -20 before considering this trade."
            )
        elif g.regime == "Positive":
            verdict = (
                f"Wait. W%R at {wr_str} (needs > -20) AND GEX Positive — "
                f"market gravitating toward max pain {mp_str}. "
                f"Two reasons to stay out. Check again next session."
            )
        elif e.pin_risk == "HIGH":
            verdict = (
                f"Wait. W%R at {wr_str} (needs > -20) and expiry pin risk is HIGH. "
                f"Do not force entries into a pinning market. Skip this cycle."
            )
        else:
            verdict = (
                f"Setup is building but W%R at {wr_str} has not reclaimed -20. "
                f"Levels are constructive: target {res_str}, floor {sup_str}. "
                f"Set an alert at W%R = -20. No entry until it confirms."
            )
    elif sizing == "SKIP" or score < 40:
        gex_warn = (
            f"GEX Positive pins market near {mp_str}. "
            if g.regime == "Positive"
            else ""
        )
        dte_warn = f"Expiry in {dte}d — too close. " if dte <= 2 else ""
        verdict = (
            f"Skip. {gex_warn}{dte_warn}"
            f"Conditions not favourable for momentum trade. Wait for next cycle."
        )
    elif sizing == "HALF":
        gex_note = (
            f"GEX Positive limits upside — market pulled toward {mp_str}. "
            if g.regime == "Positive"
            else ""
        )
        pin_note = (
            f"Expiry in {dte}d adds compression risk. " if e.pin_risk != "LOW" else ""
        )
        verdict = (
            f"Half size only. {gex_note}{pin_note}"
            f"W%R confirms momentum. Target {res_str}, hard stop below {sup_str}. "
            f"Take partial at halfway, trail the rest."
        )
    else:
        verdict = (
            f"Full size. W%R in momentum zone, GEX amplifies moves, "
            f"adequate room to {res_str}. "
            f"Enter on 4h BB confirmation. Hard stop below {sup_str}."
        )

    return dict(
        gex_line=gex_line,
        wall_line=wall_line,
        expiry_line=exp_line,
        wr_line=wr_line,
        verdict=verdict,
        sizing=sizing,
    )
