"""
modules/analytics.py
────────────────────
Options analytics + integrated viability scoring.
Now accepts PriceSignals for daily bias, vol state, WR phase.

Patch v5.1 — fixes applied:
  FIX-1  : _regime_classify — pos_gex branch (GEX POSITIVE label) [already present]
  FIX-2  : _viability hard override — W%R gate closed → SKIP not HALF
  FIX-3  : _viability — score clamped to ≤35 when sizing forced to SKIP
  FIX-4  : _regime_classify — neg_gex branches check expiry risk before FULL
  FIX-5b : _viability hard override — FIRST_DIP → SKIP / "MANAGE EXISTING"
  FIX-6  : _viability structural room — resistance below spot handled correctly
  FIX-7  : _viability hard override — DTE=0 (expiry today) → SKIP
  FIX-8  : _regime_classify — near_mp threshold widened 1.5% → 2.0%
  FIX-9  : _expiry_ctx — include DTE=0 (expiry today) in live list
  DG-1   : PCR 0.70–0.90 (mildly bearish) now penalises -5
  DG-2   : WR bars_since ≥50 / 99 → "SUSTAINED" phase (not "LATE")
  DG-3   : ps=None guard — caps score at 45 / HALF when no price feed
"""

from __future__ import annotations

import math
import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from modules.indicators import PriceSignals

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Walls:
    resistance: Optional[float] = None
    support: Optional[float] = None
    max_pain: Optional[float] = None
    pcr_oi: float = 1.0
    pcr_sentiment: str = "Neutral"
    resistance_pct: float = 0.0
    support_pct: float = 0.0
    room_to_run: bool = False


@dataclass
class GEX:
    net_gex: float = 0.0
    abs_gex: float = 0.0
    regime: str = "Neutral"
    hvl: Optional[float] = None
    by_expiry: list = field(default_factory=list)
    spot: float = 0.0


@dataclass
class ExpiryCtx:
    next_expiry: Optional[str] = None
    days_remaining: int = 99
    spot_vs_maxpain: float = 0.0
    pin_risk: str = "LOW"
    warning: Optional[str] = None


@dataclass
class RegimeClass:
    """Market regime classification for 1-week momentum strategy."""

    regime: str = (
        "UNKNOWN"  # TREND-FRIENDLY | PINNING | NEUTRAL | GEX POSITIVE | GEX NEG / EXPIRY RISK
    )
    reason: str = ""
    size_cap: str = "FULL"  # FULL | HALF | SKIP
    detail: str = ""
    color: str = "muted"  # emerald | red | yellow | muted


@dataclass
class CheckItem:
    item: str
    status: str  # pass | warn | fail | neutral
    detail: str
    implication: str


@dataclass
class Viability:
    score: int = 50
    label: str = "NEUTRAL"
    color: str = "yellow"
    sizing: str = "HALF"
    checklist: list = field(default_factory=list)
    risk_notes: list = field(default_factory=list)


@dataclass
class OptionsContext:
    walls: Walls = field(default_factory=Walls)
    gex: GEX = field(default_factory=GEX)
    expiry: ExpiryCtx = field(default_factory=ExpiryCtx)
    regime: RegimeClass = field(default_factory=RegimeClass)
    viability: Viability = field(default_factory=Viability)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def analyze(
    options_df: pd.DataFrame,
    spot: float,
    lot_size: int = 75,
    price_signals: Optional[PriceSignals] = None,
    room_thresh: float = 3.0,
) -> OptionsContext:
    ctx = OptionsContext()
    if options_df is None or options_df.empty or spot <= 0:
        return ctx
    df = _clean(options_df)
    if df.empty:
        return ctx

    ctx.walls = _walls(df, spot)
    ctx.gex = _gex(df, spot, lot_size)
    ctx.expiry = _expiry_ctx(df, spot, ctx.walls.max_pain)
    ctx.regime = _regime_classify(ctx.gex, ctx.walls, ctx.expiry, spot)
    ctx.viability = _viability(ctx, price_signals, room_thresh)
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Strike", "CE_OI", "PE_OI", "CE_IV", "PE_IV", "CE_LTP", "PE_LTP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[df["Strike"] > 0]


def _walls(df: pd.DataFrame, spot: float, oi_pct: float = 75.0) -> Walls:
    agg = (
        df.groupby("Strike")
        .agg(
            Call_OI=("CE_OI", "sum"),
            Put_OI=("PE_OI", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["Strike"] > 0]

    total_call = agg["Call_OI"].sum()
    total_put = agg["Put_OI"].sum()
    pcr = (total_put / total_call) if total_call > 0 else 1.0

    call_thresh = agg["Call_OI"].quantile(oi_pct / 100)
    put_thresh = agg["Put_OI"].quantile(oi_pct / 100)
    call_walls = agg[agg["Call_OI"] >= call_thresh]
    put_walls = agg[agg["Put_OI"] >= put_thresh]

    above = call_walls[call_walls["Strike"] >= spot]
    resistance = (
        float(above.loc[above["Call_OI"].idxmax(), "Strike"])
        if not above.empty
        else (
            float(call_walls.loc[call_walls["Call_OI"].idxmax(), "Strike"])
            if not call_walls.empty
            else None
        )
    )

    below = put_walls[put_walls["Strike"] <= spot]
    support = (
        float(below.loc[below["Put_OI"].idxmax(), "Strike"])
        if not below.empty
        else (
            float(put_walls.loc[put_walls["Put_OI"].idxmax(), "Strike"])
            if not put_walls.empty
            else None
        )
    )

    max_pain = _max_pain(agg)
    res_pct = ((resistance - spot) / spot * 100) if resistance else 0.0
    sup_pct = ((support - spot) / spot * 100) if support else 0.0
    sentiment = (
        "Bullish"
        if pcr >= 1.3
        else (
            "Mildly Bullish"
            if pcr >= 1.1
            else (
                "Neutral"
                if pcr >= 0.9
                else "Mildly Bearish" if pcr >= 0.7 else "Bearish"
            )
        )
    )
    return Walls(
        resistance=resistance,
        support=support,
        max_pain=max_pain,
        pcr_oi=round(pcr, 3),
        pcr_sentiment=sentiment,
        resistance_pct=round(res_pct, 2),
        support_pct=round(sup_pct, 2),
        room_to_run=(res_pct >= 1.5 if resistance else False),
    )


def _max_pain(agg: pd.DataFrame) -> Optional[float]:
    strikes = sorted(agg["Strike"].tolist())
    if len(strikes) < 3:
        return None
    pain = {}
    for s in strikes:
        cp = sum(
            max(0, s - k) * float(agg.loc[agg["Strike"] == k, "Call_OI"].values[0])
            for k in strikes
            if k in agg["Strike"].values
        )
        pp = sum(
            max(0, k - s) * float(agg.loc[agg["Strike"] == k, "Put_OI"].values[0])
            for k in strikes
            if k in agg["Strike"].values
        )
        pain[s] = cp + pp
    return float(min(pain, key=pain.get))


def _bs_gamma(S, K, T, r, sigma):
    try:
        if sigma <= 0 or T <= 0 or S <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return math.exp(-0.5 * d1**2) / (S * sigma * math.sqrt(2 * math.pi * T))
    except Exception:
        return 0.0


def _parse_dte(expiry_str: str, today: datetime.date) -> int:
    for fmt in ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"]:
        try:
            exp_date = datetime.datetime.strptime(expiry_str.strip(), fmt).date()
            return (exp_date - today).days
        except (ValueError, AttributeError):
            continue
    return -1


def _gex(
    df: pd.DataFrame, spot: float, lot_size: int, r: float = 0.065, max_dte: int = 90
) -> GEX:
    today = datetime.date.today()
    rows = []
    for _, row in df.iterrows():
        strike = float(row["Strike"])
        dte = _parse_dte(str(row.get("Expiry", "")), today)
        if dte <= 0 or dte > max_dte:
            continue
        T = dte / 365
        ce_iv = float(row.get("CE_IV", 0))
        pe_iv = float(row.get("PE_IV", 0))
        ce_iv = ce_iv / 100 if ce_iv > 1 else ce_iv
        pe_iv = pe_iv / 100 if pe_iv > 1 else pe_iv
        ce_oi = float(row.get("CE_OI", 0))
        pe_oi = float(row.get("PE_OI", 0))
        if ce_iv > 0.01 and ce_oi > 0:
            g = _bs_gamma(spot, strike, T, r, ce_iv)
            rows.append(
                {
                    "side": "call",
                    "strike": strike,
                    "dte": dte,
                    "gex": g * ce_oi * lot_size * spot**2 / 1e6,
                }
            )
        if pe_iv > 0.01 and pe_oi > 0:
            g = _bs_gamma(spot, strike, T, r, pe_iv)
            rows.append(
                {
                    "side": "put",
                    "strike": strike,
                    "dte": dte,
                    "gex": -g * pe_oi * lot_size * spot**2 / 1e6,
                }
            )

    if not rows:
        return GEX(spot=spot)

    gdf = pd.DataFrame(rows)
    net = float(gdf["gex"].sum())
    abs_tot = float(gdf["gex"].abs().sum())

    by_str = gdf.groupby("strike")["gex"].sum().sort_index()
    cum = by_str.cumsum()
    hvl = None
    for i in range(1, len(cum)):
        if cum.iloc[i - 1] * cum.iloc[i] < 0:
            hvl = float(cum.index[i])
            break

    regime = (
        "Negative"
        if net < -abs_tot * 0.05
        else "Positive" if net > abs_tot * 0.05 else "Neutral"
    )

    by_expiry = []
    for dte_val, grp in gdf[gdf["dte"] > 0].groupby("dte"):
        exp_net = float(grp["gex"].sum())
        exp_pct = abs(exp_net) / abs_tot * 100 if abs_tot else 0
        exp_date = (today + datetime.timedelta(days=int(dte_val))).strftime("%d-%b")
        by_expiry.append((exp_date, int(dte_val), exp_net, round(exp_pct, 1)))
    by_expiry.sort(key=lambda x: x[1])

    return GEX(
        net_gex=round(net, 0),
        abs_gex=round(abs_tot, 0),
        regime=regime,
        hvl=hvl,
        by_expiry=by_expiry,
        spot=spot,
    )


def _expiry_ctx(df: pd.DataFrame, spot: float, max_pain: Optional[float]) -> ExpiryCtx:
    today = datetime.date.today()
    expiries = df["Expiry"].dropna().unique().tolist() if "Expiry" in df.columns else []
    dtes = sorted(
        [(d, str(e)) for e in expiries if (d := _parse_dte(str(e), today)) >= 0],
        key=lambda x: x[0],
    )

    # FIX-9: include DTE=0 (expiry today) — changed d > 0 → d >= 0
    live = [(d, e) for d, e in dtes if d >= 0]
    if not live:
        return ExpiryCtx()

    next_dte, next_exp = live[0]
    mp_pct = ((spot - max_pain) / max_pain * 100) if max_pain else 0.0
    near_mp = (abs(spot - max_pain) / spot < 0.012) if (max_pain and spot) else False

    if next_dte <= 0:
        pin_risk = "HIGH"
        warning = (
            f"Expiry TODAY — strong pin near {max_pain:,.0f}"
            if max_pain
            else "Expiry today — no new entries"
        )
    elif next_dte <= 1:
        pin_risk = "HIGH"
        warning = (
            f"Expiry TOMORROW — strong pin near {max_pain:,.0f}"
            if max_pain
            else "Expiry tomorrow"
        )
    elif next_dte <= 2:
        pin_risk = "HIGH"
        warning = (
            f"Expiry in {next_dte}d — pin risk near {max_pain:,.0f}"
            if max_pain
            else f"Expiry in {next_dte}d"
        )
    elif next_dte <= 4 and near_mp:
        pin_risk = "HIGH"
        warning = f"Expiry {next_dte}d + spot within 1.2% of max pain {max_pain:,.0f} — HIGH pin risk"
    elif next_dte <= 4:
        pin_risk = "MODERATE"
        warning = (
            f"Expiry in {next_dte}d — watch for Thu/Fri compression near {max_pain:,.0f}"
            if max_pain
            else f"Expiry in {next_dte}d"
        )
    elif near_mp:
        pin_risk = "MODERATE"
        warning = (
            f"Spot within 1.2% of max pain {max_pain:,.0f} — gravitational pull risk"
        )
    else:
        pin_risk = "LOW"
        warning = None

    return ExpiryCtx(
        next_expiry=next_exp,
        days_remaining=next_dte,
        spot_vs_maxpain=round(mp_pct, 2),
        pin_risk=pin_risk,
        warning=warning,
    )


def _regime_classify(
    gex: GEX, walls: Walls, expiry: ExpiryCtx, spot: float
) -> RegimeClass:
    """
    Classify market regime for 1-week momentum strategy.

    Regimes:
      TREND-FRIENDLY       — Neg/Neutral GEX + adequate room + no pin risk
      GEX NEG / EXPIRY RISK — Neg GEX with ample room but imminent expiry
      PINNING              — Pos GEX near max pain OR high pin risk
      GEX POSITIVE         — Pos GEX not yet at max pain (still a headwind)
      NEUTRAL              — Mixed/unclear signals
    """
    mp = walls.max_pain or spot

    # FIX-8: widened near_mp threshold 1.5% → 2.0% (covers TATAMOTORS-class boundary)
    near_mp = abs(spot - mp) / spot < 0.020 if mp else False
    pos_gex = gex.regime == "Positive"
    neg_gex = gex.regime == "Negative"
    high_pin = expiry.pin_risk in ("HIGH", "MODERATE")

    # ── Positive GEX: pinning (near max pain or high expiry pin risk) ─────────
    if pos_gex and near_mp:
        return RegimeClass(
            regime="PINNING",
            reason="Strong positive GEX + spot within 2% of max pain",
            size_cap="HALF",
            detail=f"Dealers defend {mp:,.0f} — momentum trades face strong headwinds. Size down.",
            color="red",
        )

    if pos_gex and high_pin:
        return RegimeClass(
            regime="PINNING",
            reason="Positive GEX + expiry compression risk",
            size_cap="HALF",
            detail=f"Positive GEX with {expiry.days_remaining}d to expiry — pin risk elevated. Size down.",
            color="red",
        )

    # FIX-1 (already present): Positive GEX but not yet pinned
    if pos_gex:
        mp_dist_pct = abs(spot - mp) / spot * 100 if mp else 0.0
        return RegimeClass(
            regime="GEX POSITIVE",
            reason=f"Positive GEX ({gex.net_gex:+,.0f}M) — max pain gravity active",
            size_cap="HALF",
            detail=(
                f"GEX Positive ({gex.net_gex:+,.0f}M) → max pain {mp:,.0f} "
                f"acts as magnet ({mp_dist_pct:.1f}% away). "
                f"Momentum faces headwinds — size capped at HALF."
            ),
            color="yellow",
        )

    # ── Negative GEX ─────────────────────────────────────────────────────────
    if neg_gex and walls.resistance_pct >= 2.0:
        # FIX-4: expiry risk overrides FULL before granting TREND-FRIENDLY
        if expiry.pin_risk == "HIGH" or expiry.days_remaining <= 2:
            return RegimeClass(
                regime="GEX NEG / EXPIRY RISK",
                reason=f"Negative GEX + {expiry.days_remaining}d to expiry — expiry risk overrides",
                size_cap="HALF",
                detail=(
                    f"Dealers amplify moves but {expiry.days_remaining}d to expiry "
                    f"introduces pin risk — size down to HALF."
                ),
                color="yellow",
            )
        return RegimeClass(
            regime="TREND-FRIENDLY",
            reason="Negative GEX + adequate structural room",
            size_cap="FULL",
            detail=f"Dealers amplify moves, {walls.resistance_pct:.1f}% clear to {walls.resistance:,.0f}. Full size allowed.",
            color="emerald",
        )

    # ── Neutral GEX (not Positive) ────────────────────────────────────────────
    if not pos_gex and walls.resistance_pct >= 1.5 and not near_mp:
        # FIX-4: same expiry guard for neutral GEX
        if expiry.pin_risk == "HIGH" or expiry.days_remaining <= 2:
            return RegimeClass(
                regime="GEX NEG / EXPIRY RISK",
                reason=f"Neutral GEX + {expiry.days_remaining}d to expiry — expiry risk overrides",
                size_cap="HALF",
                detail=(
                    f"Neutral GEX with adequate room but {expiry.days_remaining}d to expiry "
                    f"— size down to HALF."
                ),
                color="yellow",
            )
        return RegimeClass(
            regime="TREND-FRIENDLY",
            reason="Neutral/Negative GEX + spot not pinned",
            size_cap="FULL",
            detail=f"No strong pinning forces. {walls.resistance_pct:.1f}% room to resistance.",
            color="emerald",
        )

    return RegimeClass(
        regime="NEUTRAL",
        reason="Mixed signals",
        size_cap="HALF",
        detail="No clear trend-friendly or pinning regime. Trade with standard size.",
        color="yellow",
    )


# ══════════════════════════════════════════════════════════════════════════════
# VIABILITY SCORING (integrates price signals + options context)
# ══════════════════════════════════════════════════════════════════════════════


def _viability(
    ctx: OptionsContext,
    ps: Optional[PriceSignals] = None,
    room_thresh: float = 3.0,
) -> Viability:
    """
    Two-tier rule design:
    Hard gates (BB upper band + W%R > -20) are the momentum filter.
    Without both, there is no directional trade thesis.
    Scored conditions determine conviction and sizing.

    TIER 1 — HARD GATES (capped at SKIP/ZERO if failing):
      - 4H close riding upper BB(20, 1σ)
      - Williams %R(50) > -20
    TIER 2 — SCORED CONDITIONS (each contributes pts):
      Daily bias, GEX regime, structural room, expiry/DTE, vol state.
    """
    score = 50
    checklist: list = []
    risk_notes: list = []

    def add(item, status, detail, impl):
        checklist.append(
            CheckItem(item=item, status=status, detail=detail, implication=impl)
        )

    def risk(note: str):
        risk_notes.append(note)

    # DG-3: No price signals guard — cap score and warn
    if ps is None:
        add(
            "Price Data",
            "fail",
            "No price feed — BB and W%R signals unavailable",
            "Connect live feed or upload price data for full scoring",
        )
        risk("Price signals missing — momentum checks skipped")
        score = min(score, 45)

    # ── 1. Daily Bias ──────────────────────────────────────────────────────────
    if ps is not None:
        if ps.daily_bias == "BULLISH":
            score += 15
            add(
                "Daily Trend",
                "pass",
                f"Daily 20-SMA: Bullish ({ps.daily_bias_pct:+.1f}%)",
                "Price above daily 20-SMA — long bias confirmed at higher timeframe",
            )
        elif ps.daily_bias == "BEARISH":
            score -= 20
            add(
                "Daily Trend",
                "fail",
                f"Daily 20-SMA: Bearish ({ps.daily_bias_pct:+.1f}%)",
                "Price BELOW daily 20-SMA — do NOT take new long entries",
            )
            risk("Daily bearish — no new longs")
        else:
            add(
                "Daily Trend",
                "neutral",
                f"Daily 20-SMA: Neutral ({ps.daily_bias_pct:+.1f}%)",
                "Price near daily SMA — proceed with caution, prefer pullback entries",
            )

    # ── 2. GEX Regime ─────────────────────────────────────────────────────────
    regime = ctx.gex.regime
    if regime == "Negative":
        score += 15
        add(
            "GEX Regime",
            "pass",
            "Negative GEX",
            "Dealers amplify directional moves — momentum signals MORE reliable",
        )
    elif regime == "Positive":
        score -= 15
        add(
            "GEX Regime",
            "warn",
            f"Positive GEX ({ctx.gex.net_gex:+,.0f}M)",
            f"Market gravitates toward max pain {ctx.walls.max_pain:,.0f} — size down 50%",
        )
        risk(f"GEX Positive — max pain {ctx.walls.max_pain:,.0f} acts as magnet")
    else:
        add(
            "GEX Regime",
            "neutral",
            "Neutral GEX",
            "No strong dealer bias — rely on OI walls and price action",
        )

    if ctx.regime.regime == "PINNING":
        score -= 10
        risk(ctx.regime.detail)

    # ── 3. Structural Room to Resistance ──────────────────────────────────────
    res_pct = ctx.walls.resistance_pct
    res = ctx.walls.resistance

    # FIX-6: resistance mapped BELOW spot — data issue, don't penalise
    if res and res_pct <= 0:
        add(
            "Structural Room",
            "warn",
            f"Resistance mapped BELOW spot ({res:,.0f} — {res_pct:.1f}%)",
            "Options chain may be skewed or post-breakout — treat resistance as unmapped",
        )
        risk(
            f"No resistance above spot — highest call wall {res:,.0f} is below current price"
        )
    elif res and res_pct >= room_thresh:
        score += 20
        add(
            "Structural Room",
            "pass",
            f"+{res_pct:.1f}% to resistance ({res:,.0f})",
            f"Ample room for 1-week momentum trade — target {res:,.0f}",
        )
    elif res and res_pct >= 1.5:
        score += 5
        add(
            "Structural Room",
            "warn",
            f"Only +{res_pct:.1f}% to resistance ({res:,.0f})",
            f"Below {room_thresh:.0f}% threshold — cramped for swing long, monitor closely",
        )
        risk(f"Only {res_pct:.1f}% room — resistance at {res:,.0f} may cap the move")
    elif res:
        score -= 15
        add(
            "Structural Room",
            "fail",
            f"Only +{res_pct:.1f}% — wall at {res:,.0f} too close",
            "Do not take fresh swing long — wall will likely cap any move immediately",
        )
        risk(f"Resistance at {res:,.0f} only {res_pct:.1f}% away — no room to run")
    else:
        add(
            "Structural Room",
            "neutral",
            "No resistance mapped",
            "Load options data for wall levels",
        )

    # ── 4. Expiry / Max-Pain Risk Gate ────────────────────────────────────────
    dte = ctx.expiry.days_remaining
    if dte <= 0:
        # FIX-7: expiry TODAY — hard SKIP regardless of other signals
        score -= 30
        add(
            "Expiry Gate",
            "fail",
            "Expiry TODAY (DTE=0)",
            "Do NOT open new entries — expiry day gamma pinning is extreme",
        )
        risk("Expiry today — all new entries blocked")
    elif dte <= 1:
        score -= 25
        add(
            "Expiry Gate",
            "fail",
            f"Expiry TODAY/TOMORROW ({dte}d)",
            "Do NOT open new swing entries — intraday only until next cycle",
        )
        risk("Expiry imminent — no new swing entries")
    elif dte <= 2:
        score -= 20
        add(
            "Expiry Gate",
            "fail",
            f"Only {dte}d to expiry — HIGH pin risk",
            "Avoid new positions — strong gamma pinning in effect",
        )
        risk(f"Only {dte}d to expiry — gamma pinning")
    elif dte <= 4 and ctx.expiry.pin_risk == "HIGH":
        score -= 12
        add(
            "Expiry Gate",
            "warn",
            f"{dte}d to expiry + HIGH pin risk near {ctx.walls.max_pain:,.0f}",
            "Reduce size — spot near max pain with expiry close",
        )
        risk("High pin risk — spot near max pain with expiry approaching")
    elif dte <= 4:
        add(
            "Expiry Gate",
            "warn",
            f"{dte}d to expiry — moderate caution",
            "Watch for Thu/Fri pin compression near max pain",
        )
    else:
        score += 12
        add(
            "Expiry Gate",
            "pass",
            f"{dte}d to expiry — LOW pin risk",
            "Plenty of time — expiry compression not a concern",
        )

    # ── 5. PCR ────────────────────────────────────────────────────────────────
    pcr = ctx.walls.pcr_oi
    if pcr >= 1.5:
        score += 10
        add(
            "PCR",
            "pass",
            f"PCR {pcr:.2f} — heavy put buying",
            "Heavy hedging activity → supports upside / floor below spot",
        )
    elif 0.9 <= pcr <= 1.3:
        score += 5
        add(
            "PCR",
            "pass",
            f"PCR {pcr:.2f} — balanced",
            "Balanced positioning — no extreme skew",
        )
    elif pcr < 0.7:
        score -= 10
        add(
            "PCR",
            "warn",
            f"PCR {pcr:.2f} — low / call-heavy",
            "Aggressive call buying = complacency → watch for reversal",
        )
        risk(f"Low PCR ({pcr:.2f}) — market may be complacent")
    else:
        # DG-1: PCR 0.70–0.90 (mildly bearish) — was neutral, now -5
        score -= 5
        add(
            "PCR",
            "warn",
            f"PCR {pcr:.2f} — mildly call-heavy",
            "Slightly elevated call OI — watch for complacency building",
        )
        risk(f"PCR {pcr:.2f} — mildly bearish positioning")

    # ── 6. Volatility State ───────────────────────────────────────────────────
    if ps is not None:
        vs = ps.vol_state
        if vs == "SQUEEZE":
            if ctx.regime.regime == "TREND-FRIENDLY":
                score += 12
                add(
                    "Vol State",
                    "pass",
                    f"SQUEEZE ({ps.bb_width_pctl:.0f}th pctl)",
                    "Compression + trend-friendly regime = coiled spring → very favorable for fresh momentum",
                )
            else:
                score += 5
                add(
                    "Vol State",
                    "pass",
                    f"SQUEEZE ({ps.bb_width_pctl:.0f}th pctl)",
                    "Bands compressing — breakout pending. Direction unclear — wait for confirmation",
                )
        elif vs == "EXPANDED":
            if ctx.regime.regime == "PINNING":
                score -= 8
                add(
                    "Vol State",
                    "warn",
                    f"EXPANDED ({ps.bb_width_pctl:.0f}th pctl)",
                    "Expanded volatility + pinning regime = late and choppy — harvest or skip",
                )
                risk(
                    "Late in volatility expansion with pinning regime — consider exiting"
                )
            else:
                add(
                    "Vol State",
                    "neutral",
                    f"EXPANDED ({ps.bb_width_pctl:.0f}th pctl)",
                    "Volatility expanded — still in play but watch for mean reversion",
                )
        else:
            add(
                "Vol State",
                "neutral",
                f"NORMAL ({ps.bb_width_pctl:.0f}th pctl)",
                "Volatility at normal levels — no edge from compression or expansion",
            )

    # ── 7. Williams %R Phase ──────────────────────────────────────────────────
    if ps is not None and ps.wr_value is not None:
        if not ps.wr_in_momentum:
            add(
                "W%R Gate",
                "fail",
                f"W%R {ps.wr_value:.1f} — below -20 threshold",
                "Entry gate CLOSED — wait for W%R to reclaim -20",
            )
            risk(f"W%R {ps.wr_value:.1f} — not in momentum zone")
        else:
            # DG-2: bars_since ≥ 50 (or sentinel 99) = SUSTAINED trend, not LATE
            bars = ps.wr_bars_since_cross50
            effective_phase = ps.wr_phase
            if bars >= 50:
                effective_phase = "SUSTAINED"

            if effective_phase == "FRESH":
                score += 15
                add(
                    "W%R Gate",
                    "pass",
                    f"W%R {ps.wr_value:.1f} — FRESH momentum ({bars}b ago)",
                    f"Fresh cross above -50 just {bars} bars ago — early in momentum phase",
                )
            elif effective_phase == "DEVELOPING":
                score += 8
                add(
                    "W%R Gate",
                    "pass",
                    f"W%R {ps.wr_value:.1f} — DEVELOPING ({bars}b since -50 cross)",
                    "Mid-phase momentum — still valid, monitor for signs of exhaustion",
                )
            elif effective_phase == "SUSTAINED":
                # DG-2: reward sustained momentum, not penalise
                score += 5
                add(
                    "W%R Gate",
                    "pass",
                    f"W%R {ps.wr_value:.1f} — SUSTAINED (50+ bars above -50)",
                    "Long-running momentum — structurally bullish, watch for any reversal signs",
                )
            elif effective_phase == "LATE":
                score -= 5
                add(
                    "W%R Gate",
                    "warn",
                    f"W%R {ps.wr_value:.1f} — LATE phase ({bars}b since -50 cross)",
                    "Late in momentum phase — risk of mean reversion, tighten stops",
                )
                risk(f"W%R late phase ({bars} bars since -50 cross) — reduce size")

    # ── 8. Position State (BB) ────────────────────────────────────────────────
    if ps is not None:
        if ps.position_state == "MID_BAND_BROKEN":
            score -= 15
            add(
                "BB State",
                "fail",
                "Mid-band broken — FULL EXIT zone",
                "Price closed below 20-SMA — momentum leg has ended. Exit remaining position.",
            )
            risk("Mid-band broken — momentum leg complete, exit")
        elif ps.position_state == "FIRST_DIP":
            score -= 5
            add(
                "BB State",
                "warn",
                "First dip below upper band — PARTIAL EXIT zone",
                "Scale out 50% here. Move stop to entry. Let remainder run to mid-band.",
            )
            risk("FIRST_DIP — manage existing position only, do not open new entries")
        elif ps.position_state == "RIDING_UPPER":
            score += 5
            add(
                "BB State",
                "pass",
                "Riding upper band",
                "Price holding upper band — momentum intact. Stay in position.",
            )
        elif ps.position_state == "CONSOLIDATING":
            add(
                "BB State",
                "neutral",
                "Consolidating between mid and upper",
                "Pullback within uptrend. Hold or add on reclaim of upper band.",
            )

    # ── Final score + base sizing ─────────────────────────────────────────────
    score = max(0, min(100, score))

    if score >= 78:
        label, color, sizing = "STRONG SETUP", "green", "FULL"
    elif score >= 58:
        label, color, sizing = "PROCEED", "emerald", "FULL"
    elif score >= 40:
        label, color, sizing = "CAUTION", "yellow", "HALF"
    else:
        label, color, sizing = "AVOID", "red", "SKIP"

    # ══════════════════════════════════════════════════════════════════════════
    # HARD OVERRIDES (applied in order — last write wins for sizing)
    # ══════════════════════════════════════════════════════════════════════════

    # FIX-7: expiry today → absolute block
    if ctx.expiry.days_remaining <= 0:
        sizing = "SKIP"
        label = "AVOID"
        color = "red"

    # Daily bearish → no new longs
    if ps is not None and ps.daily_bias == "BEARISH":
        sizing = "SKIP"
        label = "AVOID"
        color = "red"

    # FIX-5b: FIRST_DIP is a partial-exit signal — block ALL new entries
    if ps is not None and ps.position_state == "FIRST_DIP":
        sizing = "SKIP"
        label = "MANAGE EXISTING"
        color = "yellow"

    # GEX Positive caps at HALF
    if ctx.gex.regime == "Positive" and sizing == "FULL":
        sizing = "HALF"
        if label in ("STRONG SETUP", "PROCEED"):
            label, color = "CAUTION", "yellow"

    # Regime cap
    if ctx.regime.size_cap == "HALF" and sizing == "FULL":
        sizing = "HALF"

    # HIGH pin risk caps at HALF
    if ctx.expiry.pin_risk == "HIGH" and sizing == "FULL":
        sizing = "HALF"

    # FIX-2: W%R gate closed → SKIP (not just FULL→HALF)
    if ps is not None and ps.wr_value is not None and not ps.wr_in_momentum:
        sizing = "SKIP"
        label = "AVOID"
        color = "red"

    # Mid-band broken → exit, not entry
    if ps is not None and ps.position_state == "MID_BAND_BROKEN":
        if sizing in ("FULL", "HALF"):
            sizing = "SKIP"
            label = "AVOID"
            color = "red"

    # FIX-3: clamp score to ≤35 any time sizing is SKIP (prevents "62/AVOID/SKIP" contradiction)
    # Exception: FIRST_DIP keeps score for context (MANAGE EXISTING label), not shown as AVOID
    if sizing == "SKIP" and label == "AVOID":
        score = min(score, 35)

    return Viability(
        score=score,
        label=label,
        color=color,
        sizing=sizing,
        checklist=checklist,
        risk_notes=risk_notes,
    )
