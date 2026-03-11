"""
modules/analytics.py
────────────────────
Options analytics: OI walls, GEX, max pain, expiry context, trade viability.
All calculations are pure pandas/numpy — no Streamlit dependencies.
"""

from __future__ import annotations

import math
import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Walls:
    resistance: Optional[float] = None  # strongest call wall above spot
    support: Optional[float] = None  # strongest put wall below spot
    max_pain: Optional[float] = None
    pcr_oi: float = 1.0
    pcr_sentiment: str = "Neutral"
    resistance_pct: float = 0.0  # % distance above spot (positive)
    support_pct: float = 0.0  # % distance below spot (negative)
    room_to_run: bool = False  # resistance >= 1.5% away


@dataclass
class GEX:
    net_gex: float = 0.0
    abs_gex: float = 0.0
    regime: str = "Neutral"  # "Positive" | "Negative" | "Neutral"
    hvl: Optional[float] = None
    by_expiry: list = field(default_factory=list)  # [(label, dte, net, pct%), ...]
    spot: float = 0.0


@dataclass
class ExpiryCtx:
    next_expiry: Optional[str] = None
    days_remaining: int = 99
    spot_vs_maxpain: float = 0.0
    pin_risk: str = "LOW"
    warning: Optional[str] = None


@dataclass
class CheckItem:
    item: str
    status: str  # "pass" | "warn" | "fail" | "neutral"
    detail: str
    implication: str


@dataclass
class Viability:
    score: int = 50
    label: str = "NEUTRAL"
    color: str = "yellow"
    sizing: str = "HALF"
    checklist: list = field(default_factory=list)


@dataclass
class OptionsContext:
    walls: Walls = field(default_factory=Walls)
    gex: GEX = field(default_factory=GEX)
    expiry: ExpiryCtx = field(default_factory=ExpiryCtx)
    viability: Viability = field(default_factory=Viability)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def analyze(
    options_df: pd.DataFrame, spot: float, lot_size: int = 75
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
    ctx.viability = _viability(ctx)
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

    # Resistance: strongest call wall AT or ABOVE spot
    above = call_walls[call_walls["Strike"] >= spot]
    if not above.empty:
        resistance = float(above.loc[above["Call_OI"].idxmax(), "Strike"])
    elif not call_walls.empty:
        resistance = float(call_walls.loc[call_walls["Call_OI"].idxmax(), "Strike"])
    else:
        resistance = None

    # Support: strongest put wall AT or BELOW spot
    below = put_walls[put_walls["Strike"] <= spot]
    if not below.empty:
        support = float(below.loc[below["Put_OI"].idxmax(), "Strike"])
    elif not put_walls.empty:
        support = float(put_walls.loc[put_walls["Put_OI"].idxmax(), "Strike"])
    else:
        support = None

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


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
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

    # HVL: strike where cumulative GEX crosses zero
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

    # By-expiry summary (skip DTE=0)
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
    # Skip expired (DTE=0)
    live = [(d, e) for d, e in dtes if d > 0]
    if not live:
        return ExpiryCtx()

    next_dte, next_exp = live[0]
    mp_pct = ((spot - max_pain) / max_pain * 100) if max_pain else 0.0

    # Spot proximity to max pain (Step 3 of workflow)
    near_mp = (abs(spot - max_pain) / spot < 0.012) if (max_pain and spot) else False

    if next_dte <= 1:
        pin_risk = "HIGH"
        warning = (
            f"Expiry TODAY — strong pin near {max_pain:,.0f}"
            if max_pain
            else "Expiry today"
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
        warning = f"Expiry in {next_dte}d + spot within 1.2% of max pain {max_pain:,.0f} — HIGH pin risk"
    elif next_dte <= 4:
        pin_risk = "MODERATE"
        warning = (
            f"Expiry in {next_dte}d — watch for Thu/Fri compression near {max_pain:,.0f}"
            if max_pain
            else f"Expiry in {next_dte}d"
        )
    elif near_mp:
        pin_risk = "MODERATE"
        warning = f"Spot within 1.2% of max pain {max_pain:,.0f} — gravitational pull risk even with {next_dte}d to expiry"
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


def _viability(ctx: OptionsContext) -> Viability:
    score = 50
    checklist = []

    def add(item, status, detail, implication):
        checklist.append(
            CheckItem(item=item, status=status, detail=detail, implication=implication)
        )

    # 1. GEX regime
    regime = ctx.gex.regime
    if regime == "Negative":
        score += 15
        add(
            "GEX Regime",
            "pass",
            "Negative GEX",
            "Dealers amplify moves → BB momentum signals are MORE reliable → Size normally",
        )
    elif regime == "Positive":
        score -= 15
        add(
            "GEX Regime",
            "warn",
            "Positive GEX",
            "Market gravitates toward max pain → Momentum trades face headwinds → Size down 50%",
        )
    else:
        add(
            "GEX Regime",
            "neutral",
            "Neutral GEX",
            "No strong dealer bias — trade on OI walls and price action alone",
        )

    # 2. Wall distance
    res_pct = ctx.walls.resistance_pct
    res = ctx.walls.resistance
    if res and res_pct >= 2.5:
        score += 20
        add(
            "Wall Distance",
            "pass",
            f"+{res_pct:.1f}% to resistance",
            f"Resistance at {res:,.0f} — ample room for momentum trade",
        )
    elif res and res_pct >= 1.5:
        score += 10
        add(
            "Wall Distance",
            "pass",
            f"+{res_pct:.1f}% to resistance",
            f"Acceptable clearance. Target {res:,.0f} and monitor",
        )
    elif res:
        score -= 10
        add(
            "Wall Distance",
            "warn",
            f"Only +{res_pct:.1f}% to resistance",
            f"Wall at {res:,.0f} is too close — risk of pinning or reversal → Reduce size",
        )
    else:
        add(
            "Wall Distance",
            "neutral",
            "No resistance wall mapped",
            "Load options data to get wall levels",
        )

    # 3. Expiry
    dte = ctx.expiry.days_remaining
    if dte >= 5:
        score += 15
        add(
            "Expiry Timing",
            "pass",
            f"{dte} days to expiry",
            "Plenty of time — expiry compression is not a concern",
        )
    elif dte >= 3:
        add(
            "Expiry Timing",
            "warn",
            f"{dte} days to expiry",
            "Nearing expiry — watch for pinning behaviour Thu/Fri",
        )
    else:
        score -= 20
        add(
            "Expiry Timing",
            "fail",
            f"Only {dte} days to expiry",
            "High pin risk — reduce size or wait for next cycle",
        )

    # 4. PCR
    pcr = ctx.walls.pcr_oi
    if 0.9 <= pcr <= 1.3:
        score += 5
        add(
            "PCR",
            "pass",
            f"PCR {pcr:.2f} — {ctx.walls.pcr_sentiment}",
            "Balanced positioning — neither extreme fear nor greed",
        )
    elif pcr > 1.5:
        score += 10
        add(
            "PCR",
            "pass",
            f"PCR {pcr:.2f} — heavy put buying",
            "Heavy put buying = hedging, not conviction selling → supports upside",
        )
    elif pcr < 0.7:
        score -= 10
        add(
            "PCR",
            "warn",
            f"PCR {pcr:.2f} — low PCR",
            "Complacency / aggressive call buying → watch for reversal",
        )
    else:
        add("PCR", "neutral", f"PCR {pcr:.2f}", "Neutral positioning")

    score = max(0, min(100, score))

    if score >= 75:
        label, color, sizing = "STRONG SETUP", "green", "FULL"
    elif score >= 55:
        label, color, sizing = "PROCEED", "emerald", "FULL"
    elif score >= 40:
        label, color, sizing = "CAUTION", "yellow", "HALF"
    else:
        label, color, sizing = "AVOID", "red", "SKIP"

    # Step 1 of workflow: GEX Positive ALWAYS means HALF size regardless of score
    # The commentary already says "size down 50%" — viability must agree.
    if ctx.gex.regime == "Positive" and sizing == "FULL":
        sizing = "HALF"
        # Downgrade label if still showing STRONG/PROCEED
        if label in ("STRONG SETUP", "PROCEED"):
            label = "CAUTION"
            color = "yellow"

    # Step 3 of workflow: HIGH pin risk → force HALF or SKIP
    if ctx.expiry.pin_risk == "HIGH" and sizing == "FULL":
        sizing = "HALF"

    return Viability(
        score=score, label=label, color=color, sizing=sizing, checklist=checklist
    )
