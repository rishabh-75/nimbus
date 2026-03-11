"""
options_analytics.py — OI walls, GEX, max pain, expiry analysis
Everything the trader needs to contextualize a weekly momentum trade.
"""
from __future__ import annotations
import datetime
import math
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WallLevels:
    resistance:      Optional[float] = None   # strongest call wall above spot
    support:         Optional[float] = None   # strongest put wall below spot
    max_pain:        Optional[float] = None
    pcr_oi:          float = 1.0
    pcr_sentiment:   str = "Neutral"
    resistance_pct:  float = 0.0              # % above spot
    support_pct:     float = 0.0              # % below spot (negative)
    room_to_run:     bool = False             # resistance > 1.5% away
    walls_df:        Optional[pd.DataFrame] = None


@dataclass
class GEXData:
    net_gex:         float = 0.0
    abs_gex:         float = 0.0
    regime:          str = "Neutral"           # "Positive" | "Negative" | "Neutral"
    hvl:             Optional[float] = None    # High-Vol Level (zero-cross)
    by_expiry:       list = field(default_factory=list)  # [(expiry, net_gex, pct), ...]
    spot:            float = 0.0


@dataclass
class ExpiryContext:
    next_expiry:     Optional[str] = None
    days_remaining:  int = 99
    spot_vs_maxpain: float = 0.0              # % difference
    pin_risk:        str = "LOW"              # "LOW" | "MODERATE" | "HIGH"
    expiry_warning:  Optional[str] = None


@dataclass
class TradeViability:
    score:           int = 50                 # 0–100
    label:           str = "NEUTRAL"          # "STRONG" | "PROCEED" | "CAUTION" | "AVOID"
    color:           str = "yellow"
    checklist:       list = field(default_factory=list)


@dataclass
class OptionsContext:
    walls:    WallLevels   = field(default_factory=WallLevels)
    gex:      GEXData      = field(default_factory=GEXData)
    expiry:   ExpiryContext = field(default_factory=ExpiryContext)
    viability: TradeViability = field(default_factory=TradeViability)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze(options_df: pd.DataFrame,
            spot: float,
            lot_size: int = 75,
            oi_pct: float = 75.0,
            risk_free: float = 0.065,
            max_dte: int = 90) -> OptionsContext:
    """
    Full options context analysis. Returns OptionsContext.
    """
    ctx = OptionsContext()
    if options_df is None or options_df.empty or spot <= 0:
        return ctx

    df = _clean(options_df)
    if df.empty:
        return ctx

    ctx.walls  = _compute_walls(df, spot, oi_pct)
    ctx.gex    = _compute_gex(df, spot, lot_size, risk_free, max_dte)
    ctx.expiry = _compute_expiry_context(df, spot, ctx.walls.max_pain)
    ctx.viability = _compute_viability(ctx, spot)
    return ctx


# ── internal helpers ──────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Strike", "CE_OI", "PE_OI", "CE_IV", "PE_IV",
                "CE_LTP", "PE_LTP", "CE_Volume", "PE_Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df[df["Strike"] > 0]
    return df


def _compute_walls(df: pd.DataFrame, spot: float,
                   oi_pct: float = 75.0) -> WallLevels:
    # Aggregate across expiries
    agg = df.groupby("Strike").agg(
        Call_OI=("CE_OI", "sum"),
        Put_OI=("PE_OI", "sum"),
        Call_IV=("CE_IV", "mean"),
        Put_IV=("PE_IV", "mean"),
    ).reset_index()

    agg = agg[agg["Strike"] > 0].copy()

    total_call_oi = agg["Call_OI"].sum()
    total_put_oi  = agg["Put_OI"].sum()
    pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

    # Wall threshold
    call_threshold = agg["Call_OI"].quantile(oi_pct / 100)
    put_threshold  = agg["Put_OI"].quantile(oi_pct / 100)

    call_walls = agg[agg["Call_OI"] >= call_threshold]
    put_walls  = agg[agg["Put_OI"]  >= put_threshold]

    # Resistance: strongest call wall ABOVE spot
    cw_above = call_walls[call_walls["Strike"] >= spot]
    resistance = None
    if not cw_above.empty:
        resistance = float(cw_above.loc[cw_above["Call_OI"].idxmax(), "Strike"])
    elif not call_walls.empty:
        resistance = float(call_walls.loc[call_walls["Call_OI"].idxmax(), "Strike"])

    # Support: strongest put wall BELOW spot
    pw_below = put_walls[put_walls["Strike"] <= spot]
    support = None
    if not pw_below.empty:
        support = float(pw_below.loc[pw_below["Put_OI"].idxmax(), "Strike"])
    elif not put_walls.empty:
        support = float(put_walls.loc[put_walls["Put_OI"].idxmax(), "Strike"])

    # Max pain
    max_pain = _calculate_max_pain(agg)

    # PCR sentiment
    if pcr < 0.7:
        sentiment = "Bearish"
    elif pcr < 0.9:
        sentiment = "Mildly Bearish"
    elif pcr < 1.1:
        sentiment = "Neutral"
    elif pcr < 1.3:
        sentiment = "Mildly Bullish"
    else:
        sentiment = "Bullish"

    res_pct = ((resistance - spot) / spot * 100) if resistance else 0.0
    sup_pct = ((support - spot) / spot * 100)    if support    else 0.0
    room    = (res_pct >= 1.5) if resistance else False

    return WallLevels(
        resistance=resistance, support=support, max_pain=max_pain,
        pcr_oi=round(pcr, 3), pcr_sentiment=sentiment,
        resistance_pct=round(res_pct, 2), support_pct=round(sup_pct, 2),
        room_to_run=room, walls_df=agg,
    )


def _calculate_max_pain(agg: pd.DataFrame) -> Optional[float]:
    strikes = sorted(agg["Strike"].tolist())
    if len(strikes) < 3:
        return None
    pain = {}
    for s in strikes:
        call_pain = sum(max(0, s - k) * agg.loc[agg["Strike"] == k, "Call_OI"].values[0]
                        for k in strikes if k in agg["Strike"].values)
        put_pain  = sum(max(0, k - s) * agg.loc[agg["Strike"] == k, "Put_OI"].values[0]
                        for k in strikes if k in agg["Strike"].values)
        pain[s] = call_pain + put_pain
    return float(min(pain, key=pain.get))


def _compute_gex(df: pd.DataFrame, spot: float, lot_size: int,
                 risk_free: float, max_dte: int) -> GEXData:
    today = datetime.date.today()
    rows = []

    for _, row in df.iterrows():
        strike = float(row["Strike"])
        expiry_str = str(row.get("Expiry", ""))
        dte = _parse_dte(expiry_str, today)
        if dte < 0 or dte > max_dte:
            continue

        t = max(dte / 365, 1 / 365)
        ce_iv = float(row.get("CE_IV", 0)) / 100 if float(row.get("CE_IV", 0)) > 1 else float(row.get("CE_IV", 0))
        pe_iv = float(row.get("PE_IV", 0)) / 100 if float(row.get("PE_IV", 0)) > 1 else float(row.get("PE_IV", 0))
        ce_oi = float(row.get("CE_OI", 0))
        pe_oi = float(row.get("PE_OI", 0))

        if ce_iv > 0.01 and ce_oi > 0:
            g = _bs_gamma(spot, strike, t, risk_free, ce_iv)
            rows.append(("call", strike, dte, g * ce_oi * lot_size * spot**2 / 1e6))

        if pe_iv > 0.01 and pe_oi > 0:
            g = _bs_gamma(spot, strike, t, risk_free, pe_iv)
            rows.append(("put", strike, dte, -g * pe_oi * lot_size * spot**2 / 1e6))

    if not rows:
        return GEXData(spot=spot)

    gex_df = pd.DataFrame(rows, columns=["type", "strike", "dte", "gex"])
    net = float(gex_df["gex"].sum())
    abs_total = float(gex_df["gex"].abs().sum())

    # HVL: strike where cumulative GEX crosses zero
    by_strike = gex_df.groupby("strike")["gex"].sum().sort_index()
    cumulative = by_strike.cumsum()
    hvl = None
    for i in range(1, len(cumulative)):
        if cumulative.iloc[i-1] * cumulative.iloc[i] < 0:
            hvl = float(cumulative.index[i])
            break

    regime = "Negative" if net < -abs_total * 0.05 else \
             "Positive" if net >  abs_total * 0.05 else "Neutral"

    # GEX by expiry
    by_expiry = []
    for exp_val, grp in gex_df.groupby("dte"):
        exp_net  = float(grp["gex"].sum())
        exp_pct  = abs(exp_net) / abs_total * 100 if abs_total > 0 else 0
        exp_dte  = int(exp_val)
        exp_date = (today + datetime.timedelta(days=exp_dte)).strftime("%d-%b-%Y")
        by_expiry.append((exp_date, exp_dte, exp_net, round(exp_pct, 1)))
    by_expiry.sort(key=lambda x: x[1])

    return GEXData(
        net_gex=round(net, 0), abs_gex=round(abs_total, 0),
        regime=regime, hvl=hvl, by_expiry=by_expiry, spot=spot,
    )


def _bs_gamma(S, K, T, r, sigma):
    try:
        if sigma <= 0 or T <= 0:
            return 0.0
        from math import log, sqrt, exp
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
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


def _compute_expiry_context(df: pd.DataFrame, spot: float,
                             max_pain: Optional[float]) -> ExpiryContext:
    today = datetime.date.today()
    expiries_raw = df["Expiry"].dropna().unique().tolist() if "Expiry" in df.columns else []

    dtes = []
    for e in expiries_raw:
        dte = _parse_dte(str(e), today)
        if dte >= 0:
            dtes.append((dte, str(e)))
    dtes.sort()

    if not dtes:
        return ExpiryContext()

    next_dte, next_exp = dtes[0]
    if next_dte == 0 and len(dtes) > 1:
        next_dte, next_exp = dtes[1]

    # Spot vs max pain
    mp_pct = ((spot - max_pain) / max_pain * 100) if max_pain and max_pain > 0 else 0.0

    # Pin risk
    if next_dte <= 1:
        pin_risk = "HIGH"
        warning  = f"Expiry TODAY — expect pinning behaviour near {max_pain:.0f}"
    elif next_dte <= 2:
        pin_risk = "HIGH"
        warning  = f"Expiry in {next_dte}d — strong pin risk near {max_pain:.0f}"
    elif next_dte <= 4:
        pin_risk = "MODERATE"
        warning  = f"Expiry in {next_dte}d — watch for compression Thu/Fri"
    else:
        pin_risk = "LOW"
        warning  = None

    return ExpiryContext(
        next_expiry=next_exp, days_remaining=next_dte,
        spot_vs_maxpain=round(mp_pct, 2),
        pin_risk=pin_risk, expiry_warning=warning,
    )


def _compute_viability(ctx: OptionsContext, spot: float) -> TradeViability:
    """Score the trade setup from an options-context perspective (0–100)."""
    score = 50
    checklist = []

    # 1. GEX regime
    if ctx.gex.regime == "Negative":
        score += 15
        checklist.append({
            "item": "GEX Regime",
            "status": "pass",
            "detail": "Negative GEX",
            "implication": "Dealers amplify moves → momentum signals are MORE reliable",
        })
    elif ctx.gex.regime == "Positive":
        score -= 15
        checklist.append({
            "item": "GEX Regime",
            "status": "warn",
            "detail": "Positive GEX",
            "implication": "Market tends to pin near max pain → fade moves, don't chase",
        })
    else:
        checklist.append({
            "item": "GEX Regime",
            "status": "neutral",
            "detail": "Neutral GEX",
            "implication": "No strong dealer bias — watch OI walls for direction",
        })

    # 2. Wall distance
    res_pct = ctx.walls.resistance_pct
    if res_pct >= 2.5:
        score += 20
        checklist.append({
            "item": "Wall Distance",
            "status": "pass",
            "detail": f"+{res_pct:.1f}% to resistance",
            "implication": f"Resistance at {ctx.walls.resistance:.0f} — ample room for momentum trade",
        })
    elif res_pct >= 1.5:
        score += 10
        checklist.append({
            "item": "Wall Distance",
            "status": "pass",
            "detail": f"+{res_pct:.1f}% to resistance",
            "implication": f"Acceptable clearance. Target {ctx.walls.resistance:.0f} and monitor",
        })
    elif res_pct > 0:
        score -= 10
        checklist.append({
            "item": "Wall Distance",
            "status": "warn",
            "detail": f"Only +{res_pct:.1f}% to resistance",
            "implication": f"Wall at {ctx.walls.resistance:.0f} is too close — risk of pinning or reversal",
        })
    else:
        checklist.append({
            "item": "Wall Distance",
            "status": "neutral",
            "detail": "No resistance wall mapped",
            "implication": "Load options data to get wall levels",
        })

    # 3. Expiry risk
    dte = ctx.expiry.days_remaining
    if dte >= 5:
        score += 15
        checklist.append({
            "item": "Expiry Timing",
            "status": "pass",
            "detail": f"{dte} days to expiry",
            "implication": "Plenty of time — expiry compression not a concern",
        })
    elif dte >= 3:
        score += 0
        checklist.append({
            "item": "Expiry Timing",
            "status": "warn",
            "detail": f"{dte} days to expiry",
            "implication": "Nearing expiry — watch for pin behaviour Thu/Fri",
        })
    else:
        score -= 20
        checklist.append({
            "item": "Expiry Timing",
            "status": "fail",
            "detail": f"Only {dte} days to expiry",
            "implication": "High pin risk — reduce size or wait for next cycle",
        })

    # 4. PCR context
    pcr = ctx.walls.pcr_oi
    if 0.9 <= pcr <= 1.3:
        score += 5
        checklist.append({
            "item": "PCR Context",
            "status": "pass",
            "detail": f"PCR {pcr:.2f} — {ctx.walls.pcr_sentiment}",
            "implication": "Balanced positioning — neither extreme fear nor greed",
        })
    elif pcr > 1.5:
        score += 10
        checklist.append({
            "item": "PCR Context",
            "status": "pass",
            "detail": f"PCR {pcr:.2f} — {ctx.walls.pcr_sentiment}",
            "implication": "Heavy put buying = hedging, not conviction selling → supports upside",
        })
    elif pcr < 0.7:
        score -= 10
        checklist.append({
            "item": "PCR Context",
            "status": "warn",
            "detail": f"PCR {pcr:.2f} — {ctx.walls.pcr_sentiment}",
            "implication": "Low PCR = complacency / aggressive call buying → watch for reversal",
        })
    else:
        checklist.append({
            "item": "PCR Context",
            "status": "neutral",
            "detail": f"PCR {pcr:.2f}",
            "implication": "Neutral positioning",
        })

    score = max(0, min(100, score))

    if score >= 75:
        label, color = "STRONG SETUP", "green"
    elif score >= 55:
        label, color = "PROCEED", "emerald"
    elif score >= 40:
        label, color = "CAUTION", "yellow"
    else:
        label, color = "AVOID", "red"

    return TradeViability(score=score, label=label, color=color, checklist=checklist)
