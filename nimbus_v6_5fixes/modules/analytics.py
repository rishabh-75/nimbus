"""
modules/analytics.py
────────────────────
Options analytics + integrated viability scoring.
Now accepts PriceSignals for daily bias, vol state, WR phase.

Phase 1 additions (non-breaking):
  OptionsContext: pcr, pcr_trending, iv_skew, delta_bias, gex_rising, call_oi_wall_pct
  New private: _options_signal_state(df, spot, ctx) → dict
  analyze(): calls _options_signal_state at end, sets ctx fields via setattr
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

    regime: str = "UNKNOWN"  # TREND-FRIENDLY | PINNING | NEUTRAL
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
    # ── Phase 1: OptionsSignalState fields (all safe defaults) ──────────────
    pcr: float = 0.0  # Put-Call Ratio (OI-based)
    pcr_trending: str = "FLAT"  # "FALLING" | "RISING" | "FLAT"
    iv_skew: str = "FLAT"  # "CALL_CHEAP" | "PUT_CHEAP" | "FLAT"
    delta_bias: str = "NEUTRAL"  # "LONG" | "SHORT" | "NEUTRAL"
    gex_rising: bool = False  # net_gex > 0 (call writers dominate)
    call_oi_wall_pct: float = 0.0  # nearest call wall % distance above spot


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

    # Phase 1: derive signal-state fields from existing computed data
    for k, v in _options_signal_state(df, spot, ctx).items():
        setattr(ctx, k, v)

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — OPTIONS SIGNAL STATE
# ══════════════════════════════════════════════════════════════════════════════


def _options_signal_state(df: pd.DataFrame, spot: float, ctx: OptionsContext) -> dict:
    """
    Derive 6 signal-state fields from the already-computed ctx + raw df.
    All values are safe to fail silently — defaults are harmless.
    Column names from _clean(): Strike, CE_OI, PE_OI, CE_IV, PE_IV, CE_LTP, PE_LTP
    NSE also provides CE_Change_OI / PE_Change_OI on live sessions — used when present.
    """
    result = dict(
        pcr=0.0,
        pcr_trending="FLAT",
        iv_skew="FLAT",
        delta_bias="NEUTRAL",
        gex_rising=False,
        call_oi_wall_pct=0.0,
    )
    try:
        # ── PCR: reuse walls computation (already correct) ────────────────
        result["pcr"] = ctx.walls.pcr_oi

        # ── PCR trending: requires CE_Change_OI / PE_Change_OI columns ────
        # NSE option chain CSV and live API both provide these; present when
        # download_options() parses the NSE "changeinOpenInterest" fields.
        if "CE_Change_OI" in df.columns and "PE_Change_OI" in df.columns:
            ce_oi = df["CE_OI"].sum()
            pe_oi = df["PE_OI"].sum()
            ce_ch = pd.to_numeric(df["CE_Change_OI"], errors="coerce").fillna(0).sum()
            pe_ch = pd.to_numeric(df["PE_Change_OI"], errors="coerce").fillna(0).sum()
            if ce_oi > 0 and pe_oi > 0:
                pcr_delta = (pe_ch / pe_oi) - (ce_ch / ce_oi)
                if pcr_delta > 0.04:
                    result["pcr_trending"] = "RISING"
                elif pcr_delta < -0.04:
                    result["pcr_trending"] = "FALLING"

        # ── IV skew: CE_IV vs PE_IV in ATM ±3% range ─────────────────────
        atm_df = df[
            (df["Strike"] >= spot * 0.97) & (df["Strike"] <= spot * 1.03)
        ].copy()
        if not atm_df.empty:
            ce_iv = (
                pd.to_numeric(atm_df["CE_IV"], errors="coerce")
                .replace(0, float("nan"))
                .mean()
            )
            pe_iv = (
                pd.to_numeric(atm_df["PE_IV"], errors="coerce")
                .replace(0, float("nan"))
                .mean()
            )
            if ce_iv and pe_iv and pe_iv > 0 and not (pd.isna(ce_iv) or pd.isna(pe_iv)):
                r = ce_iv / pe_iv
                if r < 0.92:
                    result["iv_skew"] = "CALL_CHEAP"
                elif r > 1.08:
                    result["iv_skew"] = "PUT_CHEAP"

        # ── Delta bias: proxy from PCR + GEX regime ───────────────────────
        # CE_Change_OI path (higher fidelity) already handled above.
        # Without change OI, use PCR + regime as a directional proxy.
        pcr = ctx.walls.pcr_oi
        gex_regime = ctx.gex.regime  # "Negative" | "Positive" | "Neutral"
        if pcr >= 1.3 and gex_regime == "Negative":
            result["delta_bias"] = "LONG"
        elif pcr <= 0.8 and gex_regime == "Positive":
            result["delta_bias"] = "SHORT"

        # Upgrade with change OI if available
        if "CE_Change_OI" in df.columns and "PE_Change_OI" in df.columns:
            ce_oi = df["CE_OI"].sum()
            pe_oi = df["PE_OI"].sum()
            ce_ch = pd.to_numeric(df["CE_Change_OI"], errors="coerce").fillna(0).sum()
            pe_ch = pd.to_numeric(df["PE_Change_OI"], errors="coerce").fillna(0).sum()
            if ce_oi > 0 and pe_oi > 0:
                ce_flow = ce_ch / ce_oi
                pe_flow = pe_ch / pe_oi
                if ce_flow - pe_flow > 0.04:
                    result["delta_bias"] = "LONG"
                elif ce_flow - pe_flow < -0.04:
                    result["delta_bias"] = "SHORT"

        # ── GEX rising: net_gex > 0 ─────────────────────────────────────
        result["gex_rising"] = ctx.gex.net_gex > 0

        # ── Call wall distance ────────────────────────────────────────────
        if ctx.walls.resistance and spot > 0:
            result["call_oi_wall_pct"] = round(ctx.walls.resistance_pct, 2)

    except Exception as exc:
        import logging as _log

        _log.getLogger(__name__).debug("_options_signal_state: %s", exc)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL CALCULATIONS  (unchanged from original)
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

    live = [(d, e) for d, e in dtes if d > 0]
    if not live:
        return ExpiryCtx()

    next_dte, next_exp = live[0]
    mp_pct = ((spot - max_pain) / max_pain * 100) if max_pain else 0.0
    near_mp = (abs(spot - max_pain) / spot < 0.012) if (max_pain and spot) else False

    if next_dte <= 1:
        pin_risk = "HIGH"
        warning = f"Expiry TODAY/TOMORROW — no new swings"
    elif next_dte <= 3 and near_mp:
        pin_risk = "HIGH"
        warning = f"Spot near max pain ({max_pain:,.0f}) with {next_dte}d left"
    elif next_dte <= 5:
        pin_risk = "MODERATE"
        warning = f"{next_dte}d to expiry — watch pin compression"
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
    dte = expiry.days_remaining
    pin_risk = expiry.pin_risk
    res_pct = walls.resistance_pct
    gex_reg = gex.regime

    if gex_reg == "Negative":
        if pin_risk == "HIGH":
            return RegimeClass(
                "PINNING",
                "Negative GEX + HIGH pin risk",
                "HALF",
                "Gamma flip risk near expiry",
                "yellow",
            )
        return RegimeClass(
            "TREND-FRIENDLY",
            "Negative GEX — dealers short gamma",
            "FULL",
            "Trending environment favors momentum",
            "emerald",
        )

    if gex_reg == "Neutral":
        if res_pct >= 1.5:
            return RegimeClass(
                "TREND-FRIENDLY",
                "Neutral GEX + adequate room",
                "FULL",
                "Room to run to resistance",
                "emerald",
            )
        return RegimeClass(
            "PINNING",
            "Neutral GEX + resistance nearby",
            "HALF",
            "Compressed room → pinning tendency",
            "yellow",
        )

    # Positive GEX
    return RegimeClass(
        "PINNING",
        "Positive GEX — dealers long gamma",
        "HALF",
        "Mean-reversion pressure",
        "red",
    )


def _viability(
    ctx: OptionsContext, ps: Optional[PriceSignals], room_thresh: float
) -> Viability:
    score = 50
    checklist = []
    risk_notes = []

    def add(item, status, detail, implication):
        checklist.append(CheckItem(item, status, detail, implication))

    def risk(note):
        risk_notes.append(note)

    # ── 1. Regime ─────────────────────────────────────────────────────────
    reg = ctx.regime.regime
    if reg == "TREND-FRIENDLY":
        score += 15
        add(
            "Regime",
            "pass",
            "TREND-FRIENDLY — negative GEX",
            "Dealers short gamma → momentum moves amplified",
        )
    elif reg == "PINNING":
        score -= 15
        add(
            "Regime",
            "fail",
            "PINNING — positive/neutral GEX with tight room",
            "Mean reversion pressure — momentum plays risky",
        )
        risk("Pinning regime — reduce size or skip")
    else:
        add("Regime", "neutral", "Neutral regime", "No directional edge from GEX")

    # ── 2. Room to resistance ──────────────────────────────────────────────
    res_pct = ctx.walls.resistance_pct
    if res_pct >= room_thresh * 2:
        score += 15
        add(
            "Room",
            "pass",
            f"{res_pct:+.1f}% to resistance",
            "Plenty of room — full momentum run possible",
        )
    elif res_pct >= room_thresh:
        score += 8
        add(
            "Room",
            "pass",
            f"{res_pct:+.1f}% to resistance",
            "Adequate room — trade with normal targets",
        )
    elif res_pct > 0:
        score -= 10
        add(
            "Room",
            "warn",
            f"Only {res_pct:+.1f}% to resistance",
            "Tight room — resistance may cap the move quickly",
        )
        risk(f"Only {res_pct:.1f}% to call resistance")
    else:
        score -= 20
        add(
            "Room",
            "fail",
            "No clear resistance above spot",
            "Cannot define upside target — skip or wait",
        )
        risk("No resistance level identified")

    # ── 3. Daily bias (from PriceSignals) ──────────────────────────────────
    if ps is not None:
        if ps.daily_bias == "BULLISH":
            score += 10
            add(
                "Daily Bias",
                "pass",
                f"BULLISH ({ps.daily_bias_pct:+.1f}% above 20-SMA)",
                "Daily trend aligned — reduces mean-reversion risk",
            )
        elif ps.daily_bias == "BEARISH":
            score -= 20
            add(
                "Daily Bias",
                "fail",
                f"BEARISH ({ps.daily_bias_pct:+.1f}% below 20-SMA)",
                "Daily downtrend — no long trades. Wait for reclaim.",
            )
            risk("Daily bias BEARISH — no new long entries")
        else:
            add(
                "Daily Bias",
                "neutral",
                f"NEUTRAL ({ps.daily_bias_pct:+.1f}% vs 20-SMA)",
                "Flat daily trend — proceed with extra caution",
            )

    # ── 4. Expiry / Max-Pain Risk Gate ─────────────────────────────────────
    dte = ctx.expiry.days_remaining
    if dte <= 1:
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

    # ── 5. PCR ─────────────────────────────────────────────────────────────
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
        add("PCR", "neutral", f"PCR {pcr:.2f}", "Neutral positioning")

    # ── 6. Volatility State ────────────────────────────────────────────────
    if ps is not None:
        vs = ps.vol_state
        if vs == "SQUEEZE":
            if ctx.regime.regime == "TREND-FRIENDLY":
                score += 12
                add(
                    "Vol State",
                    "pass",
                    f"SQUEEZE ({ps.bb_width_pctl:.0f}th pctl)",
                    "Compression + trend-friendly regime = coiled spring → very favorable",
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

    # ── 7. Williams %R Phase ───────────────────────────────────────────────
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
            if ps.wr_phase == "FRESH":
                score += 15
                add(
                    "W%R Gate",
                    "pass",
                    f"W%R {ps.wr_value:.1f} — FRESH momentum ({ps.wr_bars_since_cross50}b ago)",
                    f"Fresh cross above -50 just {ps.wr_bars_since_cross50} bars ago — early in momentum phase",
                )
            elif ps.wr_phase == "DEVELOPING":
                score += 8
                add(
                    "W%R Gate",
                    "pass",
                    f"W%R {ps.wr_value:.1f} — DEVELOPING ({ps.wr_bars_since_cross50}b since -50 cross)",
                    "Mid-phase momentum — still valid, monitor for signs of exhaustion",
                )
            elif ps.wr_phase == "LATE":
                score -= 5
                add(
                    "W%R Gate",
                    "warn",
                    f"W%R {ps.wr_value:.1f} — LATE phase ({ps.wr_bars_since_cross50}b since -50 cross)",
                    "Late in momentum phase — risk of mean reversion, tighten stops",
                )
                risk(
                    f"W%R late phase ({ps.wr_bars_since_cross50} bars since -50 cross) — reduce size"
                )

    # ── 8. Position State (BB) ─────────────────────────────────────────────
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

    # ── Final score + sizing ───────────────────────────────────────────────
    score = max(0, min(100, score))

    if score >= 78:
        label, color, sizing = "STRONG SETUP", "green", "FULL"
    elif score >= 58:
        label, color, sizing = "PROCEED", "emerald", "FULL"
    elif score >= 40:
        label, color, sizing = "CAUTION", "yellow", "HALF"
    else:
        label, color, sizing = "AVOID", "red", "SKIP"

    # Hard overrides
    if ps is not None and ps.daily_bias == "BEARISH":
        sizing = "SKIP"
        label = "AVOID"
        color = "red"

    if ctx.gex.regime == "Positive" and sizing == "FULL":
        sizing = "HALF"
        if label in ("STRONG SETUP", "PROCEED"):
            label, color = "CAUTION", "yellow"

    if ctx.regime.size_cap == "HALF" and sizing == "FULL":
        sizing = "HALF"

    if ctx.expiry.pin_risk == "HIGH" and sizing == "FULL":
        sizing = "HALF"

    if ps is not None and ps.wr_value is not None and not ps.wr_in_momentum:
        if sizing == "FULL":
            sizing = "HALF"

    if ps is not None and ps.position_state == "MID_BAND_BROKEN":
        if sizing in ("FULL", "HALF"):
            sizing = "SKIP"
            label = "AVOID"
            color = "red"

    return Viability(
        score=score,
        label=label,
        color=color,
        sizing=sizing,
        checklist=checklist,
        risk_notes=risk_notes,
    )
