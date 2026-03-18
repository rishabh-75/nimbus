"""
modules/scanner.py
──────────────────
Universe scanner: evaluate every F&O symbol against NIMBUS entry rules.

Scoring (Section 6 of spec — exact weights, no deviation):
  Momentum (BB + W%R)  : 25 pts max
  Regime (GEX/HVL)     : 20 pts max
  Structure (walls)    : 20 pts max
  Expiry / pin risk    : 15 pts max
  Daily bias           : 10 pts max
  Vol state            : 10 pts max
  Total                : 100 pts

Entry conditions (all must pass for all_filters_pass=True):
  a. 4H close riding upper BB(20,1σ) — RIDING_UPPER only (not FIRST_DIP)
  b. W%R(50) > -20
  c. Daily 20-SMA bias = BULLISH
  d. GEX Regime = TREND_FRIENDLY
  e. % to call resistance >= 5%
  f. DTE >= 5
  g. Spot NOT pinned within 2% of max pain when DTE <= 4

Exit conditions (informational, shown in position_state):
  FIRST_DIP → partial exit (scale 50%) — NOT a new entry signal
  MID_BAND_BROKEN → full exit

Sizing:
  FULL  : all entry pass + vol=SQUEEZE or NORMAL
  HALF  : PINNING regime OR DTE 3-4 with elevated pin OR vol=EXPANDED
  SKIP  : any hard rule fails

Phase 1 additions:
  - Imports setup_classifier (classify_setup_v3, OptionsSignalState, MomentumState)
  - _analyze_symbol_inner() builds OptionsSignalState + MomentumState from ctx/ps
  - Return dict gains: setup_type, setup_detail, setup_color
  - ScanWorker sort: TRAP first, then SETUP_PRIORITY, then viability_score desc

Patch v5.2 fixes applied (unchanged):
  FIX-5    : FIRST_DIP removed from riding_upper
  FIX-8    : near_mp threshold aligned to 2.0%
  FIX-DTE0 : expiry_risk="HIGH" when DTE=0
  FIX-REGIME: Neutral GEX with resistance_pct < 1.5% → "PINNING"
  FIX-RATE : max_workers reduced 8 → 3
  FIX-DELAY: _INTER_SYMBOL_DELAY = 1.2s
"""

from __future__ import annotations

import datetime
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from modules.data import get_price_4h, download_options, infer_spot, NSE_LOT_SIZES
from modules.indicators import add_bollinger, add_williams_r, compute_price_signals
from modules.analytics import analyze, _parse_dte
from modules.setup_classifier import (
    classify_setup_v3,
    OptionsSignalState,
    MomentumState,
    SETUP_COLORS,
    SETUP_PRIORITY,
    SetupType,
)

logger = logging.getLogger(__name__)

_INTER_SYMBOL_DELAY: float = 1.2

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-SYMBOL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def analyze_symbol(symbol: str) -> Optional[dict]:
    try:
        return _analyze_symbol_inner(symbol)
    except Exception as exc:
        logger.warning(f"[scanner] {symbol} failed: {exc}")
        return None


def _analyze_symbol_inner(symbol: str) -> Optional[dict]:
    time.sleep(_INTER_SYMBOL_DELAY)

    # ── Price data ────────────────────────────────────────────────────────
    price_df, p_msg = get_price_4h(symbol, bars=120)
    if price_df is None or price_df.empty:
        return None

    price_df = add_bollinger(price_df, period=20, std_dev=1.0)
    price_df = add_williams_r(price_df, period=50)
    ps = compute_price_signals(price_df, wr_thresh=-20.0)

    last = price_df.iloc[-1]
    spot = float(last["Close"])

    # ── Options chain ─────────────────────────────────────────────────────
    lot_size = NSE_LOT_SIZES.get(symbol, 75)
    options_df, _ = download_options(symbol, max_expiries=3)

    if options_df is not None and not options_df.empty:
        opt_spot = infer_spot(options_df)
        if opt_spot and opt_spot > 0:
            spot = opt_spot

    # ── Options analytics ─────────────────────────────────────────────────
    if options_df is not None and not options_df.empty:
        ctx = analyze(
            options_df, spot=spot, lot_size=lot_size, price_signals=ps, room_thresh=5.0
        )
        has_opts = True
    else:
        ctx = None
        has_opts = False

    # ── Extract price-side fields ─────────────────────────────────────────
    bb_upper = (
        float(last.get("BB_Upper", np.nan)) if "BB_Upper" in price_df.columns else None
    )
    bb_mid = float(last.get("BB_Mid", np.nan)) if "BB_Mid" in price_df.columns else None
    wr_50 = ps.wr_value

    riding_upper = ps.position_state == "RIDING_UPPER"
    wr_above_m20 = ps.wr_in_momentum
    bars_since_m50 = (
        ps.wr_bars_since_cross50 if ps.wr_bars_since_cross50 < 99 else float("nan")
    )

    _state_map = {
        "RIDING_UPPER": "RIDING_UPPER_BAND",
        "FIRST_DIP": "FIRST_DIP",
        "MID_BAND_BROKEN": "MID_BAND_BROKEN",
        "CONSOLIDATING": "BELOW_MID",
        "UNKNOWN": "BELOW_MID",
    }
    position_state = _state_map.get(ps.position_state, "BELOW_MID")

    # ── Options-derived fields ────────────────────────────────────────────
    if has_opts and ctx:
        put_support = ctx.walls.support
        call_resistance = ctx.walls.resistance
        max_pain = ctx.walls.max_pain
        pcr_oi = ctx.walls.pcr_oi
        net_gex = ctx.gex.net_gex
        hvl = ctx.gex.hvl
        dte = ctx.expiry.days_remaining
        pin_risk = ctx.expiry.pin_risk

        pct_to_support = ctx.walls.support_pct
        pct_to_resistance = ctx.walls.resistance_pct
        pct_to_max_pain = ctx.expiry.spot_vs_maxpain

        gex_regime_raw = ctx.gex.regime

        if gex_regime_raw == "Negative":
            gex_regime = "TREND_FRIENDLY"
        elif (
            gex_regime_raw == "Neutral"
            and pct_to_resistance is not None
            and pct_to_resistance >= 1.5
        ):
            gex_regime = "TREND_FRIENDLY"
        else:
            gex_regime = "PINNING"

        near_mp = (
            max_pain is not None
            and spot > 0
            and abs(spot - max_pain) / spot < 0.020
            and dte <= 4
        )

        if dte <= 0:
            expiry_risk = "HIGH"
        elif pin_risk == "HIGH" or dte <= 2:
            expiry_risk = "HIGH"
        elif pin_risk == "MODERATE" or (dte <= 4 and near_mp):
            expiry_risk = "ELEVATED"
        else:
            expiry_risk = "LOW"
    else:
        put_support = call_resistance = max_pain = hvl = None
        pcr_oi = net_gex = None
        pct_to_support = pct_to_resistance = pct_to_max_pain = None
        dte = 99
        gex_regime = None
        expiry_risk = None
        near_mp = False

    # ── Filter booleans ───────────────────────────────────────────────────
    passes_momentum = riding_upper and wr_above_m20
    passes_structure = (
        pct_to_resistance is not None
        and pct_to_resistance >= 5.0
        and pct_to_support is not None
        and -8.0 <= pct_to_support <= -2.0
    )
    passes_regime = gex_regime == "TREND_FRIENDLY"
    passes_expiry = dte is not None and dte >= 5
    passes_bias = ps.daily_bias == "BULLISH"
    passes_pin = not near_mp

    all_filters_pass = (
        passes_momentum
        and passes_structure
        and passes_regime
        and passes_expiry
        and passes_bias
        and passes_pin
    )

    # ── Viability score ───────────────────────────────────────────────────
    if has_opts and ctx:
        score = ctx.viability.score
        label = ctx.viability.label
        size = ctx.viability.sizing
    else:
        from modules.analytics import analyze_price_only

        _ctx_no_opts = analyze_price_only(spot=spot, price_signals=ps, room_thresh=5.0)
        score = _ctx_no_opts.viability.score
        label = _ctx_no_opts.viability.label
        size = _ctx_no_opts.viability.sizing
    score = max(0, min(100, score))

    short_reason = _short_reason(
        passes_momentum,
        passes_structure,
        passes_regime,
        passes_expiry,
        passes_bias,
        ps,
        pct_to_resistance,
        dte,
        gex_regime,
        expiry_risk,
    )

    # ── ETF RS Score → viability adjustment ──────────────────────────────
    # Only applied when the symbol is a known sector ETF in SECTOR_MAP.
    # For regular F&O stocks this block is skipped entirely (sc stays None).
    sc: Optional[dict] = None
    try:
        from modules.sector_map import SECTOR_MAP

        _is_etf = any(
            t.upper().replace(".NS", "") == symbol.upper().replace(".NS", "")
            for t in SECTOR_MAP
        )
        if _is_etf:
            from modules.sector_rotation import get_sector_context

            sc = get_sector_context(symbol)  # zero extra network calls — uses cache
            if sc:
                adj = sc["adj_score"]
                rot = sc["rotation"]
                flags = sc["flags"]

                # RS tier → base contribution
                if adj > 5.0:
                    rs_contrib = +10
                elif adj > 2.0:
                    rs_contrib = +5
                elif adj > 0.0:
                    rs_contrib = +2
                elif adj > -2.0:
                    rs_contrib = -5
                else:
                    rs_contrib = -10

                # Quality flag penalties / boost
                flag_penalty = 0
                for f in flags:
                    if "PREM+" in f:
                        try:
                            prem = float(f.replace("PREM+", "").replace("%", ""))
                            flag_penalty += int(min(prem * 4, 15))  # 1.4%→5, 2.5%→10
                        except Exception:
                            flag_penalty += 5
                    if "THIN-VOL" in f:
                        flag_penalty += 8
                    if "LOW-VOL" in f:
                        flag_penalty += 3
                    if "HIGH-VOL" in f:
                        flag_penalty -= 3  # rising participation = good

                etf_delta = rs_contrib - flag_penalty
                score = max(0, min(100, score + etf_delta))

                # Override short_reason when quality is the binding constraint
                if flag_penalty > rs_contrib and score < 50:
                    short_reason = (
                        f"ETF quality: {rot} RS={sc['rs_score']:+.1f} "
                        f"adj={adj:+.1f} | " + "  ".join(flags)
                    )

                # Recompute label + size after ETF delta
                label, size = (
                    ctx.viability.__class__.from_score(score)
                    if hasattr(ctx, "viability")
                    else (label, size)
                )

    except Exception as exc:
        logger.debug("ETF RS enrichment %s: %s", symbol, exc)

    # ── Phase 1: Setup Classification ─────────────────────────────────────
    # Uses the (possibly ETF-adjusted) score so setup labels stay consistent.
    setup_type_val = SetupType.NEUTRAL
    setup_detail = ""
    setup_color = SETUP_COLORS[SetupType.NEUTRAL]

    try:
        if has_opts and ctx:
            opts_state = OptionsSignalState(
                gex_regime=ctx.gex.regime,
                gex_rising=ctx.gex_rising,
                pcr=ctx.pcr,
                pcr_trending=ctx.pcr_trending,
                iv_skew=ctx.iv_skew,
                delta_bias=ctx.delta_bias,
                call_oi_wall_pct=ctx.call_oi_wall_pct,
                pct_to_resistance=pct_to_resistance,
                pcr_oi=ctx.walls.pcr_oi,
            )
        else:
            opts_state = OptionsSignalState()

        mom_state = MomentumState(
            bb_position=ps.bb_position,
            position_state=ps.position_state,
            vol_state=ps.vol_state,
            wr_phase=ps.wr_phase,
            wr_value=ps.wr_value,
            wr_in_momentum=ps.wr_in_momentum,
        )

        fv = None

        setup_type_val, setup_detail = classify_setup_v3(
            viability_score=score,
            filing_variance=fv.variance if fv else None,
            filing_direction=fv.badge_color if fv else "NONE",
            filing_conviction=fv.conviction if fv else 0,
            filing_category=fv.category.value if fv else "OTHER",
            opts=opts_state,
            mom=mom_state,
        )
        setup_color = SETUP_COLORS[setup_type_val]

    except Exception as exc:
        logger.debug("[scanner] setup_classify %s: %s", symbol, exc)

    return {
        "symbol": symbol,
        "timestamp": datetime.datetime.now(),
        "scan_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "last_price": round(spot, 2),
        "bb_upper": (
            round(bb_upper, 2) if bb_upper and not np.isnan(bb_upper) else None
        ),
        "bb_mid": (round(bb_mid, 2) if bb_mid and not np.isnan(bb_mid) else None),
        "wr_50": (round(wr_50, 1) if wr_50 is not None else None),
        "riding_upper_band": riding_upper,
        "wr_above_minus20": wr_above_m20,
        "bars_since_wr_cross_minus50": bars_since_m50,
        "position_state": position_state,
        "daily_bias": ps.daily_bias,
        "vol_state": ps.vol_state,
        "net_gex": (round(net_gex, 0) if net_gex is not None else None),
        "gex_regime": gex_regime,
        "hvl": (round(hvl, 0) if hvl is not None else None),
        "put_support": (round(put_support, 0) if put_support is not None else None),
        "call_resistance": (
            round(call_resistance, 0) if call_resistance is not None else None
        ),
        "pct_to_support": (
            round(pct_to_support, 2) if pct_to_support is not None else None
        ),
        "pct_to_resistance": (
            round(pct_to_resistance, 2) if pct_to_resistance is not None else None
        ),
        "max_pain": (round(max_pain, 0) if max_pain is not None else None),
        "pct_to_max_pain": (
            round(pct_to_max_pain, 2) if pct_to_max_pain is not None else None
        ),
        "pcr_oi": (round(pcr_oi, 3) if pcr_oi is not None else None),
        "dte": (dte if dte is not None and dte < 99 else None),
        "expiry_risk": expiry_risk,
        "viability_score": score,
        "viability_label": label,
        "size_suggestion": size,
        "short_reason": short_reason,
        "passes_momentum": passes_momentum,
        "passes_structure": passes_structure,
        "passes_regime": passes_regime,
        "passes_expiry": passes_expiry,
        "passes_daily_bias": passes_bias,
        "all_filters_pass": all_filters_pass,
        # Phase 1
        "setup_type": setup_type_val.value,
        "setup_detail": setup_detail,
        "setup_color": setup_color,
        "setup_priority": SETUP_PRIORITY[setup_type_val],
        # Existing extended fields
        "adv_cr": round(ps.adv_cr, 2),
        "gex_regime_raw": gex_regime_raw if (has_opts and ctx) else "Neutral",
        "gex_rising": ctx.gex_rising if (has_opts and ctx) else False,
        "pcr_trending": ctx.pcr_trending if (has_opts and ctx) else "FLAT",
        "iv_skew": ctx.iv_skew if (has_opts and ctx) else "FLAT",
        "delta_bias": ctx.delta_bias if (has_opts and ctx) else "NEUTRAL",
        "call_oi_wall_pct": ctx.call_oi_wall_pct if (has_opts and ctx) else 0.0,
        "bb_position": ps.bb_position,
        "wr_phase": ps.wr_phase,
        "has_options": has_opts,
        # ETF RS enrichment (None for regular F&O stocks)
        "sector_rs": sc["adj_score"] if sc else None,
        "sector_rot": sc["rotation"] if sc else None,
        "sector_flags": sc["flags"] if sc else [],
    }


def _short_reason(
    p_mom,
    p_struct,
    p_regime,
    p_expiry,
    p_bias,
    ps,
    pct_res,
    dte,
    gex_regime,
    expiry_risk,
) -> str:
    if not p_bias:
        return "Daily bias bearish — no longs"
    if not p_mom:
        if ps.position_state == "FIRST_DIP":
            return "FIRST_DIP — manage existing position, no new entries"
        wr_str = f"W%R {ps.wr_value:.0f}" if ps.wr_value is not None else "W%R ?"
        return (
            f"Momentum gate closed ({wr_str}, "
            f"{'upper band' if ps.wr_in_momentum else 'not in zone'})"
        )
    if not p_struct:
        r = f"{pct_res:.1f}%" if pct_res is not None else "?"
        return f"Structural room {r} — below 5% minimum"
    if not p_regime:
        return "GEX Pinning regime — momentum headwinds"
    if not p_expiry:
        return f"Expiry risk: {expiry_risk or 'HIGH'} ({dte}d to expiry)"
    phase = ps.wr_phase or "?"
    res = f"{pct_res:.1f}%" if pct_res is not None else "?"
    return f"{phase} W%R momentum, {res} room, {gex_regime} regime, {expiry_risk or 'LOW'} pin risk"


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE SCAN
# ══════════════════════════════════════════════════════════════════════════════


def scan_universe(
    symbols: list[str],
    require_all_filters: bool = True,
    min_viability: int = 50,
    progress_cb=None,
    max_workers: int = 3,
) -> list[dict]:
    results = []
    total = len(symbols)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total, sym)
                except Exception:
                    pass
            try:
                row = future.result()
            except Exception as exc:
                logger.warning(f"[scan] {sym}: {exc}")
                row = None
            if row is None:
                continue
            if not row["passes_momentum"]:
                continue
            if row["viability_score"] < min_viability:
                continue
            if require_all_filters and not row["all_filters_pass"]:
                continue
            results.append(row)

    # Sort: TRAP first (setup_priority=1), then by priority asc, then score desc
    results.sort(key=lambda r: (r.get("setup_priority", 8), -r["viability_score"]))
    return results
