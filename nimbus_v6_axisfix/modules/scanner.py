"""
modules/scanner.py
──────────────────
Universe scanner: evaluate every F&O symbol against NIMBUS entry rules.

Scoring (Section 6 of spec — exact weights, no deviation):
  Momentum (BB + W%R)   : 25 pts max
  Regime (GEX/HVL)      : 20 pts max
  Structure (wall room) : 20 pts max
  Expiry / pin risk     : 15 pts max
  Daily bias            : 10 pts max
  Vol state             : 10 pts max
  Total                 : 100 pts

Entry conditions (all must pass for all_filters_pass=True):
  a. 4H close riding upper BB(20,1σ)
  b. W%R(50) > -20
  c. Daily 20-SMA bias = BULLISH
  d. GEX Regime = TREND_FRIENDLY
  e. % to call resistance >= 5%
  f. DTE >= 5
  g. Spot NOT pinned within 1.5% of max pain when DTE <= 4

Exit conditions (informational, shown in position_state):
  FIRST_DIP       → partial exit (scale 50%)
  MID_BAND_BROKEN → full exit

Sizing:
  FULL  : all entry pass + vol=SQUEEZE or NORMAL
  HALF  : PINNING regime OR DTE 3-4 with elevated pin OR vol=EXPANDED
  ZERO  : any hard rule fails
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
from modules.analytics import analyze, _parse_dte   # reuse analytics internals

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-SYMBOL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_symbol(symbol: str) -> Optional[dict]:
    """
    Run the full NIMBUS signal stack for one symbol.
    Returns a flat dict with all scanner fields, or None if data is unavailable.
    Never raises — logs and returns None on any exception.
    """
    try:
        return _analyze_symbol_inner(symbol)
    except Exception as exc:
        logger.warning(f"[scanner] {symbol} failed: {exc}")
        return None


def _analyze_symbol_inner(symbol: str) -> Optional[dict]:
    # ── Price data ────────────────────────────────────────────────────────────
    price_df, p_msg = get_price_4h(symbol, bars=120)
    if price_df is None or price_df.empty:
        return None

    price_df = add_bollinger(price_df, period=20, std_dev=1.0)
    price_df = add_williams_r(price_df, period=50)
    ps = compute_price_signals(price_df, wr_thresh=-20.0)

    last  = price_df.iloc[-1]
    spot  = float(last["Close"])

    # ── Options chain ─────────────────────────────────────────────────────────
    lot_size   = NSE_LOT_SIZES.get(symbol, 75)
    options_df, _ = download_options(symbol, max_expiries=3)

    # Infer spot from options chain if available (more accurate)
    if options_df is not None and not options_df.empty:
        opt_spot = infer_spot(options_df)
        if opt_spot and opt_spot > 0:
            spot = opt_spot

    # ── Options analytics ─────────────────────────────────────────────────────
    if options_df is not None and not options_df.empty:
        ctx = analyze(options_df, spot=spot, lot_size=lot_size,
                      price_signals=ps, room_thresh=5.0)
        has_opts = True
    else:
        ctx      = None
        has_opts = False

    # ── Extract fields ────────────────────────────────────────────────────────
    bb_upper = float(last.get("BB_Upper", np.nan)) if "BB_Upper" in price_df.columns else None
    bb_mid   = float(last.get("BB_Mid",   np.nan)) if "BB_Mid"   in price_df.columns else None
    wr_50    = ps.wr_value

    # riding_upper: use position_state from compute_price_signals() — this is the
    # confirmed 4-bar rule (last 4 closes all >= BB_Mid AND close near upper band).
    # A single "spot >= bb_upper" check is wrong — it fires on single-bar spikes
    # and misses the sustained band-riding condition the strategy requires.
    # FIRST_DIP included: trade is open, partial exit not yet triggered.
    # MID_BAND_BROKEN excluded: full exit signal — not a new entry candidate.
    riding_upper = ps.position_state in ("RIDING_UPPER", "FIRST_DIP")

    wr_above_m20   = ps.wr_in_momentum
    bars_since_m50 = ps.wr_bars_since_cross50 if ps.wr_bars_since_cross50 < 99 else float("nan")

    # position state mapping
    _state_map = {
        "RIDING_UPPER":    "RIDING_UPPER_BAND",
        "FIRST_DIP":       "FIRST_DIP",
        "MID_BAND_BROKEN": "MID_BAND_BROKEN",
        "CONSOLIDATING":   "BELOW_MID",
        "UNKNOWN":         "BELOW_MID",
    }
    position_state = _state_map.get(ps.position_state, "BELOW_MID")

    # Options-derived fields
    if has_opts and ctx:
        put_support      = ctx.walls.support
        call_resistance  = ctx.walls.resistance
        max_pain         = ctx.walls.max_pain
        pcr_oi           = ctx.walls.pcr_oi
        net_gex          = ctx.gex.net_gex
        hvl              = ctx.gex.hvl
        dte              = ctx.expiry.days_remaining
        pin_risk         = ctx.expiry.pin_risk

        pct_to_support    = ctx.walls.support_pct
        pct_to_resistance = ctx.walls.resistance_pct
        pct_to_max_pain   = ctx.expiry.spot_vs_maxpain

        gex_regime_raw = ctx.gex.regime
        # Map to scanner vocabulary: Negative/Neutral → TREND_FRIENDLY, Positive → PINNING
        gex_regime = "TREND_FRIENDLY" if gex_regime_raw in ("Negative", "Neutral") else "PINNING"

        near_mp = (
            max_pain is not None and spot > 0 and
            abs(spot - max_pain) / spot < 0.015 and dte <= 4
        )
        if pin_risk == "HIGH" or dte <= 2:
            expiry_risk = "HIGH"
        elif pin_risk == "MODERATE" or (dte <= 4 and near_mp):
            expiry_risk = "ELEVATED"
        else:
            expiry_risk = "LOW"
    else:
        # No options data — set all to None/nan, score will be 0 for these components
        put_support = call_resistance = max_pain = hvl = None
        pcr_oi = net_gex = None
        pct_to_support = pct_to_resistance = pct_to_max_pain = None
        dte = 99
        gex_regime = None
        expiry_risk = None
        near_mp = False

    # ── Filter booleans ───────────────────────────────────────────────────────
    # HARD GATE — momentum (BB riding + W%R > -20) is NON-NEGOTIABLE.
    # Without both, there is no trade thesis. This is never a checkbox the user
    # can disable. It is enforced directly in scan_universe() regardless of
    # require_all_filters or any UI toggle.
    passes_momentum = riding_upper and wr_above_m20
    passes_structure = (
        pct_to_resistance is not None and
        pct_to_resistance >= 5.0 and
        pct_to_support is not None and
        -8.0 <= pct_to_support <= -2.0
    )
    passes_regime  = (gex_regime == "TREND_FRIENDLY")
    passes_expiry  = (dte >= 5)
    passes_bias    = (ps.daily_bias == "BULLISH")
    # Entry condition g: NOT pinned within 1.5% of max pain when DTE <= 4
    passes_pin     = not near_mp

    all_filters_pass = (
        passes_momentum and passes_structure and passes_regime and
        passes_expiry and passes_bias and passes_pin
    )

    # ── Viability score — read from ctx.viability (SAME function as Dashboard) ──
    # Single source of truth. analytics.analyze() → _viability() computed it.
    # We never duplicate scoring logic here — that was the root cause of the
    # scanner/dashboard mismatch (scanner started at 0, dashboard started at 50;
    # completely different formulas on identical inputs).
    if has_opts and ctx:
        score = ctx.viability.score
        label = ctx.viability.label
        size  = ctx.viability.sizing
    else:
        # No options data: let _viability() score price signals only.
        _ctx_no_opts = analyze(pd.DataFrame(), spot=spot, lot_size=lot_size,
                               price_signals=ps, room_thresh=5.0)
        score = _ctx_no_opts.viability.score
        label = _ctx_no_opts.viability.label
        size  = _ctx_no_opts.viability.sizing
    score = max(0, min(100, score))

    # Short reason
    short_reason = _short_reason(
        passes_momentum, passes_structure, passes_regime,
        passes_expiry, passes_bias, ps, pct_to_resistance, dte,
        gex_regime, expiry_risk,
    )

    return {
        # Identity
        "symbol":           symbol,
        "timestamp":        datetime.datetime.now(),
        "scan_timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        # Price
        "last_price":       round(spot, 2),
        "bb_upper":         round(bb_upper, 2) if bb_upper and not np.isnan(bb_upper) else None,
        "bb_mid":           round(bb_mid,   2) if bb_mid   and not np.isnan(bb_mid)   else None,
        "wr_50":            round(wr_50, 1) if wr_50 is not None else None,
        "riding_upper_band":    riding_upper,
        "wr_above_minus20":     wr_above_m20,
        "bars_since_wr_cross_minus50": bars_since_m50,
        "position_state":       position_state,
        "daily_bias":           ps.daily_bias,
        "vol_state":            ps.vol_state,
        # Options
        "net_gex":          round(net_gex, 0)       if net_gex is not None else None,
        "gex_regime":       gex_regime,
        "hvl":              round(hvl, 0)            if hvl is not None else None,
        "put_support":      round(put_support, 0)    if put_support is not None else None,
        "call_resistance":  round(call_resistance,0) if call_resistance is not None else None,
        "pct_to_support":   round(pct_to_support, 2) if pct_to_support is not None else None,
        "pct_to_resistance":round(pct_to_resistance,2) if pct_to_resistance is not None else None,
        "max_pain":         round(max_pain, 0)       if max_pain is not None else None,
        "pct_to_max_pain":  round(pct_to_max_pain,2) if pct_to_max_pain is not None else None,
        "pcr_oi":           round(pcr_oi, 3)         if pcr_oi is not None else None,
        "dte":              dte if dte < 99 else None,
        "expiry_risk":      expiry_risk,
        # Viability
        "viability_score":  score,
        "viability_label":  label,
        "size_suggestion":  size,
        "short_reason":     short_reason,
        # Filter booleans
        "passes_momentum":      passes_momentum,
        "passes_structure":     passes_structure,
        "passes_regime":        passes_regime,
        "passes_expiry":        passes_expiry,
        "passes_daily_bias":    passes_bias,
        "all_filters_pass":     all_filters_pass,
    }


def _short_reason(
    p_mom, p_struct, p_regime, p_expiry, p_bias,
    ps, pct_res, dte, gex_regime, expiry_risk,
) -> str:
    if not p_bias:
        return "Daily bias bearish — no longs"
    if not p_mom:
        wr_str = f"W%R {ps.wr_value:.0f}" if ps.wr_value is not None else "W%R ?"
        return f"Momentum gate closed ({wr_str}, {'upper band' if ps.wr_in_momentum else 'not in zone'})"
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
    progress_cb=None,          # callable(done: int, total: int, symbol: str)
    max_workers: int = 8,
) -> list[dict]:
    """
    Run analyze_symbol() over all symbols in parallel.
    Returns list sorted by viability_score descending.
    """
    results = []
    total   = len(symbols)
    done    = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym  = futures[future]
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
            # HARD GATE — always enforced, regardless of require_all_filters.
            # A stock not riding the upper BB or with W%R <= -20 has no trade
            # thesis and must never appear in scanner output.
            if not row["passes_momentum"]:
                continue
            if row["viability_score"] < min_viability:
                continue
            if require_all_filters and not row["all_filters_pass"]:
                continue
            results.append(row)

    results.sort(key=lambda r: r["viability_score"], reverse=True)
    return results
