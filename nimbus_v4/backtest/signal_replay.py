"""
backtest/signal_replay.py
──────────────────────────
Walk-forward signal replay — the core of NIMBUS backtesting.

For each bar T in [warmup..N]:
  1. Take history[:T+1] (no lookahead)
  2. Compute all NIMBUS signals using existing indicator functions
  3. Record signal state at T
  4. Record forward returns at T+1, T+5, T+10

Output: DataFrame where each row = one bar with full signal snapshot + outcomes.
This is the raw material for efficacy tables, weight calibration, and correlation analysis.

CRITICAL: uses the EXACT same functions as the live pipeline.
  - add_bollinger(period=20, std_dev=1.0)
  - add_williams_r(period=50)
  - compute_price_signals(wr_thresh=-20.0)
No reimplementation. If the live pipeline is wrong, the backtest is wrong in the same way.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from modules.indicators import (
    add_bollinger,
    add_williams_r,
    compute_price_signals,
    PriceSignals,
)

logger = logging.getLogger(__name__)

# Minimum bars before first signal computation
# 50 (WR period) + 20 (BB period) + 20 (daily SMA) = 90 minimum
WARMUP_BARS = 90

# Forward return horizons (trading days)
FORWARD_HORIZONS = [1, 3, 5, 10, 20]


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-SYMBOL REPLAY
# ══════════════════════════════════════════════════════════════════════════════


def replay_signals(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    bb_period: int = 20,
    bb_std: float = 1.0,
    wr_period: int = 50,
    wr_thresh: float = -20.0,
    step: int = 1,
) -> pd.DataFrame:
    """
    Walk-forward signal replay for one symbol.

    Args:
        df: OHLCV DataFrame (daily), must have Open/High/Low/Close/Volume
        symbol: symbol name (for output labeling)
        step: compute signals every N bars (1=every bar, 5=weekly)

    Returns:
        DataFrame with columns:
          - date, symbol, close
          - All PriceSignals fields (wr_value, wr_phase, bb_position, etc.)
          - Forward returns: fwd_1d, fwd_3d, fwd_5d, fwd_10d, fwd_20d
          - fwd_max_dd_5d: max drawdown in next 5 bars
          - fwd_max_up_5d: max upside in next 5 bars
    """
    if df is None or len(df) < WARMUP_BARS + max(FORWARD_HORIZONS):
        logger.warning("%s: insufficient bars (%d) for replay", symbol, len(df) if df is not None else 0)
        return pd.DataFrame()

    records = []
    n = len(df)

    # Pre-compute full indicator columns once (for forward return lookups)
    full_close = df["Close"].values

    for t in range(WARMUP_BARS, n - max(FORWARD_HORIZONS), step):
        # ── Slice history up to T (inclusive) — NO LOOKAHEAD ──────────────
        hist = df.iloc[: t + 1].copy()

        # ── Apply indicators using EXACT live pipeline functions ──────────
        hist = add_bollinger(hist, period=bb_period, std_dev=bb_std)
        hist = add_williams_r(hist, period=wr_period)

        # ── Compute signals ───────────────────────────────────────────────
        ps = compute_price_signals(hist, wr_thresh=wr_thresh)

        # ── Forward returns (from actual future data) ─────────────────────
        spot = full_close[t]
        fwd = {}
        for h in FORWARD_HORIZONS:
            if t + h < n:
                fwd[f"fwd_{h}d"] = (full_close[t + h] / spot - 1) * 100
            else:
                fwd[f"fwd_{h}d"] = np.nan

        # Max drawdown and max upside in next 5 bars
        if t + 5 < n:
            future_5 = full_close[t + 1: t + 6]
            fwd["fwd_max_dd_5d"] = (min(future_5) / spot - 1) * 100
            fwd["fwd_max_up_5d"] = (max(future_5) / spot - 1) * 100
        else:
            fwd["fwd_max_dd_5d"] = np.nan
            fwd["fwd_max_up_5d"] = np.nan

        # ── Record ────────────────────────────────────────────────────────
        record = {
            "date": df.index[t],
            "symbol": symbol,
            "close": spot,
            # PriceSignals fields
            "daily_bias": ps.daily_bias,
            "daily_bias_pct": ps.daily_bias_pct,
            "bb_position": ps.bb_position,
            "bb_pct": ps.bb_pct,
            "position_state": ps.position_state,
            "vol_state": ps.vol_state,
            "bb_width_pctl": ps.bb_width_pctl,
            "bb_squeezing": ps.bb_squeezing,
            "wr_value": ps.wr_value,
            "wr_in_momentum": ps.wr_in_momentum,
            "wr_trend": ps.wr_trend,
            "wr_phase": ps.wr_phase,
            "wr_bars_since_cross50": ps.wr_bars_since_cross50,
            "entry_valid": ps.entry_valid,
            "adv_cr": ps.adv_cr,
            "mfi_value": ps.mfi_value,
            "mfi_state": ps.mfi_state,
            "mfi_diverge": ps.mfi_diverge,
            "mfi_reliable": ps.mfi_reliable,
            # Derived composite flags
            "momentum_pass": (
                ps.position_state == "RIDING_UPPER" and ps.wr_in_momentum
            ),
            "full_entry": (
                ps.position_state == "RIDING_UPPER"
                and ps.wr_in_momentum
                and ps.daily_bias == "BULLISH"
            ),
            **fwd,
        }
        records.append(record)

    result = pd.DataFrame(records)
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"])
        logger.info(
            "%s: replayed %d signal snapshots (%.0f%% of bars)",
            symbol, len(result), len(result) / n * 100,
        )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SYMBOL REPLAY
# ══════════════════════════════════════════════════════════════════════════════


def replay_universe(
    universe: dict[str, pd.DataFrame],
    step: int = 1,
    min_bars: int = 200,
) -> pd.DataFrame:
    """
    Run signal replay across an entire symbol universe.

    Returns a single stacked DataFrame with all symbols.
    """
    frames = []
    total = len(universe)
    for i, (sym, df) in enumerate(universe.items()):
        if len(df) < min_bars:
            logger.debug("Skipping %s: only %d bars", sym, len(df))
            continue
        result = replay_signals(df, symbol=sym, step=step)
        if not result.empty:
            frames.append(result)
        if (i + 1) % 10 == 0:
            logger.info("Replay progress: %d/%d symbols", i + 1, total)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Universe replay complete: %d snapshots across %d symbols",
        len(combined), len(frames),
    )
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL STATE ENCODER
# ══════════════════════════════════════════════════════════════════════════════


def encode_signal_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numeric encodings for categorical signal states.
    Used by weight calibration (logistic regression needs numeric features).
    """
    out = df.copy()

    # WR phase → numeric
    wr_map = {"FRESH": 3, "DEVELOPING": 2, "LATE": 1, "NONE": 0}
    out["wr_phase_n"] = out["wr_phase"].map(wr_map).fillna(0).astype(int)

    # BB position → numeric
    bb_map = {"above_upper": 4, "riding": 3, "below_mid": 1, "near_lower": 0}
    out["bb_position_n"] = out["bb_position"].map(bb_map).fillna(2).astype(int)

    # Position state → numeric
    ps_map = {
        "RIDING_UPPER": 4, "FIRST_DIP": 3, "CONSOLIDATING": 2,
        "MID_BAND_BROKEN": 0, "UNKNOWN": 1,
    }
    out["position_state_n"] = out["position_state"].map(ps_map).fillna(1).astype(int)

    # Vol state → numeric
    vol_map = {"SQUEEZE": 2, "NORMAL": 1, "EXPANDED": 0}
    out["vol_state_n"] = out["vol_state"].map(vol_map).fillna(1).astype(int)

    # Daily bias → numeric
    bias_map = {"BULLISH": 2, "NEUTRAL": 1, "BEARISH": 0}
    out["daily_bias_n"] = out["daily_bias"].map(bias_map).fillna(1).astype(int)

    # MFI state → numeric
    mfi_map = {"STRONG": 4, "RISING": 3, "NEUTRAL": 2, "FALLING": 1, "WEAK": 0}
    out["mfi_state_n"] = out["mfi_state"].map(mfi_map).fillna(2).astype(int)

    # Entry valid → int
    out["entry_valid_n"] = out["entry_valid"].astype(int)
    out["momentum_pass_n"] = out["momentum_pass"].astype(int)
    out["mfi_diverge_n"] = out["mfi_diverge"].astype(int)

    return out
