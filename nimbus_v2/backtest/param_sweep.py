"""
backtest/param_sweep.py
────────────────────────
Systematic parameter optimization for NIMBUS signals.

Tests combinations of:
  - WR period: [14, 20, 30, 50]
  - WR entry threshold: [-60, -50, -40, -30, -20]
  - BB period: [15, 20, 25]
  - BB std: [1.0, 1.5, 2.0]
  - New indicators: ADX(14), RSI(14)
  - Forward horizons: [3, 5, 10, 20]

For each parameter set:
  1. Compute indicators on full history
  2. Define entry condition
  3. Measure forward return distribution
  4. Record: N, win%, avg_ret, sharpe, profit_factor

Output: sorted table of all combinations, best parameter set per horizon.

CRITICAL: walk-forward only. No lookahead bias. Each signal state
is evaluated using data available up to that bar only.

OVERFITTING GUARD: results are split into in-sample (first 60%)
and out-of-sample (last 40%). A parameter set must work in BOTH
halves to be considered valid.
"""

from __future__ import annotations

import logging
import itertools
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# NEW INDICATORS (not in existing pipeline)
# ══════════════════════════════════════════════════════════════════════════════


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — trend strength (0-100)."""
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df[f"ADX_{period}"] = dx.rolling(period).mean()
    df[f"+DI_{period}"] = plus_di
    df[f"-DI_{period}"] = minus_di
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (0-100)."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    df = df.copy()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df[f"ATR_{period}"] = tr.rolling(period).mean()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PARAMETER SET EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_params(
    df: pd.DataFrame,
    wr_period: int = 50,
    wr_thresh: float = -20.0,
    bb_period: int = 20,
    bb_std: float = 1.0,
    use_adx: bool = False,
    adx_min: float = 20.0,
    use_rsi: bool = False,
    rsi_oversold: float = 35.0,
    entry_mode: str = "momentum",  # "momentum" | "mean_revert" | "early_momentum"
    horizons: list[int] = None,
) -> dict:
    """
    Evaluate a single parameter combination on one symbol's data.

    Entry modes:
      momentum:       WR > wr_thresh AND close > BB_Mid
      early_momentum: WR just crossed above (wr_thresh+30) with bars_since <= 3
      mean_revert:    close < BB_Mid AND RSI < rsi_oversold

    Returns dict with stats per horizon.
    """
    if horizons is None:
        horizons = [3, 5, 10, 20]

    from modules.indicators import add_bollinger, add_williams_r

    # Apply indicators
    tmp = df.copy()
    tmp = add_bollinger(tmp, period=bb_period, std_dev=bb_std)
    tmp = add_williams_r(tmp, period=wr_period)

    if use_adx:
        tmp = add_adx(tmp, 14)
    if use_rsi:
        tmp = add_rsi(tmp, 14)

    warmup = max(wr_period, bb_period) + 20
    max_horizon = max(horizons)

    if len(tmp) < warmup + max_horizon + 30:
        return {}

    closes = tmp["Close"].values
    wr_vals = tmp["WR"].values if "WR" in tmp.columns else np.full(len(tmp), np.nan)
    bb_mid = tmp["BB_Mid"].values if "BB_Mid" in tmp.columns else np.full(len(tmp), np.nan)
    bb_upper = tmp["BB_Upper"].values if "BB_Upper" in tmp.columns else np.full(len(tmp), np.nan)
    bb_width = tmp["BB_Width"].values if "BB_Width" in tmp.columns else np.full(len(tmp), np.nan)
    adx_vals = tmp.get(f"ADX_14", pd.Series(np.full(len(tmp), np.nan))).values
    rsi_vals = tmp.get(f"RSI_14", pd.Series(np.full(len(tmp), np.nan))).values

    # Precompute WR cross detection for early_momentum
    wr_cross_thresh = wr_thresh + 30  # e.g., if thresh=-20, cross at -50+30=-20... 
    # For early momentum, we want WR crossing above a LOWER level
    early_cross_level = wr_thresh - 30  # e.g., if thresh=-20, cross at -50

    entries_by_half = {"in_sample": [], "out_of_sample": []}
    mid_idx = warmup + (len(tmp) - warmup - max_horizon) * 6 // 10  # 60% split

    for t in range(warmup, len(tmp) - max_horizon):
        wr = wr_vals[t]
        close = closes[t]
        mid = bb_mid[t]
        upper = bb_upper[t]

        if np.isnan(wr) or np.isnan(mid):
            continue

        # ADX filter
        if use_adx and (np.isnan(adx_vals[t]) or adx_vals[t] < adx_min):
            continue

        # Entry condition
        triggered = False
        if entry_mode == "momentum":
            triggered = wr >= wr_thresh and close >= mid
        elif entry_mode == "early_momentum":
            # WR just crossed above early_cross_level within last 3 bars
            if wr >= early_cross_level and wr < wr_thresh:
                prev_3 = wr_vals[max(0, t-3):t]
                if len(prev_3) > 0 and np.nanmin(prev_3) < early_cross_level:
                    triggered = True
        elif entry_mode == "mean_revert":
            if use_rsi:
                triggered = close < mid and not np.isnan(rsi_vals[t]) and rsi_vals[t] < rsi_oversold
            else:
                triggered = close < mid and wr < wr_thresh - 30

        if not triggered:
            continue

        # Record forward returns
        fwd = {}
        for h in horizons:
            if t + h < len(tmp):
                fwd[h] = (closes[t + h] / close - 1) * 100

        half = "in_sample" if t < mid_idx else "out_of_sample"
        entries_by_half[half].append(fwd)

    # Compute stats per half per horizon
    result = {
        "wr_period": wr_period, "wr_thresh": wr_thresh,
        "bb_period": bb_period, "bb_std": bb_std,
        "adx_filter": use_adx, "adx_min": adx_min if use_adx else None,
        "rsi_filter": use_rsi, "entry_mode": entry_mode,
    }

    for half_name, entries in entries_by_half.items():
        n = len(entries)
        result[f"{half_name}_n"] = n
        if n < 20:
            for h in horizons:
                result[f"{half_name}_{h}d_avg"] = np.nan
                result[f"{half_name}_{h}d_win"] = np.nan
                result[f"{half_name}_{h}d_sharpe"] = np.nan
            continue

        for h in horizons:
            rets = [e[h] for e in entries if h in e]
            if not rets:
                continue
            rets = np.array(rets)
            avg = np.mean(rets)
            std = np.std(rets)
            result[f"{half_name}_{h}d_avg"] = round(avg, 4)
            result[f"{half_name}_{h}d_win"] = round((rets > 0).mean() * 100, 1)
            result[f"{half_name}_{h}d_sharpe"] = round(avg / std * 7.1, 2) if std > 0 else 0
            result[f"{half_name}_{h}d_pf"] = round(
                rets[rets > 0].sum() / abs(rets[rets <= 0].sum()), 2
            ) if (rets <= 0).any() and (rets > 0).any() else 0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FULL PARAMETER SWEEP
# ══════════════════════════════════════════════════════════════════════════════


def run_sweep(
    universe: dict[str, pd.DataFrame],
    param_grid: dict = None,
    min_bars: int = 200,
) -> pd.DataFrame:
    """
    Run parameter sweep across universe.

    Default grid tests 180 combinations per entry mode.
    Results are aggregated across all symbols.
    """
    if param_grid is None:
        param_grid = {
            "wr_period": [14, 20, 30, 50],
            "wr_thresh": [-50, -40, -30, -20],
            "bb_period": [15, 20, 25],
            "bb_std": [1.0, 1.5, 2.0],
            "entry_mode": ["momentum", "early_momentum", "mean_revert"],
            "use_adx": [False, True],
        }

    # Generate all combinations
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    logger.info("Parameter sweep: %d combinations × %d symbols", len(combos), len(universe))

    all_results = []

    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Skip nonsensical combos
        if params["entry_mode"] == "mean_revert" and params["wr_thresh"] > -30:
            continue  # mean revert needs oversold, not momentum threshold

        # Aggregate across symbols
        sym_entries = {"in_sample": [], "out_of_sample": []}

        for sym, df in universe.items():
            if len(df) < min_bars:
                continue

            result = evaluate_params(
                df,
                wr_period=params["wr_period"],
                wr_thresh=params["wr_thresh"],
                bb_period=params["bb_period"],
                bb_std=params["bb_std"],
                use_adx=params["use_adx"],
                entry_mode=params["entry_mode"],
            )

            if result:
                all_results.append(result)

        if (ci + 1) % 50 == 0:
            logger.info("Sweep progress: %d/%d combinations", ci + 1, len(combos))

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # Aggregate by parameter set (average across symbols)
    group_cols = ["wr_period", "wr_thresh", "bb_period", "bb_std", "entry_mode", "adx_filter"]
    agg_cols = [c for c in df.columns if c not in group_cols and c not in ["adx_min", "rsi_filter"]]

    agg = df.groupby(group_cols, dropna=False)[agg_cols].mean().reset_index()
    agg = agg.sort_values("out_of_sample_10d_sharpe", ascending=False, na_position="last")

    logger.info("Sweep complete: %d parameter sets evaluated", len(agg))
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# REPORT: BEST PARAMETERS PER HORIZON
# ══════════════════════════════════════════════════════════════════════════════


def sweep_report(results: pd.DataFrame) -> str:
    """Generate human-readable report from sweep results."""
    lines = []
    lines.append("NIMBUS PARAMETER SWEEP RESULTS")
    lines.append("=" * 70)

    for horizon in [3, 5, 10, 20]:
        is_col = f"in_sample_{horizon}d_sharpe"
        oos_col = f"out_of_sample_{horizon}d_sharpe"
        is_avg = f"in_sample_{horizon}d_avg"
        oos_avg = f"out_of_sample_{horizon}d_avg"
        is_win = f"in_sample_{horizon}d_win"
        oos_win = f"out_of_sample_{horizon}d_win"

        if oos_col not in results.columns:
            continue

        lines.append(f"\n{'─' * 70}")
        lines.append(f"  TOP 10 PARAMETER SETS — {horizon}-DAY FORWARD RETURN")
        lines.append(f"  (sorted by OUT-OF-SAMPLE Sharpe)")
        lines.append(f"{'─' * 70}")

        valid = results.dropna(subset=[oos_col, is_col])
        # Must have positive Sharpe in BOTH halves (overfitting guard)
        valid = valid[(valid[is_col] > 0) & (valid[oos_col] > 0)]
        top = valid.sort_values(oos_col, ascending=False).head(10)

        for _, row in top.iterrows():
            lines.append(
                f"  WR({int(row['wr_period'])}, {row['wr_thresh']:.0f}) "
                f"BB({int(row['bb_period'])}, {row['bb_std']:.1f}) "
                f"{'ADX' if row.get('adx_filter') else '   '} "
                f"{row['entry_mode']:16s}  "
                f"IS: {row.get(is_avg,0):+.3f}%/{row.get(is_win,0):.0f}%/sh={row.get(is_col,0):.2f}  "
                f"OOS: {row.get(oos_avg,0):+.3f}%/{row.get(oos_win,0):.0f}%/sh={row.get(oos_col,0):.2f}  "
                f"N_IS={row.get('in_sample_n',0):.0f} N_OOS={row.get('out_of_sample_n',0):.0f}"
            )

    # Best overall (10d OOS Sharpe, positive in both halves)
    lines.append(f"\n{'=' * 70}")
    lines.append("  RECOMMENDED PARAMETER SET")
    lines.append(f"{'=' * 70}")

    oos10 = "out_of_sample_10d_sharpe"
    is10 = "in_sample_10d_sharpe"
    if oos10 in results.columns:
        valid = results.dropna(subset=[oos10, is10])
        valid = valid[(valid[is10] > 0) & (valid[oos10] > 0)]
        if not valid.empty:
            best = valid.sort_values(oos10, ascending=False).iloc[0]
            lines.append(f"  WR period:     {int(best['wr_period'])}")
            lines.append(f"  WR threshold:  {best['wr_thresh']:.0f}")
            lines.append(f"  BB period:     {int(best['bb_period'])}")
            lines.append(f"  BB std:        {best['bb_std']:.1f}")
            lines.append(f"  ADX filter:    {best.get('adx_filter', False)}")
            lines.append(f"  Entry mode:    {best['entry_mode']}")
            lines.append(f"  10d OOS Sharpe: {best[oos10]:.2f}")

    return "\n".join(lines)
