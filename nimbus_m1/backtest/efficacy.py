"""
backtest/efficacy.py
─────────────────────
Signal efficacy analysis — produces the validation tables that tell you
whether each signal component actually predicts forward returns.

Output for each signal:
  | State         | N    | Win% 5d | Avg Ret | Sharpe | MaxDD | Profit Factor |
  | FRESH         | 342  | 64.3%   | +1.8%   | 1.42   | -4.2% | 2.1           |
  | DEVELOPING    | 518  | 51.2%   | +0.3%   | 0.31   | -6.8% | 1.1           |

This is the table a quant desk would demand before deploying any signal.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum sample size for a statistically meaningful row
MIN_SAMPLES = 30


# ══════════════════════════════════════════════════════════════════════════════
# CORE EFFICACY COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def _compute_stats(
    returns: pd.Series,
    horizon_label: str = "5d",
) -> dict:
    """Compute performance statistics for a group of forward returns."""
    n = len(returns)
    if n < MIN_SAMPLES:
        return {
            "n": n, "win_pct": np.nan, "avg_ret": np.nan,
            "med_ret": np.nan, "std_ret": np.nan, "sharpe": np.nan,
            "max_dd": np.nan, "max_up": np.nan, "profit_factor": np.nan,
            "skew": np.nan, "insufficient": True,
        }

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    avg = float(returns.mean())
    std = float(returns.std())
    sharpe = avg / std * np.sqrt(252 / 5) if std > 0 else 0.0  # annualized

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.001

    return {
        "n": n,
        "win_pct": round(len(wins) / n * 100, 1),
        "avg_ret": round(avg, 3),
        "med_ret": round(float(returns.median()), 3),
        "std_ret": round(std, 3),
        "sharpe": round(sharpe, 2),
        "max_dd": round(float(returns.min()), 2),
        "max_up": round(float(returns.max()), 2),
        "profit_factor": round(gross_profit / gross_loss, 2),
        "skew": round(float(returns.skew()), 2) if n >= 8 else np.nan,
        "insufficient": False,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPONENT TABLES
# ══════════════════════════════════════════════════════════════════════════════


def efficacy_by_column(
    replay_df: pd.DataFrame,
    signal_col: str,
    return_col: str = "fwd_5d",
    min_samples: int = MIN_SAMPLES,
) -> pd.DataFrame:
    """
    Compute efficacy table for a single signal column.

    Args:
        replay_df: output from signal_replay.replay_signals()
        signal_col: column name to group by (e.g. "wr_phase")
        return_col: forward return column (e.g. "fwd_5d")

    Returns:
        DataFrame with one row per signal state, columns = stats.
    """
    if return_col not in replay_df.columns or signal_col not in replay_df.columns:
        return pd.DataFrame()

    valid = replay_df[[signal_col, return_col]].dropna()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for state, grp in valid.groupby(signal_col):
        stats = _compute_stats(grp[return_col])
        stats["state"] = state
        rows.append(stats)

    # Add ALL baseline
    stats = _compute_stats(valid[return_col])
    stats["state"] = "ALL (baseline)"
    rows.append(stats)

    result = pd.DataFrame(rows)
    result = result.set_index("state")
    result = result.sort_values("avg_ret", ascending=False)
    return result


def full_efficacy_report(
    replay_df: pd.DataFrame,
    return_col: str = "fwd_5d",
) -> dict[str, pd.DataFrame]:
    """
    Compute efficacy tables for ALL signal components.

    Returns dict: {signal_name: efficacy_table}
    """
    signal_columns = [
        ("wr_phase", "W%R Phase"),
        ("position_state", "BB Position State"),
        ("vol_state", "Volatility State"),
        ("daily_bias", "Daily Bias"),
        ("mfi_state", "MFI State"),
        ("bb_position", "BB Position"),
        ("wr_in_momentum", "W%R In Momentum"),
        ("entry_valid", "Entry Valid"),
        ("momentum_pass", "Momentum Pass"),
        ("full_entry", "Full Entry (WR+BB+Bias)"),
        ("mfi_diverge", "MFI Divergence"),
    ]

    tables = {}
    for col, label in signal_columns:
        if col in replay_df.columns:
            tbl = efficacy_by_column(replay_df, col, return_col)
            if not tbl.empty:
                tables[label] = tbl
                logger.info("Efficacy: %s — %d states", label, len(tbl) - 1)

    return tables


# ══════════════════════════════════════════════════════════════════════════════
# COMBINATION EFFICACY (INTERACTION EFFECTS)
# ══════════════════════════════════════════════════════════════════════════════


def combination_efficacy(
    replay_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    return_col: str = "fwd_5d",
) -> pd.DataFrame:
    """
    Compute efficacy for combinations of two signals.
    Shows interaction effects (e.g., FRESH WR + SQUEEZE vol).
    """
    valid = replay_df[[col_a, col_b, return_col]].dropna()
    if valid.empty:
        return pd.DataFrame()

    valid["combo"] = valid[col_a].astype(str) + " + " + valid[col_b].astype(str)
    return efficacy_by_column(valid, "combo", return_col)


def key_combinations(
    replay_df: pd.DataFrame,
    return_col: str = "fwd_5d",
) -> dict[str, pd.DataFrame]:
    """
    Compute efficacy for the most important signal combinations.

    These are the combinations most likely to show interaction effects
    that the additive scoring model misses.
    """
    combos = [
        ("wr_phase", "position_state", "W%R Phase × BB State"),
        ("wr_phase", "vol_state", "W%R Phase × Vol State"),
        ("daily_bias", "mfi_state", "Daily Bias × MFI State"),
        ("position_state", "mfi_diverge", "BB State × MFI Divergence"),
        ("wr_in_momentum", "daily_bias", "WR Momentum × Daily Bias"),
    ]
    tables = {}
    for col_a, col_b, label in combos:
        if col_a in replay_df.columns and col_b in replay_df.columns:
            tbl = combination_efficacy(replay_df, col_a, col_b, return_col)
            if not tbl.empty:
                tables[label] = tbl
    return tables


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL STABILITY
# ══════════════════════════════════════════════════════════════════════════════


def temporal_stability(
    replay_df: pd.DataFrame,
    signal_col: str,
    signal_state: str,
    return_col: str = "fwd_5d",
    n_splits: int = 4,
) -> pd.DataFrame:
    """
    Check if a signal's efficacy is stable across time periods.

    Splits data into N equal time periods and computes stats for each.
    A signal that only works in one period is likely overfit.
    """
    valid = replay_df[replay_df[signal_col] == signal_state].copy()
    if len(valid) < MIN_SAMPLES * n_splits:
        return pd.DataFrame()

    valid = valid.sort_values("date").reset_index(drop=True)
    chunk_size = len(valid) // n_splits
    if chunk_size < MIN_SAMPLES:
        return pd.DataFrame()

    rows = []
    for i in range(n_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_splits - 1 else len(valid)
        split = valid.iloc[start:end]
        stats = _compute_stats(split[return_col])
        stats["period"] = f"Q{i + 1}"
        stats["start"] = split["date"].iloc[0].strftime("%Y-%m") if len(split) > 0 else ""
        stats["end"] = split["date"].iloc[-1].strftime("%Y-%m") if len(split) > 0 else ""
        rows.append(stats)

    return pd.DataFrame(rows).set_index("period")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY SCORECARD
# ══════════════════════════════════════════════════════════════════════════════


def signal_scorecard(
    efficacy_tables: dict[str, pd.DataFrame],
    return_col: str = "fwd_5d",
) -> pd.DataFrame:
    """
    Generate a single summary row per signal component showing:
    - Best state, worst state
    - Spread (best - worst avg return)
    - Is the signal discriminative?

    A signal with spread < 0.5% is not worth including in the scoring model.
    """
    rows = []
    for signal_name, tbl in efficacy_tables.items():
        tbl_states = tbl[tbl.index != "ALL (baseline)"]
        if tbl_states.empty:
            continue

        baseline = tbl.loc["ALL (baseline)"] if "ALL (baseline)" in tbl.index else None

        best_state = tbl_states["avg_ret"].idxmax()
        worst_state = tbl_states["avg_ret"].idxmin()
        spread = tbl_states["avg_ret"].max() - tbl_states["avg_ret"].min()
        best_sharpe = tbl_states["sharpe"].max()
        worst_sharpe = tbl_states["sharpe"].min()

        rows.append({
            "signal": signal_name,
            "best_state": best_state,
            "best_avg_ret": tbl_states.loc[best_state, "avg_ret"],
            "best_sharpe": tbl_states.loc[best_state, "sharpe"],
            "worst_state": worst_state,
            "worst_avg_ret": tbl_states.loc[worst_state, "avg_ret"],
            "spread_pct": round(spread, 3),
            "discriminative": spread >= 0.5,
            "n_states": len(tbl_states),
            "total_n": int(tbl_states["n"].sum()),
        })

    return pd.DataFrame(rows).sort_values("spread_pct", ascending=False)
