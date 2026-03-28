"""
backtest/report.py
───────────────────
Generate a consolidated backtest report for analysis.

Produces two outputs:
  1. backtest_report.txt  — human-readable, uploadable to Claude for validation
  2. backtest_report.json — machine-readable, all numbers preserved

The .txt file is designed to be uploaded as a single file with
full context — no need to send 50 separate CSVs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    output_dir: str,
    replay_df: pd.DataFrame,
    efficacy_tables: dict[str, pd.DataFrame],
    combination_tables: dict[str, pd.DataFrame],
    calibration_result,
    correlation_result,
    trade_summary: dict,
    trades: list,
    scorecard: pd.DataFrame,
    stability_tables: dict[str, pd.DataFrame],
    run_config: dict,
) -> str:
    """
    Generate consolidated report files.
    Returns path to the .txt report.
    """
    lines = []
    json_data = {}

    def section(title):
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"  {title}")
        lines.append("=" * 72)

    def subsection(title):
        lines.append("")
        lines.append(f"── {title} ──")

    # ── HEADER ────────────────────────────────────────────────────────────
    lines.append("NIMBUS BACKTEST REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Data mode: {'LIVE NSE' if run_config.get('live') else 'SYNTHETIC'}")
    lines.append(f"Symbols: {run_config.get('n_symbols', '?')}")
    lines.append(f"Total signal snapshots: {len(replay_df)}")
    lines.append(f"Date range: {replay_df['date'].min()} to {replay_df['date'].max()}")
    lines.append(f"Step interval: every {run_config.get('step', 1)} bar(s)")

    json_data["meta"] = {
        "generated": datetime.now().isoformat(),
        "mode": "LIVE" if run_config.get("live") else "SYNTHETIC",
        "n_symbols": run_config.get("n_symbols"),
        "n_snapshots": len(replay_df),
        "date_range": [str(replay_df["date"].min()), str(replay_df["date"].max())],
    }

    # ── SIGNAL DISTRIBUTION ───────────────────────────────────────────────
    section("1. SIGNAL STATE DISTRIBUTIONS")
    dist = {}
    for col in ["wr_phase", "position_state", "vol_state", "daily_bias",
                 "mfi_state", "bb_position", "mfi_diverge"]:
        if col in replay_df.columns:
            counts = replay_df[col].value_counts()
            lines.append(f"\n  {col}:")
            for state, n in counts.items():
                pct = n / len(replay_df) * 100
                lines.append(f"    {str(state):20s}  {n:6d}  ({pct:5.1f}%)")
            dist[col] = {str(k): int(v) for k, v in counts.items()}
    json_data["distributions"] = dist

    # ── SIGNAL SCORECARD ──────────────────────────────────────────────────
    section("2. SIGNAL SCORECARD (discriminative power)")
    lines.append("")
    lines.append("  Signals ranked by spread (best state avg return - worst state avg return).")
    lines.append("  Spread >= 0.5% = discriminative (✓). Below = noise (✗).")
    lines.append("")
    lines.append(f"  {'Signal':28s} {'Spread':>8s}  {'Best State':>20s} {'Avg':>8s}  {'Worst State':>20s} {'Avg':>8s}  {'N':>6s}")
    lines.append("  " + "-" * 110)

    sc_data = []
    for _, row in scorecard.iterrows():
        disc = "✓" if row["discriminative"] else "✗"
        lines.append(
            f"  {disc} {row['signal']:26s} {row['spread_pct']:+7.3f}%  "
            f"{str(row['best_state']):>20s} {row['best_avg_ret']:+7.3f}%  "
            f"{str(row['worst_state']):>20s} {row['worst_avg_ret']:+7.3f}%  "
            f"{row['total_n']:6d}"
        )
        sc_data.append({
            "signal": row["signal"],
            "spread": row["spread_pct"],
            "discriminative": bool(row["discriminative"]),
            "best_state": str(row["best_state"]),
            "best_avg": row["best_avg_ret"],
            "worst_state": str(row["worst_state"]),
            "worst_avg": row["worst_avg_ret"],
        })
    json_data["scorecard"] = sc_data

    # ── EFFICACY TABLES (5-day) ───────────────────────────────────────────
    section("3. SIGNAL EFFICACY TABLES (5-day forward returns)")
    eff_data = {}
    for signal_name, tbl in efficacy_tables.items():
        subsection(signal_name)
        lines.append(f"  {'State':20s} {'N':>6s} {'Win%':>7s} {'AvgRet':>8s} {'MedRet':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'PF':>6s} {'Skew':>6s}")
        lines.append("  " + "-" * 90)

        tbl_data = {}
        for idx in range(len(tbl)):
            state = str(tbl.index[idx])
            r = tbl.iloc[idx]
            if r.get("insufficient"):
                lines.append(f"  {state:20s} {r['n']:6d}  (insufficient data)")
            else:
                lines.append(
                    f"  {state:20s} {r['n']:6d} {r['win_pct']:6.1f}% "
                    f"{r['avg_ret']:+7.3f}% {r['med_ret']:+7.3f}% "
                    f"{r['sharpe']:6.2f} {r['max_dd']:+7.2f}% {r['profit_factor']:5.2f} "
                    f"{r['skew']:5.2f}"
                )
            tbl_data[state] = {k: _safe(v) for k, v in r.to_dict().items()}
        eff_data[signal_name] = tbl_data
    json_data["efficacy_5d"] = eff_data

    # ── COMBINATION EFFECTS ───────────────────────────────────────────────
    section("4. SIGNAL COMBINATION EFFECTS (top 5 per pair)")
    combo_data = {}
    for label, tbl in combination_tables.items():
        subsection(label)
        top = tbl[tbl.index.astype(str) != "ALL (baseline)"].head(8)
        lines.append(f"  {'Combination':40s} {'N':>6s} {'Win%':>7s} {'AvgRet':>8s} {'Sharpe':>7s}")
        lines.append("  " + "-" * 70)
        entries = []
        for idx in range(len(top)):
            state = str(top.index[idx])
            r = top.iloc[idx]
            if not r.get("insufficient"):
                lines.append(
                    f"  {state:40s} {r['n']:6d} {r['win_pct']:6.1f}% "
                    f"{r['avg_ret']:+7.3f}% {r['sharpe']:6.2f}"
                )
                entries.append({"combo": state, "n": int(r["n"]),
                                "win_pct": r["win_pct"], "avg_ret": r["avg_ret"]})
        combo_data[label] = entries
    json_data["combinations"] = combo_data

    # ── WEIGHT CALIBRATION ────────────────────────────────────────────────
    section("5. WEIGHT CALIBRATION (logistic regression)")
    cal = calibration_result
    lines.append(f"  Model accuracy: {cal.accuracy:.4f}")
    lines.append(f"  AUC-ROC:        {cal.auc_roc:.4f}")
    lines.append(f"  N samples:      {cal.n_samples}")
    lines.append("")
    lines.append(f"  {'Feature':24s} {'New Wt':>8s} {'Current':>8s} {'Delta':>8s}")
    lines.append("  " + "-" * 55)
    for feat in sorted(cal.weights.keys(), key=lambda f: abs(cal.weights.get(f, 0)), reverse=True):
        new = cal.weights.get(feat, 0)
        curr = cal.current_weights.get(feat, 0)
        delta = cal.weight_changes.get(feat, 0)
        lines.append(f"  {feat:24s} {new:+7.2f} {curr:+7.2f} {delta:+7.2f}")

    json_data["calibration"] = {
        "accuracy": cal.accuracy,
        "auc_roc": cal.auc_roc,
        "n_samples": cal.n_samples,
        "weights": cal.weights,
        "current_weights": cal.current_weights,
        "weight_changes": cal.weight_changes,
        "regime_weights": cal.regime_weights,
    }

    # ── CORRELATION ANALYSIS ──────────────────────────────────────────────
    section("6. SIGNAL CORRELATION ANALYSIS")
    corr = correlation_result
    if corr.correlated_pairs:
        lines.append("  Pairs with |r| >= 0.6 (scoring these independently double-counts):")
        lines.append("")
        lines.append(f"  {'Signal A':22s} {'Signal B':22s} {'r':>6s}  {'Discount':>8s}")
        lines.append("  " + "-" * 65)
        for fa, fb, r in corr.correlated_pairs:
            disc_a = corr.correlation_discounts.get(fa, 1.0)
            lines.append(f"  {fa:22s} {fb:22s} {r:5.3f}   {disc_a:.3f}")
    else:
        lines.append("  No highly correlated pairs found.")

    json_data["correlations"] = {
        "pairs": [{"a": a, "b": b, "r": r} for a, b, r in corr.correlated_pairs],
        "discounts": corr.correlation_discounts,
    }

    # ── REGIME-SPECIFIC WEIGHTS ───────────────────────────────────────────
    if cal.regime_weights:
        section("7. REGIME-DEPENDENT WEIGHTS")
        for regime, weights in cal.regime_weights.items():
            subsection(f"{regime} regime")
            lines.append(f"  {'Feature':24s} {'Weight':>8s}")
            lines.append("  " + "-" * 35)
            for feat in sorted(weights.keys(), key=lambda f: abs(weights[f]), reverse=True):
                lines.append(f"  {feat:24s} {weights[feat]:+7.2f}")

    # ── TRADE SIMULATION ──────────────────────────────────────────────────
    section("8. TRADE SIMULATION RESULTS")
    ts = trade_summary
    if ts.get("n_trades", 0) > 0:
        lines.append(f"  Total trades:     {ts['n_trades']}")
        lines.append(f"  Win rate:         {ts['win_rate']:.1f}%")
        lines.append(f"  Avg P&L:          {ts['avg_pnl']:+.3f}%")
        lines.append(f"  Median P&L:       {ts['med_pnl']:+.3f}%")
        lines.append(f"  Total P&L:        {ts['total_pnl']:+.2f}%")
        lines.append(f"  Profit factor:    {ts['profit_factor']:.2f}")
        lines.append(f"  Sharpe:           {ts['sharpe']:.2f}")
        lines.append(f"  Max drawdown:     {ts['max_drawdown']:.2f}%")
        lines.append(f"  Avg hold (bars):  {ts['avg_hold_bars']:.1f}")
        lines.append(f"  Final equity:     ${ts['final_equity']:.2f} (from $100)")

        subsection("Exit reasons")
        for reason, count in sorted(ts["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = count / ts["n_trades"] * 100
            lines.append(f"  {reason:30s}  {count:4d}  ({pct:.1f}%)")

        subsection("Performance by W%R phase at entry")
        for phase, stats in ts.get("by_wr_phase", {}).items():
            lines.append(f"  {phase:14s}  N={stats['n']:4d}  Win={stats['win_pct']:5.1f}%  Avg={stats['avg_pnl']:+.3f}%")

        subsection("Performance by sizing")
        for sz, stats in ts.get("by_sizing", {}).items():
            lines.append(f"  {sz:8s}  N={stats['n']:4d}  Win={stats['win_pct']:5.1f}%  Avg={stats['avg_pnl']:+.3f}%")

        subsection("Performance by vol state at entry")
        for vs, stats in ts.get("by_vol_state", {}).items():
            lines.append(f"  {vs:12s}  N={stats['n']:4d}  Win={stats['win_pct']:5.1f}%  Avg={stats['avg_pnl']:+.3f}%")

    json_data["trade_simulation"] = ts

    # ── TEMPORAL STABILITY ────────────────────────────────────────────────
    if stability_tables:
        section("9. TEMPORAL STABILITY (is the edge consistent?)")
        for label, tbl in stability_tables.items():
            subsection(label)
            lines.append(f"  {'Period':6s} {'Range':>22s} {'N':>5s} {'Win%':>6s} {'AvgRet':>8s} {'Sharpe':>7s}")
            lines.append("  " + "-" * 60)
            for idx in range(len(tbl)):
                r = tbl.iloc[idx]
                period = tbl.index[idx]
                lines.append(
                    f"  {period:6s} {r.get('start',''):>10s}–{r.get('end',''):>10s} "
                    f"{r['n']:5d} {r['win_pct']:5.1f}% {r['avg_ret']:+7.3f}% {r['sharpe']:6.2f}"
                )

    # ── ACTIONABLE SUMMARY ────────────────────────────────────────────────
    section("10. ACTIONABLE SUMMARY FOR PIPELINE CHANGES")
    lines.append("")

    # Identify discriminative signals
    disc_signals = scorecard[scorecard["discriminative"] == True]["signal"].tolist()
    noise_signals = scorecard[scorecard["discriminative"] == False]["signal"].tolist()

    lines.append("  DISCRIMINATIVE signals (keep, validate weights):")
    for s in disc_signals:
        lines.append(f"    ✓ {s}")
    if not disc_signals:
        lines.append("    (none found — all signals have spread < 0.5%)")

    lines.append("")
    lines.append("  NON-DISCRIMINATIVE signals (consider removing from scoring):")
    for s in noise_signals:
        lines.append(f"    ✗ {s}")

    lines.append("")
    lines.append("  CORRELATED pairs (apply discount to avoid double-counting):")
    for fa, fb, r in corr.correlated_pairs:
        lines.append(f"    {fa} ↔ {fb}: r={r:.3f}")

    lines.append("")
    lines.append("  WEIGHT CHANGES (largest delta from current):")
    for feat in sorted(cal.weight_changes.keys(),
                       key=lambda f: abs(cal.weight_changes.get(f, 0)), reverse=True)[:5]:
        delta = cal.weight_changes[feat]
        lines.append(f"    {feat}: {delta:+.2f}")

    lines.append("")
    lines.append("── END OF REPORT ──")

    json_data["actionable"] = {
        "discriminative_signals": disc_signals,
        "noise_signals": noise_signals,
        "top_weight_changes": {
            f: cal.weight_changes[f]
            for f in sorted(cal.weight_changes.keys(),
                            key=lambda f: abs(cal.weight_changes.get(f, 0)), reverse=True)[:5]
        },
    }

    # ── WRITE FILES ───────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "backtest_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    json_path = os.path.join(output_dir, "backtest_report.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=_safe)

    logger.info("Report written: %s (%d lines)", txt_path, len(lines))
    logger.info("Report JSON: %s", json_path)
    return txt_path


def _safe(v):
    """Make a value JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if not np.isnan(v) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v
