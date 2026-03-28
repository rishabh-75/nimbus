"""
run_backtest.py — NIMBUS Historical Validation Framework
─────────────────────────────────────────────────────────

End-to-end backtest pipeline:
  1. Load data (synthetic or live)
  2. Replay signals bar-by-bar (no lookahead)
  3. Compute signal efficacy tables
  4. Calibrate weights via logistic regression
  5. Analyze signal correlations
  6. Run trade simulation
  7. Generate comprehensive report

Usage:
    python3 run_backtest.py                          # synthetic data (offline)
    python3 run_backtest.py --live --symbols NIFTY100 # real NSE data

Output: data/backtest_results/ directory with:
    - signal_efficacy.csv     (per-signal performance tables)
    - combinations.csv        (interaction effect tables)
    - calibration.json        (logistic regression weights)
    - correlation_matrix.csv  (signal correlation matrix)
    - trades.csv              (simulated trade log)
    - summary.json            (aggregate statistics)
    - scorecard.csv           (signal discriminative power)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest.data_loader import generate_universe, generate_synthetic, download_batch
from backtest.signal_replay import replay_signals, replay_universe, encode_signal_states
from backtest.efficacy import (
    full_efficacy_report,
    key_combinations,
    signal_scorecard,
    temporal_stability,
)
from backtest.weight_calibrator import (
    calibrate_weights,
    analyze_correlations,
    calibrate_by_regime,
)
from backtest.trade_simulator import (
    simulate_trades,
    simulate_universe,
    trade_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest")

_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "backtest_results"
)


def _save(obj, filename: str):
    """Save DataFrame/dict/string to output directory."""
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(_OUTPUT_DIR, filename)

    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path)
        logger.info("Saved %s (%d rows)", filename, len(obj))
    elif isinstance(obj, dict):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        logger.info("Saved %s", filename)
    elif isinstance(obj, str):
        with open(path, "w") as f:
            f.write(obj)
        logger.info("Saved %s", filename)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run(
    use_live: bool = False,
    n_synthetic: int = 20,
    symbols: list[str] = None,
    years: int = 3,
    step: int = 1,
):
    """
    Full backtest pipeline.

    Args:
        use_live: download real NSE data (requires internet + yfinance)
        n_synthetic: number of synthetic symbols if not live
        symbols: specific symbols to test (live mode)
        years: years of history
        step: signal computation interval (1=every bar, 5=weekly)
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("NIMBUS BACKTEST FRAMEWORK")
    logger.info("=" * 60)

    # ── PHASE 1: Data Loading ─────────────────────────────────────────────
    logger.info("\n── PHASE 1: Data Loading ──")

    if use_live and symbols:
        universe = download_batch(symbols, years=years)
    elif use_live:
        from modules.data import NIFTY100_SYMBOLS
        universe = download_batch(NIFTY100_SYMBOLS[:30], years=years)
    else:
        logger.info("Generating %d synthetic symbols (750 bars each)", n_synthetic)
        universe = generate_universe(n_symbols=n_synthetic, n_bars=750)

    logger.info("Universe: %d symbols loaded", len(universe))

    # ── PHASE 2: Signal Replay ────────────────────────────────────────────
    logger.info("\n── PHASE 2: Signal Replay ──")

    replay_df = replay_universe(universe, step=step)
    if replay_df.empty:
        logger.error("Signal replay produced no data — aborting")
        return

    _save(replay_df, "signal_replay.csv")
    logger.info("Signal replay: %d snapshots", len(replay_df))

    # ── PHASE 3: Encode signals ───────────────────────────────────────────
    logger.info("\n── PHASE 3: Encoding Signals ──")

    encoded = encode_signal_states(replay_df)
    _save(encoded, "encoded_signals.csv")

    # ── PHASE 4: Signal Efficacy Tables ───────────────────────────────────
    logger.info("\n── PHASE 4: Signal Efficacy ──")

    for horizon in ["fwd_1d", "fwd_5d", "fwd_10d"]:
        tables = full_efficacy_report(replay_df, return_col=horizon)
        # Save each table
        for signal_name, tbl in tables.items():
            safe_name = signal_name.replace(" ", "_").replace("%", "pct").lower()
            _save(tbl, f"efficacy_{safe_name}_{horizon}.csv")

        # Print summary for 5d horizon
        if horizon == "fwd_5d":
            logger.info("\n  ═══ SIGNAL EFFICACY (5-day forward returns) ═══")
            for signal_name, tbl in tables.items():
                logger.info("\n  %s:", signal_name)
                for idx in range(len(tbl)):
                    state = tbl.index[idx]
                    row = tbl.iloc[idx]
                    state_str = str(state)
                    if row.get("insufficient"):
                        logger.info("    %-20s  N=%-5d  (insufficient data)", state_str, row["n"])
                    else:
                        logger.info(
                            "    %-20s  N=%-5d  Win=%-5.1f%%  Avg=%+.3f%%  Sharpe=%-5.2f  MaxDD=%-6.2f%%  PF=%.2f",
                            state_str, row["n"], row["win_pct"], row["avg_ret"],
                            row["sharpe"], row["max_dd"], row["profit_factor"],
                        )

    # Scorecard
    tables_5d = full_efficacy_report(replay_df, return_col="fwd_5d")
    scorecard = signal_scorecard(tables_5d)
    _save(scorecard, "signal_scorecard.csv")
    logger.info("\n  ═══ SIGNAL SCORECARD ═══")
    for _, row in scorecard.iterrows():
        disc = "✓" if row["discriminative"] else "✗"
        logger.info(
            "  %s %-25s  spread=%.3f%%  best=%s(%.3f%%)  worst=%s(%.3f%%)  N=%d",
            disc, row["signal"], row["spread_pct"],
            row["best_state"], row["best_avg_ret"],
            row["worst_state"], row["worst_avg_ret"],
            row["total_n"],
        )

    # ── PHASE 5: Combination Effects ──────────────────────────────────────
    logger.info("\n── PHASE 5: Combination Effects ──")

    combos = key_combinations(replay_df)
    for label, tbl in combos.items():
        safe_name = label.replace(" ", "_").replace("×", "x").replace("%", "pct").lower()
        _save(tbl, f"combo_{safe_name}.csv")
        # Show top 5 best combinations
        top = tbl[tbl.index.astype(str) != "ALL (baseline)"].head(5)
        logger.info("  %s — top combos:", label)
        for idx in range(len(top)):
            state = str(top.index[idx])
            r = top.iloc[idx]
            if not r.get("insufficient"):
                logger.info("    %-35s  N=%d  Win=%.1f%%  Avg=%+.3f%%", state, r["n"], r["win_pct"], r["avg_ret"])

    # ── PHASE 6: Weight Calibration ───────────────────────────────────────
    logger.info("\n── PHASE 6: Weight Calibration ──")

    cal_result = calibrate_by_regime(replay_df)
    cal_path = os.path.join(_OUTPUT_DIR, "calibration.json")
    cal_result.to_json(cal_path)

    logger.info("  Logistic Regression: accuracy=%.3f  AUC=%.3f", cal_result.accuracy, cal_result.auc_roc)
    logger.info("  Calibrated weights (0-100 scale):")
    for feat, w in sorted(cal_result.weights.items(), key=lambda x: abs(x[1]), reverse=True):
        curr = cal_result.current_weights.get(feat, 0)
        delta = cal_result.weight_changes.get(feat, 0)
        logger.info("    %-22s  new=%+7.2f  current=%+7.2f  Δ=%+.2f", feat, w, curr, delta)

    if cal_result.regime_weights:
        for regime, weights in cal_result.regime_weights.items():
            logger.info("  %s regime weights:", regime)
            for feat, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                logger.info("    %-22s  %+.2f", feat, w)

    # ── PHASE 7: Correlation Analysis ─────────────────────────────────────
    logger.info("\n── PHASE 7: Correlation Analysis ──")

    corr_result = analyze_correlations(replay_df)
    if corr_result.correlation_matrix is not None:
        _save(corr_result.correlation_matrix, "correlation_matrix.csv")

    if corr_result.correlated_pairs:
        logger.info("  Correlated pairs (>0.6):")
        for fa, fb, r in corr_result.correlated_pairs:
            disc = corr_result.correlation_discounts.get(fa, 1.0)
            logger.info("    %s ↔ %s: r=%.3f → discount=%.2f", fa, fb, r, disc)
    else:
        logger.info("  No highly correlated pairs found")

    # ── PHASE 8: Trade Simulation ─────────────────────────────────────────
    logger.info("\n── PHASE 8: Trade Simulation ──")

    all_trades = simulate_universe(universe, min_bars=200)
    if all_trades:
        trades_df = pd.DataFrame([t.to_dict() for t in all_trades])
        _save(trades_df, "trades.csv")

        summary = trade_summary(all_trades)
        _save(summary, "trade_summary.json")

        logger.info("  ═══ TRADE SIMULATION RESULTS ═══")
        logger.info("  Total trades:    %d", summary["n_trades"])
        logger.info("  Win rate:        %.1f%%", summary["win_rate"])
        logger.info("  Avg P&L:         %+.3f%%", summary["avg_pnl"])
        logger.info("  Profit factor:   %.2f", summary["profit_factor"])
        logger.info("  Sharpe:          %.2f", summary["sharpe"])
        logger.info("  Max drawdown:    %.2f%%", summary["max_drawdown"])
        logger.info("  Avg hold:        %.1f bars", summary["avg_hold_bars"])
        logger.info("  Final equity:    $%.2f (from $100)", summary["final_equity"])

        logger.info("  Exit reasons:")
        for reason, count in sorted(summary["exit_reasons"].items(), key=lambda x: -x[1]):
            logger.info("    %-25s  %d", reason, count)

        logger.info("  By W%%R phase:")
        for phase, stats in summary["by_wr_phase"].items():
            logger.info("    %-12s  N=%d  Win=%.1f%%  Avg=%+.3f%%", phase, stats["n"], stats["win_pct"], stats["avg_pnl"])

        logger.info("  By sizing:")
        for sz, stats in summary["by_sizing"].items():
            logger.info("    %-6s  N=%d  Win=%.1f%%  Avg=%+.3f%%", sz, stats["n"], stats["win_pct"], stats["avg_pnl"])
    else:
        logger.info("  No trades generated (entry conditions may be too strict)")

    # ── PHASE 9: Temporal Stability ───────────────────────────────────────
    logger.info("\n── PHASE 9: Temporal Stability ──")

    # Check if key signals are stable across time periods
    # ── Collect stability tables ─────────────────────────────────────────
    stability_tables_all = {}
    for signal_col, state in [
        ("wr_phase", "FRESH"),
        ("position_state", "RIDING_UPPER"),
        ("mfi_state", "STRONG"),
    ]:
        stab = temporal_stability(replay_df, signal_col, state)
        if not stab.empty:
            stability_tables_all[f"{signal_col}={state}"] = stab
            _save(stab, f"stability_{signal_col}_{state}.csv")
            logger.info("  %s=%s across periods:", signal_col, state)
            for period in stab.index:
                r = stab.loc[period]
                logger.info(
                    "    %s (%s to %s): N=%d  Win=%.1f%%  Avg=%+.3f%%  Sharpe=%.2f",
                    period, r.get("start", "?"), r.get("end", "?"),
                    r["n"], r["win_pct"], r["avg_ret"], r["sharpe"],
                )

    # ── PHASE 10: Consolidated Report ─────────────────────────────────────
    logger.info("\n── PHASE 10: Generating Consolidated Report ──")

    from backtest.report import generate_report

    report_path = generate_report(
        output_dir=_OUTPUT_DIR,
        replay_df=replay_df,
        efficacy_tables=tables_5d,
        combination_tables=combos,
        calibration_result=cal_result,
        correlation_result=corr_result,
        trade_summary=summary if all_trades else {},
        trades=all_trades,
        scorecard=scorecard,
        stability_tables=stability_tables_all,
        run_config={
            "live": use_live,
            "n_symbols": len(universe),
            "step": step,
            "years": years,
        },
    )

    # ── DONE ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST COMPLETE in %.1f seconds", elapsed)
    logger.info("Results saved to: %s", _OUTPUT_DIR)
    logger.info("")
    logger.info("TO VALIDATE: Upload this file to Claude:")
    logger.info("  %s", report_path)
    logger.info("")
    logger.info("Or upload the full results zip:")
    logger.info("  cd %s && zip -r backtest_results.zip .", _OUTPUT_DIR)
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIMBUS Backtest Framework")
    parser.add_argument("--live", action="store_true", help="Use live NSE data (requires internet)")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backtest")
    parser.add_argument("--n-synthetic", type=int, default=20, help="Number of synthetic symbols")
    parser.add_argument("--years", type=int, default=3, help="Years of history")
    parser.add_argument("--step", type=int, default=1, help="Signal computation interval")
    args = parser.parse_args()

    run(
        use_live=args.live,
        n_synthetic=args.n_synthetic,
        symbols=args.symbols,
        years=args.years,
        step=args.step,
    )
