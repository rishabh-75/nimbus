#!/usr/bin/env python3
"""
run_param_sweep.py — Optimise NIMBUS signal parameters
────────────────────────────────────────────────────────

Uses the signal_replay.csv from a previous backtest run,
OR downloads fresh data for specified symbols.

Usage:
  python3 run_param_sweep.py                    # use existing replay data
  python3 run_param_sweep.py --fresh --n 10     # fresh synthetic data

Output: data/backtest_results/param_sweep_results.csv
        data/backtest_results/param_sweep_report.txt
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sweep")

_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "backtest_results"
)


def main():
    parser = argparse.ArgumentParser(description="NIMBUS Parameter Sweep")
    parser.add_argument("--fresh", action="store_true", help="Generate fresh data")
    parser.add_argument("--n", type=int, default=10, help="Synthetic symbols")
    parser.add_argument("--live", action="store_true", help="Use live NSE data")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols")
    args = parser.parse_args()

    from backtest.param_sweep import run_sweep, sweep_report

    t0 = time.time()

    # Load or generate data
    if args.fresh or args.live:
        if args.live and args.symbols:
            from backtest.data_loader import download_batch
            universe = download_batch(args.symbols, years=3)
        elif args.live:
            from backtest.data_loader import download_batch
            from modules.data import NIFTY100_SYMBOLS
            universe = download_batch(NIFTY100_SYMBOLS[:30], years=3)
        else:
            from backtest.data_loader import generate_universe
            universe = generate_universe(n_symbols=args.n, n_bars=750)
    else:
        # Try to load cached parquet files from previous backtest
        cache_dir = os.path.join(os.path.dirname(__file__), "data", "backtest_cache")
        universe = {}
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith("_1d.parquet"):
                    sym = f.replace("_1d.parquet", "")
                    try:
                        df = pd.read_parquet(os.path.join(cache_dir, f))
                        if len(df) >= 200:
                            universe[sym] = df
                    except Exception:
                        pass

        if not universe:
            logger.info("No cached data found, generating synthetic")
            from backtest.data_loader import generate_universe
            universe = generate_universe(n_symbols=args.n, n_bars=750)

    logger.info("Universe: %d symbols", len(universe))

    # Define parameter grid
    param_grid = {
        "wr_period": [14, 20, 30, 50],
        "wr_thresh": [-50, -40, -30, -20],
        "bb_period": [15, 20, 25],
        "bb_std": [1.0, 1.5, 2.0],
        "entry_mode": ["momentum", "early_momentum", "mean_revert"],
        "use_adx": [False, True],
    }

    # Run sweep
    results = run_sweep(universe, param_grid)

    if results.empty:
        logger.error("Sweep produced no results")
        return

    # Save results
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    results.to_csv(os.path.join(_OUTPUT_DIR, "param_sweep_results.csv"), index=False)
    logger.info("Saved %d parameter sets", len(results))

    # Generate and save report
    report = sweep_report(results)
    report_path = os.path.join(_OUTPUT_DIR, "param_sweep_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report: %s", report_path)

    # Print report
    print()
    print(report)

    elapsed = time.time() - t0
    logger.info("\nSweep complete in %.1f seconds", elapsed)
    logger.info("Upload param_sweep_report.txt for analysis")


if __name__ == "__main__":
    main()
