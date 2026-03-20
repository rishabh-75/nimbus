#!/usr/bin/env python3
"""
run_resolve.py — Resolve outcomes for tracked NIMBUS signals.

Run daily (after market close) to:
  1. Fetch close prices for signals that are ≥10 trading days old
  2. Mark them as resolved with P&L
  3. Check for regime drift
  4. Optionally export a report for analysis

Usage:
    python3 run_resolve.py             # resolve + drift check
    python3 run_resolve.py --export    # also export report to file
    python3 run_resolve.py --status    # just show current performance
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("resolve")


def resolve_mature_signals():
    """Fetch outcomes for signals using new exit rules: +5% PT, BBW contraction, or max 25d hold."""
    from modules.signal_tracker import get_unresolved, resolve_signal
    from modules.dual_mode import PROFIT_TARGET, MAX_HOLD

    unresolved = get_unresolved()
    if not unresolved:
        logger.info("No unresolved signals")
        return 0

    # Signals need at least 5 trading days to evaluate exit conditions
    min_days_cutoff = date.today() - timedelta(days=8)
    # Signals at max hold (30 trading days ≈ 42 calendar days)
    max_hold_cutoff = date.today() - timedelta(days=MAX_HOLD + 12)

    candidates = [
        s for s in unresolved
        if s["signal_date"] <= min_days_cutoff.isoformat()
    ]

    if not candidates:
        logger.info(
            "%d signals pending, none old enough yet (cutoff: %s)",
            len(unresolved), min_days_cutoff,
        )
        return 0

    logger.info("Evaluating %d signals for exit", len(candidates))

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance required for resolution")
        return 0

    resolved = 0
    for sig in candidates:
        symbol = sig["symbol"]
        signal_date = sig["signal_date"]
        entry_price = sig["entry_price"]

        try:
            ticker = f"{symbol}.NS" if not symbol.startswith("^") else symbol
            df = yf.download(
                tickers=ticker, start=signal_date, interval="1d",
                auto_adjust=True, progress=False,
            )
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)

            if df.empty or "Close" not in df.columns:
                logger.warning("No price data for %s from %s", symbol, signal_date)
                continue

            closes = df["Close"].dropna()
            if len(closes) < 5:
                continue

            # Compute BBW slope and SMA for exit detection
            sma_20 = df["Close"].rolling(20, min_periods=10).mean()
            bb_ma = df["Close"].rolling(20, min_periods=10).mean()
            bb_sd = df["Close"].rolling(20, min_periods=10).std()
            bbw = ((bb_ma + 2*bb_sd) - (bb_ma - 2*bb_sd)) / bb_ma.replace(0, float('nan')) * 100
            bbw_slope = bbw.diff(5)

            # Walk forward from entry, check exit conditions each day
            exit_price = None
            exit_bar = None
            exit_reason = None

            for t in range(5, min(MAX_HOLD, len(closes))):
                c = float(closes.iloc[t])
                pnl_pct = (c / entry_price - 1) * 100

                # Exit 1: Profit target
                if pnl_pct >= PROFIT_TARGET:
                    exit_price = c; exit_bar = t; exit_reason = "PROFIT_TARGET"
                    break

                # Exit 2: BBW contraction + above SMA
                if t < len(sma_20) and t < len(bbw_slope):
                    sma_val = sma_20.iloc[t]
                    slope_val = bbw_slope.iloc[t]
                    if (not pd.isna(sma_val) and c > sma_val and
                        not pd.isna(slope_val) and slope_val < 0):
                        exit_price = c; exit_bar = t; exit_reason = "BBW_CONTRACT"
                        break

            # Exit 3: Max hold
            if exit_price is None:
                if len(closes) >= MAX_HOLD:
                    exit_price = float(closes.iloc[MAX_HOLD - 1])
                    exit_bar = MAX_HOLD - 1
                    exit_reason = "MAX_HOLD"
                elif sig["signal_date"] <= max_hold_cutoff.isoformat():
                    # Past max hold period, use last available close
                    exit_price = float(closes.iloc[-1])
                    exit_bar = len(closes) - 1
                    exit_reason = "MAX_HOLD"
                else:
                    # Not old enough for max hold yet, skip
                    continue

            peak = float(closes.iloc[:exit_bar+1].max())
            pnl = (exit_price / entry_price - 1) * 100

            resolve_signal(
                signal_id=sig["id"],
                exit_price=exit_price,
                peak_price=peak,
                exit_reason=exit_reason,
            )
            logger.info(
                "  %s %s: entry=%.2f exit=%.2f P&L=%+.2f%% (%s, %dd)",
                symbol, signal_date, entry_price, exit_price, pnl,
                exit_reason, exit_bar,
            )
            resolved += 1

        except Exception as exc:
            logger.warning("Resolution failed for %s: %s", symbol, exc)

    return resolved


def show_status():
    """Display current performance and drift status."""
    from modules.signal_tracker import get_performance, check_drift, EXPECTED

    print("\n" + "=" * 60)
    print("NIMBUS FORWARD VALIDATION STATUS")
    print("=" * 60)

    for mode_key, mode_name in [("PRIMARY", "Primary Entry"), ("SECONDARY", "Secondary Entry")]:
        perf = get_performance(mode=mode_key)
        exp = EXPECTED[mode_key]
        print(f"\n── Tier {mode_key}: {mode_name} ──")

        if perf["n_signals"] == 0:
            print("  No resolved signals yet. Monitoring...")
            continue

        # Performance vs expectations
        wr = perf["win_rate"]
        avg = perf["avg_pnl"]
        wr_delta = wr - exp["win_rate"]
        avg_delta = avg - exp["avg_return"]

        print(f"  Signals:    {perf['n_signals']}")
        print(f"  Win rate:   {wr:5.1f}%  (expected {exp['win_rate']:.0f}%  Δ={wr_delta:+.1f}%)")
        print(f"  Avg P&L:    {avg:+.3f}%  (expected {exp['avg_return']:+.2f}%  Δ={avg_delta:+.2f}%)")
        print(f"  Median:     {perf['med_pnl']:+.3f}%")
        print(f"  Prof. fac:  {perf['profit_factor']:.2f}")
        print(f"  Sharpe:     {perf['sharpe']:.2f}  (expected {exp['sharpe']:.2f})")

    # Drift
    drift = check_drift()
    print(f"\n── Drift Detection ──")
    for mode_key, d in drift.items():
        icon = {"OK": "✓", "WARNING": "⚠", "ALERT": "✗", "NO_DATA": "○"}
        print(f"  {icon.get(d['status'], '?')} {d['message']}")

    print()


def export():
    """Export full report to file."""
    from modules.signal_tracker import export_report

    report = export_report()
    out_dir = os.path.join(os.path.dirname(__file__), "data", "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "forward_validation_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nReport exported: {path}")
    print("Upload this file for analysis.\n")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="NIMBUS Signal Resolution")
    parser.add_argument("--export", action="store_true", help="Export report")
    parser.add_argument("--status", action="store_true", help="Show status only")
    args = parser.parse_args()

    from modules.signal_tracker import init_tracker
    init_tracker()

    if args.status:
        show_status()
        return

    # Resolve
    n = resolve_mature_signals()
    logger.info("Resolved %d signals", n)

    # Status
    show_status()

    # Export if requested
    if args.export:
        export()


if __name__ == "__main__":
    main()
