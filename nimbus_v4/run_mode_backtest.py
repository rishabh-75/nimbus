#!/usr/bin/env python3
"""
run_mode_backtest.py — Compare mode classification strategies
──────────────────────────────────────────────────────────────

Tests 4 approaches to deciding Mode A vs Mode B per stock:
  1. HARDCODED:  current sector_map classification
  2. HURST:      H < 0.5 → Mode B, H > 0.5 → Mode A
  3. AUTOCORR:   negative 5d return autocorrelation → Mode B
  4. BEST_SCORE: compute both, pick whichever scores higher

For each approach, simulates entries on the same universe and
measures 10-day forward return, win rate, Sharpe.

Also tests a 5th approach:
  5. BENCHMARK:  compare stock returns vs equal-weighted basket,
                 stocks that revert toward the basket → Mode B

Usage:
    python3 run_mode_backtest.py                    # synthetic (offline)
    python3 run_mode_backtest.py --live --n 30      # live NSE data
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("mode_bt")


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION METHODS
# ══════════════════════════════════════════════════════════════════════════════


def classify_hardcoded(symbol: str, daily_df: pd.DataFrame) -> str:
    """Current approach: sector-based lookup."""
    from modules.sector_map import get_segment
    seg = get_segment(symbol)
    return "A" if seg == "INSTITUTIONAL" else "B"


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Compute Hurst exponent via R/S analysis. H<0.5=mean-revert, H>0.5=trend."""
    if len(series) < max_lag * 2:
        return 0.5
    vals = series.dropna().values
    lags = range(2, max_lag + 1)
    rs_vals = []
    for lag in lags:
        chunks = [vals[i:i+lag] for i in range(0, len(vals) - lag, lag)]
        if not chunks:
            continue
        rs_list = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_vals.append((np.log(lag), np.log(np.mean(rs_list))))
    if len(rs_vals) < 3:
        return 0.5
    x = np.array([v[0] for v in rs_vals])
    y = np.array([v[1] for v in rs_vals])
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def classify_hurst(symbol: str, daily_df: pd.DataFrame) -> str:
    """H < 0.5 = mean reverting → Mode B, H > 0.5 = trending → Mode A."""
    if len(daily_df) < 50:
        return "B"
    returns = daily_df["Close"].pct_change().dropna()
    h = hurst_exponent(returns)
    return "A" if h > 0.5 else "B"


def return_autocorrelation(series: pd.Series, lag: int = 5, window: int = 60) -> float:
    """Autocorrelation of N-day returns over a rolling window."""
    if len(series) < window + lag:
        return 0.0
    rets = series.pct_change(lag).dropna()
    if len(rets) < window:
        return 0.0
    recent = rets.iloc[-window:]
    return float(recent.autocorr(lag=1))


def classify_autocorr(symbol: str, daily_df: pd.DataFrame) -> str:
    """Negative autocorrelation = mean reverting → Mode B."""
    if len(daily_df) < 70:
        return "B"
    ac = return_autocorrelation(daily_df["Close"])
    return "A" if ac > 0.05 else "B"


def classify_best_score(symbol: str, daily_df: pd.DataFrame) -> str:
    """Compute both modes, pick whichever scores higher."""
    from modules.dual_mode import DualModeSignal, _score_mode_a, _score_mode_b
    from modules.indicators import add_williams_r, add_adx

    if len(daily_df) < 35:
        return "B"

    sma = daily_df["Close"].rolling(20).mean()
    close = float(daily_df["Close"].iloc[-1])
    sma_val = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else close
    above_sma = close > sma_val
    pct_sma = (close / sma_val - 1) * 100 if sma_val > 0 else 0

    # Mode A signals
    df_a = add_williams_r(daily_df.copy(), period=20)
    df_a = add_adx(df_a, period=14)
    wr20 = float(df_a["WR"].dropna().iloc[-1]) if not df_a["WR"].dropna().empty else -100
    adx_col = "ADX_14"
    adx = float(df_a[adx_col].dropna().iloc[-1]) if adx_col in df_a.columns and not df_a[adx_col].dropna().empty else 0

    arr = df_a["WR"].dropna().values[-10:]
    cross = 99
    for i in range(len(arr)-1, 0, -1):
        if arr[i] >= -50 and arr[i-1] < -50:
            cross = len(arr)-1-i; break

    sig_a = DualModeSignal(mode="A", segment="INSTITUTIONAL",
                           wr_20=wr20, wr_20_cross_bars=cross,
                           adx_14=adx, adx_trending=(adx >= 20),
                           above_sma=above_sma, pct_from_sma=pct_sma)
    score_a = _score_mode_a(sig_a)

    # Mode B signals
    df_b = add_williams_r(daily_df.copy(), period=30)
    wr30 = float(df_b["WR"].dropna().iloc[-1]) if not df_b["WR"].dropna().empty else -100

    sig_b = DualModeSignal(mode="B", segment="CYCLICAL",
                           wr_30=wr30, adx_14=adx,
                           above_sma=above_sma, pct_from_sma=pct_sma)
    score_b = _score_mode_b(sig_b)

    return "A" if score_a > score_b else "B"


def compute_benchmark_returns(universe: dict, window: int = 10) -> pd.DataFrame:
    """Compute equal-weighted basket returns for relative comparison."""
    all_rets = {}
    for sym, df in universe.items():
        r = df["Close"].pct_change(window)
        all_rets[sym] = r
    rets_df = pd.DataFrame(all_rets)
    rets_df["BASKET"] = rets_df.mean(axis=1)
    return rets_df


def classify_benchmark(symbol: str, daily_df: pd.DataFrame,
                       basket_rets: pd.Series = None) -> str:
    """
    Compare stock vs equal-weighted basket.
    If stock tends to revert toward basket → Mode B.
    If stock tends to diverge from basket → Mode A.
    """
    if basket_rets is None or len(daily_df) < 70:
        return "B"
    stock_rets = daily_df["Close"].pct_change(10)
    # Align indices
    aligned = pd.DataFrame({"stock": stock_rets, "basket": basket_rets}).dropna()
    if len(aligned) < 30:
        return "B"
    # Compute relative return and its autocorrelation
    aligned["relative"] = aligned["stock"] - aligned["basket"]
    recent = aligned["relative"].iloc[-60:]
    if len(recent) < 20:
        return "B"
    ac = recent.autocorr(lag=1)
    # Negative autocorrelation of relative returns = reverts toward basket
    return "A" if ac > 0.05 else "B"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY + FORWARD RETURN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_entry_and_returns(
    daily_df: pd.DataFrame,
    mode: str,
    horizons: list = None,
) -> list:
    """
    Given daily data and assigned mode, compute entry signals and forward returns.
    Returns list of dicts with signal state + outcomes.
    """
    if horizons is None:
        horizons = [5, 10, 20]

    from modules.indicators import add_williams_r, add_adx

    if len(daily_df) < 50:
        return []

    closes = daily_df["Close"].values
    sma = daily_df["Close"].rolling(20).mean().values

    if mode == "A":
        df_ind = add_williams_r(daily_df.copy(), period=20)
        df_ind = add_adx(df_ind, period=14)
        wr_vals = df_ind["WR"].values
        adx_col = "ADX_14"
        adx_vals = df_ind[adx_col].values if adx_col in df_ind.columns else np.full(len(df_ind), 0)
    else:
        df_ind = add_williams_r(daily_df.copy(), period=30)
        df_ind = add_adx(df_ind, period=14)
        wr_vals = df_ind["WR"].values
        adx_col = "ADX_14"
        adx_vals = df_ind[adx_col].values if adx_col in df_ind.columns else np.full(len(df_ind), 0)

    max_h = max(horizons)
    records = []

    for t in range(40, len(daily_df) - max_h):
        wr = wr_vals[t]
        adx = adx_vals[t]
        close = closes[t]
        sma_val = sma[t]

        if np.isnan(wr) or np.isnan(sma_val) or sma_val == 0:
            continue

        above_sma = close > sma_val
        pct_sma = (close / sma_val - 1) * 100

        # Check entry condition
        triggered = False
        if mode == "A":
            # WR(20) cross above -50 within last 3 bars
            if wr >= -50 and not np.isnan(adx) and adx >= 20:
                prev = wr_vals[max(0, t-3):t]
                if len(prev) > 0 and np.nanmin(prev) < -50:
                    triggered = True
        else:
            # WR(30) < -30 AND below SMA
            triggered = wr < -30 and not above_sma

        if not triggered:
            continue

        # Forward returns
        fwd = {}
        for h in horizons:
            if t + h < len(daily_df):
                fwd[f"fwd_{h}d"] = (closes[t + h] / close - 1) * 100

        records.append({
            "bar": t,
            "close": close,
            "wr": wr,
            "adx": adx if not np.isnan(adx) else 0,
            "pct_sma": pct_sma,
            "mode": mode,
            **fwd,
        })

    return records


def _compute_entries_custom_wr(
    daily_df: pd.DataFrame,
    wr_period: int,
    wr_thresh: float,
    horizons: list = None,
) -> list:
    """
    Compute Mode B entries with custom WR period and threshold.
    Entry: WR(period) < thresh AND close < 20-SMA.
    """
    if horizons is None:
        horizons = [5, 10, 20]

    from modules.indicators import add_williams_r

    if len(daily_df) < wr_period + 20:
        return []

    df = add_williams_r(daily_df.copy(), period=wr_period)
    closes = df["Close"].values
    sma = df["Close"].rolling(20).mean().values
    wr_vals = df["WR"].values

    max_h = max(horizons)
    records = []

    for t in range(max(wr_period, 30), len(df) - max_h):
        wr = wr_vals[t]
        close = closes[t]
        sma_val = sma[t]

        if np.isnan(wr) or np.isnan(sma_val) or sma_val == 0:
            continue

        # Mode B entry: WR < threshold AND below SMA
        if wr >= wr_thresh or close >= sma_val:
            continue

        fwd = {}
        for h in horizons:
            if t + h < len(df):
                fwd[f"fwd_{h}d"] = (closes[t + h] / close - 1) * 100

        records.append({
            "bar": t,
            "close": close,
            "wr": wr,
            "pct_sma": (close / sma_val - 1) * 100,
            **fwd,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ══════════════════════════════════════════════════════════════════════════════


def run(use_live: bool = False, n_symbols: int = 20):
    from modules.indicators import _resample_daily

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("MODE CLASSIFICATION BACKTEST")
    logger.info("=" * 60)

    # Load data
    if use_live:
        from backtest.data_loader import download_batch
        try:
            from modules.data import NIFTY100_SYMBOLS
            symbols = NIFTY100_SYMBOLS[:n_symbols]
        except Exception:
            symbols = [
                "RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK",
                "SBIN", "AXISBANK", "BAJFINANCE", "TATAMOTORS", "MARUTI",
                "TATASTEEL", "JSWSTEEL", "HINDALCO", "WIPRO", "TECHM",
                "SUNPHARMA", "CIPLA", "LT", "NTPC", "BHARTIARTL",
                "ITC", "HINDUNILVR", "ASIANPAINT", "DRREDDY", "KOTAKBANK",
                "BAJAJ-AUTO", "HEROMOTOCO", "M&M", "BPCL", "COALINDIA",
            ][:n_symbols]
        universe = download_batch(symbols, years=3)
    else:
        from backtest.data_loader import generate_universe
        universe = generate_universe(n_symbols=n_symbols, n_bars=750)

    logger.info("Universe: %d symbols loaded", len(universe))

    # Compute basket returns for benchmark method
    basket_rets_df = compute_benchmark_returns(universe)
    basket_rets = basket_rets_df["BASKET"] if "BASKET" in basket_rets_df.columns else None

    # Classification methods to test
    methods = {
        "HARDCODED": lambda sym, df: classify_hardcoded(sym, df),
        "HURST": lambda sym, df: classify_hurst(sym, df),
        "AUTOCORR": lambda sym, df: classify_autocorr(sym, df),
        "BEST_SCORE": lambda sym, df: classify_best_score(sym, df),
        "BENCHMARK": lambda sym, df: classify_benchmark(sym, df, basket_rets),
    }

    # Also test WR period variants (daily-equivalent of 4H lookbacks)
    # WR(8) on daily ≈ WR(30) on 4H (NSE has ~2 bars/day at 4H)
    # WR(14) on daily ≈ WR(50) on 4H (original system's WR period)
    wr_periods_to_test = {
        "B_WR8":  (8,  -30),   # fast: ~1.5 weeks lookback
        "B_WR14": (14, -30),   # medium: ~3 weeks lookback  
        "B_WR20": (20, -30),   # medium-slow: ~4 weeks
        "B_WR30": (30, -30),   # current: ~6 weeks (sweep-validated)
        "B_WR50": (50, -30),   # slow: ~10 weeks
        "B_WR8_deep":  (8,  -50),  # fast + deeper threshold
        "B_WR14_deep": (14, -50),  # medium + deeper threshold
        "B_WR30_deep": (30, -50),  # current + deeper threshold
    }

    results = {}

    for method_name, classifier in methods.items():
        logger.info("\n── Testing: %s ──", method_name)
        all_entries = []
        mode_counts = {"A": 0, "B": 0}
        class_map = {}

        for sym, df in universe.items():
            if len(df) < 100:
                continue

            # Classify
            mode = classifier(sym, df)
            mode_counts[mode] += 1
            class_map[sym] = mode

            # Compute entries with assigned mode
            entries = compute_entry_and_returns(df, mode)
            for e in entries:
                e["symbol"] = sym
                e["method"] = method_name
            all_entries.extend(entries)

        # Compute stats
        if not all_entries:
            results[method_name] = {"n": 0, "status": "NO_ENTRIES"}
            continue

        edf = pd.DataFrame(all_entries)
        stats = {"n": len(edf), "mode_split": mode_counts}

        for h in [5, 10, 20]:
            col = f"fwd_{h}d"
            if col not in edf.columns:
                continue
            vals = edf[col].dropna()
            if len(vals) < 10:
                continue
            avg = float(vals.mean())
            std = float(vals.std())
            stats[f"{h}d_n"] = len(vals)
            stats[f"{h}d_avg"] = round(avg, 4)
            stats[f"{h}d_win"] = round((vals > 0).mean() * 100, 1)
            stats[f"{h}d_sharpe"] = round(avg / std * 7.1, 2) if std > 0 else 0
            stats[f"{h}d_pf"] = round(
                float(vals[vals > 0].sum()) / abs(float(vals[vals <= 0].sum())), 2
            ) if (vals <= 0).any() and (vals > 0).any() else 0

        # Per-mode breakdown
        for mode_key in ("A", "B"):
            sub = edf[edf["mode"] == mode_key]
            for h in [10]:
                col = f"fwd_{h}d"
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna()
                if len(vals) >= 5:
                    stats[f"mode_{mode_key}_{h}d_n"] = len(vals)
                    stats[f"mode_{mode_key}_{h}d_avg"] = round(float(vals.mean()), 4)
                    stats[f"mode_{mode_key}_{h}d_win"] = round((vals > 0).mean() * 100, 1)

        # Hurst/autocorr values for diagnostics
        if method_name in ("HURST", "AUTOCORR"):
            diag = {}
            for sym, df in universe.items():
                if len(df) < 70:
                    continue
                if method_name == "HURST":
                    val = hurst_exponent(df["Close"].pct_change().dropna())
                else:
                    val = return_autocorrelation(df["Close"])
                diag[sym] = round(val, 3)
            stats["diagnostics"] = diag

        results[method_name] = stats
        logger.info("  %s: %d entries, split=%s", method_name, stats["n"], mode_counts)
        for h in [5, 10, 20]:
            k = f"{h}d_avg"
            if k in stats:
                logger.info(
                    "    %dd: N=%d  Win=%.1f%%  Avg=%+.3f%%  Sharpe=%.2f  PF=%.2f",
                    h, stats.get(f"{h}d_n", 0), stats.get(f"{h}d_win", 0),
                    stats.get(f"{h}d_avg", 0), stats.get(f"{h}d_sharpe", 0),
                    stats.get(f"{h}d_pf", 0),
                )

    # ── WR PERIOD SWEEP (all stocks as Mode B with varying lookbacks) ─────
    logger.info("\n" + "=" * 70)
    logger.info("WR PERIOD SWEEP (Mode B mean-reversion, varying lookback)")
    logger.info("=" * 70)

    for wr_label, (wr_period, wr_thresh) in wr_periods_to_test.items():
        all_entries = []
        for sym, df in universe.items():
            if len(df) < max(wr_period + 20, 60):
                continue
            entries = _compute_entries_custom_wr(df, wr_period, wr_thresh)
            for e in entries:
                e["symbol"] = sym
            all_entries.extend(entries)

        if not all_entries:
            results[wr_label] = {"n": 0, "status": "NO_ENTRIES", "mode_split": {"A": 0, "B": len(universe)}}
            continue

        edf = pd.DataFrame(all_entries)
        stats = {"n": len(edf), "mode_split": {"A": 0, "B": len(universe)}}

        # Split into in-sample (first 60%) and out-of-sample (last 40%)
        edf = edf.sort_values("bar")
        split_idx = int(len(edf) * 0.6)

        for half_name, half_df in [("IS", edf.iloc[:split_idx]), ("OOS", edf.iloc[split_idx:])]:
            for h in [5, 10, 20]:
                col = f"fwd_{h}d"
                if col not in half_df.columns:
                    continue
                vals = half_df[col].dropna()
                if len(vals) < 10:
                    continue
                avg = float(vals.mean())
                std = float(vals.std())
                stats[f"{half_name}_{h}d_n"] = len(vals)
                stats[f"{half_name}_{h}d_avg"] = round(avg, 4)
                stats[f"{half_name}_{h}d_win"] = round((vals > 0).mean() * 100, 1)
                stats[f"{half_name}_{h}d_sharpe"] = round(avg / std * 7.1, 2) if std > 0 else 0

        # Combined stats (for comparison table)
        for h in [5, 10, 20]:
            col = f"fwd_{h}d"
            if col not in edf.columns:
                continue
            vals = edf[col].dropna()
            if len(vals) < 10:
                continue
            avg = float(vals.mean())
            std = float(vals.std())
            stats[f"{h}d_n"] = len(vals)
            stats[f"{h}d_avg"] = round(avg, 4)
            stats[f"{h}d_win"] = round((vals > 0).mean() * 100, 1)
            stats[f"{h}d_sharpe"] = round(avg / std * 7.1, 2) if std > 0 else 0
            stats[f"{h}d_pf"] = round(
                float(vals[vals > 0].sum()) / abs(float(vals[vals <= 0].sum())), 2
            ) if (vals <= 0).any() and (vals > 0).any() else 0

        results[wr_label] = stats
        logger.info(
            "  %-14s WR(%d) thresh=%d: N=%d",
            wr_label, wr_period, wr_thresh, stats["n"],
        )
        for h in [10]:
            is_sh = stats.get(f"IS_{h}d_sharpe", 0)
            oos_sh = stats.get(f"OOS_{h}d_sharpe", 0)
            logger.info(
                "    %dd: IS(N=%d avg=%+.3f%% sh=%.2f) OOS(N=%d avg=%+.3f%% sh=%.2f)",
                h,
                stats.get(f"IS_{h}d_n", 0), stats.get(f"IS_{h}d_avg", 0), is_sh,
                stats.get(f"OOS_{h}d_n", 0), stats.get(f"OOS_{h}d_avg", 0), oos_sh,
            )

    # ── COMPARISON TABLE ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: 10-DAY FORWARD RETURNS")
    logger.info("=" * 70)
    logger.info(
        "%-12s %6s %6s %8s %7s %6s  %s",
        "Method", "N", "Win%", "AvgRet", "Sharpe", "PF", "A/B Split"
    )
    logger.info("-" * 70)

    ranked = sorted(
        results.items(),
        key=lambda x: x[1].get("10d_sharpe", -99),
        reverse=True,
    )

    for method, stats in ranked:
        if stats.get("n", 0) == 0:
            logger.info("%-12s  (no entries)", method)
            continue
        split = stats.get("mode_split", {})
        logger.info(
            "%-12s %6d %5.1f%% %+7.3f%% %6.2f %5.2f  A=%d B=%d",
            method,
            stats.get("10d_n", 0),
            stats.get("10d_win", 0),
            stats.get("10d_avg", 0),
            stats.get("10d_sharpe", 0),
            stats.get("10d_pf", 0),
            split.get("A", 0),
            split.get("B", 0),
        )

    # Per-mode breakdown for the best method
    best_method = ranked[0][0] if ranked else None
    if best_method:
        bs = results[best_method]
        logger.info("\n── BEST METHOD: %s ──", best_method)
        for mk in ("A", "B"):
            n = bs.get(f"mode_{mk}_10d_n", 0)
            if n > 0:
                logger.info(
                    "  Mode %s: N=%d  Win=%.1f%%  Avg=%+.3f%%",
                    mk, n,
                    bs.get(f"mode_{mk}_10d_win", 0),
                    bs.get(f"mode_{mk}_10d_avg", 0),
                )

    # ── WR PERIOD COMPARISON TABLE ────────────────────────────────────────
    wr_results = {k: v for k, v in results.items() if k.startswith("B_WR")}
    if wr_results:
        logger.info("\n" + "=" * 85)
        logger.info("WR PERIOD COMPARISON (Mode B mean-reversion, 10-day returns)")
        logger.info("=" * 85)
        logger.info(
            "%-16s %6s %6s %8s %7s  |  %6s %6s %8s %7s",
            "Config", "IS_N", "IS_W%", "IS_Avg", "IS_Sh",
            "OOS_N", "OOS_W%", "OOS_Avg", "OOS_Sh",
        )
        logger.info("-" * 85)

        wr_ranked = sorted(
            wr_results.items(),
            key=lambda x: x[1].get("OOS_10d_sharpe", -99),
            reverse=True,
        )

        for label, stats in wr_ranked:
            if stats.get("n", 0) == 0:
                logger.info("%-16s  (no entries)", label)
                continue
            logger.info(
                "%-16s %6d %5.1f%% %+7.3f%% %6.2f  |  %6d %5.1f%% %+7.3f%% %6.2f",
                label,
                stats.get("IS_10d_n", 0), stats.get("IS_10d_win", 0),
                stats.get("IS_10d_avg", 0), stats.get("IS_10d_sharpe", 0),
                stats.get("OOS_10d_n", 0), stats.get("OOS_10d_win", 0),
                stats.get("OOS_10d_avg", 0), stats.get("OOS_10d_sharpe", 0),
            )

        # Best WR config (positive Sharpe in both halves)
        best_wr = None
        for label, stats in wr_ranked:
            is_sh = stats.get("IS_10d_sharpe", -1)
            oos_sh = stats.get("OOS_10d_sharpe", -1)
            if is_sh > 0 and oos_sh > 0:
                best_wr = label
                break

        if best_wr:
            logger.info("\n  RECOMMENDED WR CONFIG: %s", best_wr)
            bw = wr_results[best_wr]
            logger.info(
                "    IS:  Sharpe=%.2f  Win=%.1f%%  Avg=%+.3f%%",
                bw.get("IS_10d_sharpe", 0), bw.get("IS_10d_win", 0), bw.get("IS_10d_avg", 0),
            )
            logger.info(
                "    OOS: Sharpe=%.2f  Win=%.1f%%  Avg=%+.3f%%",
                bw.get("OOS_10d_sharpe", 0), bw.get("OOS_10d_win", 0), bw.get("OOS_10d_avg", 0),
            )

    # Save report
    out_dir = os.path.join(os.path.dirname(__file__), "data", "backtest_results")
    os.makedirs(out_dir, exist_ok=True)

    lines = ["MODE CLASSIFICATION BACKTEST RESULTS", "=" * 60, ""]
    for method, stats in ranked:
        lines.append(f"Method: {method}")
        for k, v in sorted(stats.items()):
            if k != "diagnostics":
                lines.append(f"  {k}: {v}")
        lines.append("")

    report_path = os.path.join(out_dir, "mode_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("\nReport: %s", report_path)

    elapsed = time.time() - t0
    logger.info("Complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    run(use_live=args.live, n_symbols=args.n)
