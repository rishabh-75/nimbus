#!/usr/bin/env python3
"""
run_walkforward.py — Walk-forward optimization + validation.

For each fold:
  1. TRAIN window: sweep key parameters, pick best by IS Sharpe
  2. TEST window: run best params OOS, record trades
  3. Advance 6 months, repeat

This answers:
  - Do optimal parameters stay stable or drift?
  - What's the true OOS edge with honest parameter selection?
  - How many OOS trades across all folds?

Usage:
    python3 run_walkforward.py --live --n 70 --years 5
"""
import argparse, datetime, itertools, logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("wf")


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df):
    d = df.copy(); c=d["Close"]; h=d["High"]; l=d["Low"]
    v = d["Volume"] if "Volume" in d.columns else pd.Series(0, index=d.index)
    for p in [10, 20]: d[f"SMA_{p}"] = c.rolling(p).mean()
    for p in [14, 20, 30]:
        hh = h.rolling(p).max(); ll = l.rolling(p).min()
        d[f"WR_{p}"] = ((hh - c) / (hh - ll).replace(0, np.nan)) * -100
    tp = (h+l+c)/3; mf = tp*v
    pmf = pd.Series(np.where(tp>tp.shift(1), mf, 0), index=d.index)
    nmf = pd.Series(np.where(tp<tp.shift(1), mf, 0), index=d.index)
    mr = pmf.rolling(14).sum() / nmf.rolling(14).sum().replace(0, np.nan)
    d["MFI"] = 100 - (100 / (1 + mr))
    bb_ma = c.rolling(20).mean(); bb_sd = c.rolling(20).std()
    d["BBW"] = ((bb_ma + 2*bb_sd) - (bb_ma - 2*bb_sd)) / bb_ma.replace(0, np.nan) * 100
    d["BBW_slope"] = d["BBW"].diff(5)
    d["VOL_RATIO"] = v / v.rolling(20).mean().replace(0, np.nan)
    d["HIGH_50"] = h.rolling(50, min_periods=20).max()
    d["DD"] = ((c - d["HIGH_50"]) / d["HIGH_50"]) * 100
    red = (c < c.shift(1)).astype(int)
    d["RED_STREAK"] = red.groupby((red != red.shift()).cumsum()).cumsum()
    return d


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION (parameterized)
# ══════════════════════════════════════════════════════════════════════════════

def simulate(df, start_idx, end_idx, params):
    """Run entry/exit simulation with given params between start_idx and end_idx."""
    n = len(df)
    if n < 60: return []

    wr_col = f"WR_{params['wr_period']}"
    sma_col = f"SMA_{params['sma_period']}"
    if wr_col not in df.columns or sma_col not in df.columns: return []

    c = df["Close"].values; sma = df[sma_col].values; wr = df[wr_col].values
    mfi = df["MFI"].values; vol_r = df["VOL_RATIO"].values
    dd = df["DD"].values; streak = df["RED_STREAK"].values
    bbw_slope = df["BBW_slope"].values

    wt = params["wr_thresh"]; mfi_min = params["mfi_min"]
    dd_thresh = params["dd_thresh"]; streak_min = params["streak_min"]
    pt = params["profit_target"]; max_hold = params["max_hold"]

    trades = []; in_t = False; eb = 0; ep = 0.0; tier = ""

    for t in range(max(60, start_idx), min(end_idx, n)):
        if not in_t:
            w = wr[t]; cl = c[t]; s = sma[t]; m = mfi[t]
            if np.isnan(w) or np.isnan(s) or s == 0 or np.isnan(m): continue
            if w >= wt or cl >= s or m < mfi_min: continue

            d_val = dd[t] if not np.isnan(dd[t]) else 0
            r_val = streak[t] if not np.isnan(streak[t]) else 0
            v_val = vol_r[t] if not np.isnan(vol_r[t]) else 1.0

            if d_val <= dd_thresh and r_val >= streak_min and v_val >= 0.5:
                tier = "PRIMARY"
            else:
                tier = "SECONDARY"
            in_t = True; eb = t; ep = cl

        else:
            bars = t - eb; cl = c[t]; s = sma[t]; ex = False; reason = ""

            if ep > 0 and (cl / ep - 1) * 100 >= pt:
                ex = True; reason = "PT"
            if not ex and bars >= 5 and not np.isnan(bbw_slope[t]) and bbw_slope[t] < 0:
                if not np.isnan(s) and cl > s:
                    ex = True; reason = "BBW"
            if not ex and bars >= max_hold:
                ex = True; reason = "MAX"

            if ex:
                pnl = (cl / ep - 1) * 100
                trades.append({
                    "entry_bar": eb, "bars_held": bars, "pnl_pct": round(pnl, 4),
                    "tier": tier, "exit_reason": reason,
                    "entry_date": str(df.index[eb].date()) if hasattr(df.index[eb], 'date') else "",
                })
                in_t = False
    return trades


def calc_stats(pnls):
    if len(pnls) < 3:
        return {"n": len(pnls), "avg": 0, "win": 0, "sharpe": 0, "pf": 0}
    arr = np.array(pnls); avg = float(np.mean(arr)); std = float(np.std(arr))
    w = arr[arr > 0]; lo = arr[arr <= 0]
    return {
        "n": len(arr), "avg": round(avg, 4),
        "win": round((arr > 0).mean() * 100, 1),
        "sharpe": round(avg / std * np.sqrt(252/10), 2) if std > 0 else 0,
        "pf": round(float(w.sum()) / abs(float(lo.sum())), 2) if len(lo) > 0 and lo.sum() != 0 else 99.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER GRID (swept within each fold's training window)
# ══════════════════════════════════════════════════════════════════════════════

def build_param_grid():
    """Build parameter combinations to sweep in each fold."""
    configs = []
    for wr_p in [14, 20, 30]:
        for wr_t in [-30, -40, -50]:
            for sma_p in [10, 20]:
                for mfi_min in [30, 40]:
                    for dd_t in [-5, -8]:
                        for streak_m in [2, 3]:
                            for pt in [3.0, 5.0]:
                                for mh in [25, 30]:
                                    configs.append({
                                        "wr_period": wr_p, "wr_thresh": wr_t,
                                        "sma_period": sma_p, "mfi_min": mfi_min,
                                        "dd_thresh": dd_t, "streak_min": streak_m,
                                        "profit_target": pt, "max_hold": mh,
                                    })
    return configs


def param_label(p):
    return (f"WR({p['wr_period']},{p['wr_thresh']}) SMA{p['sma_period']} "
            f"MFI≥{p['mfi_min']} DD≤{p['dd_thresh']} Str≥{p['streak_min']} "
            f"PT{p['profit_target']}% MH{p['max_hold']}")


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run(use_live=False, n_symbols=70, years=5):
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("WALK-FORWARD OPTIMIZATION + VALIDATION")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    if use_live:
        from backtest.data_loader import download_batch
        try:
            from modules.data import NIFTY100_SYMBOLS
            symbols = NIFTY100_SYMBOLS[:n_symbols]
        except Exception:
            symbols = [
                "RELIANCE","HDFCBANK","TCS","INFY","ICICIBANK","SBIN","AXISBANK",
                "BAJFINANCE","TATAMOTORS","MARUTI","TATASTEEL","JSWSTEEL","HINDALCO",
                "WIPRO","TECHM","SUNPHARMA","CIPLA","LT","NTPC","BHARTIARTL",
                "ITC","HINDUNILVR","ASIANPAINT","DRREDDY","KOTAKBANK","BAJAJ-AUTO",
                "HEROMOTOCO","M&M","BPCL","COALINDIA","ONGC","POWERGRID","DIVISLAB",
                "TITAN","ULTRACEMCO","NESTLEIND","HCLTECH","APOLLOHOSP","ADANIENT",
                "ADANIPORTS","BRITANNIA","PIDILITIND","SBILIFE","HDFCLIFE",
                "BAJAJFINSV","GRASIM","INDUSINDBK","EICHERMOT","SHRIRAMFIN",
                "CHOLAFIN","TRENT","CANBK","BANKBARODA","PNB","IOC",
                "IRCTC","HAL","BEL","MOTHERSON","TATAPOWER",
                "ABB","SIEMENS","GODREJCP","COLPAL","HAVELLS","VOLTAS",
                "MFSL","VBL","MUTHOOTFIN","IDFCFIRSTB",
            ][:n_symbols]
        raw = download_batch(symbols, years=years)
    else:
        from backtest.data_loader import generate_universe
        raw = generate_universe(n_symbols=n_symbols, n_bars=int(years * 252))

    logger.info("Loaded %d symbols", len(raw))

    # ── Precompute indicators ─────────────────────────────────────────────
    logger.info("Computing indicators...")
    universe = {}
    for sym, df in raw.items():
        if len(df) >= 100:
            universe[sym] = compute_indicators(df)
    logger.info("Ready: %d symbols, %d–%d bars",
                len(universe),
                min(len(v) for v in universe.values()),
                max(len(v) for v in universe.values()))

    # ── Walk-forward folds ────────────────────────────────────────────────
    TRAIN_BARS = 378    # 18 months
    TEST_BARS = 126     # 6 months
    MIN_BARS_NEEDED = TRAIN_BARS + 2 * TEST_BARS  # at least 2 folds

    # Filter out stocks with insufficient history
    short = [s for s, df in universe.items() if len(df) < MIN_BARS_NEEDED]
    if short:
        logger.info("Dropping %d symbols with < %d bars: %s",
                     len(short), MIN_BARS_NEEDED, ", ".join(short[:10]))
        for s in short:
            del universe[s]

    if len(universe) < 5:
        logger.error("Only %d symbols with enough data. Need at least 5.", len(universe))
        return

    max_bars = min(len(df) for df in universe.values())
    n_folds = (max_bars - TRAIN_BARS) // TEST_BARS
    if n_folds < 2:
        logger.error("Need %d bars, have %d", TRAIN_BARS + 2*TEST_BARS, max_bars)
        return
    logger.info("After filter: %d symbols, %d–%d bars (min needed: %d)",
                len(universe), min(len(v) for v in universe.values()),
                max(len(v) for v in universe.values()), MIN_BARS_NEEDED)
    logger.info("Folds: %d (train=%dd, test=%dd)", n_folds, TRAIN_BARS, TEST_BARS)

    param_grid = build_param_grid()
    logger.info("Param grid: %d combinations per fold", len(param_grid))

    # ── Run each fold ─────────────────────────────────────────────────────
    fold_results = []
    all_oos_trades = []
    chosen_params_log = []

    for fold in range(n_folds):
        train_start = fold * TEST_BARS
        train_end = train_start + TRAIN_BARS
        test_start = train_end
        test_end = test_start + TEST_BARS
        if test_end > max_bars: break

        sample_df = list(universe.values())[0]
        train_dates = f"{sample_df.index[train_start].date()} → {sample_df.index[train_end-1].date()}"
        test_dates = f"{sample_df.index[test_start].date()} → {sample_df.index[test_end-1].date()}"

        logger.info("\n  Fold %d: TRAIN %s  |  TEST %s", fold+1, train_dates, test_dates)

        # ── Phase 1: Sweep params on training window ──────────────────────
        best_sharpe = -999; best_params = None; best_is_stats = None

        for pi, params in enumerate(param_grid):
            is_trades = []
            for sym, df in universe.items():
                if len(df) < test_end: continue
                is_trades.extend(simulate(df, train_start, train_end, params))

            if len(is_trades) < 15: continue
            pnls = np.array([t["pnl_pct"] for t in is_trades])
            avg = float(np.mean(pnls)); std = float(np.std(pnls))
            sharpe = avg / std * np.sqrt(252/10) if std > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()
                best_is_stats = calc_stats(pnls)

        if best_params is None:
            logger.warning("    No viable params found in training window")
            continue

        logger.info("    BEST IS: %s", param_label(best_params))
        logger.info("    IS stats: N=%d  Sh=%.2f  W=%.1f%%  Avg=%+.3f%%",
                     best_is_stats["n"], best_is_stats["sharpe"],
                     best_is_stats["win"], best_is_stats["avg"])

        # ── Phase 2: Test best params on OOS window ───────────────────────
        oos_trades = []
        for sym, df in universe.items():
            if len(df) < test_end: continue
            trades = simulate(df, test_start, test_end, best_params)
            for t in trades: t["symbol"] = sym; t["fold"] = fold + 1
            oos_trades.extend(trades)

        oos_stats = calc_stats([t["pnl_pct"] for t in oos_trades])
        pri_stats = calc_stats([t["pnl_pct"] for t in oos_trades if t["tier"] == "PRIMARY"])
        sec_stats = calc_stats([t["pnl_pct"] for t in oos_trades if t["tier"] == "SECONDARY"])

        logger.info("    OOS: N=%d  Sh=%.2f  W=%.1f%%  Avg=%+.3f%%  (PRI=%d, SEC=%d)",
                     oos_stats["n"], oos_stats["sharpe"], oos_stats["win"], oos_stats["avg"],
                     pri_stats["n"], sec_stats["n"])

        fold_results.append({
            "fold": fold+1, "train_dates": train_dates, "test_dates": test_dates,
            "best_params": best_params, "is_stats": best_is_stats,
            "oos_all": oos_stats, "oos_primary": pri_stats, "oos_secondary": sec_stats,
        })
        chosen_params_log.append({"fold": fold+1, **best_params})
        all_oos_trades.extend(oos_trades)

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 90)
    logger.info("WALK-FORWARD RESULTS")
    logger.info("=" * 90)

    # Aggregate OOS
    all_pnls = [t["pnl_pct"] for t in all_oos_trades]
    pri_pnls = [t["pnl_pct"] for t in all_oos_trades if t["tier"] == "PRIMARY"]
    sec_pnls = [t["pnl_pct"] for t in all_oos_trades if t["tier"] == "SECONDARY"]

    logger.info("\n── AGGREGATE OOS (all folds, honestly optimized) ──")
    a = calc_stats(all_pnls); p = calc_stats(pri_pnls); s = calc_stats(sec_pnls)
    logger.info("  ALL:       N=%-5d  Sh=%5.2f  W=%5.1f%%  Avg=%+.3f%%  PF=%.2f",
                a["n"], a["sharpe"], a["win"], a["avg"], a["pf"])
    logger.info("  PRIMARY:   N=%-5d  Sh=%5.2f  W=%5.1f%%  Avg=%+.3f%%  PF=%.2f",
                p["n"], p["sharpe"], p["win"], p["avg"], p["pf"])
    logger.info("  SECONDARY: N=%-5d  Sh=%5.2f  W=%5.1f%%  Avg=%+.3f%%  PF=%.2f",
                s["n"], s["sharpe"], s["win"], s["avg"], s["pf"])

    # Per-fold table
    logger.info("\n── PER-FOLD BREAKDOWN ──")
    logger.info("%-4s %-27s | %5s %5s %7s %5s | %5s %5s | %-30s",
                "Fold", "Test Period", "N", "W%", "Avg", "Sh", "PriN", "SecN", "Best Params")
    logger.info("-" * 120)
    for f in fold_results:
        o = f["oos_all"]; bp = f["best_params"]
        short = f"WR({bp['wr_period']},{bp['wr_thresh']}) SMA{bp['sma_period']} MFI≥{bp['mfi_min']} PT{bp['profit_target']}%"
        logger.info("%-4d %-27s | %5d %4.1f%% %+6.3f%% %5.2f | %5d %5d | %s",
                     f["fold"], f["test_dates"],
                     o["n"], o["win"], o["avg"], o["sharpe"],
                     f["oos_primary"]["n"], f["oos_secondary"]["n"], short)

    # Parameter stability
    logger.info("\n── PARAMETER STABILITY (do optimal params drift?) ──")
    params_df = pd.DataFrame(chosen_params_log)
    for col in ["wr_period", "wr_thresh", "sma_period", "mfi_min", "dd_thresh",
                "streak_min", "profit_target", "max_hold"]:
        vals = params_df[col].values
        if len(set(vals)) == 1:
            logger.info("  %-14s: STABLE → %s (same across all folds)", col, vals[0])
        else:
            from collections import Counter
            counts = Counter(vals)
            most = counts.most_common(1)[0]
            logger.info("  %-14s: VARIES → %s (mode=%s in %d/%d folds)",
                         col, dict(counts), most[0], most[1], len(vals))

    # Stability metrics
    logger.info("\n── STABILITY METRICS ──")
    fold_sh = [f["oos_all"]["sharpe"] for f in fold_results if f["oos_all"]["n"] >= 5]
    fold_w = [f["oos_all"]["win"] for f in fold_results if f["oos_all"]["n"] >= 5]
    fold_a = [f["oos_all"]["avg"] for f in fold_results if f["oos_all"]["n"] >= 5]
    if fold_sh:
        pos = sum(1 for x in fold_sh if x > 0)
        logger.info("  Positive Sharpe folds: %d / %d (%.0f%%)", pos, len(fold_sh), pos/len(fold_sh)*100)
        logger.info("  Sharpe:   min=%.2f  max=%.2f  median=%.2f", min(fold_sh), max(fold_sh), np.median(fold_sh))
        logger.info("  Win%%:     min=%.1f  max=%.1f  median=%.1f", min(fold_w), max(fold_w), np.median(fold_w))
        logger.info("  Avg ret:  min=%+.3f  max=%+.3f  median=%+.3f", min(fold_a), max(fold_a), np.median(fold_a))

    # Walk-forward efficiency ratio
    if fold_sh and best_is_stats:
        avg_oos_sharpe = np.mean(fold_sh)
        avg_is_sharpe = np.mean([f["is_stats"]["sharpe"] for f in fold_results])
        wfe = avg_oos_sharpe / avg_is_sharpe * 100 if avg_is_sharpe > 0 else 0
        logger.info("  WF Efficiency: %.0f%% (OOS/IS Sharpe ratio — >50%% is good, >70%% is excellent)", wfe)

    # Exit reasons
    logger.info("\n── EXIT REASONS (OOS) ──")
    from collections import Counter
    reasons = Counter(t["exit_reason"] for t in all_oos_trades)
    for r, cnt in reasons.most_common():
        rs = calc_stats([t["pnl_pct"] for t in all_oos_trades if t["exit_reason"] == r])
        logger.info("  %-4s: N=%4d (%4.1f%%)  W=%5.1f%%  Avg=%+.3f%%",
                     r, cnt, cnt/len(all_oos_trades)*100, rs["win"], rs["avg"])

    # ── Save ──────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), "data", "backtest_results")
    os.makedirs(out, exist_ok=True)

    pd.DataFrame(all_oos_trades).to_csv(os.path.join(out, "wf_oos_trades.csv"), index=False)
    params_df.to_csv(os.path.join(out, "wf_chosen_params.csv"), index=False)

    lines = [
        "WALK-FORWARD OPTIMIZATION REPORT",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Universe: {len(universe)} symbols, {years}yr daily",
        f"Folds: {len(fold_results)} (train={TRAIN_BARS}d, test={TEST_BARS}d)",
        f"Param grid: {len(param_grid)} per fold",
        "",
        "AGGREGATE OOS:",
        f"  ALL:       N={a['n']}  Sh={a['sharpe']}  W={a['win']}%  Avg={a['avg']:+.3f}%  PF={a['pf']}",
        f"  PRIMARY:   N={p['n']}  Sh={p['sharpe']}  W={p['win']}%  Avg={p['avg']:+.3f}%  PF={p['pf']}",
        f"  SECONDARY: N={s['n']}  Sh={s['sharpe']}  W={s['win']}%  Avg={s['avg']:+.3f}%  PF={s['pf']}",
        "",
    ]
    for f in fold_results:
        o = f["oos_all"]; bp = f["best_params"]
        lines.append(f"Fold {f['fold']}: {f['test_dates']}  OOS N={o['n']} Sh={o['sharpe']} W={o['win']}% Avg={o['avg']:+.3f}%")
        lines.append(f"  Best: {param_label(bp)}")
        lines.append(f"  IS: N={f['is_stats']['n']} Sh={f['is_stats']['sharpe']} W={f['is_stats']['win']}%")
        lines.append("")

    if fold_sh:
        lines.append("STABILITY:")
        lines.append(f"  Positive folds: {pos}/{len(fold_sh)}")
        lines.append(f"  Sharpe: {min(fold_sh):.2f} – {max(fold_sh):.2f} (median {np.median(fold_sh):.2f})")
        if avg_is_sharpe > 0:
            lines.append(f"  WF Efficiency: {wfe:.0f}%")

    lines.append("\nPARAMETER STABILITY:")
    for col in ["wr_period","wr_thresh","sma_period","mfi_min","dd_thresh","streak_min","profit_target","max_hold"]:
        vals = params_df[col].values
        lines.append(f"  {col}: {list(vals)}")

    with open(os.path.join(out, "wf_report.txt"), "w") as f:
        f.write("\n".join(lines))

    logger.info("\nSaved to %s", out)
    logger.info("Complete in %.1f min", (time.time()-t0)/60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--n", type=int, default=70)
    p.add_argument("--years", type=int, default=5)
    a = p.parse_args()
    run(use_live=a.live, n_symbols=a.n, years=a.years)
