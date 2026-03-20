#!/usr/bin/env python3
"""
run_etf_momentum.py — Parameter sweep + walk-forward for ETF momentum.

Hypothesis: ETFs (gold, commodities, sectoral) trend more than equities.
Entry: price above SMA, WR rising from oversold, ADX confirming trend.
Exit: trailing stop (ATR-based), SMA break, or BBW contraction.

Phase 1: Sweep entry/exit params on NSE ETFs
Phase 2: Walk-forward validation (18mo train, 6mo test)

Usage:
    python3 run_etf_momentum.py --live --years 5
"""
import argparse, datetime, logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("etf")

# ── NSE ETF Universe (grouped by category) ────────────────────────────────────
# Note: yfinance uses SYMBOL.NS for NSE ETFs
ETF_SYMBOLS = [
    # ── Gold (strongest trending category) ────────────────────────────────
    "GOLDBEES",          # Nippon Gold BeES — highest volume gold ETF
    "GOLDCASE",          # UTI Gold ETF
    "GOLDETF",           # SBI Gold ETF
    "GOLDSHARE",         # SBI Gold ETF (alternate)
    "IVZINGOLD",         # Invesco India Gold ETF

    # ── Silver ────────────────────────────────────────────────────────────
    "SILVERBEES",        # Nippon Silver BeES
    "SILVERIETF",        # ICICI Silver ETF

    # ── Broad Index ───────────────────────────────────────────────────────
    "NIFTYBEES",         # Nippon Nifty 50 BeES — most traded ETF
    "SETFNIF50",         # SBI Nifty 50 ETF — largest AUM
    "JUNIORBEES",        # Nippon Nifty Next 50
    "NIFTYIETF",         # ICICI Nifty 50
    "NV20IETF",          # ICICI Nifty 50 Value 20

    # ── Banking & Financial ───────────────────────────────────────────────
    "BANKBEES",          # Nippon Bank BeES
    "SETFNIFBK",         # SBI Nifty Bank
    "PSUBNKBEES",        # Nippon PSU Bank BeES
    "FINIETF",           # ICICI Financial Services

    # ── Sectoral (well-established) ───────────────────────────────────────
    "ITBEES",            # Nippon IT BeES
    "PHARMABEES",        # Nippon Pharma BeES
    "INFRABEES",         # Nippon Infra BeES
    "CPSEETF",           # CPSE ETF (public sector enterprises)
    "AUTOIETF",          # ICICI Auto ETF
    "FMCGIETF",          # ICICI FMCG ETF
    "HEALTHIETF",        # ICICI Healthcare ETF
    "CONSIETF",          # ICICI Consumer Durables

    # ── Commodities & Sectoral (newer but high volume) ────────────────────
    "COMMOIETF",         # ICICI Commodities (started Dec 2022)
    "METALIETF",         # ICICI Metal (started Aug 2024 — short history)
    "OILIETF",           # ICICI Oil & Gas (started Jul 2024 — short history)

    # ── Thematic / Factor ─────────────────────────────────────────────────
    "MOM100",            # Motilal Oswal Momentum 100
    "LOWVOLIETF",        # ICICI Low Volatility
    "ALPHAETF",          # ICICI Alpha Low Vol 30
    "MIDCAPIETF",        # ICICI Midcap 150
    "SMALLCAPIETF",      # ICICI Smallcap 250

    # ── International ─────────────────────────────────────────────────────
    "N100",              # Motilal Oswal Nasdaq 100
    "MON100",            # Motilal Oswal Nasdaq 100 (alternate)
    "MAFANG",            # Mirae Asset FANG+
    "HNGSNGBEES",        # Nippon Hang Seng BeES
    "MOM50",             # Motilal Oswal S&P 500

    # ── Debt / Liquid (benchmark, low trend) ──────────────────────────────
    "LIQUIDBEES",        # Nippon Liquid BeES
    "ITETF",             # ICICI 10yr Govt Bond
    "NETFGILT5Y",        # Nippon Gilt 5yr
]


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df):
    d = df.copy(); c = d["Close"]; h = d["High"]; l = d["Low"]
    v = d["Volume"] if "Volume" in d.columns else pd.Series(0, index=d.index)

    for p in [10, 20, 50]:
        d[f"SMA_{p}"] = c.rolling(p).mean()

    for p in [14, 20, 30]:
        hh = h.rolling(p).max(); ll = l.rolling(p).min()
        d[f"WR_{p}"] = ((hh - c) / (hh - ll).replace(0, np.nan)) * -100

    # ADX
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); d["ATR"] = atr
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=d.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=d.index)
    pdi = 100 * pdm.rolling(14).mean() / atr.replace(0, np.nan)
    mdi = 100 * mdm.rolling(14).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["ADX"] = dx.rolling(14).mean()

    # MFI
    tp = (h+l+c)/3; mf = tp*v
    pmf = pd.Series(np.where(tp>tp.shift(1), mf, 0), index=d.index)
    nmf = pd.Series(np.where(tp<tp.shift(1), mf, 0), index=d.index)
    mr = pmf.rolling(14).sum() / nmf.rolling(14).sum().replace(0, np.nan)
    d["MFI"] = 100 - (100 / (1 + mr))

    # BB
    bb_ma = c.rolling(20).mean(); bb_sd = c.rolling(20).std()
    d["BB_Upper"] = bb_ma + 2*bb_sd; d["BB_Lower"] = bb_ma - 2*bb_sd
    d["BBW"] = ((d["BB_Upper"] - d["BB_Lower"]) / bb_ma.replace(0, np.nan) * 100)
    d["BBW_slope"] = d["BBW"].diff(5)

    # Momentum-specific: rate of change
    d["ROC_10"] = c.pct_change(10) * 100
    d["ROC_20"] = c.pct_change(20) * 100

    # Above/below SMA streak
    for p in [20, 50]:
        above = (c > d[f"SMA_{p}"]).astype(int)
        d[f"ABOVE_SMA{p}_STREAK"] = above.groupby((above != above.shift()).cumsum()).cumsum()

    return d


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate(df, start_idx, end_idx, params):
    n = len(df)
    if n < 60: return []

    sma_col = f"SMA_{params['sma_period']}"
    wr_col = f"WR_{params['wr_period']}"
    if sma_col not in df.columns or wr_col not in df.columns: return []

    c = df["Close"].values; sma = df[sma_col].values; wr = df[wr_col].values
    adx = df["ADX"].values; mfi = df["MFI"].values; atr = df["ATR"].values
    bbw_slope = df["BBW_slope"].values
    roc = df[f"ROC_{params.get('roc_period', 10)}"].values if f"ROC_{params.get('roc_period', 10)}" in df.columns else df["ROC_10"].values

    entry_mode = params["entry_mode"]
    exit_mode = params["exit_mode"]
    trail_atr = params.get("trail_atr", 2.5)
    max_hold = params.get("max_hold", 40)

    trades = []; in_t = False; eb = 0; ep = 0.0; peak = 0.0

    for t in range(max(60, start_idx), min(end_idx, n)):
        if not in_t:
            cl = c[t]; s = sma[t]; w = wr[t]; a = adx[t]; m = mfi[t]; r = roc[t]
            if np.isnan(s) or np.isnan(w) or s == 0: continue

            entered = False

            if entry_mode == "sma_cross":
                # Price above SMA + WR rising from oversold (crossed -50 recently)
                if cl > s and w > params.get("wr_cross_level", -50):
                    # Check WR was below threshold within last N bars
                    lookback = min(t, params.get("cross_lookback", 10))
                    recent_wr = wr[t-lookback:t]
                    if len(recent_wr) > 0 and np.nanmin(recent_wr) < params.get("wr_cross_level", -50):
                        entered = True

            elif entry_mode == "breakout":
                # Price above SMA + ADX rising + ROC positive
                adx_min = params.get("adx_min", 20)
                if cl > s and not np.isnan(a) and a >= adx_min and not np.isnan(r) and r > 0:
                    entered = True

            elif entry_mode == "pullback_in_trend":
                # Above SMA(50) + WR dips below threshold then recovers
                sma50 = df[f"SMA_50"].values[t] if "SMA_50" in df.columns else np.nan
                if not np.isnan(sma50) and cl > sma50 and w > -30:
                    lookback = min(t, 10)
                    recent_wr = wr[t-lookback:t]
                    if len(recent_wr) > 0 and np.nanmin(recent_wr) < params.get("wr_dip_level", -50):
                        entered = True

            elif entry_mode == "mfi_momentum":
                # Above SMA + MFI > 50 (strong flow) + ADX > threshold
                adx_min = params.get("adx_min", 15)
                if cl > s and not np.isnan(m) and m > 50 and not np.isnan(a) and a >= adx_min:
                    entered = True

            if entered:
                in_t = True; eb = t; ep = cl; peak = cl

        else:
            bars = t - eb; cl = c[t]; s = sma[t]; ex = False; reason = ""
            if cl > peak: peak = cl

            # Exit: trailing stop
            if exit_mode in ("trail", "trail_sma"):
                if not np.isnan(atr[t]) and atr[t] > 0:
                    trail_level = peak - trail_atr * atr[t]
                    if cl < trail_level:
                        ex = True; reason = "TRAIL"

            # Exit: SMA break
            if not ex and exit_mode in ("sma_break", "trail_sma"):
                if not np.isnan(s) and cl < s and bars >= 3:
                    ex = True; reason = "SMA_BREAK"

            # Exit: BBW contraction (momentum fading)
            if not ex and exit_mode == "bbw_fade":
                if bars >= 5 and not np.isnan(bbw_slope[t]) and bbw_slope[t] < -0.5:
                    ex = True; reason = "BBW_FADE"

            # Exit: profit target
            pt = params.get("profit_target")
            if not ex and pt is not None and ep > 0:
                if (cl / ep - 1) * 100 >= pt:
                    ex = True; reason = "PT"

            # Exit: max hold
            if not ex and bars >= max_hold:
                ex = True; reason = "MAX"

            if ex:
                pnl = (cl / ep - 1) * 100
                trades.append({
                    "entry_bar": eb, "bars_held": bars, "pnl_pct": round(pnl, 4),
                    "peak_pnl": round((peak/ep-1)*100, 4),
                    "exit_reason": reason,
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
        "sharpe": round(avg / std * np.sqrt(252/15), 2) if std > 0 else 0,
        "pf": round(float(w.sum()) / abs(float(lo.sum())), 2) if len(lo) > 0 and lo.sum() != 0 else 99.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER GRID
# ══════════════════════════════════════════════════════════════════════════════

def build_param_grid():
    configs = []
    for entry_mode in ["sma_cross", "breakout", "pullback_in_trend", "mfi_momentum"]:
        for sma_p in [20, 50]:
            for wr_p in [14, 20]:
                for exit_mode in ["trail", "sma_break", "trail_sma", "bbw_fade"]:
                    for trail_atr in [2.0, 3.0]:
                        for max_hold in [30, 50]:
                            base = {
                                "entry_mode": entry_mode, "sma_period": sma_p,
                                "wr_period": wr_p, "exit_mode": exit_mode,
                                "trail_atr": trail_atr, "max_hold": max_hold,
                            }
                            if entry_mode == "sma_cross":
                                for cross_lvl in [-30, -50]:
                                    for lookback in [5, 10]:
                                        configs.append({**base, "wr_cross_level": cross_lvl,
                                                        "cross_lookback": lookback})
                            elif entry_mode == "breakout":
                                for adx_min in [15, 25]:
                                    base["roc_period"] = 10
                                    configs.append({**base, "adx_min": adx_min})
                            elif entry_mode == "pullback_in_trend":
                                for wr_dip in [-40, -60]:
                                    configs.append({**base, "wr_dip_level": wr_dip})
                            elif entry_mode == "mfi_momentum":
                                for adx_min in [15, 20]:
                                    configs.append({**base, "adx_min": adx_min})
    # Also test with profit targets
    for cfg in list(configs):
        for pt in [5.0, 10.0]:
            configs.append({**cfg, "profit_target": pt})
    return configs


def param_label(p):
    parts = [p["entry_mode"], f"SMA{p['sma_period']}", f"WR{p['wr_period']}",
             p["exit_mode"], f"TR{p['trail_atr']}ATR", f"MH{p['max_hold']}"]
    if p.get("profit_target"): parts.append(f"PT{p['profit_target']}%")
    if p.get("adx_min"): parts.append(f"ADX≥{p['adx_min']}")
    if p.get("wr_cross_level"): parts.append(f"Xcross{p['wr_cross_level']}")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(use_live=False, years=5):
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("ETF MOMENTUM: PARAMETER SWEEP + WALK-FORWARD")
    logger.info("=" * 70)

    # Load data
    if use_live:
        from backtest.data_loader import download_batch
        raw = download_batch(ETF_SYMBOLS, years=years)
    else:
        from backtest.data_loader import generate_universe
        raw = generate_universe(n_symbols=15, n_bars=int(years * 252))

    logger.info("Loaded %d ETFs", len(raw))

    # Precompute
    universe = {}
    for sym, df in raw.items():
        if len(df) >= 100:
            universe[sym] = compute_indicators(df)
    logger.info("Ready: %d ETFs", len(universe))
    for sym in sorted(universe.keys()):
        logger.info("  %-15s %d bars (%s → %s)", sym, len(universe[sym]),
                     universe[sym].index[0].date(), universe[sym].index[-1].date())

    if len(universe) < 3:
        logger.error("Need at least 3 ETFs with data")
        return

    # ── Phase 1: Full sweep (per-ETF split for different histories) ─────
    grid = build_param_grid()
    logger.info("Phase 1: Sweeping %d param combos across %d ETFs", len(grid), len(universe))

    sweep_results = []
    for i, params in enumerate(grid):
        if i % 500 == 0:
            el = time.time() - t0
            eta = (el / max(i, 1)) * (len(grid) - i) / 60 if i > 0 else 0
            logger.info("  Sweep: %d/%d (%.0f%%) ETA: %.1fm", i, len(grid), i/len(grid)*100, eta)

        is_trades = []; oos_trades = []
        for sym, df in universe.items():
            n = len(df)
            split = int(n * 0.6)
            is_trades.extend(simulate(df, 60, split, params))
            oos_trades.extend(simulate(df, split, n, params))

        if len(is_trades) < 10 or len(oos_trades) < 5: continue

        is_s = calc_stats([t["pnl_pct"] for t in is_trades])
        oos_s = calc_stats([t["pnl_pct"] for t in oos_trades])

        sweep_results.append({
            "params": params, "label": param_label(params),
            "is": is_s, "oos": oos_s,
            "avg_hold": round(np.mean([t["bars_held"] for t in is_trades + oos_trades]), 1),
        })

    sweep_results.sort(key=lambda x: x["oos"]["sharpe"], reverse=True)

    logger.info("\n" + "=" * 100)
    logger.info("PHASE 1: TOP 20 CONFIGS BY OOS SHARPE")
    logger.info("=" * 100)
    logger.info("%-3s %-55s | %5s %5s %7s %5s | %5s %5s %7s %5s | %4s",
                "#", "Config", "IS_N", "IS_W", "IS_Avg", "IS_Sh",
                "OOS_N", "OOS_W", "OOS_Av", "OOSSh", "Hold")
    logger.info("-" * 110)
    for i, r in enumerate(sweep_results[:20], 1):
        logger.info("%-3d %-55s | %5d %4.1f%% %+6.3f%% %5.2f | %5d %4.1f%% %+6.3f%% %5.2f | %4.1f",
                     i, r["label"][:55],
                     r["is"]["n"], r["is"]["win"], r["is"]["avg"], r["is"]["sharpe"],
                     r["oos"]["n"], r["oos"]["win"], r["oos"]["avg"], r["oos"]["sharpe"],
                     r["avg_hold"])

    # Per-dimension analysis
    for dim_name, dim_key in [("ENTRY MODE", "entry_mode"), ("EXIT MODE", "exit_mode"),
                               ("SMA PERIOD", "sma_period"), ("TRAIL ATR", "trail_atr")]:
        logger.info(f"\n── BEST BY {dim_name} ──")
        seen = {}
        for r in sweep_results:
            v = str(r["params"].get(dim_key, "none"))
            if v not in seen or r["oos"]["sharpe"] > seen[v]["oos"]["sharpe"]:
                seen[v] = r
        for v, r in sorted(seen.items(), key=lambda x: -x[1]["oos"]["sharpe"]):
            logger.info("  %-20s OOS: Sh=%5.2f W=%5.1f%% Avg=%+.3f%% N=%d",
                         v, r["oos"]["sharpe"], r["oos"]["win"], r["oos"]["avg"], r["oos"]["n"])

    # ── Phase 2: Walk-forward with top params ─────────────────────────────
    TRAIN_BARS = 378; TEST_BARS = 126
    MIN_BARS = TRAIN_BARS + 2 * TEST_BARS

    # Filter to ETFs with enough history for walk-forward
    wf_universe = {s: df for s, df in universe.items() if len(df) >= MIN_BARS}
    short = [s for s in universe if s not in wf_universe]
    if short:
        logger.info("WF: Dropping %d ETFs with < %d bars: %s", len(short), MIN_BARS, ", ".join(short))

    if len(wf_universe) < 3:
        logger.warning("Not enough ETFs with %d+ bars for walk-forward (%d available)", MIN_BARS, len(wf_universe))
    else:
        wf_max = min(len(df) for df in wf_universe.values())
        n_folds = (wf_max - TRAIN_BARS) // TEST_BARS
        wf_grid = build_param_grid()

        logger.info("\n" + "=" * 100)
        logger.info("PHASE 2: WALK-FORWARD (%d folds, %d ETFs, %d params/fold)",
                     n_folds, len(wf_universe), len(wf_grid))
        logger.info("=" * 100)

        all_oos = []
        chosen_params = []

        for fold in range(n_folds):
            ts = fold * TEST_BARS; te = ts + TRAIN_BARS
            os_start = te; os_end = os_start + TEST_BARS
            if os_end > wf_max: break

            best_sh = -999; best_p = None
            for params in wf_grid:
                trades = []
                for sym, df in wf_universe.items():
                    trades.extend(simulate(df, ts, te, params))
                if len(trades) < 5: continue
                pnls = np.array([t["pnl_pct"] for t in trades])
                avg = float(np.mean(pnls)); std = float(np.std(pnls))
                sh = avg / std * np.sqrt(252/15) if std > 0 else 0
                if sh > best_sh: best_sh = sh; best_p = params.copy()

            if best_p is None: continue

            oos_trades = []
            for sym, df in wf_universe.items():
                trades = simulate(df, os_start, os_end, best_p)
                for t in trades: t["symbol"] = sym; t["fold"] = fold + 1
                oos_trades.extend(trades)

            oos_s = calc_stats([t["pnl_pct"] for t in oos_trades])
            sample_df = list(wf_universe.values())[0]
            test_dates = f"{sample_df.index[os_start].date()} → {sample_df.index[min(os_end-1, len(sample_df)-1)].date()}"

            logger.info("  Fold %d: %s | OOS N=%d Sh=%.2f W=%.1f%% Avg=%+.3f%% | %s",
                         fold+1, test_dates, oos_s["n"], oos_s["sharpe"], oos_s["win"],
                         oos_s["avg"], best_p["entry_mode"])

            all_oos.extend(oos_trades)
            chosen_params.append({"fold": fold+1, **best_p})

        if all_oos:
            agg = calc_stats([t["pnl_pct"] for t in all_oos])
            logger.info("\n── AGGREGATE WF OOS ──")
            logger.info("  N=%d  Sh=%.2f  W=%.1f%%  Avg=%+.3f%%  PF=%.2f",
                         agg["n"], agg["sharpe"], agg["win"], agg["avg"], agg["pf"])

            # Parameter stability
            logger.info("\n── PARAMETER STABILITY ──")
            pdf = pd.DataFrame(chosen_params)
            for col in ["entry_mode", "exit_mode", "sma_period", "trail_atr", "max_hold"]:
                if col in pdf.columns:
                    vals = pdf[col].values
                    from collections import Counter
                    counts = Counter(vals)
                    most = counts.most_common(1)[0]
                    logger.info("  %-14s: %s (mode=%s in %d/%d)",
                                 col, dict(counts), most[0], most[1], len(vals))

    # ── Save ──────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), "data", "backtest_results")
    os.makedirs(out, exist_ok=True)

    lines = [
        "ETF MOMENTUM RESEARCH REPORT",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"ETFs: {len(universe)}, {years}yr daily",
        "",
        "TOP 10 CONFIGS (Phase 1 sweep):",
    ]
    for i, r in enumerate(sweep_results[:10], 1):
        lines.append(f"  #{i}: {r['label']}")
        lines.append(f"    IS:  N={r['is']['n']} Sh={r['is']['sharpe']} W={r['is']['win']}%")
        lines.append(f"    OOS: N={r['oos']['n']} Sh={r['oos']['sharpe']} W={r['oos']['win']}% Avg={r['oos']['avg']:+.3f}%")
        lines.append("")

    with open(os.path.join(out, "etf_momentum_report.txt"), "w") as f:
        f.write("\n".join(lines))

    if all_oos:
        pd.DataFrame(all_oos).to_csv(os.path.join(out, "etf_momentum_trades.csv"), index=False)
        pd.DataFrame(chosen_params).to_csv(os.path.join(out, "etf_momentum_params.csv"), index=False)

    logger.info("\nSaved to %s", out)
    logger.info("Complete in %.1f min", (time.time()-t0)/60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--years", type=int, default=5)
    a = p.parse_args()
    run(use_live=a.live, years=a.years)
