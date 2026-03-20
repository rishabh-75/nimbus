#!/usr/bin/env python3
"""
run_full_sweep.py — Comprehensive parameter sweep (optimized)
Precomputes indicators once per stock, then sweeps thresholds fast.
Usage: python3 run_full_sweep.py --live --n 30
"""
import argparse, datetime, logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("sweep")

def compute_all_indicators(df):
    d = df.copy(); c = d["Close"]; h = d["High"]; l = d["Low"]
    v = d["Volume"] if "Volume" in d.columns else pd.Series(0, index=d.index)
    for p in [10, 20, 50]: d[f"SMA_{p}"] = c.rolling(p).mean()
    for p in [8, 14, 20, 30]:
        hh = h.rolling(p).max(); ll = l.rolling(p).min()
        d[f"WR_{p}"] = ((hh - c) / (hh - ll).replace(0, np.nan)) * -100
    bb_ma = c.rolling(20).mean(); bb_sd = c.rolling(20).std()
    d["BB_Upper"] = bb_ma + 2.0 * bb_sd; d["BB_Lower"] = bb_ma - 2.0 * bb_sd
    d["BBW"] = ((d["BB_Upper"] - d["BB_Lower"]) / bb_ma.replace(0, np.nan) * 100)
    d["BBW_pctl"] = d["BBW"].rolling(100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    d["BBW_slope"] = d["BBW"].diff(5)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(np.where((up>dn)&(up>0),up,0), index=d.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0),dn,0), index=d.index)
    pdi = 100*pdm.rolling(14).mean()/atr.replace(0,np.nan)
    mdi = 100*mdm.rolling(14).mean()/atr.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d["ADX"] = dx.rolling(14).mean()
    tp = (h+l+c)/3; mf = tp*v
    pmf = pd.Series(np.where(tp>tp.shift(1),mf,0), index=d.index)
    nmf = pd.Series(np.where(tp<tp.shift(1),mf,0), index=d.index)
    mr = pmf.rolling(14).sum()/nmf.rolling(14).sum().replace(0,np.nan)
    d["MFI"] = 100-(100/(1+mr)); d["MFI_slope"] = d["MFI"].diff(3)
    return d

def simulate_config(df, cfg):
    n = len(df)
    if n < 80: return []
    wr_col = f"WR_{cfg['wr_period']}"; sma_col = f"SMA_{cfg['sma_period']}"
    if wr_col not in df.columns or sma_col not in df.columns: return []
    wr=df[wr_col].values; sma=df[sma_col].values; closes=df["Close"].values
    adx=df["ADX"].values; mfi=df["MFI"].values; mfi_slope=df["MFI_slope"].values
    bbw_pctl=df["BBW_pctl"].values; bbw_slope=df["BBW_slope"].values
    wt=cfg["wr_thresh"]; af=cfg.get("adx_filter"); am=cfg.get("adx_mode")
    mf=cfg.get("mfi_filter"); bf=cfg.get("bbw_filter")
    em=cfg["exit_mode"]; hd=cfg.get("hold_days",10); mh=cfg.get("max_hold",25)
    trades=[]; in_t=False; eb=0; ep=0.0
    for t in range(60, n):
        if not in_t:
            w=wr[t]; c=closes[t]; s=sma[t]
            if np.isnan(w) or np.isnan(s) or s==0: continue
            if w>=wt or c>=s: continue
            if af is not None and not np.isnan(adx[t]):
                if am=="below" and adx[t]>af: continue
                elif am=="above" and adx[t]<af: continue
            if mf is not None and not np.isnan(mfi[t]):
                if mf=="not_weak" and mfi[t]<30: continue
                elif mf=="strong" and mfi[t]<50: continue
                elif mf=="accumulating" and mfi_slope[t]<=0: continue
            if bf is not None and not np.isnan(bbw_pctl[t]):
                if bf=="squeeze" and bbw_pctl[t]>30: continue
                elif bf=="not_expanded" and bbw_pctl[t]>70: continue
            in_t=True; eb=t; ep=c
        else:
            bars=t-eb; c=closes[t]; s=sma[t]; ex=False
            if em=="fixed": ex=bars>=hd
            elif em=="sma_break":
                if bars>=mh: ex=True
                elif bars>=3 and not np.isnan(s) and c>s: ex=True
            elif em=="bbw_contract":
                if bars>=mh: ex=True
                elif bars>=5 and not np.isnan(bbw_slope[t]) and bbw_slope[t]<0 and not np.isnan(s) and c>s: ex=True
            elif em=="mfi_exit":
                if bars>=mh: ex=True
                elif bars>=5:
                    if not np.isnan(mfi[t]) and mfi[t]<30: ex=True
                    elif not np.isnan(mfi_slope[t]) and mfi_slope[t]<-15: ex=True
            elif em=="adaptive":
                if bars>=mh: ex=True
                elif bars>=5:
                    bs=not np.isnan(s) and c<s; bf2=not np.isnan(bbw_slope[t]) and bbw_slope[t]<0
                    mw=not np.isnan(mfi[t]) and mfi[t]<35
                    if bs and (bf2 or mw): ex=True
                    pnl=(c/ep-1)*100
                    if pnl>3 and not np.isnan(s) and c>s: ex=True
            elif em=="momentum_ride":
                if bars>=mh: ex=True
                elif bars>=5:
                    if not np.isnan(s) and c<s: ex=True
                    elif bars>=10 and not np.isnan(bbw_slope[t]) and bbw_slope[t]<-0.5: ex=True
            if ex:
                pnl=(c/ep-1)*100; pk=float(np.max(closes[eb:t+1])); tr=float(np.min(closes[eb:t+1]))
                trades.append({"entry_bar":eb,"bars_held":bars,"pnl_pct":round(pnl,4),
                    "peak_pnl":round((pk/ep-1)*100,4),"max_dd":round((tr/ep-1)*100,4)})
                in_t=False
    return trades

def run(use_live=False, n_symbols=30):
    t0=time.time()
    logger.info("="*70); logger.info("COMPREHENSIVE PARAMETER SWEEP"); logger.info("="*70)
    if use_live:
        from backtest.data_loader import download_batch
        try:
            from modules.data import NIFTY100_SYMBOLS; symbols=NIFTY100_SYMBOLS[:n_symbols]
        except: symbols=["RELIANCE","HDFCBANK","TCS","INFY","ICICIBANK","SBIN","AXISBANK",
            "BAJFINANCE","TATAMOTORS","MARUTI","TATASTEEL","JSWSTEEL","HINDALCO","WIPRO",
            "TECHM","SUNPHARMA","CIPLA","LT","NTPC","BHARTIARTL","ITC","HINDUNILVR",
            "ASIANPAINT","DRREDDY","KOTAKBANK","BAJAJ-AUTO","HEROMOTOCO","M&M","BPCL","COALINDIA"][:n_symbols]
        raw=download_batch(symbols, years=3)
    else:
        from backtest.data_loader import generate_universe; raw=generate_universe(n_symbols=n_symbols,n_bars=750)
    logger.info("Loaded %d symbols",len(raw))
    logger.info("Precomputing indicators...")
    universe={}
    for sym,df in raw.items():
        if len(df)>=100: universe[sym]=compute_all_indicators(df)
    logger.info("Indicators ready for %d symbols",len(universe))
    configs=[]
    for wr_p in [14,20,30]:
      for wr_t in [-30,-40,-50]:
        for sma_p in [10,20]:
          for mfi_f in [None,"not_weak","accumulating"]:
            for bbw_f in [None,"squeeze","not_expanded"]:
              for adx_v,adx_m in [(None,None),(25,"below"),(20,"above")]:
                for exit_cfg in [
                    {"exit_mode":"fixed","hold_days":10},{"exit_mode":"fixed","hold_days":15},
                    {"exit_mode":"fixed","hold_days":20},{"exit_mode":"sma_break","max_hold":25},
                    {"exit_mode":"bbw_contract","max_hold":30},{"exit_mode":"mfi_exit","max_hold":25},
                    {"exit_mode":"adaptive","max_hold":30},{"exit_mode":"momentum_ride","max_hold":40}]:
                  configs.append({"wr_period":wr_p,"wr_thresh":wr_t,"sma_period":sma_p,
                    "mfi_filter":mfi_f,"bbw_filter":bbw_f,"adx_filter":adx_v,"adx_mode":adx_m,**exit_cfg})
    logger.info("Grid: %d configs x %d stocks",len(configs),len(universe))
    results=[]
    for i,cfg in enumerate(configs):
        if i%200==0:
            el=time.time()-t0; pct=i/len(configs)*100; eta=(el/max(i,1))*(len(configs)-i)/60
            logger.info("  %d/%d (%.0f%%) ETA: %.1fm",i,len(configs),pct,eta)
        trades=[]
        for sym,df in universe.items(): trades.extend(simulate_config(df,cfg))
        if len(trades)<20: continue
        pnls=np.array([t["pnl_pct"] for t in trades]); holds=np.array([t["bars_held"] for t in trades])
        peaks=np.array([t["peak_pnl"] for t in trades]); dds=np.array([t["max_dd"] for t in trades])
        bars_arr=np.array([t["entry_bar"] for t in trades]); order=np.argsort(bars_arr)
        sp=int(len(order)*0.6); is_i=order[:sp]; oos_i=order[sp:]
        def st(a):
            if len(a)<10: return {"n":len(a),"avg":0,"win":0,"sharpe":0,"pf":0}
            av=float(np.mean(a)); sd=float(np.std(a)); w=a[a>0]; lo=a[a<=0]
            return {"n":len(a),"avg":round(av,4),"win":round((a>0).mean()*100,1),
                "sharpe":round(av/sd*np.sqrt(252/10),2) if sd>0 else 0,
                "pf":round(float(w.sum())/abs(float(lo.sum())),2) if len(lo)>0 and lo.sum()!=0 else 0}
        lbl=(f"WR({cfg['wr_period']},{cfg['wr_thresh']}) SMA{cfg['sma_period']}"
            +(f" MFI:{cfg['mfi_filter']}" if cfg.get("mfi_filter") else "")
            +(f" BBW:{cfg['bbw_filter']}" if cfg.get("bbw_filter") else "")
            +(f" ADX{cfg['adx_mode'][0]}{cfg['adx_filter']}" if cfg.get("adx_filter") else "")
            +f" {cfg['exit_mode']}"+(f"{cfg.get('hold_days','')}" if cfg['exit_mode']=='fixed' else ""))
        results.append({"label":lbl,"config":cfg,"total":st(pnls),"is":st(pnls[is_i]),"oos":st(pnls[oos_i]),
            "avg_hold":round(float(holds.mean()),1),"avg_peak":round(float(peaks.mean()),2),"avg_dd":round(float(dds.mean()),2)})
    logger.info("Done: %d configs with results",len(results))
    results.sort(key=lambda x:x["oos"]["sharpe"],reverse=True)
    logger.info("\n"+"="*130); logger.info("TOP 20 BY OOS SHARPE"); logger.info("="*130)
    logger.info("%-3s %-65s | %5s %5s %7s %5s | %5s %5s %7s %5s | %4s %5s",
        "#","Config","IS_N","IS_W","IS_Avg","IS_Sh","OOS_N","OOS_W","OOS_Av","OOSSh","Hold","Peak")
    logger.info("-"*130)
    for i,r in enumerate(results[:20],1):
        logger.info("%-3d %-65s | %5d %4.1f%% %+6.3f%% %5.2f | %5d %4.1f%% %+6.3f%% %5.2f | %4.1f %4.1f%%",
            i,r["label"][:65],r["is"]["n"],r["is"]["win"],r["is"]["avg"],r["is"]["sharpe"],
            r["oos"]["n"],r["oos"]["win"],r["oos"]["avg"],r["oos"]["sharpe"],r["avg_hold"],r["avg_peak"])
    for dn,dk in [("EXIT MODE","exit_mode"),("MFI FILTER","mfi_filter"),("BBW FILTER","bbw_filter"),
        ("WR PERIOD","wr_period"),("WR THRESH","wr_thresh"),("SMA PERIOD","sma_period")]:
        logger.info(f"\n-- BEST BY {dn} --")
        seen={}
        for r in results:
            v=str(r["config"].get(dk) or "none")
            if v not in seen or r["oos"]["sharpe"]>seen[v]["oos"]["sharpe"]: seen[v]=r
        for v,r in sorted(seen.items(),key=lambda x:-x[1]["oos"]["sharpe"]):
            logger.info("  %-16s OOS: Sh=%5.2f W=%5.1f%% Avg=%+.3f%% N=%5d Hold=%.1fd",
                v,r["oos"]["sharpe"],r["oos"]["win"],r["oos"]["avg"],r["oos"]["n"],r["avg_hold"])
    logger.info("\n-- BEST BY ADX --")
    adx_s={}
    for r in results:
        af=r["config"].get("adx_filter"); am=r["config"].get("adx_mode")
        k=f"{am}_{af}" if af else "none"
        if k not in adx_s or r["oos"]["sharpe"]>adx_s[k]["oos"]["sharpe"]: adx_s[k]=r
    for v,r in sorted(adx_s.items(),key=lambda x:-x[1]["oos"]["sharpe"]):
        logger.info("  %-16s OOS: Sh=%5.2f W=%5.1f%% Avg=%+.3f%% N=%5d",
            v,r["oos"]["sharpe"],r["oos"]["win"],r["oos"]["avg"],r["oos"]["n"])
    out=os.path.join(os.path.dirname(__file__),"data","backtest_results"); os.makedirs(out,exist_ok=True)
    rows=[]
    for r in results:
        row={**{f"cfg_{k}":v for k,v in r["config"].items()},"label":r["label"],
            **{f"is_{k}":v for k,v in r["is"].items()},**{f"oos_{k}":v for k,v in r["oos"].items()},
            "avg_hold":r["avg_hold"],"avg_peak":r["avg_peak"],"avg_dd":r["avg_dd"]}
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out,"full_sweep_results.csv"),index=False)
    lines=["FULL SWEEP REPORT",f"Date: {datetime.datetime.now()}",f"Configs: {len(configs)}",f"Results: {len(results)}",""]
    for i,r in enumerate(results[:30],1):
        lines.append(f"#{i}: {r['label']}")
        lines.append(f"  IS:  Sh={r['is']['sharpe']} W={r['is']['win']}% Avg={r['is']['avg']:+.3f}% N={r['is']['n']}")
        lines.append(f"  OOS: Sh={r['oos']['sharpe']} W={r['oos']['win']}% Avg={r['oos']['avg']:+.3f}% N={r['oos']['n']}")
        lines.append(f"  Hold={r['avg_hold']}d Peak={r['avg_peak']}% DD={r['avg_dd']}%")
        lines.append("")
    with open(os.path.join(out,"full_sweep_report.txt"),"w") as f: f.write("\n".join(lines))
    logger.info("\nSaved to %s",out); logger.info("Time: %.1f min",(time.time()-t0)/60)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--live",action="store_true"); p.add_argument("--n",type=int,default=30)
    a=p.parse_args(); run(use_live=a.live,n_symbols=a.n)
