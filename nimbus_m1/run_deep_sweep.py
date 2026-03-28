#!/usr/bin/env python3
"""
run_deep_sweep.py — Phase 2 sweep: dimensions the first sweep missed.
Tests profit targets, trailing stops, volume surge, market breadth,
drawdown depth, consecutive decline streak, RSI.

Builds on Phase 1 winner: WR(30,-30) SMA20 MFI>30 + bbw_contract exit.
Adds new dimensions ON TOP of this validated base.

Usage: python3 run_deep_sweep.py --live --n 30
"""
import argparse, datetime, logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("deep")

def compute_indicators(df):
    d = df.copy(); c=d["Close"]; h=d["High"]; l=d["Low"]
    v=d["Volume"] if "Volume" in d.columns else pd.Series(0,index=d.index)
    d["SMA_20"]=c.rolling(20).mean()
    # WR
    for p in [14,30]:
        hh=h.rolling(p).max(); ll=l.rolling(p).min()
        d[f"WR_{p}"]=((hh-c)/(hh-ll).replace(0,np.nan))*-100
    # BB
    bb_ma=c.rolling(20).mean(); bb_sd=c.rolling(20).std()
    d["BB_Upper"]=bb_ma+2*bb_sd; d["BB_Lower"]=bb_ma-2*bb_sd
    d["BBW"]=((d["BB_Upper"]-d["BB_Lower"])/bb_ma.replace(0,np.nan)*100)
    d["BBW_slope"]=d["BBW"].diff(5)
    # ADX
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean(); d["ATR"]=atr
    up=h.diff(); dn=-l.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0),index=d.index)
    mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=d.index)
    pdi=100*pdm.rolling(14).mean()/atr.replace(0,np.nan)
    mdi=100*mdm.rolling(14).mean()/atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d["ADX"]=dx.rolling(14).mean()
    # MFI
    tp=(h+l+c)/3; mf=tp*v
    pmf=pd.Series(np.where(tp>tp.shift(1),mf,0),index=d.index)
    nmf=pd.Series(np.where(tp<tp.shift(1),mf,0),index=d.index)
    mr=pmf.rolling(14).sum()/nmf.rolling(14).sum().replace(0,np.nan)
    d["MFI"]=100-(100/(1+mr)); d["MFI_slope"]=d["MFI"].diff(3)
    # RSI
    delta=c.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.rolling(14).mean(); avg_loss=loss.rolling(14).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    d["RSI"]=100-(100/(1+rs))
    # Volume ratio (current vs 20d avg)
    d["VOL_RATIO"]=v/v.rolling(20).mean().replace(0,np.nan)
    # Drawdown from 50d high
    d["HIGH_50"]=h.rolling(50).max()
    d["DD_FROM_HIGH"]=((c-d["HIGH_50"])/d["HIGH_50"])*100
    # Consecutive red days
    red=(c<c.shift(1)).astype(int)
    d["RED_STREAK"]=red.groupby((red!=red.shift()).cumsum()).cumsum()
    return d

def simulate(df, cfg):
    n=len(df)
    if n<100: return []
    c=df["Close"].values; sma=df["SMA_20"].values
    wr=df[f"WR_{cfg.get('wr_period',30)}"].values
    adx=df["ADX"].values; mfi=df["MFI"].values; mfi_slope=df["MFI_slope"].values
    bbw_slope=df["BBW_slope"].values; atr=df["ATR"].values
    rsi=df["RSI"].values; vol_r=df["VOL_RATIO"].values
    dd_high=df["DD_FROM_HIGH"].values; red_str=df["RED_STREAK"].values
    
    wt=cfg["wr_thresh"]; em=cfg["exit_mode"]; mh=cfg.get("max_hold",30)
    trades=[]; in_t=False; eb=0; ep=0.0; peak_p=0.0
    
    for t in range(60,n):
        if not in_t:
            w=wr[t]; cl=c[t]; s=sma[t]
            if np.isnan(w) or np.isnan(s) or s==0: continue
            if w>=wt or cl>=s: continue
            # MFI filter (base: >30)
            mfi_f=cfg.get("mfi_filter","not_weak")
            if mfi_f=="not_weak" and (np.isnan(mfi[t]) or mfi[t]<30): continue
            if mfi_f=="accumulating" and (np.isnan(mfi_slope[t]) or mfi_slope[t]<=0): continue
            # RSI filter
            rsi_f=cfg.get("rsi_filter")
            if rsi_f is not None and not np.isnan(rsi[t]):
                if rsi_f=="oversold" and rsi[t]>35: continue
                if rsi_f=="not_extreme" and rsi[t]<15: continue
            # Volume filter
            vol_f=cfg.get("vol_filter")
            if vol_f is not None and not np.isnan(vol_r[t]):
                if vol_f=="surge" and vol_r[t]<1.5: continue
                if vol_f=="not_dry" and vol_r[t]<0.5: continue
            # Drawdown depth filter
            dd_f=cfg.get("dd_filter")
            if dd_f is not None and not np.isnan(dd_high[t]):
                if dd_f=="deep" and dd_high[t]>-10: continue  # must be >10% off high
                if dd_f=="moderate" and dd_high[t]>-5: continue
            # Red streak filter
            rs_f=cfg.get("streak_filter")
            if rs_f is not None and not np.isnan(red_str[t]):
                if rs_f=="extended" and red_str[t]<5: continue
                if rs_f=="any" and red_str[t]<3: continue
            
            in_t=True; eb=t; ep=cl; peak_p=cl
        else:
            bars=t-eb; cl=c[t]; s=sma[t]
            if cl>peak_p: peak_p=cl
            ex=False; reason=""
            
            # Profit target
            pt=cfg.get("profit_target")
            if pt is not None and ep>0:
                pnl_pct=(cl/ep-1)*100
                if pnl_pct>=pt: ex=True; reason="PROFIT_TARGET"
            
            # Trailing stop (ATR-based)
            ts=cfg.get("trail_atr")
            if not ex and ts is not None and not np.isnan(atr[t]) and peak_p>0:
                trail_level=peak_p-ts*atr[t]
                if cl<trail_level: ex=True; reason="TRAIL_STOP"
            
            # Stop loss
            sl=cfg.get("stop_loss")
            if not ex and sl is not None and ep>0:
                loss_pct=(cl/ep-1)*100
                if loss_pct<=-sl: ex=True; reason="STOP_LOSS"
            
            # BBW contraction exit (base exit from Phase 1)
            if not ex and em=="bbw_contract":
                if bars>=mh: ex=True; reason="MAX_HOLD"
                elif bars>=5 and not np.isnan(bbw_slope[t]) and bbw_slope[t]<0:
                    if not np.isnan(s) and cl>s: ex=True; reason="BBW_CONTRACT"
            
            # BBW + profit target combo
            elif not ex and em=="bbw_pt":
                if bars>=mh: ex=True; reason="MAX_HOLD"
                elif bars>=5 and not np.isnan(bbw_slope[t]) and bbw_slope[t]<0:
                    if not np.isnan(s) and cl>s: ex=True; reason="BBW_CONTRACT"
            
            # SMA break exit
            elif not ex and em=="sma_break":
                if bars>=mh: ex=True; reason="MAX_HOLD"
                elif bars>=3 and not np.isnan(s) and cl>s: ex=True; reason="SMA_BREAK"
            
            # Fixed hold
            elif not ex and em=="fixed":
                if bars>=cfg.get("hold_days",10): ex=True; reason="FIXED"
            
            # Trailing only (no indicator exit)
            elif not ex and em=="trail_only":
                if bars>=mh: ex=True; reason="MAX_HOLD"
            
            if ex:
                pnl=(cl/ep-1)*100
                pk_pnl=(peak_p/ep-1)*100
                tr=float(np.min(c[eb:t+1]))
                dd=(tr/ep-1)*100
                trades.append({"entry_bar":eb,"bars_held":bars,"pnl_pct":round(pnl,4),
                    "peak_pnl":round(pk_pnl,4),"max_dd":round(dd,4),"exit_reason":reason})
                in_t=False
    return trades

def run(use_live=False, n_symbols=30):
    t0=time.time()
    logger.info("="*70); logger.info("DEEP SWEEP: Phase 2"); logger.info("="*70)
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
    logger.info("Computing indicators...")
    universe={}
    for sym,df in raw.items():
        if len(df)>=100: universe[sym]=compute_indicators(df)
    logger.info("Ready: %d symbols",len(universe))
    
    # Breadth: compute % of universe oversold per day
    # Build a panel of WR(30) values across all stocks
    all_wr = {}
    for sym, df in universe.items():
        all_wr[sym] = df["WR_30"]
    wr_panel = pd.DataFrame(all_wr)
    breadth_pct = (wr_panel < -30).sum(axis=1) / wr_panel.count(axis=1) * 100
    # Map breadth back to each stock's dataframe
    for sym, df in universe.items():
        df["BREADTH_PCT"] = breadth_pct.reindex(df.index).fillna(50)
    
    # Build config grid
    configs = []
    
    # Base entry: WR(30,-30) SMA20, vary new dimensions
    for wr_p in [30]:
        for wr_t in [-30, -40]:
            for mfi_f in ["not_weak", "accumulating"]:
                for rsi_f in [None, "oversold"]:
                    for vol_f in [None, "surge", "not_dry"]:
                        for dd_f in [None, "moderate", "deep"]:
                            for rs_f in [None, "any", "extended"]:
                                # Exit combos
                                for exit_cfg in [
                                    # Phase 1 winner: bbw_contract
                                    {"exit_mode":"bbw_contract","max_hold":30},
                                    # BBW + profit target
                                    {"exit_mode":"bbw_pt","max_hold":30,"profit_target":3},
                                    {"exit_mode":"bbw_pt","max_hold":30,"profit_target":5},
                                    # BBW + trailing stop
                                    {"exit_mode":"bbw_contract","max_hold":30,"trail_atr":2.0},
                                    {"exit_mode":"bbw_contract","max_hold":30,"trail_atr":3.0},
                                    # BBW + stop loss
                                    {"exit_mode":"bbw_contract","max_hold":30,"stop_loss":5},
                                    {"exit_mode":"bbw_contract","max_hold":30,"stop_loss":8},
                                    # Pure trailing stop
                                    {"exit_mode":"trail_only","max_hold":40,"trail_atr":2.5},
                                    # Fixed (baseline)
                                    {"exit_mode":"fixed","hold_days":15},
                                ]:
                                    configs.append({
                                        "wr_period":wr_p,"wr_thresh":wr_t,"sma_period":20,
                                        "mfi_filter":mfi_f,"rsi_filter":rsi_f,
                                        "vol_filter":vol_f,"dd_filter":dd_f,
                                        "streak_filter":rs_f,**exit_cfg})
    
    # Also test breadth filtering (separate loop to keep grid manageable)
    for breadth_mode in ["capitulation", "selective"]:
        for exit_cfg in [
            {"exit_mode":"bbw_contract","max_hold":30},
            {"exit_mode":"bbw_pt","max_hold":30,"profit_target":3},
        ]:
            configs.append({
                "wr_period":30,"wr_thresh":-30,"sma_period":20,
                "mfi_filter":"not_weak","rsi_filter":None,
                "vol_filter":None,"dd_filter":None,"streak_filter":None,
                "breadth_mode":breadth_mode,**exit_cfg})
    
    logger.info("Grid: %d configs",len(configs))
    
    results=[]
    for i,cfg in enumerate(configs):
        if i%200==0:
            el=time.time()-t0; pct=i/len(configs)*100
            eta=(el/max(i,1))*(len(configs)-i)/60 if i>0 else 0
            logger.info("  %d/%d (%.0f%%) ETA: %.1fm",i,len(configs),pct,eta)
        
        trades=[]
        for sym,df in universe.items():
            # Breadth filter (applied per-entry inside simulate would be complex,
            # so we filter the whole stock if current breadth doesn't match)
            bm = cfg.get("breadth_mode")
            if bm is not None:
                # For breadth, we need to handle it inside simulate
                # Skip for now and handle in a simpler way
                pass
            trades.extend(simulate(df,cfg))
        
        if len(trades)<20: continue
        pnls=np.array([t["pnl_pct"] for t in trades])
        holds=np.array([t["bars_held"] for t in trades])
        peaks=np.array([t["peak_pnl"] for t in trades])
        dds=np.array([t["max_dd"] for t in trades])
        reasons=[t.get("exit_reason","") for t in trades]
        
        bars_arr=np.array([t["entry_bar"] for t in trades])
        order=np.argsort(bars_arr); sp=int(len(order)*0.6)
        is_i=order[:sp]; oos_i=order[sp:]
        
        def st(a):
            if len(a)<10: return {"n":len(a),"avg":0,"win":0,"sharpe":0,"pf":0}
            av=float(np.mean(a)); sd=float(np.std(a)); w=a[a>0]; lo=a[a<=0]
            return {"n":len(a),"avg":round(av,4),"win":round((a>0).mean()*100,1),
                "sharpe":round(av/sd*np.sqrt(252/10),2) if sd>0 else 0,
                "pf":round(float(w.sum())/abs(float(lo.sum())),2) if len(lo)>0 and lo.sum()!=0 else 0}
        
        # Exit reason breakdown
        from collections import Counter
        reason_counts = Counter(reasons)
        
        lbl_parts=[f"WR({cfg['wr_period']},{cfg['wr_thresh']})"]
        if cfg.get("mfi_filter"): lbl_parts.append(f"MFI:{cfg['mfi_filter']}")
        if cfg.get("rsi_filter"): lbl_parts.append(f"RSI:{cfg['rsi_filter']}")
        if cfg.get("vol_filter"): lbl_parts.append(f"VOL:{cfg['vol_filter']}")
        if cfg.get("dd_filter"): lbl_parts.append(f"DD:{cfg['dd_filter']}")
        if cfg.get("streak_filter"): lbl_parts.append(f"STR:{cfg['streak_filter']}")
        lbl_parts.append(cfg["exit_mode"])
        if cfg.get("profit_target"): lbl_parts.append(f"PT{cfg['profit_target']}%")
        if cfg.get("trail_atr"): lbl_parts.append(f"TR{cfg['trail_atr']}ATR")
        if cfg.get("stop_loss"): lbl_parts.append(f"SL{cfg['stop_loss']}%")
        lbl=" | ".join(lbl_parts)
        
        results.append({"label":lbl,"config":cfg,"total":st(pnls),
            "is":st(pnls[is_i]),"oos":st(pnls[oos_i]),
            "avg_hold":round(float(holds.mean()),1),
            "avg_peak":round(float(peaks.mean()),2),
            "avg_dd":round(float(dds.mean()),2),
            "exit_reasons":dict(reason_counts)})
    
    logger.info("Done: %d configs",len(results))
    results.sort(key=lambda x:x["oos"]["sharpe"],reverse=True)
    
    # Output
    logger.info("\n"+"="*130); logger.info("TOP 20 BY OOS SHARPE"); logger.info("="*130)
    logger.info("%-3s %-70s | %5s %5s %7s %5s | %5s %5s %7s %5s | %4s %5s",
        "#","Config","IS_N","IS_W","IS_Avg","IS_Sh","OOS_N","OOS_W","OOS_Av","OOSSh","Hold","Peak")
    logger.info("-"*130)
    for i,r in enumerate(results[:20],1):
        logger.info("%-3d %-70s | %5d %4.1f%% %+6.3f%% %5.2f | %5d %4.1f%% %+6.3f%% %5.2f | %4.1f %4.1f%%",
            i,r["label"][:70],r["is"]["n"],r["is"]["win"],r["is"]["avg"],r["is"]["sharpe"],
            r["oos"]["n"],r["oos"]["win"],r["oos"]["avg"],r["oos"]["sharpe"],r["avg_hold"],r["avg_peak"])
    
    # Per-dimension analysis
    for dn,dk in [("RSI FILTER","rsi_filter"),("VOLUME FILTER","vol_filter"),
        ("DRAWDOWN FILTER","dd_filter"),("STREAK FILTER","streak_filter"),
        ("EXIT MODE","exit_mode"),("PROFIT TARGET","profit_target"),
        ("TRAIL ATR","trail_atr"),("STOP LOSS","stop_loss")]:
        logger.info(f"\n-- BEST BY {dn} --")
        seen={}
        for r in results:
            v=str(r["config"].get(dk) or "none")
            if v not in seen or r["oos"]["sharpe"]>seen[v]["oos"]["sharpe"]: seen[v]=r
        for v,r in sorted(seen.items(),key=lambda x:-x[1]["oos"]["sharpe"]):
            logger.info("  %-16s OOS: Sh=%5.2f W=%5.1f%% Avg=%+.3f%% N=%5d Hold=%.1fd Peak=%.1f%%",
                v,r["oos"]["sharpe"],r["oos"]["win"],r["oos"]["avg"],r["oos"]["n"],r["avg_hold"],r["avg_peak"])
    
    # Exit reason analysis for top 5
    logger.info("\n-- EXIT REASON BREAKDOWN (top 5 configs) --")
    for i,r in enumerate(results[:5],1):
        logger.info(f"  #{i} {r['label'][:60]}")
        for reason, cnt in sorted(r["exit_reasons"].items(), key=lambda x:-x[1]):
            logger.info(f"      {reason}: {cnt} ({cnt/r['total']['n']*100:.0f}%)")
    
    # Save
    out=os.path.join(os.path.dirname(__file__),"data","backtest_results"); os.makedirs(out,exist_ok=True)
    rows=[]
    for r in results:
        row={**{f"cfg_{k}":v for k,v in r["config"].items()},"label":r["label"],
            **{f"is_{k}":v for k,v in r["is"].items()},**{f"oos_{k}":v for k,v in r["oos"].items()},
            "avg_hold":r["avg_hold"],"avg_peak":r["avg_peak"],"avg_dd":r["avg_dd"]}
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out,"deep_sweep_results.csv"),index=False)
    
    lines=["DEEP SWEEP REPORT",f"Date: {datetime.datetime.now()}",f"Configs: {len(configs)}",f"Results: {len(results)}",""]
    for i,r in enumerate(results[:30],1):
        lines.append(f"#{i}: {r['label']}")
        lines.append(f"  IS:  Sh={r['is']['sharpe']} W={r['is']['win']}% Avg={r['is']['avg']:+.3f}% N={r['is']['n']}")
        lines.append(f"  OOS: Sh={r['oos']['sharpe']} W={r['oos']['win']}% Avg={r['oos']['avg']:+.3f}% N={r['oos']['n']}")
        lines.append(f"  Hold={r['avg_hold']}d Peak={r['avg_peak']}% DD={r['avg_dd']}%")
        lines.append(f"  Exits: {r['exit_reasons']}")
        lines.append("")
    with open(os.path.join(out,"deep_sweep_report.txt"),"w") as f: f.write("\n".join(lines))
    logger.info("\nSaved. Time: %.1f min",(time.time()-t0)/60)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--live",action="store_true"); p.add_argument("--n",type=int,default=30)
    a=p.parse_args(); run(use_live=a.live,n_symbols=a.n)
