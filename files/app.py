"""
NSE Trading Signal Dashboard
─────────────────────────────
Options Wall + Institutional Activity Signal Generator
"""

from __future__ import annotations
import sys
import time
import datetime
import threading
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from options_wall import OptionsWallCalculator
from insider_detector import InsiderWallDetector
from signal_engine import SignalEngine
from data_manager import DataManager
from alert_manager import AlertManager
from trade_recommender import TradeRecommender
import chart_builder as cb

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Dark base ── */
html, body, [class*="css"] {
    background-color: #0d0d1a !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp { background-color: #0d0d1a; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d24 !important;
    border-right: 1px solid #1e1e3a;
}

/* ── KPI Cards ── */
.kpi-card {
    background: #14142a;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    position: relative;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #e0e0e0; }
.kpi-label { font-size: 0.78rem; color: #9e9e9e; margin-bottom: 4px; letter-spacing: 0.05em; }
.kpi-delta { font-size: 0.82rem; margin-top: 4px; }
.kpi-tooltip {
    font-size: 0.7rem; color: #616161; margin-top: 6px;
    font-style: italic; line-height: 1.3;
}

/* ── Signal Cards ── */
.signal-card {
    background: #14142a;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #1e1e3a;
    height: 100%;
}
.signal-card h4 { margin-bottom: 10px; }

/* ── Master Signal Card ── */
.master-card {
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    border: 2px solid;
}
.master-card h2 { font-size: 2rem; margin-bottom: 8px; }
.master-card .key-reason { font-size: 1rem; opacity: 0.85; margin-bottom: 16px; }
.master-card ul { list-style: none; padding: 0; margin: 0; }
.master-card li { font-size: 0.9rem; opacity: 0.85; margin-bottom: 6px; }
.master-card li::before { content: "• "; color: inherit; }

/* ── Tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d0d1a;
    border-bottom: 1px solid #1e1e3a;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #14142a;
    border-radius: 8px 8px 0 0;
    color: #9e9e9e;
    padding: 8px 20px;
    font-size: 0.88rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #1e1e3a !important;
    color: #00e676 !important;
    border-bottom: 2px solid #00e676;
}

/* ── Freshness Badge ── */
.freshness { font-size: 0.78rem; padding: 3px 10px; border-radius: 20px; }
.demo-banner {
    background: rgba(255,215,64,0.15);
    border: 1px solid #ffd740;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: #ffd740;
    margin-bottom: 12px;
}

/* ── Buttons ── */
.stButton > button {
    background: #1e1e3a;
    color: #00e676;
    border: 1px solid #00e676;
    border-radius: 8px;
    font-size: 0.85rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00e676;
    color: #0d0d1a;
}

/* ── Metric deltas ── */
[data-testid="stMetricDelta"] { font-size: 0.78rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; color: #e0e0e0; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── session state defaults ───────────────────────────────────────────────────


def init_session():
    defaults = {
        "symbol": "SBIN",
        "date": datetime.date.today(),
        "wall_pct": 75,
        "proximity_pct": 1.5,
        "min_score": 40,
        "score_alert_threshold": 70,
        "oi_spike_pct": 20.0,
        "entity_filter": [
            "FII/FPI",
            "DII/MF",
            "Promoter/Insider",
            "Institutional Broker",
        ],
        "recent_symbols": ["SBIN", "NIFTY", "BANKNIFTY", "RELIANCE", "TCS"],
        "last_refresh": None,
        "options_df": None,
        "deals_df": None,
        "walls_df": None,
        "matched_df": None,
        "zones_df": None,
        "call_walls": None,
        "put_walls": None,
        "pcr_data": None,
        "iv_skew": None,
        "max_pain": None,
        "key_levels": None,
        "cmp": None,
        "composite": None,
        "options_signal": None,
        "institutional_signal": None,
        "summary": None,
        "master_card": None,
        "is_demo": False,
        "countdown": 300,
        "alert_log": [],
        "prior_walls": None,
        "prior_pcr": None,
        "prior_signal": None,
        "prior_max_pain": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# ─── DataManager ─────────────────────────────────────────────────────────────
dm = DataManager(
    data_folder="./data",
    options_folder="./data/options",
    deals_folder="./data/deals",
    price_folder="./data/price",
)


# ─── compute pipeline ─────────────────────────────────────────────────────────


def run_compute_pipeline(
    options_uploaded=None,
    deals_uploaded=None,
    download_deals: bool = False,
):
    symbol = st.session_state["symbol"]
    date = st.session_state["date"]
    wall_pct = st.session_state["wall_pct"]
    proximity_pct = st.session_state["proximity_pct"]
    min_score = st.session_state["min_score"]
    entity_filter = st.session_state["entity_filter"]

    # 1. Load options
    options_df, is_demo_opt = dm.load_options_chain(
        symbol, date, uploaded_file=options_uploaded
    )
    st.session_state["options_df"] = options_df
    st.session_state["is_demo"] = is_demo_opt

    # 2. Load deals
    deals_df, is_demo_deal = dm.load_deals(
        symbol, uploaded_file=deals_uploaded, download=download_deals
    )
    if entity_filter and not deals_df.empty:
        if "Entity" in deals_df.columns:
            deals_df = deals_df[deals_df["Entity"].isin(entity_filter)]
    st.session_state["deals_df"] = deals_df
    if is_demo_deal:
        st.session_state["is_demo"] = True

    # 3. Options wall computation
    calc = OptionsWallCalculator(options_df)
    walls_df = calc.consolidate_walls()
    call_walls, put_walls = calc.identify_walls(pct=wall_pct)
    pcr_data = calc.analyze_pcr()
    iv_skew = calc.analyze_iv_skew()
    max_pain = calc.calculate_max_pain()

    # 4. CMP
    cmp = dm.get_cmp(symbol)
    key_levels = calc.identify_key_levels(cmp=cmp, pct=wall_pct)

    st.session_state.update(
        {
            "walls_df": walls_df,
            "call_walls": call_walls,
            "put_walls": put_walls,
            "pcr_data": pcr_data,
            "iv_skew": iv_skew,
            "max_pain": max_pain,
            "key_levels": key_levels,
            "cmp": cmp,
        }
    )

    # 5. Insider detection
    detector = InsiderWallDetector(walls_df, deals_df, proximity_pct=proximity_pct)
    matched_df = detector.match_deals_to_walls()
    zones_df = detector.detect_zones()
    summary = detector.get_summary()
    aggregated = detector.aggregate_by_level()

    if not matched_df.empty and min_score > 0:
        matched_df = matched_df[matched_df["Score"] >= min_score]

    st.session_state.update(
        {
            "matched_df": matched_df,
            "zones_df": zones_df,
            "summary": summary,
            "aggregated_df": aggregated,
        }
    )

    # 6. Signal engine
    engine = SignalEngine()
    options_signal = engine.compute_options_signal(
        walls_df, pcr_data["pcr_oi"], iv_skew, max_pain, cmp
    )
    institutional_signal = engine.compute_institutional_signal(zones_df, matched_df)
    iv_score = engine.iv_skew_score(iv_skew.get("skew", 0))
    composite = engine.composite_signal(
        options_signal["score"], institutional_signal["score"], iv_score
    )
    master_card = engine.build_master_card(
        composite,
        options_signal,
        institutional_signal,
        pcr_data["pcr_oi"],
        max_pain,
        cmp,
        summary,
    )

    st.session_state.update(
        {
            "options_signal": options_signal,
            "institutional_signal": institutional_signal,
            "composite": composite,
            "master_card": master_card,
        }
    )

    # 7. Alerts
    alert_mgr = AlertManager(
        score_threshold=st.session_state["score_alert_threshold"],
        oi_spike_pct=st.session_state["oi_spike_pct"],
    )
    alert_mgr.run_all_checks(
        matched_df=matched_df,
        walls_df=walls_df,
        pcr=pcr_data["pcr_oi"],
        max_pain=max_pain,
        signal_label=composite["label"],
    )

    st.session_state["last_refresh"] = time.time()


# ─── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    st.divider()

    # Symbol
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol_input = st.text_input(
            "Symbol", value=st.session_state["symbol"], key="symbol_input"
        ).upper()
    with col2:
        recent = st.selectbox(
            "Recent", st.session_state["recent_symbols"], label_visibility="collapsed"
        )
    if st.button("Set Symbol", use_container_width=True):
        st.session_state["symbol"] = symbol_input
        if symbol_input not in st.session_state["recent_symbols"]:
            st.session_state["recent_symbols"].insert(0, symbol_input)

    # Date
    sel_date = st.date_input("Date", value=st.session_state["date"])
    st.session_state["date"] = sel_date

    st.divider()
    st.markdown("**📊 Wall Parameters**")
    wall_pct = st.slider(
        "Wall OI Percentile", 50, 95, st.session_state["wall_pct"], step=5
    )
    st.session_state["wall_pct"] = wall_pct

    proximity = st.slider(
        "Proximity Tolerance (%)", 0.5, 3.0, st.session_state["proximity_pct"], step=0.1
    )
    st.session_state["proximity_pct"] = proximity

    min_score = st.slider(
        "Min Signal Score", 0, 100, st.session_state["min_score"], step=5
    )
    st.session_state["min_score"] = min_score

    st.divider()
    st.markdown("**🏦 Entity Filter**")
    all_entities = [
        "FII/FPI",
        "DII/MF",
        "Promoter/Insider",
        "Institutional Broker",
        "Retail/Unknown",
    ]
    selected_entities = []
    for e in all_entities:
        default = e in st.session_state["entity_filter"]
        if st.checkbox(e, value=default, key=f"chk_{e}"):
            selected_entities.append(e)
    st.session_state["entity_filter"] = selected_entities

    st.divider()
    st.markdown("**🔔 Alert Thresholds**")
    st.session_state["score_alert_threshold"] = st.slider(
        "Score Alert (≥)", 40, 100, st.session_state["score_alert_threshold"], step=5
    )
    st.session_state["oi_spike_pct"] = st.slider(
        "OI Spike Alert (%)", 5.0, 50.0, st.session_state["oi_spike_pct"], step=5.0
    )

    st.divider()
    st.markdown("**📁 Data Upload**")
    options_file = st.file_uploader("Options Chain CSV", type=["csv"], key="opt_upload")
    deals_file = st.file_uploader(
        "Bulk/Block Deals CSV", type=["csv"], key="deals_upload"
    )

    st.divider()
    st.markdown("**📂 Data Folder Path**")
    st.text_input("Options Folder", value="./data/options")

    st.divider()
    col_dl, col_ref = st.columns(2)
    with col_dl:
        download_nse = st.button("⬇️ NSE Deals", use_container_width=True)
    with col_ref:
        manual_refresh = st.button("🔄 Refresh", use_container_width=True)


# ─── header ──────────────────────────────────────────────────────────────────

header_col1, header_col2, header_col3 = st.columns([4, 2, 2])
with header_col1:
    st.markdown(f"# 📈 NSE Signal Dashboard — {st.session_state['symbol']}")
with header_col2:
    freshness = dm.freshness_badge(st.session_state.get("last_refresh"))
    st.markdown(f"**Data Freshness:** {freshness}")
with header_col3:
    if st.session_state.get("is_demo"):
        st.markdown(
            '<div class="demo-banner">⚠️ DEMO DATA — Load real CSVs to begin</div>',
            unsafe_allow_html=True,
        )

# ─── auto-load on first run ───────────────────────────────────────────────────

if (
    st.session_state["last_refresh"] is None
    or manual_refresh
    or options_file
    or deals_file
    or download_nse
):
    with st.spinner("🔄 Computing walls & signals…"):
        try:
            run_compute_pipeline(
                options_uploaded=options_file,
                deals_uploaded=deals_file,
                download_deals=download_nse,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

# ─── grab state refs ─────────────────────────────────────────────────────────

walls_df = st.session_state.get("walls_df", pd.DataFrame())
call_walls = st.session_state.get("call_walls", pd.DataFrame())
put_walls = st.session_state.get("put_walls", pd.DataFrame())
pcr_data = st.session_state.get("pcr_data", {})
iv_skew = st.session_state.get("iv_skew", {})
max_pain = st.session_state.get("max_pain", 0.0)
key_levels = st.session_state.get("key_levels", {})
cmp = st.session_state.get("cmp")
matched_df = st.session_state.get("matched_df", pd.DataFrame())
zones_df = st.session_state.get("zones_df", pd.DataFrame())
summary = st.session_state.get("summary", {})
aggregated_df = st.session_state.get("aggregated_df", pd.DataFrame())
options_signal = st.session_state.get("options_signal", {})
institutional_signal = st.session_state.get("institutional_signal", {})
composite = st.session_state.get("composite", {})
master_card = st.session_state.get("master_card", {})

# ─── tabs ─────────────────────────────────────────────────────────────────────

tabs = st.tabs(
    [
        "🏦 Wall Overview",
        "🕵️ Insider Activity",
        "🎯 Signal Dashboard",
        "💡 Trade Ideas",
        "📊 Historical",
        "⚙️ Settings",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Wall Overview
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    if walls_df is None or walls_df.empty:
        st.warning("No options data loaded. Upload a CSV or wait for demo data.")
    else:
        # KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)

        pcr_val = pcr_data.get("pcr_oi", 0)
        sentiment = pcr_data.get("sentiment", "N/A")
        sent_emoji = {
            "Bullish": "🟢",
            "Mildly Bullish": "🟡",
            "Mildly Bearish": "🟠",
            "Bearish": "🔴",
        }.get(sentiment, "⚪")

        with k1:
            st.metric(
                "PCR OI",
                f"{pcr_val:.3f}",
                help="Put/Call OI ratio. <0.7 bullish, >1.3 bearish",
            )
        with k2:
            st.metric(
                "Sentiment",
                f"{sent_emoji} {sentiment}",
                help="Market sentiment derived from PCR",
            )
        with k3:
            st.metric(
                "Max Pain",
                f"₹{max_pain:.0f}",
                help="Strike where option writers lose least (OI-weighted)",
            )
        with k4:
            primary_support = key_levels.get("primary_support")
            st.metric(
                "Primary Support",
                f"₹{primary_support:.0f}" if primary_support else "N/A",
                help="Highest Put Wall strike = key support floor",
            )
        with k5:
            primary_res = key_levels.get("primary_resistance")
            st.metric(
                "Primary Resistance",
                f"₹{primary_res:.0f}" if primary_res else "N/A",
                help="Highest Call OI wall = key resistance ceiling",
            )

        st.divider()

        # Main OI Chart
        fig_oi = cb.oi_wall_chart(walls_df, call_walls, put_walls, cmp, max_pain)
        st.plotly_chart(fig_oi, use_container_width=True)

        st.divider()

        # PCR + IV Smile
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(cb.pcr_by_strike_chart(walls_df), use_container_width=True)
        with col_right:
            st.plotly_chart(cb.iv_smile_chart(walls_df), use_container_width=True)

        # Total OI + Wall Strength
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.plotly_chart(cb.total_oi_area_chart(walls_df), use_container_width=True)
        with col_r2:
            st.plotly_chart(cb.wall_strength_chart(walls_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Insider Activity
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    if walls_df is None or walls_df.empty:
        st.warning("No data loaded.")
    else:
        # KPI Row
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.metric(
                "Total Deals Today",
                summary.get("total_deals", 0),
                help="Total bulk/block deals matched for this symbol today",
            )
        with i2:
            st.metric(
                "At Wall Levels",
                summary.get("at_wall_levels", 0),
                help="Deals with signal score ≥ 40 (near a wall strike)",
            )
        with i3:
            st.metric(
                "High Conviction (≥60)",
                summary.get("high_conviction", 0),
                help="Deals scoring ≥ 60 / 100",
            )
        with i4:
            st.metric(
                "Top Entity",
                summary.get("top_entity", "N/A"),
                help="Most active institutional entity today",
            )

        st.divider()

        # OI + Deal Overlay
        fig_overlay = cb.oi_deal_overlay_chart(
            walls_df, call_walls, put_walls, matched_df, zones_df, cmp, max_pain
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

        st.divider()

        # Score Scatter + Net Qty
        sc_l, sc_r = st.columns(2)
        with sc_l:
            st.plotly_chart(
                cb.signal_score_scatter(matched_df), use_container_width=True
            )
        with sc_r:
            st.plotly_chart(
                cb.net_qty_bar_chart(aggregated_df), use_container_width=True
            )

        st.divider()

        # Deals Table
        st.markdown("### 📋 Matched Deals")
        if matched_df is not None and not matched_df.empty:
            display_cols = [
                c
                for c in [
                    "Date",
                    "Client Name",
                    "Entity",
                    "Buy/Sell",
                    "Quantity Traded",
                    "Trade Price/Wght. Avg. Price",
                    "Wall_Type_Hit",
                    "Nearest_Strike",
                    "Score",
                    "Signal_Strength",
                    "Interpretation",
                ]
                if c in matched_df.columns
            ]

            def _style_row(row):
                if row.get("Buy/Sell") == "BUY" and row.get("Score", 0) >= 60:
                    return ["background-color: rgba(0,230,118,0.08)"] * len(row)
                elif row.get("Buy/Sell") == "SELL" and row.get("Score", 0) >= 60:
                    return ["background-color: rgba(255,82,82,0.08)"] * len(row)
                return [""] * len(row)

            tdf = matched_df[display_cols].copy()
            if "Quantity Traded" in tdf.columns:
                tdf["Quantity Traded"] = tdf["Quantity Traded"].apply(
                    lambda x: f"{x:,.0f}"
                )

            st.dataframe(
                tdf.style.apply(_style_row, axis=1),
                use_container_width=True,
                height=350,
            )

            csv = matched_df[display_cols].to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Deals CSV", csv, "matched_deals.csv", "text/csv"
            )
        else:
            st.info(
                "No matched deals found. Lower the Min Signal Score or adjust proximity tolerance."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Signal Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    if composite:
        # Master Signal Card
        col = composite.get("colour", "#ffd740")
        emoji = composite.get("emoji", "⚪")
        label = composite.get("label", "NEUTRAL")
        confidence = composite.get("confidence", 0)
        key_reason = master_card.get("key_reason", "")
        bullets = master_card.get("bullets", [])

        bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
        st.markdown(
            f"""
<div class="master-card" style="background: rgba({
    '0,230,118' if 'BULL' in label.upper() else ('255,82,82' if 'BEAR' in label.upper() else '255,215,64')
},0.08); border-color: {col};">
  <h2 style="color:{col}">{emoji} {label} — Confidence: {confidence}%</h2>
  <p class="key-reason" style="color:#b0bec5">{key_reason}</p>
  <ul style="color:#b0bec5">{bullets_html}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

        # Three signal panels
        p1, p2, p3 = st.columns(3)

        with p1:
            opt_score = options_signal.get("score", 0)
            opt_color = (
                "#00e676"
                if opt_score > 0
                else ("#ff5252" if opt_score < 0 else "#ffd740")
            )
            st.markdown(
                f"""
<div class="signal-card" style="border-color:{opt_color}">
<h4 style="color:{opt_color}">📊 Options Flow Signal</h4>
<p>PCR OI: <b>{pcr_val:.3f}</b> — {sentiment}</p>
<p>IV Skew: {iv_skew.get('direction','N/A').title()} ({iv_skew.get('skew',0):+.2f})</p>
<p>Max Pain: <b>₹{max_pain:.0f}</b> vs CMP: {f'₹{cmp:.0f}' if cmp else 'N/A'}</p>
<p>Support: <b>{f'₹{key_levels.get("primary_support",0):.0f}' if key_levels.get("primary_support") else 'N/A'}</b></p>
<p>Resistance: <b>{f'₹{key_levels.get("primary_resistance",0):.0f}' if key_levels.get("primary_resistance") else 'N/A'}</b></p>
<hr style="border-color:#1e1e3a;margin:10px 0">
<p style="color:{opt_color};font-size:1.3rem;font-weight:700">Score: {opt_score:+d}</p>
</div>""",
                unsafe_allow_html=True,
            )

        with p2:
            inst_score = institutional_signal.get("score", 0)
            inst_color = (
                "#00e676"
                if inst_score > 0
                else ("#ff5252" if inst_score < 0 else "#ffd740")
            )
            net_qty = summary.get("net_fii_dii_qty", 0)
            net_cr = net_qty / 1e7
            st.markdown(
                f"""
<div class="signal-card" style="border-color:{inst_color}">
<h4 style="color:{inst_color}">🏦 Institutional Signal</h4>
<p>Accumulation Zones: <b>{summary.get('accumulation_zones',0)}</b></p>
<p>Distribution Zones: <b>{summary.get('distribution_zones',0)}</b></p>
<p>Top Entity: <b>{summary.get('top_entity','N/A')}</b></p>
<p>Net FII+DII: <b style="color:{'#00e676' if net_qty>=0 else '#ff5252'}">{'+'if net_qty>=0 else ''}{net_cr:.2f}Cr</b></p>
<p>High Conviction: <b>{summary.get('high_conviction',0)}</b> deals ≥60</p>
<hr style="border-color:#1e1e3a;margin:10px 0">
<p style="color:{inst_color};font-size:1.3rem;font-weight:700">Score: {inst_score:+d}</p>
</div>""",
                unsafe_allow_html=True,
            )

        with p3:
            comp_score = composite.get("score", 0)
            gauge_fig = cb.composite_gauge(comp_score, label)
            st.markdown(
                """<div class="signal-card">
<h4 style="color:#ce93d8">🎯 Composite Conviction</h4>""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown(
                f"""
<p style="text-align:center;font-size:0.8rem;color:#9e9e9e">
Options(40%) + Institutional(40%) + IV(20%)
</p></div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # Zone Table
        st.markdown("### 🗺️ Zone Analysis")
        if zones_df is not None and not zones_df.empty:

            def zone_color(row):
                if row["Zone_Type"] == "ACCUMULATION":
                    return ["background-color: rgba(0,230,118,0.07)"] * len(row)
                elif row["Zone_Type"] == "DISTRIBUTION":
                    return ["background-color: rgba(255,82,82,0.07)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                zones_df.style.apply(zone_color, axis=1),
                use_container_width=True,
                height=300,
            )
            csv_zones = zones_df.to_csv(index=False).encode()
            st.download_button("⬇️ Export Zones CSV", csv_zones, "zones.csv", "text/csv")
        else:
            st.info(
                "No zone data available. Load deal data to see institutional zones."
            )

        st.divider()

        # Alert Log
        st.markdown("### 🔔 Alert Log")
        alert_mgr = AlertManager(
            score_threshold=st.session_state.get("score_alert_threshold", 70),
            oi_spike_pct=st.session_state.get("oi_spike_pct", 20.0),
        )
        alert_mgr.render_alert_log()
        if st.button("🗑️ Clear Alert Log"):
            alert_mgr.clear_log()
            st.rerun()

    else:
        st.warning("Run the pipeline first (click Refresh in the sidebar).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 💡 Trade Ideas
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("## 💡 Trade Recommendations")

    if walls_df is None or walls_df.empty:
        st.warning("Load options data first.")
    else:
        # ── run recommender ──────────────────────────────────────────────────
        recommender = TradeRecommender(
            symbol=st.session_state["symbol"],
            walls_df=walls_df,
            call_walls=call_walls,
            put_walls=put_walls,
            zones_df=zones_df,
            matched_df=matched_df,
            pcr_data=pcr_data,
            iv_skew=iv_skew,
            max_pain=max_pain,
            key_levels=key_levels,
            composite=composite,
            options_signal=options_signal,
            institutional_signal=institutional_signal,
            cmp=cmp,
        )
        setups = recommender.generate()
        setups_df = recommender.to_dataframe()

        if not setups:
            st.info(
                "No trade setups meet the minimum R:R threshold of 1.5x right now. "
                "Try lowering the Wall Percentile in the sidebar or loading a different symbol."
            )
        else:
            # ── KPI row ──────────────────────────────────────────────────────
            longs = [s for s in setups if s.direction == "LONG"]
            shorts = [s for s in setups if s.direction == "SHORT"]
            top = setups[0]  # highest conviction
            avg_rr = sum(s.rr_ratio for s in setups) / len(setups)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Setups", len(setups))
            k2.metric("Long Ideas", len(longs), help="Bullish setups")
            k3.metric("Short Ideas", len(shorts), help="Bearish setups")
            k4.metric("Avg R:R", f"{avg_rr:.2f}x")
            k5.metric(
                "Top Conviction",
                f"{top.conviction}/100",
                help=f"{top.direction} {top.strategy}",
            )

            st.divider()

            # ── filter controls ───────────────────────────────────────────────
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                dir_filter = st.multiselect(
                    "Direction",
                    ["LONG", "SHORT"],
                    default=["LONG", "SHORT"],
                    key="rec_dir",
                )
            with fc2:
                tf_filter = st.multiselect(
                    "Timeframe",
                    ["Intraday", "Swing", "Positional"],
                    default=["Intraday", "Swing", "Positional"],
                    key="rec_tf",
                )
            with fc3:
                min_conv = st.slider("Min Conviction", 0, 100, 40, key="rec_conv")

            filtered = [
                s
                for s in setups
                if s.direction in dir_filter
                and s.timeframe in tf_filter
                and s.conviction >= min_conv
            ]
            filtered_df = (
                setups_df[setups_df["ID"].isin([s.id for s in filtered])]
                if not setups_df.empty
                else pd.DataFrame()
            )

            st.divider()

            if not filtered:
                st.info("No setups match current filters.")
            else:
                # ── conviction + R:R overview charts ─────────────────────────
                ov1, ov2 = st.columns(2)
                with ov1:
                    st.plotly_chart(
                        cb.conviction_bar_chart(filtered_df),
                        use_container_width=True,
                    )
                with ov2:
                    st.plotly_chart(
                        cb.rr_scatter_chart(filtered_df),
                        use_container_width=True,
                    )

                st.divider()

                # ── individual setup cards ────────────────────────────────────
                st.markdown("### 📋 Detailed Setup Cards")

                for setup in filtered:
                    dir_color = "#00e676" if setup.direction == "LONG" else "#ff5252"
                    dir_arrow = "▲" if setup.direction == "LONG" else "▼"
                    tf_badge = {
                        "Intraday": "🟡 Intraday",
                        "Swing": "🟠 Swing (2–5d)",
                        "Positional": "🔵 Positional (1–2w)",
                    }.get(setup.timeframe, setup.timeframe)
                    signal_badge = {
                        "Options": "📊 Options",
                        "Institutional": "🏦 Institutional",
                        "Options + Institutional": "📊+🏦 Combined",
                        "Options + Composite": "📊+🎯 Options+Signal",
                    }.get(setup.signal_source, setup.signal_source)

                    with st.expander(
                        f"#{setup.id}  {dir_arrow} {setup.direction}  |  "
                        f"{setup.strategy}  |  {tf_badge}  |  "
                        f"Conviction: {setup.conviction}/100  |  R:R: {setup.rr_ratio:.1f}x",
                        expanded=(setup.conviction >= 65),
                    ):
                        col_chart, col_info = st.columns([3, 2])

                        with col_chart:
                            st.plotly_chart(
                                cb.trade_setup_chart(setup, walls_df, cmp),
                                use_container_width=True,
                            )

                        with col_info:
                            st.markdown(
                                f"""
<div style="background:#14142a;border:1px solid {dir_color};border-radius:12px;padding:16px">
<h4 style="color:{dir_color};margin-bottom:12px">
  {dir_arrow} {setup.direction} — {setup.strategy}
</h4>
<table style="width:100%;font-size:0.85rem;color:#e0e0e0;border-collapse:collapse">
<tr><td style="color:#9e9e9e;padding:4px 0">Signal Source</td>
    <td style="text-align:right">{signal_badge}</td></tr>
<tr><td style="color:#9e9e9e;padding:4px 0">Timeframe</td>
    <td style="text-align:right">{tf_badge}</td></tr>
<tr><td style="color:#9e9e9e;padding:4px 0">Entry Zone</td>
    <td style="text-align:right;color:{dir_color}">
      {setup.entry_low:.2f} – {setup.entry_high:.2f}</td></tr>
<tr><td style="color:#9e9e9e;padding:4px 0">Target 1</td>
    <td style="text-align:right;color:#ffd740">{setup.target1:.2f}
      <span style="color:#9e9e9e;font-size:0.75rem">
        ({setup.expected_move_pct:+.2f}%)</span></td></tr>
<tr><td style="color:#9e9e9e;padding:4px 0">Target 2</td>
    <td style="text-align:right;color:#00e676">{setup.target2:.2f}</td></tr>
<tr><td style="color:#9e9e9e;padding:4px 0">Stop Loss</td>
    <td style="text-align:right;color:#ff5252">{setup.stop_loss:.2f}
      <span style="color:#9e9e9e;font-size:0.75rem">
        (-{setup.risk_pct:.2f}%)</span></td></tr>
<tr><td style="color:#9e9e9e;padding:4px 0">R:R Ratio</td>
    <td style="text-align:right;color:#ffd740;font-weight:700">
      {setup.rr_ratio:.2f}x</td></tr>
</table>
<hr style="border-color:#1e1e3a;margin:12px 0">
<p style="color:#9e9e9e;font-size:0.78rem;margin-bottom:6px">WHY THIS TRADE:</p>
{"".join(f"<p style=\"color:#b0bec5;font-size:0.82rem;margin:3px 0\">• {r}</p>" for r in setup.reasoning)}
<hr style="border-color:#1e1e3a;margin:12px 0">
<p style="color:#ff7043;font-size:0.78rem">
  ⚠️ INVALIDATION: {setup.invalidation}
</p>
</div>
""",
                                unsafe_allow_html=True,
                            )

                st.divider()

                # ── setups summary table ──────────────────────────────────────
                st.markdown("### 📊 All Setups Summary")

                display_cols = [
                    "ID",
                    "Direction",
                    "Strategy",
                    "Timeframe",
                    "Entry Zone",
                    "Target 1",
                    "Target 2",
                    "Stop Loss",
                    "R:R",
                    "Conviction",
                    "Move %",
                    "Risk %",
                    "Signal",
                ]
                display_df = filtered_df[
                    [c for c in display_cols if c in filtered_df.columns]
                ]

                def _style_setup(row):
                    if row.get("Direction") == "LONG":
                        return ["background-color: rgba(0,230,118,0.07)"] * len(row)
                    return ["background-color: rgba(255,82,82,0.07)"] * len(row)

                st.dataframe(
                    display_df.style.apply(_style_setup, axis=1),
                    use_container_width=True,
                    height=300,
                )

                csv_setups = filtered_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Export Trade Ideas CSV",
                    csv_setups,
                    "trade_ideas.csv",
                    "text/csv",
                )

                st.divider()
                st.caption(
                    "⚠️ **Disclaimer:** These are algorithmic setups derived from options OI structure "
                    "and institutional deal data. They are NOT financial advice. Always do your own "
                    "research and use appropriate position sizing and risk management."
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Historical Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    symbol = st.session_state["symbol"]
    history = dm.load_all_options_history(symbol, max_days=10)

    if len(history) < 2:
        st.info(
            "📁 Need at least 2 days of options data in ./data/options/ for historical comparison. Demo data shown below."
        )

        # Generate synthetic history for demo
        dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(5)]
        pcr_vals = [0.82, 0.79, 1.01, 1.15, 0.93]
        mp_vals = [590, 585, 580, 595, 600]

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.plotly_chart(
                cb.pcr_trend_chart(pd.Series(pcr_vals), dates), use_container_width=True
            )
        with col_h2:
            st.plotly_chart(
                cb.max_pain_migration_chart(mp_vals, dates), use_container_width=True
            )
    else:
        sorted_dates = sorted(history.keys(), reverse=True)
        today_date = sorted_dates[0]
        yesterday_date = sorted_dates[1] if len(sorted_dates) > 1 else today_date

        try:
            calc_today = OptionsWallCalculator(
                dm._normalise_options(history[today_date])
            )
            walls_today = calc_today.consolidate_walls()

            calc_yest = OptionsWallCalculator(
                dm._normalise_options(history[yesterday_date])
            )
            walls_yest = calc_yest.consolidate_walls()

            # OI Change
            st.plotly_chart(
                cb.oi_change_chart(walls_today, walls_yest), use_container_width=True
            )

            # PCR & Max Pain trends
            pcr_series = []
            mp_series = []
            dates_list = sorted(history.keys())
            for d in dates_list:
                try:
                    c = OptionsWallCalculator(dm._normalise_options(history[d]))
                    c.consolidate_walls()
                    pcr_series.append(c.analyze_pcr()["pcr_oi"])
                    mp_series.append(c.calculate_max_pain())
                except Exception:
                    pcr_series.append(np.nan)
                    mp_series.append(np.nan)

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.plotly_chart(
                    cb.pcr_trend_chart(pd.Series(pcr_series), dates_list),
                    use_container_width=True,
                )
            with col_h2:
                st.plotly_chart(
                    cb.max_pain_migration_chart(mp_series, dates_list),
                    use_container_width=True,
                )

            # New OI built today
            st.markdown("### 🆕 New OI Built Today (>10% increase)")
            merged = walls_today[["Strike", "Total_OI"]].merge(
                walls_yest[["Strike", "Total_OI"]].rename(
                    columns={"Total_OI": "Prior_OI"}
                ),
                on="Strike",
                how="inner",
            )
            merged["Change_%"] = (
                (merged["Total_OI"] - merged["Prior_OI"])
                / merged["Prior_OI"].clip(lower=1)
                * 100
            )
            new_oi = merged[merged["Change_%"] > 10].sort_values(
                "Change_%", ascending=False
            )
            if not new_oi.empty:
                st.dataframe(
                    new_oi.style.background_gradient(
                        subset=["Change_%"], cmap="Greens"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No significant new OI buildup detected.")

        except Exception as e:
            st.error(f"Historical analysis error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Settings & Data Health
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:  # Settings
    st.markdown("## ⚙️ Settings & Data Health")

    # Data quality report
    st.markdown("### 🔍 Data Quality Report")
    col_dq1, col_dq2 = st.columns(2)

    with col_dq1:
        st.markdown("**Options Chain**")
        options_df = st.session_state.get("options_df")
        if options_df is not None and not options_df.empty:
            null_counts = options_df.isnull().sum()
            total_rows = len(options_df)
            coverage = (1 - null_counts / total_rows * 100).round(1)
            st.metric("Total Rows", total_rows)
            st.metric(
                "Strikes",
                options_df["Strike"].nunique() if "Strike" in options_df.columns else 0,
            )
            st.metric(
                "Expiries",
                options_df["Expiry"].nunique() if "Expiry" in options_df.columns else 0,
            )
            st.dataframe(
                pd.DataFrame({"Null Count": null_counts, "Coverage %": coverage}),
                use_container_width=True,
            )
        else:
            st.info("No options data loaded.")

    with col_dq2:
        st.markdown("**Deals Data**")
        deals_df = st.session_state.get("deals_df")
        if deals_df is not None and not deals_df.empty:
            st.metric("Total Deal Records", len(deals_df))
            if "Entity" in deals_df.columns:
                st.dataframe(
                    deals_df["Entity"].value_counts().to_frame("Count"),
                    use_container_width=True,
                )
            null_counts_d = deals_df.isnull().sum()
            st.dataframe(
                pd.DataFrame({"Null Count": null_counts_d}),
                use_container_width=True,
            )
        else:
            st.info("No deals data loaded.")

    st.divider()
    st.markdown("### 👁️ Raw Data Preview")

    preview_tab = st.radio(
        "Show:", ["Options Chain", "Deals", "Walls"], horizontal=True
    )

    if preview_tab == "Options Chain" and options_df is not None:
        st.dataframe(options_df.head(50), use_container_width=True)
    elif preview_tab == "Deals" and deals_df is not None:
        st.dataframe(deals_df.head(50), use_container_width=True)
    elif preview_tab == "Walls" and walls_df is not None:
        st.dataframe(walls_df, use_container_width=True)

    st.divider()
    st.markdown("### 🕐 Session Info")
    lr = st.session_state.get("last_refresh")
    if lr:
        st.write(
            f"Last refreshed: {datetime.datetime.fromtimestamp(lr).strftime('%H:%M:%S')}"
        )
        st.write(
            f"Mode: {'⚠️ DEMO' if st.session_state.get('is_demo') else '✅ Live Data'}"
        )
    st.write(f"Symbol: {st.session_state.get('symbol')}")
    st.write(f"Date: {st.session_state.get('date')}")

    st.divider()
    st.markdown("### 📋 Column Mapping Editor")
    st.info(
        "If your CSV uses non-standard column names, rename them to match: "
        "`Strike, Expiry, OptionType, OpenInterest, Volume, IV, LTP` (for options) or "
        "`Date, Symbol, Client Name, Buy/Sell, Quantity Traded, Trade Price/Wght. Avg. Price` (for deals)."
    )

    st.markdown("### ⏱️ Refresh Schedule")
    market_open = dm.is_market_open()
    st.write(f"Market Status: {'🟢 OPEN' if market_open else '🔴 CLOSED'}")
    st.info(
        "Auto-refresh every 5 min during market hours (09:15–15:30 IST). Reload page to refresh manually outside market hours."
    )


# ─── auto-refresh ────────────────────────────────────────────────────────────
# streamlit-autorefresh triggers a full page rerun every N milliseconds.
# We only activate it during market hours so it doesn't hammer your machine
# when the market is closed.

try:
    from streamlit_autorefresh import st_autorefresh

    REFRESH_INTERVAL_MS = 5 * 60 * 1000  # 5 minutes in milliseconds

    if dm.is_market_open():
        refresh_count = st_autorefresh(
            interval=REFRESH_INTERVAL_MS,
            limit=None,  # refresh indefinitely
            key="mkt_autorefresh",
        )

        # Every time the component fires a rerun, re-run the pipeline
        if refresh_count and refresh_count > 0:
            with st.spinner("⏱️ Auto-refreshing data…"):
                try:
                    run_compute_pipeline()
                except Exception as e:
                    st.warning(f"Auto-refresh error: {e}")

        # Sidebar countdown display
        with st.sidebar:
            st.markdown("---")
            now = time.time()
            last = st.session_state.get("last_refresh") or now
            elapsed = int(now - last)
            remaining = max(0, 300 - elapsed)
            mins, secs = divmod(remaining, 60)
            st.markdown(f"**⏱️ Next refresh: {mins}:{secs:02d}**")
            st.caption("Auto-refresh active (market hours)")
    else:
        with st.sidebar:
            st.markdown("---")
            st.caption("🔴 Market closed — manual refresh only")

except ImportError:
    # streamlit-autorefresh not installed — show install hint in sidebar
    with st.sidebar:
        st.markdown("---")
        st.warning(
            "Install **streamlit-autorefresh** for auto-refresh during market hours:\n\n"
            "```\npip install streamlit-autorefresh\n```"
        )
        if dm.is_market_open():
            st.caption("Market is OPEN — refresh manually with 🔄 button above")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trade Ideas (injected at bottom, tabs[3] block)
# ══════════════════════════════════════════════════════════════════════════════
