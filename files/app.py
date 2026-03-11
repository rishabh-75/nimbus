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
from market_scanner import (
    MarketScanner,
    DEFAULT_WATCHLIST,
    NSE_LARGE_CAP,
    NSE_MIDCAP,
    NSE_INDICES,
)
from gex_calculator import GEXCalculator, NSE_LOT_SIZES, DEFAULT_LOT_SIZE
import chart_builder as cb

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Signal Dashboard — Emerald Slate",
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
    download_options: bool = False,
):
    symbol = st.session_state["symbol"]
    date = st.session_state["date"]
    wall_pct = st.session_state["wall_pct"]
    proximity_pct = st.session_state["proximity_pct"]
    min_score = st.session_state["min_score"]
    entity_filter = st.session_state["entity_filter"]

    # 1. Load options  (auto_download on first load, symbol change, or button press)
    _first_load = st.session_state.get("last_refresh") is None
    _auto_dl = bool(download_options) or _first_load
    if _auto_dl and not options_uploaded:
        _dl_note = st.empty()
        _dl_note.info(
            f"⬇️ Fetching live options chain for **{symbol}** from NSE (all expiries)...⋯"
        )
    options_df, is_demo_opt = dm.load_options_chain(
        symbol,
        date,
        uploaded_file=options_uploaded,
        auto_download=_auto_dl,
    )
    if _auto_dl and not options_uploaded:
        _dl_note.empty()
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

    # Surface NSE download results so UI can display them
    st.session_state["last_download_errors"] = getattr(dm, "last_download_errors", [])
    st.session_state["last_download_success"] = getattr(dm, "last_download_success", [])

    # Show a permanent success toast when options were freshly downloaded
    if is_demo_opt:
        st.session_state["options_status"] = "demo"
    elif getattr(dm, "last_download_success", []):
        st.session_state["options_status"] = "downloaded"
    else:
        st.session_state["options_status"] = "file"

    # 3. Options wall computation
    calc = OptionsWallCalculator(options_df)
    walls_df = calc.consolidate_walls()
    call_walls, put_walls = calc.identify_walls(pct=wall_pct)
    pcr_data = calc.analyze_pcr()
    iv_skew = calc.analyze_iv_skew()
    max_pain = calc.calculate_max_pain()

    # 4. CMP
    cmp = dm.get_cmp(symbol, options_df=options_df)
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

    if not matched_df.empty and min_score > 0 and "Score" in matched_df.columns:
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

    # ── Symbol selector ──────────────────────────────────────────────────────
    all_presets = [
        "NIFTY",
        "BANKNIFTY",
        "FINNIFTY",
        "MIDCPNIFTY",
        "NIFTYNXT50",
        "RELIANCE",
        "TCS",
        "HDFCBANK",
        "INFY",
        "ICICIBANK",
        "SBIN",
        "AXISBANK",
        "KOTAKBANK",
        "LT",
        "WIPRO",
        "ITC",
        "BHARTIARTL",
        "MARUTI",
        "TITAN",
        "BAJFINANCE",
    ]
    recent = st.session_state["recent_symbols"]
    options = list(dict.fromkeys(recent + [s for s in all_presets if s not in recent]))

    # ── on_change callback — fires BEFORE the next render cycle,
    #    so session_state["symbol"] is already correct when the
    #    pipeline trigger runs further down the page. ─────────────────────────
    def _on_sym_change():
        new_sym = st.session_state["_sym_select"]
        st.session_state["symbol"] = new_sym
        st.session_state["last_refresh"] = None  # force pipeline re-run
        if new_sym not in st.session_state["recent_symbols"]:
            st.session_state["recent_symbols"].insert(0, new_sym)

    cur_idx = (
        options.index(st.session_state["symbol"])
        if st.session_state["symbol"] in options
        else 0
    )

    st.selectbox(
        "Symbol",
        options,
        index=cur_idx,
        key="_sym_select",
        on_change=_on_sym_change,
        help="Selecting a symbol immediately loads its data",
    )

    # Custom symbol — for anything not in the preset list
    custom_sym = st.text_input(
        "Or type any symbol", placeholder="e.g. NESTLEIND", key="_sym_custom"
    )
    if st.button("🔍 Load Symbol", use_container_width=True, key="_load_sym_btn"):
        s = custom_sym.strip().upper()
        if s:
            if s not in st.session_state["recent_symbols"]:
                st.session_state["recent_symbols"].insert(0, s)
            st.session_state["symbol"] = s
            st.session_state["last_refresh"] = None

    # Date
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
    st.markdown("**📁 Options Chain**")

    download_options = st.button(
        "⬇️ Try Auto-Download from NSE",
        use_container_width=True,
        help="Fetches live options chain from NSE (all expiries). Also runs automatically on first load and symbol change.",
    )

    options_file = st.file_uploader(
        "📂 Upload Options Chain CSV",
        type=["csv"],
        key="opt_upload",
        help="Upload a CSV downloaded from NSE. Wide format (CE/PE columns) and long format both accepted.",
    )

    with st.expander("📋 How to get options data", expanded=False):
        sym_link = st.session_state.get("symbol", "NIFTY").upper()
        st.markdown("**Step 1 — Open this link in your browser:**")
        st.code("https://www.nseindia.com/option-chain")
        st.markdown(
            f"**Step 2 — Search for `{sym_link}` and click ‘Download CSV’ (top right of the chain table)**"
        )
        st.markdown("**Step 3 — Upload the downloaded file using the uploader above**")
        st.divider()
        st.markdown(
            "**Format the dashboard expects** (or just upload the NSE CSV as-is):"
        )
        st.code(
            "Strike, Expiry, OptionType (CE/PE), OpenInterest, Volume, IV, LTP\n"
            "22000,  27-Mar-2025, CE,          152340,       45210,   14.2, 310.5",
            language="text",
        )
        st.caption(
            "NSE wide format (CE OI | PE OI columns side by side) is also accepted automatically."
        )

    st.divider()
    st.divider()
    st.markdown("**📄 Deals Data**")
    deals_file = st.file_uploader(
        "Bulk/Block Deals CSV", type=["csv"], key="deals_upload"
    )

    st.divider()
    st.markdown("**📂 Data Folder Path**")
    st.text_input("Options Folder", value="./data/options")

    st.divider()
    st.markdown("**🔄 Refresh**")
    manual_refresh = st.button("🔄 Refresh Now", use_container_width=True)

    st.divider()
    st.markdown("**⬇️ NSE Bulk/Block Deals**")
    download_nse = st.button(
        "⬇️ Try Auto-Download",
        use_container_width=True,
        help="Attempts 3 strategies to fetch NSE bulk/block CSVs. NSE bot protection may block all attempts.",
    )
    with st.expander("📥 If auto-download fails (click to expand)", expanded=False):
        st.info(
            "NSE blocks most automated downloads. Open these links in your browser, "
            "save the files, then upload using the file uploader below."
        )
        st.markdown("**Bulk Deals CSV:**")
        st.code("https://archives.nseindia.com/content/equities/bulk.csv")
        st.markdown("**Block Deals CSV:**")
        st.code("https://archives.nseindia.com/content/equities/block.csv")
        st.caption(
            "Save downloaded files anywhere, then upload using 'Bulk/Block Deals CSV' "
            "uploader below. Or drop them into ./data/deals/ and click Refresh."
        )


# ─── header ──────────────────────────────────────────────────────────────────

header_col1, header_col2, header_col3 = st.columns([4, 2, 2])
with header_col1:
    st.markdown(f"# 📈 NSE Signal Dashboard — {st.session_state['symbol']}")
    st.markdown(
        "<span style='font-size:12px;color:#4ade80;letter-spacing:0.12em;"
        "font-weight:600;text-transform:uppercase'>⬡ Emerald Slate</span>",
        unsafe_allow_html=True,
    )
with header_col2:
    freshness = dm.freshness_badge(st.session_state.get("last_refresh"))
    st.markdown(f"**Data Freshness:** {freshness}")
with header_col3:
    opt_status = st.session_state.get("options_status", "demo")

    if st.session_state.get("is_demo"):
        _dl_errs = st.session_state.get("last_download_errors", [])
        if _dl_errs:
            # Auto-download was tried and failed — give actionable next step
            with st.expander(
                "⚠️ DEMO DATA — NSE auto-download blocked. Click to get real data →",
                expanded=True,
            ):
                st.warning(
                    "NSE blocks automated downloads. "
                    "**Get real options data in 30 seconds:**\n\n"
                    "1. Open [nseindia.com/option-chain](https://www.nseindia.com/option-chain) in your browser\n"
                    "2. Search your symbol and click **Download CSV**\n"
                    "3. Upload the file via **Options Chain** uploader in the sidebar"
                )
        else:
            st.markdown(
                '<div class="demo-banner">⚠️ DEMO DATA — '
                "Upload real options CSV via sidebar to begin</div>",
                unsafe_allow_html=True,
            )
    elif opt_status == "downloaded":
        st.success("✅ Live options chain loaded from NSE")

# Deals download results
_dl_errors = st.session_state.get("last_download_errors", [])
_dl_success = st.session_state.get("last_download_success", [])

for _msg in _dl_success:
    if "rows downloaded" in _msg:  # deals success (not options)
        st.success(f"✅ {_msg}")

# ─── auto-load on first run ───────────────────────────────────────────────────

if (
    st.session_state["last_refresh"] is None
    or manual_refresh
    or options_file
    or deals_file
    or download_nse
    or download_options
):
    with st.spinner(f"🔄 Loading {st.session_state['symbol']}…"):
        try:
            run_compute_pipeline(
                options_uploaded=options_file,
                deals_uploaded=deals_file,
                download_deals=download_nse,
                download_options=download_options,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

# ─── grab state refs ─────────────────────────────────────────────────────────

options_df = st.session_state.get("options_df", pd.DataFrame())
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
        "🔍 Scanner",
        "⚡ GEX",
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
        # ── KPI Row ───────────────────────────────────────────────────────────
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
                help="Highest Put OI wall below spot",
            )
        with k5:
            primary_res = key_levels.get("primary_resistance")
            st.metric(
                "Primary Resistance",
                f"₹{primary_res:.0f}" if primary_res else "N/A",
                help="Highest Call OI wall above spot",
            )

        st.divider()

        # ── OI Wall ───────────────────────────────────────────────────────────
        st.plotly_chart(
            cb.oi_wall_chart(walls_df, call_walls, put_walls, cmp, max_pain),
            use_container_width=True,
            key="ov_oi_wall",
        )

        st.divider()

        # ── GEX section (inline — no tab switching needed) ────────────────────
        st.markdown("### ⚡ Gamma Exposure")

        _sym_upper = st.session_state["symbol"].upper()
        _default_lot = NSE_LOT_SIZES.get(_sym_upper, DEFAULT_LOT_SIZE)

        _ov_c1, _ov_c2, _ov_c3 = st.columns(3)
        with _ov_c1:
            _ov_lot = st.number_input(
                "Lot Size", 1, 50000, _default_lot, 25, key="ov_gex_lot"
            )
        with _ov_c2:
            _ov_rf = st.number_input(
                "Risk-Free (%)", 0.0, 20.0, 6.5, 0.25, key="ov_gex_rf"
            )
        with _ov_c3:
            _ov_dte = st.slider("Max DTE", 7, 180, 90, 7, key="ov_gex_dte")

        @st.cache_data(ttl=300, show_spinner=False)
        def _ov_compute_gex(_hash, symbol, spot, lot, rf, dte):
            df = st.session_state.get("options_df", pd.DataFrame())
            from gex_calculator import GEXCalculator

            return GEXCalculator(
                df,
                symbol=symbol,
                spot=spot,
                lot_size=lot,
                risk_free=rf / 100,
                max_dte_days=dte,
            ).compute()

        with st.spinner("⚡ Computing GEX…"):
            _ov_gex = _ov_compute_gex(
                id(options_df),
                _sym_upper,
                cmp or 0.0,
                _ov_lot,
                _ov_rf,
                _ov_dte,
            )

        if _ov_gex.expiries:
            # KPI strip for GEX
            _gk1, _gk2, _gk3, _gk4 = st.columns(4)
            _all_df = _ov_gex.all_strikes
            _net_gex = float(_all_df["Net_GEX"].sum()) if not _all_df.empty else 0
            _abs_gex = float(_all_df["Net_GEX"].abs().sum()) if not _all_df.empty else 0
            _g_hvl = _ov_gex.overall_hvl
            _g_regime = (
                "🟢 Positive (Range)" if _net_gex >= 0 else "🔴 Negative (Trend)"
            )
            _gk1.metric(
                "Net GEX", f"{_net_gex:+,.0f}", help="Positive = dealers long gamma"
            )
            _gk2.metric("|GEX| Total", f"{_abs_gex:,.0f}", help="Total gamma exposure")
            _gk3.metric(
                "HVL",
                f"₹{_g_hvl:.0f}" if _g_hvl else "N/A",
                help="High Volatility Level — gamma zero-cross",
            )
            _gk4.metric("GEX Regime", _g_regime)

            st.divider()

            # All expiries combined
            _oi_res = key_levels.get("primary_resistance")
            _oi_sup = key_levels.get("primary_support")
            st.plotly_chart(
                cb.gex_aggregate_chart(
                    _ov_gex, oi_resistance=_oi_res, oi_support=_oi_sup
                ),
                use_container_width=True,
                key="ov_gex_agg",
            )

            st.divider()

            # Per-expiry 4-panel
            st.markdown("#### Per-Expiry Panels")
            _figs = cb.gex_4panel(
                _ov_gex, height_each=440, oi_resistance=_oi_res, oi_support=_oi_sup
            )
            _r1, _r2 = st.columns(2)
            with _r1:
                st.plotly_chart(_figs[0], use_container_width=True, key="ov_gex_p0")
            with _r2:
                st.plotly_chart(_figs[1], use_container_width=True, key="ov_gex_p1")
            _r3, _r4 = st.columns(2)
            with _r3:
                st.plotly_chart(_figs[2], use_container_width=True, key="ov_gex_p2")
            with _r4:
                st.plotly_chart(_figs[3], use_container_width=True, key="ov_gex_p3")
        else:
            st.info(
                "GEX data unavailable — IV column may be missing from the options chain."
            )


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
        st.plotly_chart(fig_overlay, use_container_width=True, key="ins_overlay")

        st.divider()

        # Score Scatter + Net Qty
        sc_l, sc_r = st.columns(2)
        with sc_l:
            st.plotly_chart(
                cb.signal_score_scatter(matched_df),
                use_container_width=True,
                key="ins_scatter",
            )
        with sc_r:
            st.plotly_chart(
                cb.net_qty_bar_chart(aggregated_df),
                use_container_width=True,
                key="ins_netqty",
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
            st.plotly_chart(gauge_fig, use_container_width=True, key="sig_gauge")
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
                        key="tr_setup_a",
                    )
                with ov2:
                    st.plotly_chart(
                        cb.rr_scatter_chart(filtered_df),
                        use_container_width=True,
                        key="tr_setup_b",
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
                                key=f"tr_setup_{setup.id}",
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
# TAB 4 — 🔍 Market Scanner
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("## 🔍 Market Scanner")
    st.caption(
        "Runs the full options wall + institutional + signal pipeline across your watchlist and surfaces the best trade setups."
    )

    # ── watchlist editor ─────────────────────────────────────────────────────
    with st.expander("📋 Manage Watchlist", expanded=False):
        wl_preset = st.radio(
            "Quick preset",
            ["Custom", "NSE Indices", "Large Cap", "Mid Cap", "Indices + Large Cap"],
            horizontal=True,
            key="wl_preset",
        )
        preset_map = {
            "NSE Indices": NSE_INDICES,
            "Large Cap": NSE_LARGE_CAP,
            "Mid Cap": NSE_MIDCAP,
            "Indices + Large Cap": DEFAULT_WATCHLIST,
        }
        if wl_preset != "Custom":
            default_wl = ", ".join(preset_map[wl_preset])
        else:
            default_wl = st.session_state.get(
                "custom_watchlist_str", "SBIN, NIFTY, BANKNIFTY"
            )

        wl_input = st.text_area(
            "Symbols (comma-separated)",
            value=default_wl,
            height=80,
            key="wl_input",
            help="Enter NSE symbols separated by commas. You need a matching CSV in ./data/options/ for each.",
        )
        st.session_state["custom_watchlist_str"] = wl_input

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            scan_wall_pct = st.slider(
                "Wall Percentile", 50, 95, 75, key="scan_wall_pct"
            )
        with sc2:
            scan_min_conv = st.slider("Min Conviction", 0, 100, 40, key="scan_min_conv")
        with sc3:
            scan_min_rr = st.slider(
                "Min R:R", 1.0, 3.0, 1.5, step=0.1, key="scan_min_rr"
            )

    watchlist = [s.strip().upper() for s in wl_input.split(",") if s.strip()]
    st.markdown(
        f"**Watchlist ({len(watchlist)} symbols):** "
        + " · ".join(f"`{s}`" for s in watchlist)
    )

    # ── run scan button ───────────────────────────────────────────────────────
    sc_btn_col1, sc_btn_col2 = st.columns(2)
    with sc_btn_col1:
        run_scan = st.button(
            "🔍 Run Full Scan",
            use_container_width=True,
            type="primary",
            help="Analyses locally cached data for all symbols",
        )
    with sc_btn_col2:
        download_all = st.button(
            "⬇️ Download All Chains from NSE",
            use_container_width=True,
            help=(
                "Fetches live options chain for every watchlist symbol concurrently. "
                "Uses 3 parallel threads with rate-limiting to stay within NSE limits. "
                "Takes ~30–90 seconds depending on watchlist size."
            ),
        )

    if download_all:
        dl_progress = st.progress(0, text="Initialising NSE downloads…")
        dl_status = st.empty()
        dl_log = st.empty()
        dl_messages: list[str] = []

        def _dl_progress(sym, status, done, total):
            pct = int(done / total * 100)
            icon = "✅" if status == "ok" else "❌"
            dl_messages.append(f"{icon} {sym}")
            dl_progress.progress(pct, text=f"Downloading {sym} ({done}/{total})…")
            dl_log.markdown("  ".join(dl_messages[-12:]))  # show last 12

        with st.spinner("Downloading options chains from NSE…"):
            bulk_results = dm.download_watchlist_chains(
                symbols=watchlist,
                max_workers=3,
                inter_symbol_delay=1.5,
                progress_callback=_dl_progress,
            )

        dl_progress.empty()
        dl_status.empty()
        dl_log.empty()

        ok_syms = [s for s, (df, _) in bulk_results.items() if df is not None]
        err_syms = [s for s, (df, _) in bulk_results.items() if df is None]

        if ok_syms:
            st.success(f"✅ Downloaded {len(ok_syms)} symbols: {', '.join(ok_syms)}")
        if err_syms:
            with st.expander(f"❌ {len(err_syms)} symbols failed"):
                for sym in err_syms:
                    _, msg = bulk_results[sym]
                    st.caption(f"**{sym}**: {msg}")

        st.toast(
            f"Download complete — {len(ok_syms)}/{len(watchlist)} succeeded. Click Run Full Scan to analyse."
        )

    # Cache scan results in session state
    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = None
    if "scan_setups_df" not in st.session_state:
        st.session_state["scan_setups_df"] = None
    if "scan_summary_df" not in st.session_state:
        st.session_state["scan_summary_df"] = None
    if "scan_ts" not in st.session_state:
        st.session_state["scan_ts"] = None

    if run_scan:
        progress_bar = st.progress(0, text="Initialising scanner…")
        status_text = st.empty()

        def _progress(symbol, idx, total):
            pct = int((idx + 1) / total * 100)
            progress_bar.progress(pct, text=f"Scanning {symbol} ({idx+1}/{total})…")
            status_text.caption(f"⏳ Processing {symbol}…")

        scanner = MarketScanner(
            watchlist=watchlist,
            data_manager=dm,
            date=st.session_state["date"],
            wall_pct=scan_wall_pct,
            proximity_pct=st.session_state["proximity_pct"],
            min_rr=scan_min_rr,
        )

        with st.spinner("Running scan…"):
            results = scanner.scan(progress_callback=_progress)

        progress_bar.empty()
        status_text.empty()

        st.session_state["scan_results"] = results
        st.session_state["scan_setups_df"] = scanner.all_setups_df
        st.session_state["scan_summary_df"] = scanner.summary_dataframe()
        st.session_state["scan_ts"] = time.time()
        st.toast(
            f"✅ Scan complete — {len(results)} symbols, {len(scanner.all_setups_df)} setups found"
        )

    # ── display results ───────────────────────────────────────────────────────
    scan_results = st.session_state.get("scan_results")
    scan_setups_df = st.session_state.get("scan_setups_df")
    scan_summary_df = st.session_state.get("scan_summary_df")
    scan_ts = st.session_state.get("scan_ts")

    if scan_results is None:
        st.info(
            "👆 Configure your watchlist above and click **Run Full Scan** to see cross-market opportunities."
        )
    else:
        # Freshness
        if scan_ts:
            age = int(time.time() - scan_ts)
            st.caption(
                f"Last scanned: {datetime.datetime.fromtimestamp(scan_ts).strftime('%H:%M:%S')} — {age}s ago"
            )

        ok_results = [r for r in scan_results if r.status == "ok"]
        err_results = [r for r in scan_results if r.status != "ok"]
        syms_with = [r.symbol for r in ok_results if r.total_setups > 0]
        bulls = [r.symbol for r in ok_results if "BULL" in r.composite_label.upper()]
        bears = [r.symbol for r in ok_results if "BEAR" in r.composite_label.upper()]

        # KPI row
        km1, km2, km3, km4, km5, km6 = st.columns(6)
        km1.metric("Symbols Scanned", len(scan_results))
        km2.metric("With Setups", len(syms_with))
        km3.metric("Bullish", len(bulls), help=" · ".join(bulls[:5]))
        km4.metric("Bearish", len(bears), help=" · ".join(bears[:5]))
        km5.metric(
            "Total Setups", len(scan_setups_df) if scan_setups_df is not None else 0
        )
        km6.metric("Errors / No Data", len(err_results))

        st.divider()

        # Symbols with plays — highlight cards
        if syms_with:
            st.markdown("### 🎯 Symbols With Active Trade Setups")
            cols = st.columns(min(len(syms_with), 4))
            for i, sym in enumerate(syms_with[:8]):
                r = next((x for x in ok_results if x.symbol == sym), None)
                if r is None:
                    continue
                bs = r.best_setup
                dir_color = "#00e676" if bs and bs.direction == "LONG" else "#ff5252"
                dir_arrow = "▲" if bs and bs.direction == "LONG" else "▼"
                with cols[i % 4]:
                    st.markdown(
                        f"""
<div style="background:#14142a;border:1px solid {dir_color};
            border-radius:12px;padding:14px;margin-bottom:10px;text-align:center">
  <p style="font-size:1.3rem;font-weight:700;color:#e0e0e0;margin:0">{sym}</p>
  <p style="font-size:1.1rem;color:{dir_color};margin:4px 0">
    {dir_arrow} {bs.direction if bs else "—"}
  </p>
  <p style="font-size:0.82rem;color:#9e9e9e;margin:2px 0">
    {bs.strategy if bs else "—"}
  </p>
  <p style="font-size:0.82rem;color:#ffd740;margin:2px 0">
    R:R {r.best_rr:.1f}x &nbsp;|&nbsp; Conv {bs.conviction if bs else 0}/100
  </p>
  <p style="font-size:0.78rem;color:#9e9e9e;margin:2px 0">
    {r.composite_emoji} {r.composite_label}
  </p>
  <p style="font-size:0.78rem;color:#9e9e9e;margin:2px 0">
    PCR {r.pcr:.3f if r.pcr else "—"} &nbsp;|&nbsp; {r.total_setups} setup(s)
  </p>
</div>""",
                        unsafe_allow_html=True,
                    )

            # Quick-select to jump to single-symbol analysis
            st.markdown("**Jump to symbol →**")
            jump_col1, jump_col2 = st.columns([2, 4])
            with jump_col1:
                jump_sym = st.selectbox(
                    "Select symbol to analyse",
                    options=syms_with,
                    key="scanner_jump_sym",
                    label_visibility="collapsed",
                )
            with jump_col2:
                if st.button(
                    f"📈 Load {jump_sym} in main dashboard", key="scanner_jump_btn"
                ):
                    st.session_state["symbol"] = jump_sym
                    if jump_sym not in st.session_state["recent_symbols"]:
                        st.session_state["recent_symbols"].insert(0, jump_sym)
                    with st.spinner(f"Loading {jump_sym}…"):
                        run_compute_pipeline()
                    st.toast(
                        f"Switched to {jump_sym} — check Wall Overview, Insider, Signal and Trade Ideas tabs"
                    )

        else:
            st.info(
                "No symbols generated valid setups with current filters. Try lowering Min Conviction or Min R:R."
            )

        st.divider()

        # Charts row
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(
                cb.scanner_bubble_chart(scan_summary_df),
                use_container_width=True,
                key="scan_bubble",
            )
        with ch2:
            st.plotly_chart(
                cb.top_setups_bar(scan_setups_df),
                use_container_width=True,
                key="scan_topbar",
            )

        st.divider()

        # Full market overview grid
        st.markdown("### 📋 Full Market Overview")
        if scan_summary_df is not None and not scan_summary_df.empty:
            disp_cols = [
                "Symbol",
                "Signal",
                "PCR",
                "Setups",
                "Best Setup",
                "Best R:R",
                "Conviction",
                "Acc Zones",
                "Dist Zones",
                "Support",
                "Resistance",
                "Max Pain",
                "IV Skew",
            ]
            show_df = scan_summary_df[
                [c for c in disp_cols if c in scan_summary_df.columns]
            ]

            def _style_scanner_row(row):
                sig = str(row.get("Signal", ""))
                if "BULL" in sig.upper():
                    return ["background-color: rgba(0,230,118,0.06)"] * len(row)
                elif "BEAR" in sig.upper():
                    return ["background-color: rgba(255,82,82,0.06)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                show_df.style.apply(_style_scanner_row, axis=1),
                use_container_width=True,
                height=400,
            )
            csv_scan = scan_summary_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export Scanner CSV", csv_scan, "scanner_results.csv", "text/csv"
            )

        st.divider()

        # All setups ranked table
        st.markdown("### 🏆 All Setups Across Watchlist — Ranked by Conviction")
        if scan_setups_df is not None and not scan_setups_df.empty:
            # Filters
            f1, f2, f3 = st.columns(3)
            with f1:
                dir_f = st.multiselect(
                    "Direction",
                    ["LONG", "SHORT"],
                    default=["LONG", "SHORT"],
                    key="scan_dir_f",
                )
            with f2:
                tf_f = st.multiselect(
                    "Timeframe",
                    ["Intraday", "Swing", "Positional"],
                    default=["Intraday", "Swing", "Positional"],
                    key="scan_tf_f",
                )
            with f3:
                sym_f = st.multiselect(
                    "Symbols",
                    scan_setups_df["Symbol"].unique().tolist(),
                    default=scan_setups_df["Symbol"].unique().tolist(),
                    key="scan_sym_f",
                )

            filt = scan_setups_df[
                scan_setups_df["Direction"].isin(dir_f)
                & scan_setups_df["Timeframe"].isin(tf_f)
                & scan_setups_df["Symbol"].isin(sym_f)
            ]

            all_cols = [
                "Symbol",
                "Direction",
                "Strategy",
                "Timeframe",
                "Entry Zone",
                "Target 1",
                "Stop Loss",
                "R:R",
                "Conviction",
                "Move %",
                "Risk %",
                "PCR",
                "Signal",
            ]
            show_all = filt[[c for c in all_cols if c in filt.columns]]

            def _style_all(row):
                if row.get("Direction") == "LONG":
                    return ["background-color: rgba(0,230,118,0.06)"] * len(row)
                return ["background-color: rgba(255,82,82,0.06)"] * len(row)

            st.dataframe(
                show_all.style.apply(_style_all, axis=1),
                use_container_width=True,
                height=450,
            )
            csv_all = filt.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export All Setups CSV", csv_all, "all_setups.csv", "text/csv"
            )

        # Errors / no data
        if err_results:
            with st.expander(f"⚠️ {len(err_results)} symbols with no data or errors"):
                for r in err_results:
                    st.caption(
                        f"**{r.symbol}** — {r.status}: {r.error_msg or 'No options CSV found in ./data/options/'}"
                    )

        st.divider()
        st.caption(
            "⚠️ Scanner uses the same demo data generation for symbols without CSVs. "
            "For real signals, place `{SYMBOL}_options_{YYYYMMDD}.csv` files in `./data/options/`."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ⚡ GEX (Gamma Exposure)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("## ⚡ Gamma Exposure (GEX)")
    st.caption(
        "GEX measures dealer hedging pressure at each strike. "
        "**Positive GEX** (green) → dealers long gamma → they dampen moves (pin). "
        "**Negative GEX** (red) → dealers short gamma → they amplify moves (trending). "
        "**HVL** = level where net gamma flips — above it is rangebound, below it is volatile."
    )

    if options_df is None or options_df.empty:
        st.warning("Load options data first.")
    else:
        # ── Controls row ──────────────────────────────────────────────────────
        gx1, gx2, gx3 = st.columns(3)
        with gx1:
            sym_upper = st.session_state["symbol"].upper()
            default_lot = NSE_LOT_SIZES.get(sym_upper, DEFAULT_LOT_SIZE)
            lot_size = st.number_input(
                "Lot Size",
                min_value=1,
                max_value=50000,
                value=default_lot,
                step=25,
                key="gex_lot",
                help="Contract lot size for this symbol. Pre-filled from NSE defaults.",
            )
        with gx2:
            risk_free = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=6.5,
                step=0.25,
                key="gex_rf",
                help="RBI repo rate proxy (default 6.5%)",
            )
        with gx3:
            max_dte = st.slider(
                "Max DTE (days)",
                7,
                180,
                90,
                step=7,
                key="gex_maxdte",
                help="Ignore expiries further than this many days out",
            )

        # ── Compute GEX ───────────────────────────────────────────────────────
        @st.cache_data(ttl=300, show_spinner=False)
        def _compute_gex(_options_hash, symbol, spot, lot_size, risk_free, max_dte):
            # We pass options_df via session state since it can't be hashed
            df = st.session_state.get("options_df", pd.DataFrame())
            calc = GEXCalculator(
                df,
                symbol=symbol,
                spot=spot,
                lot_size=lot_size,
                risk_free=risk_free / 100,
                max_dte_days=max_dte,
            )
            return calc.compute()

        with st.spinner("⚡ Computing GEX across all expiries…"):
            gex_result = _compute_gex(
                id(options_df),  # cache buster — changes when df changes
                sym_upper,
                cmp or 0.0,
                lot_size,
                risk_free,
                max_dte,
            )

        if not gex_result.expiries:
            st.warning(
                "Could not compute GEX — options data may be missing IV column, "
                "or all expiries are beyond the Max DTE filter."
            )
        else:
            # ── KPI row ───────────────────────────────────────────────────────
            all_df = gex_result.all_strikes
            net_gex = float(all_df["Net_GEX"].sum()) if not all_df.empty else 0.0
            abs_gex = float(all_df["Net_GEX"].abs().sum()) if not all_df.empty else 0.0
            hvl = gex_result.overall_hvl
            cr = gex_result.overall_call_resistance
            ps = gex_result.overall_put_support
            n_exp = len(gex_result.expiries)

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric(
                "Net GEX",
                f"{net_gex:+.2f}B",
                help="Positive = dealers long gamma (range); Negative = short gamma (trend)",
            )
            k2.metric(
                "|GEX| Total",
                f"{abs_gex:.2f}B",
                help="Total gamma exposure magnitude across all expiries",
            )
            k3.metric(
                "HVL",
                f"₹{hvl:.0f}" if hvl else "N/A",
                help="High Volatility Level — gamma zero-cross. Below = volatile regime.",
            )
            k4.metric(
                "Call Resistance",
                f"₹{cr:.0f}" if cr else "N/A",
                help="Strongest positive GEX strike above spot",
            )
            k5.metric(
                "Put Support",
                f"₹{ps:.0f}" if ps else "N/A",
                help="Strongest negative GEX strike below spot",
            )
            k6.metric("Expiries", n_exp, help="Number of expiry dates analysed")

            # GEX regime badge
            if net_gex > 0:
                st.success(
                    "🟢 **Positive GEX Regime** — Dealers long gamma, expect mean-reversion / range-bound price action. Strong support/resistance at walls."
                )
            else:
                st.error(
                    "🔴 **Negative GEX Regime** — Dealers short gamma, expect trend amplification and volatile moves. Levels break more easily."
                )

            # ── AI Commentary ──────────────────────────────────────────────────
            st.divider()
            st.markdown("### 🧠 AI Commentary")

            # Build a compact data summary to send to Claude
            _exp_summary = []
            for _e in gex_result.expiries[:6]:
                _cr_str = f"{_e.call_resistance:.0f}" if _e.call_resistance else "N/A"
                _ps_str = f"{_e.put_support:.0f}" if _e.put_support else "N/A"
                _hvl_str = f"{_e.hvl:.0f}" if _e.hvl else "N/A"
                _exp_summary.append(
                    f"  - {_e.expiry} (DTE {_e.dte_days}): net_gex={_e.net_gex:+,.0f}, "
                    f"|gex|={_e.total_gex:,.0f}, {_e.gex_pct_of_total:.1f}% of total, "
                    f"CR={_cr_str}, PS={_ps_str}, HVL={_hvl_str}"
                )

            _hvl_str_all = f"{hvl:.0f}" if hvl else "N/A"
            _cr_str_all = f"{cr:.0f}" if cr else "N/A"
            _ps_str_all = f"{ps:.0f}" if ps else "N/A"
            _regime_str = (
                "POSITIVE (dealer long gamma)"
                if net_gex > 0
                else "NEGATIVE (dealer short gamma)"
            )

            _gex_prompt = (
                f"You are a professional NSE options analyst. Analyse the following GEX "
                f"(Gamma Exposure) data for {sym_upper} and provide a concise but thorough commentary.\n\n"
                f"SYMBOL: {sym_upper}\n"
                f"SPOT: {gex_result.spot:.1f}\n"
                f"NET GEX (all expiries): {net_gex:+,.0f}\n"
                f"|GEX| TOTAL: {abs_gex:,.0f}\n"
                f"OVERALL HVL: {_hvl_str_all}\n"
                f"OVERALL CALL RESISTANCE: {_cr_str_all}\n"
                f"OVERALL PUT SUPPORT: {_ps_str_all}\n"
                f"GEX REGIME: {_regime_str}\n\n"
                f"PER-EXPIRY BREAKDOWN:\n" + "\n".join(_exp_summary) + "\n\n"
                "Write your commentary in this exact structure (use markdown headers):\n\n"
                "### What the GEX is telling us\n"
                "2-3 sentences explaining the overall gamma positioning and what it means for price action right now.\n\n"
                "### Key levels to watch\n"
                "Bullet points for each significant level (Call Resistance, Put Support, HVL) "
                "with a specific trading implication for each.\n\n"
                "### Regime analysis\n"
                "Is this a pinning/ranging environment or a trending/volatile one? "
                "Which expiry is driving the most gamma pressure and what does that imply about the upcoming move?\n\n"
                "### Directional bias\n"
                "Based purely on GEX structure: bullish / bearish / neutral, and why. "
                "Include what would change this view (e.g. what happens if price crosses HVL or a key wall).\n\n"
                "Keep the tone professional, specific to the numbers, and actionable. "
                "No generic statements. Under 350 words total."
            )

            _commentary_key = f"gex_commentary_{sym_upper}_{round(net_gex/1e6)}"
            if _commentary_key not in st.session_state:
                st.session_state[_commentary_key] = None

            _col_btn, _col_refresh = st.columns([3, 1])
            with _col_refresh:
                _regen = st.button("🔄 Regenerate", key="gex_regen")

            if st.session_state[_commentary_key] is None or _regen:
                with st.spinner("🧠 Analysing GEX structure…"):
                    try:
                        import requests as _req

                        _resp = _req.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": "claude-sonnet-4-20250514",
                                "max_tokens": 1000,
                                "messages": [{"role": "user", "content": _gex_prompt}],
                            },
                            timeout=30,
                        )
                        _data = _resp.json()
                        _text = "".join(
                            b.get("text", "")
                            for b in _data.get("content", [])
                            if b.get("type") == "text"
                        )
                        st.session_state[_commentary_key] = (
                            _text or "No commentary returned."
                        )
                    except Exception as _ce:
                        st.session_state[_commentary_key] = (
                            f"Commentary unavailable: {_ce}"
                        )

            if st.session_state.get(_commentary_key):
                _commentary_text = st.session_state[_commentary_key]
                st.markdown(
                    "<div style='background:#0f0f1f;border:1px solid #2a2a4a;"
                    "border-left:3px solid #7c5cbf;border-radius:10px;"
                    "padding:20px 24px;margin:8px 0'>",
                    unsafe_allow_html=True,
                )
                st.markdown(_commentary_text)
                st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

            # ── 4-panel layout ─────────────────────────────────────────────────
            st.markdown("### Per-Expiry GEX Panels")
            _wall_res = key_levels.get("primary_resistance")
            _wall_sup = key_levels.get("primary_support")
            figs = cb.gex_4panel(
                gex_result,
                height_each=460,
                oi_resistance=_wall_res,
                oi_support=_wall_sup,
            )

            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                st.plotly_chart(figs[0], use_container_width=True, key="gex_p0")
            with row1_c2:
                st.plotly_chart(figs[1], use_container_width=True, key="gex_p1")

            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                st.plotly_chart(figs[2], use_container_width=True, key="gex_p2")
            with row2_c2:
                st.plotly_chart(figs[3], use_container_width=True, key="gex_p3")

            st.divider()

            # ── GEX by expiry bar ──────────────────────────────────────────────
            st.plotly_chart(
                cb.gex_all_expiry_chart(gex_result),
                use_container_width=True,
                key="gex_expiry_bar",
            )

            st.divider()

            # ── Aggregate across all expiries ──────────────────────────────────
            st.markdown("### All Expiries Combined")
            st.plotly_chart(
                cb.gex_aggregate_chart(
                    gex_result, oi_resistance=_wall_res, oi_support=_wall_sup
                ),
                use_container_width=True,
                key="gex_agg",
            )

            st.divider()

            # ── Expiry detail table ────────────────────────────────────────────
            with st.expander("📋 All Expiries Detail Table"):
                exp_rows = [
                    {
                        "Expiry": e.expiry,
                        "DTE": e.dte_days,
                        "Net GEX (B)": round(e.net_gex, 4),
                        "|GEX| (B)": round(e.total_gex, 4),
                        "% of Total": f"{e.gex_pct_of_total:.1f}%",
                        "Call Resistance": (
                            round(e.call_resistance, 0) if e.call_resistance else "—"
                        ),
                        "Put Support": (
                            round(e.put_support, 0) if e.put_support else "—"
                        ),
                        "HVL": round(e.hvl, 0) if e.hvl else "—",
                    }
                    for e in gex_result.expiries
                ]
                st.dataframe(pd.DataFrame(exp_rows), use_container_width=True)

            st.divider()
            st.caption(
                "⚠️ GEX assumes standard Black-Scholes gamma. "
                "Accuracy depends on IV quality from the options chain. "
                "Lot sizes are pre-set but can be adjusted above."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Historical Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
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
                cb.pcr_trend_chart(pd.Series(pcr_vals), dates),
                use_container_width=True,
                key="hist_pcr_trend",
            )
        with col_h2:
            st.plotly_chart(
                cb.max_pain_migration_chart(mp_vals, dates),
                use_container_width=True,
                key="hist_mp_migrate",
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
                cb.oi_change_chart(walls_today, walls_yest),
                use_container_width=True,
                key="hist_oi_change",
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
                    key="hist_pcr_trend2",
                )
            with col_h2:
                st.plotly_chart(
                    cb.max_pain_migration_chart(mp_series, dates_list),
                    use_container_width=True,
                    key="hist_mp_migrate2",
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

with tabs[7]:  # Settings
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

        # Every time the component fires a rerun, re-download + re-run the pipeline
        if refresh_count and refresh_count > 0:
            with st.spinner("⏱️ Auto-refreshing live data from NSE…"):
                try:
                    run_compute_pipeline(download_options=True)
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

# ─── footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;padding:8px 0 4px;color:#4a4a6a;font-size:11px;"
    "letter-spacing:0.08em'>"
    "⬡ <b style='color:#4ade80'>EMERALD SLATE</b> &nbsp;·&nbsp; "
    "NSE Signal Dashboard &nbsp;·&nbsp; "
    "For informational purposes only. Not financial advice."
    "</div>",
    unsafe_allow_html=True,
)
