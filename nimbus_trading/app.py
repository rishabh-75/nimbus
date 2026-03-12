"""
NIMBUS — Weekly Momentum Dashboard  ⬡ Emerald Slate  v5
Tabs: Dashboard · Scanner · Watchlist
"""

import io
import time
import datetime
import logging

import streamlit as st
import pandas as pd

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="NIMBUS · Emerald Slate",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
  --bg:#080c10;--surface:#0d1117;--surface2:#111827;--border:#1e2937;
  --em:#10b981;--em2:#059669;--red:#ef4444;--gold:#f59e0b;--white:#e2e8f0;
  --muted:#64748b;--green:#22c55e;--violet:#8b5cf6;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'JetBrains Mono',monospace!important;color:var(--white)!important}
[data-testid="stHeader"]{background:var(--bg)!important}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)}
#MainMenu,footer,.stDeployButton{visibility:hidden}
h1,h2,h3{font-family:'Syne',sans-serif!important}

[data-testid="stMetric"]{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 14px!important}
[data-testid="stMetricValue"]{font-size:1.2rem!important;font-weight:600!important;color:var(--white)!important;font-family:'JetBrains Mono',monospace!important}
[data-testid="stMetricLabel"]{font-size:.56rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--muted)!important}
[data-testid="stMetricDelta"]{font-size:.68rem!important}

[data-testid="stButton"]>button{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--white)!important;font-family:'JetBrains Mono',monospace!important;font-size:.72rem!important;letter-spacing:.06em!important;border-radius:4px!important}
[data-testid="stButton"]>button:hover{border-color:var(--em)!important;color:var(--em)!important}
[data-testid="stSelectbox"]>div,[data-testid="stTextInput"]>div>div{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:4px!important;font-size:.8rem!important}
[data-testid="stPlotlyChart"]{border:1px solid var(--border);border-radius:6px}
hr{border-color:var(--border)!important;margin:6px 0!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:var(--surface)!important;border-bottom:1px solid var(--border)!important;gap:2px}
[data-testid="stTabs"] [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-family:'JetBrains Mono',monospace!important;font-size:.72rem!important;padding:8px 16px!important;border-radius:4px 4px 0 0!important}
[data-testid="stTabs"] [aria-selected="true"]{background:var(--surface2)!important;color:var(--em)!important;border-bottom:2px solid var(--em)!important}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px}

.card{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:14px 16px;margin-bottom:8px}
.card-title{font-family:'Syne',sans-serif;font-size:.54rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.regime-tile{border-radius:6px;padding:12px 16px;margin-bottom:8px;border-left:3px solid}
.regime-trend{background:rgba(16,185,129,.08);border-color:var(--em)}
.regime-pin{background:rgba(239,68,68,.08);border-color:var(--red)}
.regime-neutral{background:rgba(245,158,11,.08);border-color:var(--gold)}
.regime-label{font-family:'Syne',sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.regime-detail{font-size:.7rem;color:var(--muted);margin-top:4px;line-height:1.6}
.score-num{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;line-height:1;letter-spacing:-.04em}
.score-lbl{font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;font-weight:600;margin-top:3px}
.wf-line{font-size:.72rem;line-height:1.7;padding:5px 0;border-bottom:1px solid var(--border)}
.wf-line:last-child{border-bottom:none}
.wf-label{font-size:.52rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:1px}
.verdict{background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.3);border-left:3px solid var(--em);border-radius:4px;padding:10px 14px;font-size:.75rem;line-height:1.75;margin-top:10px}
.verdict-half{background:rgba(245,158,11,.06);border-color:rgba(245,158,11,.3);border-left-color:var(--gold)}
.verdict-skip{background:rgba(239,68,68,.06);border-color:rgba(239,68,68,.3);border-left-color:var(--red)}
.chk{display:flex;align-items:flex-start;gap:8px;padding:4px 0;font-size:.70rem;border-bottom:1px solid var(--border)}
.chk:last-child{border-bottom:none}
.chk-sub{color:var(--muted);font-size:.62rem;margin-top:2px}
.risk-note{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2);border-left:2px solid var(--red);border-radius:3px;padding:4px 10px;font-size:.67rem;color:#fca5a5;margin:2px 0}
.pill{display:inline-block;padding:2px 8px;border-radius:3px;font-size:.58rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600}
.p-g{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
.p-r{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.p-y{background:rgba(245,158,11,.15);color:var(--gold);border:1px solid rgba(245,158,11,.3)}
.p-v{background:rgba(139,92,246,.15);color:var(--violet);border:1px solid rgba(139,92,246,.3)}
.p-m{background:rgba(100,116,139,.15);color:var(--muted);border:1px solid rgba(100,116,139,.3)}
.p-b{background:rgba(16,185,129,.15);color:var(--em);border:1px solid rgba(16,185,129,.3)}
.lvl{display:grid;grid-template-columns:90px 1fr auto;align-items:center;padding:5px 0;font-size:.72rem;border-bottom:1px solid var(--border);gap:4px}
.lvl:last-child{border-bottom:none}
.lvl-name{color:var(--muted);font-size:.56rem;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}
.lvl-val{font-weight:600;font-family:'JetBrains Mono',monospace;text-align:right}
.lvl-pct{font-size:.66rem;text-align:right;min-width:48px}
.hdr{display:flex;align-items:baseline;gap:12px;padding:4px 0 12px;border-bottom:1px solid var(--border);margin-bottom:12px}
.hdr-logo{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--em);letter-spacing:-.02em}
.hdr-sym{font-size:.9rem;font-weight:600}
.hdr-price{font-size:.9rem;color:var(--em)}
.hdr-sub{font-size:.53rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-left:auto}
.risk-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:.72rem;border-bottom:1px solid var(--border)}
.risk-row:last-child{border-bottom:none}
.risk-label{color:var(--muted);font-size:.56rem;text-transform:uppercase;letter-spacing:.06em}
.alert-error{background:rgba(239,68,68,.10);border:1px solid rgba(239,68,68,.35);border-left:3px solid var(--red);border-radius:4px;padding:8px 14px;font-size:.72rem;margin:4px 0;color:#fca5a5}
.alert-warn{background:rgba(245,158,11,.10);border:1px solid rgba(245,158,11,.35);border-left:3px solid var(--gold);border-radius:4px;padding:8px 14px;font-size:.72rem;margin:4px 0;color:#fcd34d}
.wl-row{display:grid;grid-template-columns:100px 90px 60px 80px 100px 60px 60px 60px 50px;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid var(--border);font-size:.72rem}
.wl-header{color:var(--muted);font-size:.54rem;letter-spacing:.08em;text-transform:uppercase}
/* Refresh button — fixed width regardless of sidebar state */
div[data-testid="column"]:last-child > div > div[data-testid="stButton"] > button {
  min-width:88px!important;max-width:88px!important;white-space:nowrap!important;overflow:hidden!important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Imports ───────────────────────────────────────────────────────────────────
from modules.data import (
    download_options,
    get_price_4h,
    parse_uploaded_csv,
    infer_spot,
    is_market_open,
    NSE_LOT_SIZES,
    get_universe,
    NIFTY100_SYMBOLS,
)
from modules.indicators import add_bollinger, add_williams_r, compute_price_signals
from modules.analytics import analyze
from modules.commentary import get_commentary
from modules.scanner import scan_universe
import modules.charts as ch
import modules.watchlist as wl_mod

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = dict(
    symbol="NIFTY",
    price_df=pd.DataFrame(),
    options_df=pd.DataFrame(),
    spot=None,
    ctx=None,
    commentary=None,
    ps=None,
    last_refresh=None,
    price_status="",
    options_status="",
    lot_size=75,
    bb_period=20,
    bb_std=1.0,
    wr_period=50,
    wr_thresh=-20.0,
    room_thresh=3.0,
    account_size=500000,
    risk_pct=0.5,
    # Scanner
    scan_results=[],
    scan_ts=None,
    scan_symbols_count=0,
    active_tab=0,
    _switch_to_dashboard=False,
    # Watchlist
    watchlist=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load watchlist once
if st.session_state["watchlist"] is None:
    st.session_state["watchlist"] = wl_mod.load_watchlist()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE (Dashboard tab)
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(fetch_options: bool = False):
    sym = st.session_state["symbol"]
    with st.spinner("Fetching price data…"):
        df_p, msg_p = get_price_4h(sym)
    if not df_p.empty:
        df_p = add_bollinger(
            df_p,
            period=st.session_state["bb_period"],
            std_dev=st.session_state["bb_std"],
        )
        df_p = add_williams_r(df_p, period=st.session_state["wr_period"])
    st.session_state["price_df"] = df_p
    st.session_state["price_status"] = msg_p

    spot = infer_spot(st.session_state["options_df"])
    if spot is None and not df_p.empty:
        spot = float(df_p["Close"].iloc[-1])
    st.session_state["spot"] = spot

    if fetch_options:
        prog = st.empty()

        def _cb(msg):
            prog.caption(f"Downloading: {msg}")

        with st.spinner("Downloading options chain…"):
            df_o, msg_o = download_options(sym, progress_cb=_cb)
        prog.empty()
        if df_o is not None and not df_o.empty:
            st.session_state["options_df"] = df_o
            st.session_state["options_status"] = msg_o
            s2 = infer_spot(df_o)
            if s2:
                st.session_state["spot"] = s2
                spot = s2
        else:
            st.error(
                f"Options unavailable: {msg_o}\n\nUpload an NSE CSV via the sidebar."
            )

    df_o = st.session_state["options_df"]
    spot = st.session_state["spot"]

    ps = (
        compute_price_signals(df_p, wr_thresh=st.session_state["wr_thresh"])
        if not df_p.empty
        else None
    )
    st.session_state["ps"] = ps

    if not df_o.empty and spot:
        ctx = analyze(
            df_o,
            spot=spot,
            lot_size=st.session_state["lot_size"],
            price_signals=ps,
            room_thresh=st.session_state["room_thresh"],
        )
        st.session_state["ctx"] = ctx
        st.session_state["commentary"] = get_commentary(
            ctx, ps=ps, symbol=sym, price_df=df_p
        )
    else:
        st.session_state["ctx"] = None
        st.session_state["commentary"] = None

    st.session_state["last_refresh"] = time.time()


# ── Trigger pipeline ──────────────────────────────────────────────────────────
if st.session_state["last_refresh"] is None or st.session_state["price_df"].empty:
    run_pipeline(fetch_options=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style='padding:8px 0 16px'>
      <span style='font-family:Syne,sans-serif;font-size:.95rem;font-weight:800;color:#10b981'>⬡ NIMBUS</span>
      <span style='font-size:.54rem;letter-spacing:.12em;text-transform:uppercase;color:#64748b;margin-left:8px'>EMERALD SLATE v5</span>
    </div>""",
        unsafe_allow_html=True,
    )

    SYMS = [
        "NIFTY",
        "BANKNIFTY",
        "FINNIFTY",
        "MIDCPNIFTY",
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "AXISBANK",
        "KOTAKBANK",
        "BAJFINANCE",
    ]
    sym_sel = st.selectbox(
        "Symbol",
        SYMS,
        index=(
            SYMS.index(st.session_state["symbol"])
            if st.session_state["symbol"] in SYMS
            else 0
        ),
        key="sym_dropdown",
    )

    # Custom symbol — separate text input + Go button so entry is explicit
    _c1, _c2 = st.columns([4, 1])
    with _c1:
        custom = st.text_input(
            "Custom symbol",
            placeholder="e.g. NESTLEIND",
            label_visibility="collapsed",
            key="custom_sym_input",
        )
    with _c2:
        custom_go = st.button(
            "→",
            key="custom_sym_go",
            help="Load custom symbol",
            use_container_width=True,
        )

    # Determine active symbol: custom Go button wins, otherwise dropdown
    if custom_go and custom.strip():
        new_sym = custom.strip().upper()
    elif not custom.strip():
        # Dropdown changed (no custom text)
        new_sym = sym_sel
    else:
        # There's text in the box but Go wasn't pressed → keep current symbol
        new_sym = st.session_state["symbol"]

    if new_sym != st.session_state["symbol"]:
        st.session_state["symbol"] = new_sym
        st.session_state["options_df"] = pd.DataFrame()
        st.session_state["last_refresh"] = None
        st.rerun()  # Force full pipeline re-execution with new symbol

    st.divider()
    st.session_state["lot_size"] = st.number_input(
        "Lot Size",
        value=NSE_LOT_SIZES.get(st.session_state["symbol"], 75),
        min_value=1,
        step=25,
    )
    st.divider()
    st.markdown("**Indicators**")
    st.session_state["bb_period"] = st.slider("BB Period", 10, 50, 20)
    st.session_state["bb_std"] = st.slider("BB Std Dev", 0.5, 2.5, 1.0, 0.25)
    st.session_state["wr_period"] = st.slider("W%R Period", 14, 100, 50)
    st.session_state["wr_thresh"] = float(st.slider("%R Threshold", -50, -10, -20))
    st.session_state["room_thresh"] = float(
        st.slider("Min room to resistance (%)", 1.0, 6.0, 3.0, 0.5)
    )
    st.divider()
    st.markdown("**Risk Module**")
    st.session_state["account_size"] = st.number_input(
        "Account size (₹)",
        value=st.session_state["account_size"],
        min_value=10000,
        step=50000,
        format="%d",
    )
    st.session_state["risk_pct"] = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
    st.divider()
    st.markdown("**Options Chain**")
    uploaded = st.file_uploader(
        "Upload NSE CSV", type=["csv"], label_visibility="collapsed"
    )
    if uploaded:
        df_up, msg_up = parse_uploaded_csv(uploaded)
        if df_up is not None:
            st.session_state["options_df"] = df_up
            st.session_state["options_status"] = msg_up
            s = infer_spot(df_up)
            if s:
                st.session_state["spot"] = s
            st.success(msg_up)
    if st.button("Download from NSE", use_container_width=True):
        run_pipeline(fetch_options=True)
        st.rerun()
    st.divider()
    st.markdown(
        """<div style='font-size:.54rem;color:#64748b;line-height:1.9'>
      <b style='color:#10b981'>⬡ EMERALD SLATE</b><br>
      4H BB(20,1σ) + W%%R(50) + Daily Bias<br>
      Options context for sizing/veto<br><br>
      <i>Not financial advice.</i>
    </div>""",
        unsafe_allow_html=True,
    )


# ── Recompute if param sliders changed ────────────────────────────────────────
price_df = st.session_state["price_df"].copy()
if not price_df.empty:
    price_df = add_bollinger(
        price_df,
        period=st.session_state["bb_period"],
        std_dev=st.session_state["bb_std"],
    )
    price_df = add_williams_r(price_df, period=st.session_state["wr_period"])
    ps = compute_price_signals(price_df, wr_thresh=st.session_state["wr_thresh"])
else:
    ps = st.session_state["ps"]

symbol = st.session_state["symbol"]
spot = st.session_state["spot"]
ctx = st.session_state["ctx"]
commentary = st.session_state["commentary"]
last_ref = st.session_state["last_refresh"]

cmp_str = f"{spot:,.2f}" if spot else "—"
cmp_delta = None
chg = 0.0
if not price_df.empty and len(price_df) >= 2:
    chg = float(price_df["Close"].iloc[-1]) - float(price_df["Close"].iloc[-2])
    chg_pct = chg / float(price_df["Close"].iloc[-2]) * 100
    cmp_delta = f"{chg:+.2f} ({chg_pct:+.2f}%)"

age_str = f"Refreshed {int(time.time()-last_ref)}s ago" if last_ref else ""
mkt = "MARKET OPEN" if is_market_open() else "MARKET CLOSED"
mkt_dot = "🟢" if is_market_open() else "🔴"
ts = datetime.datetime.now().strftime("%H:%M  %d %b %Y")
chg_col = "#22c55e" if chg >= 0 else "#ef4444"
chg_html = (
    f"<span style='color:{chg_col};font-size:.82rem'>{cmp_delta}</span>"
    if cmp_delta
    else ""
)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER (above tabs — always visible)
# ══════════════════════════════════════════════════════════════════════════════
hdr_col, ref_col = st.columns([10, 1])
with hdr_col:
    st.markdown(
        f"""
    <div class='hdr'>
      <span class='hdr-logo'>⬡ NIMBUS</span>
      <span class='hdr-sym'>{symbol}</span>
      <span class='hdr-price'>{cmp_str}</span>
      {chg_html}
      <span class='hdr-sub'>{mkt_dot} {mkt} &nbsp;·&nbsp; {ts} &nbsp;·&nbsp; {age_str}</span>
    </div>""",
        unsafe_allow_html=True,
    )
with ref_col:
    if st.button("⟳ Refresh", use_container_width=False):
        run_pipeline(fetch_options=False)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_dash, tab_scan, tab_wl = st.tabs(["📊 Dashboard", "🔍 Scanner", "📋 Watchlist"])

# ── Programmatic tab switch (triggered by Scanner "Open in Dashboard" button) ─
if st.session_state.get("_switch_to_dashboard", False):
    st.session_state["_switch_to_dashboard"] = False
    import streamlit.components.v1 as _components

    _components.html(
        """
    <script>
    setTimeout(function(){
      var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
      if (tabs && tabs[0]) { tabs[0].click(); }
    }, 250);
    </script>
    """,
        height=0,
        scrolling=False,
    )


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 1 — DASHBOARD
# ╚══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # ── Badge row ────────────────────────────────────────────────────────────
    def _bias_pill(b):
        return {"BULLISH": "p-g", "BEARISH": "p-r", "NEUTRAL": "p-y"}.get(b, "p-m")

    def _vs_pill(v):
        return {"SQUEEZE": "p-v", "EXPANDED": "p-y", "NORMAL": "p-m"}.get(v, "p-m")

    def _state_pill(s):
        return {
            "RIDING_UPPER": "p-b",
            "FIRST_DIP": "p-y",
            "MID_BAND_BROKEN": "p-r",
            "CONSOLIDATING": "p-m",
        }.get(s, "p-m")

    def _regime_pill(r):
        return {"TREND-FRIENDLY": "p-g", "PINNING": "p-r", "NEUTRAL": "p-y"}.get(
            r, "p-m"
        )

    def _wr_phase_pill(p):
        return {"FRESH": "p-g", "DEVELOPING": "p-b", "LATE": "p-y", "NONE": "p-r"}.get(
            p, "p-m"
        )

    badges = "<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center'>"
    if ps:
        badges += (
            f"<span class='pill {_bias_pill(ps.daily_bias)}'>Daily: {ps.daily_bias}</span> "
            f"<span class='pill {_vs_pill(ps.vol_state)}'>Vol: {ps.vol_state}</span> "
            f"<span class='pill {_state_pill(ps.position_state)}'>{ps.position_state.replace('_',' ')}</span> "
        )
        wr_lbl = f"W%R {ps.wr_value:.1f}" if ps.wr_value is not None else "W%R —"
        ph_lbl = ps.wr_phase if ps.wr_phase != "NONE" else "NOT IN ZONE"
        badges += f"<span class='pill {_wr_phase_pill(ps.wr_phase)}'>{wr_lbl} · {ph_lbl}</span> "
    if ctx:
        badges += f"<span class='pill {_regime_pill(ctx.regime.regime)}'>Regime: {ctx.regime.regime}</span> "
    badges += "</div>"
    st.markdown(badges, unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = ch.main_chart(price_df, ctx=ctx, ps=ps, symbol=symbol)
    st.plotly_chart(fig, use_container_width=True, key="main_chart")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    with k1:
        st.metric("SPOT", cmp_str, cmp_delta or "—")
    with k2:
        wr_v = f"{ps.wr_value:.1f}" if ps and ps.wr_value else "—"
        st.metric("W%R(50)", wr_v, ps.wr_phase if ps else "—")
    with k3:
        st.metric(
            "BB %B",
            f"{ps.bb_pct:.2f}" if ps else "—",
            ps.bb_position.replace("_", " ").title() if ps else "—",
        )
    with k4:
        st.metric(
            "DAILY BIAS",
            ps.daily_bias if ps else "—",
            f"{ps.daily_bias_pct:+.1f}% vs SMA" if ps and ps.daily_sma else "—",
        )
    with k5:
        st.metric(
            "PCR OI",
            f"{ctx.walls.pcr_oi:.3f}" if ctx else "—",
            ctx.walls.pcr_sentiment if ctx else "—",
        )
    with k6:
        st.metric(
            "NET GEX",
            f"{ctx.gex.net_gex:+,.0f}M" if ctx else "—",
            ctx.gex.regime if ctx else "—",
        )
    with k7:
        st.metric(
            "TO EXPIRY",
            (
                f"{ctx.expiry.days_remaining}d"
                if ctx and ctx.expiry.days_remaining < 99
                else "—"
            ),
            f"{ctx.expiry.pin_risk} pin" if ctx else "—",
        )
    with k8:
        st.metric(
            "VOL STATE",
            ps.vol_state if ps else "—",
            f"{ps.bb_width_pctl:.0f}th pctl" if ps else "—",
        )

    st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)

    # ── Intelligence row ──────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([42, 28, 30], gap="medium")

    with col_a:
        if ctx:
            r = ctx.regime
            tc = {"TREND-FRIENDLY": "regime-trend", "PINNING": "regime-pin"}.get(
                r.regime, "regime-neutral"
            )
            rc = {
                "TREND-FRIENDLY": "#10b981",
                "PINNING": "#ef4444",
                "NEUTRAL": "#f59e0b",
            }.get(r.regime, "#64748b")
            cp = {"FULL": "p-g", "HALF": "p-y", "SKIP": "p-r"}.get(r.size_cap, "p-m")
            st.markdown(
                f"""
            <div class='regime-tile {tc}'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <span class='regime-label' style='color:{rc}'>Regime: {r.regime}</span>
                <span class='pill {cp}'>Max: {r.size_cap}</span>
              </div>
              <div class='regime-detail'>{r.detail}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        if commentary:
            sz = commentary.get("sizing", "HALF")
            vc = {"SKIP": "verdict-skip", "HALF": "verdict-half"}.get(sz, "")
            sp = {"FULL": "p-g", "HALF": "p-y", "SKIP": "p-r"}.get(sz, "p-m")
            bl = commentary.get("bias_line", "")
            st.markdown(
                f"""
            <div class='card'>
              <div class='card-title'>Workflow Analysis
                <span class='pill {sp}' style='float:right'>SIZE: {sz}</span>
              </div>
              {"<div class='wf-line'><span class='wf-label'>Trend & State</span>"+bl+"</div>" if bl else ""}
              <div class='wf-line'><span class='wf-label'>GEX Regime</span>{commentary.get('gex_line','—')}</div>
              <div class='wf-line'><span class='wf-label'>OI Walls</span>{commentary.get('wall_line','—')}</div>
              <div class='wf-line'><span class='wf-label'>Expiry</span>{commentary.get('expiry_line','—')}</div>
              <div class='wf-line'><span class='wf-label'>Williams %R</span>{commentary.get('wr_line','—')}</div>
              <div class='verdict {vc}'>{commentary.get('verdict','—')}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class='card'><div class='card-title'>Workflow Analysis</div>
              <div style='color:#64748b;font-size:.74rem;line-height:2'>
                Fetch or upload options chain to unlock full analysis.
              </div></div>""",
                unsafe_allow_html=True,
            )

    with col_b:
        if ctx:
            v = ctx.viability
            sc = {
                "green": "#22c55e",
                "emerald": "#10b981",
                "yellow": "#f59e0b",
                "red": "#ef4444",
            }.get(v.color, "#64748b")
            sp = {"FULL": "p-g", "HALF": "p-y", "SKIP": "p-r"}.get(v.sizing, "p-m")
            st.markdown(
                f"""
            <div class='card'>
              <div class='card-title'>Trade Viability</div>
              <div style='display:flex;align-items:baseline;gap:10px'>
                <span class='score-num' style='color:{sc}'>{v.score}</span>
                <div>
                  <div class='score-lbl' style='color:{sc}'>{v.label}</div>
                  <div style='font-size:.55rem;color:#64748b;margin-top:2px'>/100</div>
                </div>
                <span class='pill {sp}' style='margin-left:auto'>SIZE: {v.sizing}</span>
              </div>
              {"<div style='margin-top:8px'>"+"".join(f"<div class='risk-note'>⚠ {n}</div>" for n in v.risk_notes[:3])+"</div>" if v.risk_notes else ""}
            </div>""",
                unsafe_allow_html=True,
            )

            w, g, e = ctx.walls, ctx.gex, ctx.expiry

            def _lvl(name, val, color):
                if val is None:
                    return ""
                pct = ((val - spot) / spot * 100) if spot else 0
                return (
                    f"<div class='lvl'><span class='lvl-name'>{name}</span>"
                    f"<span class='lvl-val' style='color:{color}'>{val:,.0f}</span>"
                    f"<span class='lvl-pct' style='color:{color}'>{pct:+.1f}%</span></div>"
                )

            lvls = (
                _lvl("Resistance", w.resistance, "#ef4444")
                + _lvl("Support", w.support, "#10b981")
                + _lvl("Max Pain", w.max_pain, "#f59e0b")
                + _lvl("GEX HVL", g.hvl, "#8b5cf6")
            )
            if ps and ps.daily_sma:
                lvls += _lvl(
                    "Daily SMA",
                    ps.daily_sma,
                    "#10b981" if ps.daily_bias == "BULLISH" else "#ef4444",
                )
            pc = "p-g" if w.pcr_oi >= 1.1 else ("p-r" if w.pcr_oi < 0.8 else "p-m")
            pi = {"HIGH": "p-r", "MODERATE": "p-y", "LOW": "p-g"}.get(e.pin_risk, "p-m")
            gc = {"Positive": "p-r", "Negative": "p-g", "Neutral": "p-m"}.get(
                g.regime, "p-m"
            )
            dte_h = (
                f"<span class='pill p-m'>+{e.days_remaining}d</span>"
                if e.days_remaining < 99
                else ""
            )
            st.markdown(
                f"""
            <div class='card'>
              <div class='card-title'>Key Levels</div>
              {lvls}
              <div style='margin-top:8px;display:flex;gap:5px;flex-wrap:wrap'>
                <span class='pill {pc}'>PCR {w.pcr_oi:.2f}</span>
                <span class='pill {pi}'>PIN {e.pin_risk}</span>
                <span class='pill {gc}'>GEX {g.regime.upper()}</span>
                {dte_h}
              </div>
            </div>""",
                unsafe_allow_html=True,
            )

            if spot and w.support:
                acct = st.session_state["account_size"]
                r_pct = st.session_state["risk_pct"]
                entry = spot
                stop = w.support
                stop_d = (stop - entry) / entry * 100
                target = w.resistance or (entry * 1.05)
                rr = abs((target - entry) / (entry - stop)) if entry != stop else 0
                risk_a = acct * r_pct / 100
                lots = int(
                    risk_a / max(abs(entry - stop) * st.session_state["lot_size"], 1)
                )
                st.markdown(
                    f"""
                <div class='card'>
                  <div class='card-title'>Risk Calculator</div>
                  <div class='risk-row'><span class='risk-label'>Entry (last close)</span><span>{entry:,.0f}</span></div>
                  <div class='risk-row'><span class='risk-label'>Stop (put wall)</span><span style='color:#ef4444'>{stop:,.0f} ({stop_d:+.1f}%)</span></div>
                  <div class='risk-row'><span class='risk-label'>Target (call wall)</span><span style='color:#10b981'>{target:,.0f}</span></div>
                  <div class='risk-row'><span class='risk-label'>R:R</span><span style='color:#f59e0b'>1 : {rr:.1f}</span></div>
                  <div class='risk-row'><span class='risk-label'>Risk ({r_pct:.1f}%)</span><span>₹{risk_a:,.0f}</span></div>
                  <div class='risk-row'><span class='risk-label'>Max lots</span><span style='font-weight:600'>{lots}</span></div>
                </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """<div class='card'><div class='card-title'>Trade Viability</div>
              <div style='color:#64748b;font-size:.74rem'>Load options chain to score setup</div></div>""",
                unsafe_allow_html=True,
            )

    with col_c:
        if ctx and ctx.viability.checklist:
            icons = {"pass": "✅", "warn": "⚠️", "fail": "❌", "neutral": "◻️"}
            rows = "".join(
                f"<div class='chk'><span style='min-width:18px'>{icons.get(c.status,'◻️')}</span>"
                f"<div><div>{c.detail}</div><div class='chk-sub'>{c.implication}</div></div></div>"
                for c in ctx.viability.checklist
            )
            st.markdown(
                f"""<div class='card'><div class='card-title'>Pre-Trade Checklist</div>{rows}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class='card'><div class='card-title'>Pre-Trade Checklist</div>
              <div style='color:#64748b;font-size:.74rem'>Load options chain to generate checklist</div></div>""",
                unsafe_allow_html=True,
            )
        if ctx:
            st.plotly_chart(
                ch.gex_expiry_bar(ctx), use_container_width=True, key="gex_bar"
            )


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 2 — SCANNER
# ╚══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    # ── Control bar ───────────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4, sc5 = st.columns([2, 2, 2, 2, 4])
    with sc1:
        run_scan = st.button("▶ Run Scanner", use_container_width=True)
    with sc2:
        min_via = st.slider("Min Viability", 0, 100, 50, key="scan_min_via")
    with sc3:
        min_room = st.slider(
            "Min % to Resistance", 0.0, 15.0, 5.0, 0.5, key="scan_min_room"
        )
    with sc4:
        req_all = st.toggle("Require All Filters", value=True, key="scan_req_all")
    with sc5:
        univ_sel = st.selectbox(
            "Universe",
            ["NIFTY 100", "F&O Full", "Watchlist", "Custom"],
            key="scan_universe",
        )

    # Status line
    sr = st.session_state["scan_results"]
    sts = st.session_state["scan_ts"]
    sc_n = st.session_state["scan_symbols_count"]
    if sts:
        st.caption(
            f"Last scan: {sts.strftime('%H:%M:%S %d %b')} · {sc_n} symbols evaluated · {len(sr)} passed filters"
        )

    # ── Filter chip toggles ───────────────────────────────────────────────────
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        st.checkbox(
            "Momentum (BB + W%R)",
            value=True,
            disabled=True,
            key="fc_mom",
            help="Hard gate — always enforced. BB riding + W%R > −20 are non-negotiable entry conditions.",
        )
        f_mom = True  # always True — hard gate enforced in scan_universe()
    with fc2:
        f_str = st.checkbox("Structure (Room to Wall)", value=True, key="fc_str")
    with fc3:
        f_reg = st.checkbox("Regime (Trend-Friendly)", value=True, key="fc_reg")
    with fc4:
        f_exp = st.checkbox("Expiry (DTE ≥ 5)", value=True, key="fc_exp")
    with fc5:
        f_bias = st.checkbox("Daily Bias (Bullish)", value=True, key="fc_bias")

    # ── Run scan ──────────────────────────────────────────────────────────────
    if run_scan:
        wl_entries = st.session_state["watchlist"]
        if univ_sel == "NIFTY 100":
            syms_to_scan = NIFTY100_SYMBOLS
        elif univ_sel == "F&O Full":
            with st.spinner("Fetching F&O universe from NSE…"):
                syms_to_scan = get_universe()
        elif univ_sel == "Watchlist":
            syms_to_scan = [e["symbol"] for e in wl_entries]
        else:  # Custom
            custom_input = st.text_input(
                "Enter symbols (comma-separated)",
                key="scan_custom_input",
                placeholder="RELIANCE,TCS,INFY",
            )
            syms_to_scan = (
                [s.strip().upper() for s in custom_input.split(",") if s.strip()]
                if "scan_custom_input" in st.session_state
                else NIFTY100_SYMBOLS
            )

        n_syms = len(syms_to_scan)
        prog_bar = st.progress(0, text="Starting scan…")
        prog_text = st.empty()

        def _progress(done, total, sym):
            pct = done / total
            prog_bar.progress(pct, text=f"Scanning {sym}… ({done}/{total})")

        with st.spinner(f"Scanning {n_syms} symbols…"):
            raw_results = scan_universe(
                symbols=syms_to_scan,
                require_all_filters=req_all,  # honouring the UI toggle for non-hard-gate filters
                min_viability=min_via,
                progress_cb=_progress,
                max_workers=8,
            )

        prog_bar.empty()
        prog_text.empty()

        # Apply chip filters
        def _passes(row):
            if f_mom and not row["passes_momentum"]:
                return False
            if f_str and not row["passes_structure"]:
                return False
            if f_reg and not row["passes_regime"]:
                return False
            if f_exp and not row["passes_expiry"]:
                return False
            if f_bias and not row["passes_daily_bias"]:
                return False
            if (
                row["pct_to_resistance"] is not None
                and row["pct_to_resistance"] < min_room
            ):
                return False
            return True

        filtered = [r for r in raw_results if _passes(r)]

        st.session_state["scan_results"] = filtered
        st.session_state["scan_ts"] = datetime.datetime.now()
        st.session_state["scan_symbols_count"] = n_syms
        st.rerun()

    # ── Results table ─────────────────────────────────────────────────────────
    results = st.session_state["scan_results"]
    if not results:
        st.markdown(
            """<div style='padding:40px;text-align:center;color:#64748b;font-size:.8rem'>
          No results yet. Click <b style='color:#10b981'>▶ Run Scanner</b> to scan the universe.
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        # Build display dataframe
        rows = []
        for r in results:
            rows.append(
                {
                    "Symbol": r["symbol"],
                    "Last": f"₹{r['last_price']:,.1f}" if r["last_price"] else "—",
                    "Score": r["viability_score"],
                    "Size": r["size_suggestion"],
                    "Regime": r["gex_regime"] or "—",
                    "%→Res": (
                        f"{r['pct_to_resistance']:+.1f}%"
                        if r["pct_to_resistance"]
                        else "—"
                    ),
                    "%→Sup": (
                        f"{r['pct_to_support']:+.1f}%" if r["pct_to_support"] else "—"
                    ),
                    "W%R": f"{r['wr_50']:.0f}" if r["wr_50"] else "—",
                    "DTE": str(r["dte"]) if r["dte"] else "—",
                    "Bias": r["daily_bias"],
                    "Vol": r["vol_state"],
                    "Scanned": r.get("scan_timestamp", "—"),
                    "Reason": r["short_reason"],
                }
            )
        df_display = pd.DataFrame(rows)

        def _colour_score(val):
            if isinstance(val, int):
                if val >= 70:
                    return "background-color:#1a2e1a;color:#22c55e;font-weight:700"
                if val >= 50:
                    return "background-color:#2e2a1a;color:#f59e0b;font-weight:700"
            return ""

        def _colour_size(val):
            c = {"FULL": "#22c55e", "HALF": "#f59e0b", "ZERO": "#ef4444"}.get(val, "")
            return f"color:{c};font-weight:600" if c else ""

        styled = df_display.style.applymap(_colour_score, subset=["Score"]).applymap(
            _colour_size, subset=["Size"]
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width=80),
                "Score": st.column_config.NumberColumn("Viability", format="%d"),
                "Scanned": st.column_config.TextColumn(
                    "Scanned at",
                    width=90,
                    help="Time this row's data was fetched. Dashboard re-fetches fresh data.",
                ),
                "Reason": st.column_config.TextColumn("Reason", width=280),
            },
        )

        # Per-row actions
        st.markdown("**Actions** — select a row:")
        act1, act2, act3 = st.columns([3, 2, 2])
        with act1:
            sel_sym = st.selectbox(
                "Symbol to act on",
                [r["symbol"] for r in results],
                key="scan_sel_sym",
                label_visibility="collapsed",
            )
        with act2:
            if st.button("📊 Open in Dashboard", use_container_width=True):
                st.session_state["symbol"] = sel_sym
                st.session_state["options_df"] = pd.DataFrame()
                st.session_state["last_refresh"] = None
                st.session_state["_switch_to_dashboard"] = True
                run_pipeline(fetch_options=True)
                st.rerun()
        with act3:
            if st.button("➕ Add to Watchlist", use_container_width=True):
                row_data = next((r for r in results if r["symbol"] == sel_sym), {})
                st.session_state["watchlist"] = wl_mod.add_entry(
                    st.session_state["watchlist"],
                    sel_sym,
                    tags=["scanner"],
                    notes=row_data.get("short_reason", ""),
                )
                st.success(f"{sel_sym} added to watchlist.")

        # Explain the expected score difference when opening dashboard
        _sel_row = next((r for r in results if r["symbol"] == sel_sym), {})
        _scan_ts = _sel_row.get("scan_timestamp", "")
        st.caption(
            f"ℹ️ Score shown is from scan data ({_scan_ts}). "
            f"Dashboard always fetches the **latest** price bar — "
            f"if conditions changed since the scan, the score will differ. "
            f"This is correct behaviour, not a bug."
        )

        # CSV export
        csv_buf = io.StringIO()
        df_display.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"nimbus_scan_{datetime.date.today()}.csv",
            mime="text/csv",
        )


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 3 — WATCHLIST
# ╚══════════════════════════════════════════════════════════════════════════════
with tab_wl:
    wl = st.session_state["watchlist"]

    # ── Alerts banner ─────────────────────────────────────────────────────────
    alerts = wl_mod.detect_alerts(wl)
    for a in alerts:
        cls = "alert-error" if a["severity"] == "error" else "alert-warn"
        icon = "🚨" if a["severity"] == "error" else "⚠️"
        st.markdown(
            f"<div class='{cls}'>{icon} {a['message']}</div>", unsafe_allow_html=True
        )

    # ── Add symbol form ───────────────────────────────────────────────────────
    with st.expander("➕ Add to Watchlist", expanded=len(wl) == 0):
        wa1, wa2, wa3, wa4, wa5 = st.columns([2, 1.5, 1.5, 1.5, 3])
        with wa1:
            wl_sym = (
                st.text_input("Symbol", key="wl_add_sym", placeholder="SBIN")
                .upper()
                .strip()
            )
        with wa2:
            wl_entry = st.number_input("Entry ₹", value=0.0, min_value=0.0, key="wl_ep")
        with wa3:
            wl_stop = st.number_input("Stop ₹", value=0.0, min_value=0.0, key="wl_sp")
        with wa4:
            wl_target = st.number_input(
                "Target ₹", value=0.0, min_value=0.0, key="wl_tp"
            )
        with wa5:
            wl_notes = st.text_input(
                "Notes", key="wl_notes", placeholder="Entry thesis…"
            )
        if st.button("+ Add to Watchlist", use_container_width=False):
            if wl_sym:
                st.session_state["watchlist"] = wl_mod.add_entry(
                    st.session_state["watchlist"],
                    wl_sym,
                    entry_price=wl_entry or None,
                    stop_price=wl_stop or None,
                    target_price=wl_target or None,
                    notes=wl_notes,
                )
                wl = st.session_state["watchlist"]
                st.success(f"{wl_sym} added.")
                st.rerun()
            else:
                st.warning("Enter a symbol first.")

    # ── Refresh button ────────────────────────────────────────────────────────
    wl_r1, wl_r2 = st.columns([2, 8])
    with wl_r1:
        do_refresh = st.button("🔄 Refresh Watchlist", use_container_width=True)

    if do_refresh and wl:
        prog_wl = st.progress(0, text="Refreshing…")

        def _wl_prog(done, total, sym):
            prog_wl.progress(done / total, text=f"Refreshing {sym}… ({done}/{total})")

        with st.spinner("Re-evaluating watchlist…"):
            wl = wl_mod.refresh_watchlist(wl, progress_cb=_wl_prog)
        prog_wl.empty()
        st.session_state["watchlist"] = wl
        wl_mod.save_watchlist(wl)
        st.rerun()

    # ── Watchlist table ───────────────────────────────────────────────────────
    if not wl:
        st.markdown(
            """<div style='padding:40px;text-align:center;color:#64748b;font-size:.8rem'>
          Watchlist is empty. Add symbols above or use ➕ Add to Watchlist in the Scanner.
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        for idx, entry in enumerate(wl):
            sym = entry["symbol"]
            live = entry.get("live", {})
            ep = entry.get("entry_price")
            sp_ = entry.get("stop_price")
            tp = entry.get("target_price")
            notes = entry.get("notes", "")
            pnl = wl_mod.calc_pnl(entry)
            added = entry.get("added_at", "")[:10]

            lp = live.get("last_price")
            l_score = live.get("viability_score")
            l_state = live.get("position_state", "—")
            l_wr = live.get("wr_50")
            l_res = live.get("pct_to_resistance")
            l_sup = live.get("pct_to_support")
            l_dte = live.get("dte")
            l_bias = live.get("daily_bias", "—")
            l_reg = live.get("gex_regime", "—")

            # State pill colour
            state_c = {
                "RIDING_UPPER_BAND": "#10b981",
                "FIRST_DIP": "#f59e0b",
                "MID_BAND_BROKEN": "#ef4444",
            }.get(l_state, "#64748b")
            score_c = (
                "#22c55e"
                if (l_score or 0) >= 70
                else ("#f59e0b" if (l_score or 0) >= 50 else "#ef4444")
            )
            pnl_c = "#22c55e" if (pnl or 0) >= 0 else "#ef4444"

            # Low viability flag
            flag = " ⚑" if l_score is not None and l_score < 40 else ""

            with st.container():
                st.markdown(
                    f"""
                <div style='background:var(--surface2);border:1px solid var(--border);
                  border-radius:6px;padding:10px 14px;margin-bottom:6px'>
                  <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap'>
                    <span style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;color:var(--white)'>{sym}{flag}</span>
                    <span style='font-size:.7rem;color:#64748b'>Added {added}</span>
                    {"<span style='font-size:.8rem;color:#e2e8f0'>₹"+f"{lp:,.1f}"+"</span>" if lp else ""}
                    {"<span style='font-size:.75rem;font-weight:700;color:"+score_c+"'>Score "+str(l_score)+"</span>" if l_score is not None else ""}
                    <span style='font-size:.72rem;color:{state_c}'>{l_state.replace("_"," ")}</span>
                    {"<span style='font-size:.72rem'>W%R "+str(round(l_wr,0))+"</span>" if l_wr else ""}
                    {"<span style='font-size:.72rem'>Res "+f"{l_res:+.1f}%"+"</span>" if l_res else ""}
                    {"<span style='font-size:.72rem'>DTE "+str(l_dte)+"d</span>" if l_dte else ""}
                    {"<span style='font-size:.72rem;color:#64748b'>Entry ₹"+f"{ep:,.1f}"+"</span>" if ep else ""}
                    {"<span style='font-size:.72rem;color:#ef4444'>Stop ₹"+f"{sp_:,.1f}"+"</span>" if sp_ else ""}
                    {"<span style='font-size:.72rem;color:#10b981'>Target ₹"+f"{tp:,.1f}"+"</span>" if tp else ""}
                    {"<span style='font-size:.8rem;font-weight:700;color:"+pnl_c+"'>P&L "+f"{pnl:+.1f}%"+"</span>" if pnl is not None else ""}
                  </div>
                  {"<div style='font-size:.66rem;color:#64748b;margin-top:4px'>"+notes+"</div>" if notes else ""}
                </div>""",
                    unsafe_allow_html=True,
                )

                # Action buttons per row
                ba1, ba2, ba3, ba4 = st.columns([2, 2, 2, 6])
                with ba1:
                    if st.button("📊 View", key=f"wl_view_{sym}_{idx}"):
                        st.session_state["symbol"] = sym
                        st.session_state["options_df"] = pd.DataFrame()
                        st.session_state["last_refresh"] = None
                        st.session_state["_switch_to_dashboard"] = True
                        run_pipeline(fetch_options=True)
                        st.rerun()
                with ba2:
                    if st.button("✏️ Edit", key=f"wl_edit_{sym}_{idx}"):
                        st.session_state[f"wl_editing_{sym}"] = True
                with ba3:
                    if st.button("🗑️ Remove", key=f"wl_del_{sym}_{idx}"):
                        st.session_state["watchlist"] = wl_mod.remove_entry(
                            st.session_state["watchlist"], sym
                        )
                        st.rerun()

                # Inline edit form
                if st.session_state.get(f"wl_editing_{sym}"):
                    with st.form(key=f"wl_form_{sym}_{idx}"):
                        e1, e2, e3, e4 = st.columns(4)
                        with e1:
                            new_ep = st.number_input(
                                "Entry", value=ep or 0.0, min_value=0.0
                            )
                        with e2:
                            new_sp = st.number_input(
                                "Stop", value=sp_ or 0.0, min_value=0.0
                            )
                        with e3:
                            new_tp = st.number_input(
                                "Target", value=tp or 0.0, min_value=0.0
                            )
                        with e4:
                            new_notes = st.text_input("Notes", value=notes)
                        if st.form_submit_button("Save"):
                            st.session_state["watchlist"] = wl_mod.update_entry(
                                st.session_state["watchlist"],
                                sym,
                                entry_price=new_ep or None,
                                stop_price=new_sp or None,
                                target_price=new_tp or None,
                                notes=new_notes,
                            )
                            st.session_state[f"wl_editing_{sym}"] = False
                            st.rerun()


# ── Auto-refresh (market hours) ───────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh

    if is_market_open():
        count = st_autorefresh(interval=5 * 60 * 1000, key="autorefresh")
        if count and count > 0:
            run_pipeline(fetch_options=True)
except ImportError:
    pass

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style='text-align:center;padding:16px 0 6px;border-top:1px solid #1e2937;margin-top:10px'>
  <span style='font-family:Syne,sans-serif;font-size:.56rem;letter-spacing:.16em;color:#10b981;font-weight:700'>⬡ EMERALD SLATE v5</span>
  <span style='color:#1e2937;margin:0 8px'>|</span>
  <span style='font-size:.52rem;color:#64748b'>4H BB(1σ) + W%R(50) + DAILY BIAS · SCANNER · WATCHLIST</span>
  <span style='color:#1e2937;margin:0 8px'>|</span>
  <span style='font-size:.52rem;color:#64748b'>NOT FINANCIAL ADVICE</span>
</div>""",
    unsafe_allow_html=True,
)
