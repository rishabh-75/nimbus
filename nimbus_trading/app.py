"""
NIMBUS — Weekly Momentum Dashboard  ⬡ Emerald Slate
────────────────────────────────────────────────────
4hr BB (1σ, 20-period) + Williams %R (50-period) + Options Context

One screen. Four questions answered before every trade:
  1. Is the momentum signal valid?        BB position + WR zone
  2. Will the market trend or pin?        GEX regime
  3. How far can price run?               OI walls
  4. Is expiry timing for or against us?  Days to expiry + pin risk
"""

import time
import datetime

import streamlit as st
import pandas as pd

# ── page config — MUST be first Streamlit call ────────────────────────────────
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
  --bg:      #080c10; --surface: #0d1117; --surface2: #111827;
  --border:  #1e2937; --em:      #10b981; --em2:      #059669;
  --red:     #ef4444; --gold:    #f59e0b; --white:    #e2e8f0;
  --muted:   #64748b; --green:   #22c55e; --violet:   #8b5cf6;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'JetBrains Mono',monospace!important;color:var(--white)!important}
[data-testid="stHeader"]{background:var(--bg)!important}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)}
#MainMenu,footer,.stDeployButton{visibility:hidden}
h1,h2,h3{font-family:'Syne',sans-serif!important;letter-spacing:-0.02em}

/* metrics */
[data-testid="stMetric"]{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 14px!important}
[data-testid="stMetricValue"]{font-size:1.3rem!important;font-weight:600!important;color:var(--white)!important;font-family:'JetBrains Mono',monospace!important}
[data-testid="stMetricLabel"]{font-size:0.6rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--muted)!important}
[data-testid="stMetricDelta"]{font-size:.72rem!important}

/* buttons */
[data-testid="stButton"]>button{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--white)!important;font-family:'JetBrains Mono',monospace!important;font-size:.75rem!important;letter-spacing:.06em!important;border-radius:4px!important;transition:border-color .2s}
[data-testid="stButton"]>button:hover{border-color:var(--em)!important;color:var(--em)!important}

/* selectbox / inputs */
[data-testid="stSelectbox"]>div,[data-testid="stTextInput"]>div>div{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:4px!important;font-family:'JetBrains Mono',monospace!important;font-size:.8rem!important}

/* plotly */
[data-testid="stPlotlyChart"]{border:1px solid var(--border);border-radius:6px}

hr{border-color:var(--border)!important;margin:6px 0!important}

.card{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:14px 16px;margin-bottom:10px}
.card-title{font-family:'Syne',sans-serif;font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}

/* score badge */
.score-num{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;line-height:1;letter-spacing:-.04em}
.score-lbl{font-size:.65rem;letter-spacing:.16em;text-transform:uppercase;font-weight:600;margin-top:4px}

/* workflow lines */
.wf-line{font-size:.76rem;line-height:1.75;padding:5px 0;border-bottom:1px solid var(--border)}
.wf-line:last-child{border-bottom:none}
.wf-label{font-size:.56rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:2px}

/* verdict */
.verdict{background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.3);border-left:3px solid var(--em);border-radius:4px;padding:10px 14px;font-size:.78rem;line-height:1.7}
.verdict-half{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.3);border-left:3px solid var(--gold)}
.verdict-skip{background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.3);border-left:3px solid var(--red)}

/* checklist */
.chk{display:flex;align-items:flex-start;gap:8px;padding:4px 0;font-size:.73rem;border-bottom:1px solid var(--border)}
.chk:last-child{border-bottom:none}
.chk-sub{color:var(--muted);font-size:.67rem;margin-top:2px}

/* pills */
.pill{display:inline-block;padding:2px 8px;border-radius:3px;font-size:.62rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600}
.p-g{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
.p-r{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.p-y{background:rgba(245,158,11,.15);color:var(--gold);border:1px solid rgba(245,158,11,.3)}
.p-v{background:rgba(139,92,246,.15);color:var(--violet);border:1px solid rgba(139,92,246,.3)}
.p-m{background:rgba(100,116,139,.15);color:var(--muted);border:1px solid rgba(100,116,139,.3)}

/* header */
.hdr{display:flex;align-items:baseline;gap:12px;padding:4px 0 12px;border-bottom:1px solid var(--border);margin-bottom:14px}
.hdr-logo{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:var(--em);letter-spacing:-.02em}
.hdr-sym{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600}
.hdr-price{font-family:'JetBrains Mono',monospace;font-size:1rem;color:var(--em)}
.hdr-sub{font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-left:auto}

/* level rows */
.lvl{display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:.76rem;border-bottom:1px solid var(--border)}
.lvl:last-child{border-bottom:none}
.lvl-name{color:var(--muted);font-size:.6rem;letter-spacing:.08em;text-transform:uppercase}
</style>
""",
    unsafe_allow_html=True,
)

# ── module imports after config ───────────────────────────────────────────────
from modules.data import (
    download_options,
    get_price_4h,
    parse_uploaded_csv,
    infer_spot,
    is_market_open,
    NSE_LOT_SIZES,
)
from modules.indicators import add_bollinger, add_williams_r, bb_signal, wr_signal
from modules.analytics import analyze
from modules.commentary import get_commentary
import modules.charts as ch

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    symbol="NIFTY",
    price_df=pd.DataFrame(),
    options_df=pd.DataFrame(),
    spot=None,
    ctx=None,
    commentary=None,
    last_refresh=None,
    price_status="",
    options_status="",
    lot_size=75,
    bb_period=20,
    bb_std=1.0,
    wr_period=50,
    wr_thresh=-20.0,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(fetch_options: bool = False) -> None:
    sym = st.session_state["symbol"]
    lot_size = st.session_state["lot_size"]

    # 1. Price
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

    # 2. Spot
    spot = infer_spot(st.session_state["options_df"])
    if spot is None and not df_p.empty:
        spot = float(df_p["Close"].iloc[-1])
    st.session_state["spot"] = spot

    # 3. Options chain
    if fetch_options:
        _prog = st.empty()

        def _cb(msg):
            _prog.caption(f"⬇ {msg}")

        with st.spinner("Downloading options chain from NSE…"):
            df_o, msg_o = download_options(sym, progress_cb=_cb)
        _prog.empty()
        if df_o is not None and not df_o.empty:
            st.session_state["options_df"] = df_o
            st.session_state["options_status"] = msg_o
            # Update spot from freshly downloaded data
            s2 = infer_spot(df_o)
            if s2:
                st.session_state["spot"] = s2
                spot = s2
        else:
            st.error(
                f"⚠ Options unavailable: {msg_o}\n\n"
                "💡 Upload an NSE options chain CSV from the sidebar instead."
            )

    # 4. Analytics
    df_o = st.session_state["options_df"]
    spot = st.session_state["spot"]
    if not df_o.empty and spot:
        ctx = analyze(df_o, spot=spot, lot_size=lot_size)
        st.session_state["ctx"] = ctx
        # 5. Commentary
        df_p2 = st.session_state["price_df"]
        st.session_state["commentary"] = get_commentary(ctx, df_p2, sym)
    else:
        st.session_state["ctx"] = None
        st.session_state["commentary"] = None

    st.session_state["last_refresh"] = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div style='padding:8px 0 16px'>
      <span style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:#10b981'>⬡ NIMBUS</span>
      <span style='font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:#64748b;margin-left:8px'>EMERALD SLATE</span>
    </div>""",
        unsafe_allow_html=True,
    )

    # Symbol
    SYMBOLS = [
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
        SYMBOLS,
        index=(
            SYMBOLS.index(st.session_state["symbol"])
            if st.session_state["symbol"] in SYMBOLS
            else 0
        ),
        label_visibility="visible",
    )
    custom = st.text_input(
        "Custom symbol", placeholder="e.g. NESTLEIND", label_visibility="collapsed"
    )
    new_sym = custom.strip().upper() if custom.strip() else sym_sel

    if new_sym != st.session_state["symbol"]:
        st.session_state["symbol"] = new_sym
        st.session_state["options_df"] = pd.DataFrame()
        st.session_state["last_refresh"] = None  # force re-run

    st.divider()

    # Lot size
    default_lot = NSE_LOT_SIZES.get(st.session_state["symbol"], 75)
    st.session_state["lot_size"] = st.number_input(
        "Lot Size", value=default_lot, min_value=1, step=25
    )

    st.divider()

    # Indicator params
    st.markdown("**Indicators**")
    st.session_state["bb_period"] = st.slider("BB Period", 10, 50, 20)
    st.session_state["bb_std"] = st.slider(
        "BB Std Dev", 0.5, 2.5, 1.0, 0.25, help="1.0 = momentum; 2.0 = classic"
    )
    st.session_state["wr_period"] = st.slider(
        "Williams %R Period", 14, 100, 50, help="50 = ~2.5-week lookback on 4h"
    )
    st.session_state["wr_thresh"] = float(
        st.slider(
            "%R Threshold",
            -50,
            -10,
            -20,
            help="Above = momentum zone. -20 is standard.",
        )
    )

    st.divider()

    # Options chain
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

    fetch_btn = st.button(
        "⬇  Fetch from NSE",
        use_container_width=True,
        help="Downloads live options chain. Uses proven session pattern.",
    )

    st.divider()
    st.markdown(
        """
    <div style='font-size:.58rem;color:#64748b;line-height:1.9'>
      <b style='color:#10b981'>⬡ EMERALD SLATE</b><br>
      NIMBUS v3 · Weekly Momentum<br>
      4H BB(20,1σ) + W%%R(50)<br><br>
      <i>Not financial advice.</i>
    </div>""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TRIGGER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
first_load = st.session_state["last_refresh"] is None
if first_load or fetch_btn:
    run_pipeline(fetch_options=True)  # always try to fetch on first load
elif st.session_state["price_df"].empty:
    run_pipeline(fetch_options=False)

# Recompute indicators if params changed without re-fetching
price_df = st.session_state["price_df"]
if not price_df.empty:
    price_df = add_bollinger(
        price_df,
        period=st.session_state["bb_period"],
        std_dev=st.session_state["bb_std"],
    )
    price_df = add_williams_r(price_df, period=st.session_state["wr_period"])

# ══════════════════════════════════════════════════════════════════════════════
# STATE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
symbol = st.session_state["symbol"]
spot = st.session_state["spot"]
ctx = st.session_state["ctx"]
commentary = st.session_state["commentary"]
last_ref = st.session_state["last_refresh"]

cmp_str = f"{spot:,.2f}" if spot else "—"
cmp_delta = None
if not price_df.empty and len(price_df) >= 2:
    chg = float(price_df["Close"].iloc[-1]) - float(price_df["Close"].iloc[-2])
    chg_pct = chg / float(price_df["Close"].iloc[-2]) * 100
    cmp_delta = f"{chg:+.2f} ({chg_pct:+.2f}%)"

age_str = ""
if last_ref:
    age = int(time.time() - last_ref)
    age_str = f"Last refresh: {age}s ago"

mkt = "🟢 MARKET OPEN" if is_market_open() else "🔴 MARKET CLOSED"
ts = datetime.datetime.now().strftime("%H:%M:%S  %d %b %Y")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
chg_html = (
    f"<span style='color:{'#22c55e' if cmp_delta and chg >= 0 else '#ef4444'};font-size:.85rem'>"
    f"{cmp_delta}</span>"
    if cmp_delta
    else ""
)
st.markdown(
    f"""
<div class='hdr'>
  <span class='hdr-logo'>⬡ NIMBUS</span>
  <span class='hdr-sym'>{symbol}</span>
  <span class='hdr-price'>{cmp_str}</span>
  {chg_html}
  <span class='hdr-sub'>{mkt} &nbsp;·&nbsp; {ts} &nbsp;·&nbsp; {age_str}</span>
</div>""",
    unsafe_allow_html=True,
)


# Refresh button row
_hcol, _rcol = st.columns([11, 1])
with _rcol:
    if st.button("REFRESH", use_container_width=True):
        run_pipeline(fetch_options=False)
        st.rerun()

# ── CHART: full width ─────────────────────────────────────────────────────────
fig = ch.main_chart(price_df, ctx=ctx, symbol=symbol)
st.plotly_chart(fig, use_container_width=True, key="main_chart")

# ── KPI STRIP ─────────────────────────────────────────────────────────────────
bb = bb_signal(price_df) if not price_df.empty else {}
wr = (
    wr_signal(price_df, threshold=st.session_state["wr_thresh"])
    if not price_df.empty
    else {}
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("SPOT", cmp_str, cmp_delta or "—")
with k2:
    wr_v = f"{wr['wr_value']:.1f}" if wr else "—"
    wr_s = ("Momentum" if wr.get("in_momentum") else "No Momentum") if wr else "—"
    st.metric("WILLIAMS %R", wr_v, wr_s)
with k3:
    bp = f"{bb['bb_pct']:.2f}" if bb else "—"
    bs = bb.get("position", "—").replace("_", " ").title() if bb else "—"
    st.metric("BB %B", bp, bs)
with k4:
    st.metric(
        "PCR OI",
        f"{ctx.walls.pcr_oi:.3f}" if ctx else "—",
        ctx.walls.pcr_sentiment if ctx else "—",
    )
with k5:
    st.metric(
        "NET GEX",
        f"{ctx.gex.net_gex:+,.0f}M" if ctx else "—",
        ctx.gex.regime if ctx else "—",
    )
with k6:
    dte_v = (
        f"{ctx.expiry.days_remaining}d"
        if ctx and ctx.expiry.days_remaining < 99
        else "—"
    )
    dte_s = f"{ctx.expiry.pin_risk} pin" if ctx else "—"
    st.metric("TO EXPIRY", dte_v, dte_s)

st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)

# ── INTELLIGENCE ROW: 3 columns below chart ───────────────────────────────────
col_a, col_b, col_c = st.columns([42, 28, 30], gap="medium")

# COL A: Workflow Analysis + Verdict
with col_a:
    if commentary:
        sizing = commentary.get("sizing", "HALF")
        v_cls = {"SKIP": "verdict-skip", "HALF": "verdict-half"}.get(sizing, "")
        sz_pill = {"FULL": "p-g", "HALF": "p-y", "SKIP": "p-r"}.get(sizing, "p-m")
        st.markdown(
            f"""
        <div class='card'>
          <div class='card-title'>Workflow Analysis
            <span class='pill {sz_pill}' style='float:right'>SIZE: {sizing}</span>
          </div>
          <div class='wf-line'><span class='wf-label'>GEX Regime</span>{commentary.get('gex_line','—')}</div>
          <div class='wf-line'><span class='wf-label'>OI Walls</span>{commentary.get('wall_line','—')}</div>
          <div class='wf-line'><span class='wf-label'>Expiry</span>{commentary.get('expiry_line','—')}</div>
          <div class='wf-line'><span class='wf-label'>Williams %R</span>{commentary.get('wr_line','—')}</div>
          <div class='verdict {v_cls}'>{commentary.get('verdict','—')}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class='card'>
          <div class='card-title'>Workflow Analysis</div>
          <div style='color:#64748b;font-size:.74rem;line-height:2'>
            Fetch or upload an options chain to unlock:<br>
            &nbsp;· GEX regime &rarr; momentum amplified or dampened<br>
            &nbsp;· OI wall distance &rarr; target and stop placement<br>
            &nbsp;· Expiry timing &rarr; pin risk assessment<br>
            &nbsp;· Position sizing recommendation
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

# COL B: Viability Score + Key Levels
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
              <div style='font-size:.56rem;color:#64748b;margin-top:2px'>/100 · options context</div>
            </div>
            <span class='pill {sp}' style='margin-left:auto'>SIZE: {v.sizing}</span>
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

        w, g, e = ctx.walls, ctx.gex, ctx.expiry

        def _level_row(name, val, color):
            if val is None:
                return ""
            pct = ((val - spot) / spot * 100) if spot else 0
            return (
                f"<div class='lvl'>"
                f"<span class='lvl-name'>{name}</span>"
                f"<span style='font-weight:600;font-family:JetBrains Mono,monospace;"
                f"color:{color}'>{val:,.0f}</span>"
                f"<span style='font-size:.66rem;color:{color}'>{pct:+.1f}%</span>"
                f"</div>"
            )

        levels_html = (
            _level_row("Resistance", w.resistance, "#ef4444")
            + _level_row("Support", w.support, "#10b981")
            + _level_row("Max Pain", w.max_pain, "#f59e0b")
            + _level_row("GEX HVL", g.hvl, "#8b5cf6")
        )
        pin_cls = {"HIGH": "p-r", "MODERATE": "p-y", "LOW": "p-g"}.get(
            e.pin_risk, "p-m"
        )
        gex_cls = {"Positive": "p-r", "Negative": "p-g", "Neutral": "p-m"}.get(
            g.regime, "p-m"
        )
        pcr_cls = "p-g" if w.pcr_oi >= 1.1 else ("p-r" if w.pcr_oi < 0.8 else "p-m")
        dte_html = (
            f"<span class='pill p-m'>+{e.days_remaining}d</span>"
            if e.days_remaining < 99
            else ""
        )
        st.markdown(
            f"""
        <div class='card'>
          <div class='card-title'>Key Levels</div>
          {levels_html}
          <div style='margin-top:8px;display:flex;gap:5px;flex-wrap:wrap'>
            <span class='pill {pcr_cls}'>PCR {w.pcr_oi:.2f}</span>
            <span class='pill {pin_cls}'>PIN {e.pin_risk}</span>
            <span class='pill {gex_cls}'>GEX {g.regime.upper()}</span>
            {dte_html}
          </div>
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class='card'>
          <div class='card-title'>Trade Viability</div>
          <div style='color:#64748b;font-size:.74rem'>Load options chain to score setup</div>
        </div>""",
            unsafe_allow_html=True,
        )

# COL C: Pre-Trade Checklist + GEX bar
with col_c:
    if ctx and ctx.viability.checklist:
        icons = {"pass": "✅", "warn": "⚠️", "fail": "❌", "neutral": "◻️"}
        rows = "".join(
            f"<div class='chk'>"
            f"<span style='min-width:18px'>{icons.get(c.status,'◻️')}</span>"
            f"<div><div>{c.detail}</div>"
            f"<div class='chk-sub'>{c.implication}</div></div>"
            f"</div>"
            for c in ctx.viability.checklist
        )
        st.markdown(
            f"""
        <div class='card'>
          <div class='card-title'>Pre-Trade Checklist</div>
          {rows}
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class='card'>
          <div class='card-title'>Pre-Trade Checklist</div>
          <div style='color:#64748b;font-size:.74rem'>Load options chain to generate checklist</div>
        </div>""",
            unsafe_allow_html=True,
        )

    if ctx:
        st.plotly_chart(ch.gex_expiry_bar(ctx), use_container_width=True, key="gex_bar")


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH (market hours only)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from streamlit_autorefresh import st_autorefresh

    if is_market_open():
        count = st_autorefresh(interval=5 * 60 * 1000, key="autorefresh")
        if count and count > 0:
            run_pipeline(fetch_options=True)
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div style='text-align:center;padding:18px 0 6px;border-top:1px solid #1e2937;margin-top:14px'>
  <span style='font-family:Syne,sans-serif;font-size:.6rem;letter-spacing:.16em;color:#10b981;font-weight:700'>⬡ EMERALD SLATE</span>
  <span style='color:#1e2937;margin:0 10px'>|</span>
  <span style='font-size:.56rem;color:#64748b;letter-spacing:.08em'>NIMBUS v3 · WEEKLY MOMENTUM · 4H BB(1σ) + W%R(50) + OPTIONS CONTEXT</span>
  <span style='color:#1e2937;margin:0 10px'>|</span>
  <span style='font-size:.56rem;color:#64748b'>NOT FINANCIAL ADVICE</span>
</div>""",
    unsafe_allow_html=True,
)
