"""
ui/dashboard_tab.py — Each intelligence panel individually scrollable.
Regime name wraps (no fixedHeight). Viability score properly aligned.
"""

from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
)
from PyQt6.QtGui import QFont
from ui.theme import (
    BG,
    SURFACE,
    S2,
    BORDER,
    EM,
    RED,
    GOLD,
    MUTED,
    WHITE,
    GREEN,
    BLUE,
    VIOLET,
    FONT_MONO,
    FONT_UI,
    score_color,
    wr_color,
)
from ui.chart_widget import NimbusChart
from ui.components import KPITile, PanelFrame, Badge, ChecklistRow, LevelRow, RiskNote
from modules.analytics import OptionsContext
from modules.etf_analyzer import ETFContext
from modules.indicators import PriceSignals

logger = logging.getLogger(__name__)


def _h(t, c=MUTED, s=7):
    l = QLabel(t.upper())
    l.setFont(QFont(FONT_UI, s))
    l.setStyleSheet(f"color: {c}; letter-spacing: 0.8px;")
    return l


def _t(t="", c=WHITE, s=8, mono=False, bold=False):
    l = QLabel(t)
    f = QFont(FONT_MONO if mono else FONT_UI, s)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {c};")
    l.setWordWrap(True)
    return l


def _sep():
    s = QFrame()
    s.setFixedHeight(1)
    s.setStyleSheet(f"background: {BORDER};")
    return s


def _make_scroll_panel(widget: QWidget, height: int = 280) -> QScrollArea:
    """Wrap a widget in a thin-scrollbar QScrollArea."""
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFixedHeight(height)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    sa.setStyleSheet(
        "QScrollArea { border: none; background: transparent; }"
        "QScrollBar:vertical { background: #0D1117; width: 5px; border: none; }"
        "QScrollBar::handle:vertical { background: #1E2937; border-radius: 2px; min-height: 20px; }"
        "QScrollBar::handle:vertical:hover { background: #10B981; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
    )
    sa.setWidget(widget)

    from PyQt6.QtWidgets import QSizePolicy

    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    widget.setMinimumWidth(0)
    sa.viewport().setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )

    return sa


class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._symbol = ""
        self._spot = 0.0
        self._price_df: Optional[pd.DataFrame] = None
        self._ps: Optional[PriceSignals] = None
        self._ctx: Optional[OptionsContext] = None
        self._fv = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 4)
        root.setSpacing(5)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QFrame()
        hdr.setFixedHeight(30)
        hdr.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px;"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 12, 0)
        self._sym = _t("--", WHITE, 12, mono=True, bold=True)
        hl.addWidget(self._sym)
        hl.addSpacing(8)
        self._spot_lbl = _t("", WHITE, 11, mono=True)
        hl.addWidget(self._spot_lbl)
        hl.addSpacing(6)
        self._chg = _t("", MUTED, 9, mono=True)
        hl.addWidget(self._chg)
        hl.addStretch(1)
        self._ts = _t("", MUTED, 7, mono=True)
        hl.addWidget(self._ts)
        root.addWidget(hdr)

        # ── Badge row ─────────────────────────────────────────────────────────
        br = QHBoxLayout()
        br.setSpacing(4)
        self._badges = {}
        for k in ("DAILY", "VOL", "STATE", "W%R", "REGIME"):
            b = Badge("--")
            br.addWidget(b)
            self._badges[k] = b
        br.addStretch(1)
        root.addLayout(br)

        # ── Chart (stretch=1) ─────────────────────────────────────────────────
        self._chart = NimbusChart()
        root.addWidget(self._chart, stretch=1)

        # ── KPI tiles ─────────────────────────────────────────────────────────
        kr = QHBoxLayout()
        kr.setSpacing(5)
        self._kpi = {}
        for k in (
            "SPOT",
            "W%R(50)",
            "BB %B",
            "DAILY BIAS",
            "PCR OI",
            "NET GEX",
            "TO EXPIRY",
            "VOL STATE",
        ):
            t = KPITile(k)
            kr.addWidget(t)
            self._kpi[k] = t
        root.addLayout(kr)

        # ── Intelligence row: each panel individually scrollable ──────────────
        intel = QHBoxLayout()
        intel.setSpacing(5)

        pa_widget = QWidget()
        self._build_panel_a(pa_widget)
        intel.addWidget(_make_scroll_panel(pa_widget, 260), 42)

        pb_widget = QWidget()
        self._build_panel_b(pb_widget)
        intel.addWidget(_make_scroll_panel(pb_widget, 260), 28)

        pc_widget = QWidget()
        self._build_panel_c(pc_widget)
        intel.addWidget(_make_scroll_panel(pc_widget, 260), 30)

        fstrip = QFrame()
        fstrip.setFixedHeight(30)
        fstrip.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px;"
        )
        fl = QHBoxLayout(fstrip)
        fl.setContentsMargins(8, 0, 8, 0)
        fl.setSpacing(6)
        self._setup_badge = Badge("SETUP: —")
        self._filing_badge = Badge("NO FILING")
        fl.addWidget(self._setup_badge)
        fl.addWidget(self._filing_badge)
        self._conviction_lbl = QLabel("○○○○○○○○○○")
        self._conviction_lbl.setFont(QFont(FONT_MONO, 7))
        self._conviction_lbl.setStyleSheet(f"color: {MUTED};")
        fl.addWidget(self._conviction_lbl)
        fl.addSpacing(6)
        self._filing_detail_lbl = QLabel("Fetching filings…")
        self._filing_detail_lbl.setFont(QFont(FONT_UI, 7))
        self._filing_detail_lbl.setStyleSheet(f"color: {MUTED};")
        fl.addWidget(self._filing_detail_lbl, stretch=1)
        self._filing_recency_lbl = QLabel("")
        self._filing_recency_lbl.setFont(QFont(FONT_MONO, 7))
        self._filing_recency_lbl.setStyleSheet(f"color: {MUTED};")
        fl.addWidget(self._filing_recency_lbl)
        root.addWidget(fstrip)

        root.addLayout(intel)

    # ── Panel A: Regime + Workflow ────────────────────────────────────────────
    def _build_panel_a(self, container):
        self._pa = PanelFrame("", accent=EM)
        pa = self._pa.content_layout()
        pa.setSpacing(4)

        # Regime tile — NO fixedHeight so name wraps naturally
        rt = QFrame()
        rt.setStyleSheet(
            f"background: rgba(16,185,129,0.06); border-left: 3px solid {EM}; "
            f"border-radius: 4px; padding: 4px 8px;"
        )
        self._regime_frame = rt
        rl = QVBoxLayout(rt)
        rl.setContentsMargins(4, 2, 4, 2)
        rl.setSpacing(2)
        rh = QHBoxLayout()
        self._regime_lbl = _t("REGIME: --", EM, 8, bold=True)
        self._regime_lbl.setWordWrap(True)
        rh.addWidget(self._regime_lbl, stretch=1)
        self._regime_cap = Badge("--")
        rh.addWidget(self._regime_cap)
        rl.addLayout(rh)
        self._regime_detail = _t("", MUTED, 7)
        rl.addWidget(self._regime_detail)
        pa.addWidget(rt)

        pa.addWidget(_sep())

        wh = QHBoxLayout()
        wh.addWidget(_h("Workflow Analysis"))
        wh.addStretch(1)
        self._wf_badge = Badge("--")
        wh.addWidget(self._wf_badge)
        pa.addLayout(wh)

        self._wf = {}
        for key in ("TREND & STATE", "GEX REGIME", "OI WALLS", "EXPIRY", "WILLIAMS %R"):
            pa.addWidget(_h(key, MUTED, 6))
            body = _t("", WHITE, 7)
            pa.addWidget(body)
            self._wf[key] = body

        self._verdict = QLabel("")
        self._verdict.setFont(QFont(FONT_UI, 7))
        self._verdict.setWordWrap(True)
        self._verdict.setStyleSheet(
            f"background: rgba(16,185,129,0.06); color: {WHITE}; "
            f"border-left: 3px solid {EM}; border-radius: 3px; padding: 5px 8px;"
        )
        pa.addWidget(self._verdict)

        pa.addWidget(_sep())
        pa.addWidget(_h("Filing Intelligence"))
        sh = QHBoxLayout()
        self._filing_setup_lbl = _t("SETUP: —", MUTED, 7, bold=True)
        sh.addWidget(self._filing_setup_lbl, stretch=1)
        self._filing_conv_badge = Badge("CONV —")
        sh.addWidget(self._filing_conv_badge)
        pa.addLayout(sh)
        self._filing_subject_lbl = _t("", MUTED, 7)
        self._filing_deal_lbl = _t("", WHITE, 7)
        pa.addWidget(self._filing_subject_lbl)
        pa.addWidget(self._filing_deal_lbl)

        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._pa)

    # ── Panel B: Viability + Levels ───────────────────────────────────────────
    def _build_panel_b(self, container):
        self._pb = PanelFrame("Trade Viability", accent=GOLD)
        pb = self._pb.content_layout()
        pb.setSpacing(4)

        # Score row — no fixedWidth, natural alignment
        sr = QHBoxLayout()
        sr.setSpacing(6)
        self._score = _t("--", MUTED, 24, mono=True, bold=True)
        sr.addWidget(self._score)

        sv = QVBoxLayout()
        sv.setSpacing(0)
        self._vlabel = _t("", MUTED, 8, bold=True)
        sv.addWidget(self._vlabel)
        sv.addWidget(_t("/100", MUTED, 6))
        sr.addLayout(sv)
        sr.addStretch(1)
        self._size_badge = Badge("--")
        self._size_badge.setMaximumWidth(90)
        self._size_badge.setMinimumWidth(60)
        sr.addWidget(self._size_badge)
        pb.addLayout(sr)

        self._risk_container = QVBoxLayout()
        self._risk_container.setSpacing(2)
        pb.addLayout(self._risk_container)
        pb.addWidget(_sep())
        pb.addWidget(_h("Key Levels"))

        self._levels: list[LevelRow] = []
        for _ in range(5):
            lr = LevelRow()
            lr.setVisible(False)
            pb.addWidget(lr)
            self._levels.append(lr)

        bb = QHBoxLayout()
        bb.setSpacing(3)
        self._bot = {}
        for k in ("PCR", "PIN", "GEX", "DTE"):
            b = Badge("--")
            b.setFixedHeight(16)
            bb.addWidget(b)
            self._bot[k] = b
        bb.addStretch(1)
        pb.addLayout(bb)

        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._pb)

    # ── Panel C: Checklist ────────────────────────────────────────────────────
    def _build_panel_c(self, container):
        self._pc = PanelFrame("Pre-Trade Checklist", accent=BORDER)
        pc = self._pc.content_layout()
        pc.setSpacing(2)
        self._checks: list[ChecklistRow] = []
        for _ in range(10):
            cr = ChecklistRow()
            cr.setVisible(False)
            pc.addWidget(cr)
            self._checks.append(cr)

        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._pc)

    # ══════════════════════════════════════════════════════════════════════════
    # DATA SLOTS (same as before — compact)
    # ══════════════════════════════════════════════════════════════════════════
    def on_price_updated(self, symbol, df):
        self._symbol = symbol
        self._fv = None
        self._refresh_filing()
        self._price_df = df
        self._sym.setText(symbol)
        if df is not None and not df.empty:
            last = df.iloc[-1]
            self._spot = float(last["Close"])
            self._spot_lbl.setText(f"{self._spot:,.2f}")
            op = float(last["Open"])
            if op > 0:
                ch = self._spot - op
                p = ch / op * 100
                c = EM if ch >= 0 else RED
                self._chg.setText(f"{ch:+,.2f} ({p:+.2f}%)")
                self._chg.setStyleSheet(f"color: {c};")
            try:
                ts = df.index[-1]
                if hasattr(ts, "strftime"):
                    self._ts.setText(ts.strftime("%d %b %H:%M IST"))
            except:
                pass
        self._update_chart()

    def on_ps_updated(self, symbol, ps):
        self._ps = ps
        self._refresh_kpi()
        self._refresh_badges()
        self._update_chart()
        if self._fv:
            self._refresh_filing()

    def on_context_updated(self, symbol, ctx):
        self._ctx = ctx
        self._refresh_kpi()

        from modules.etf_analyzer import ETFContext

        if isinstance(ctx, ETFContext):
            self._refresh_intel_etf(ctx)
        else:
            self._refresh_intel()
        self._update_chart()
        if self._fv:
            self._refresh_filing()

    def on_spot_updated(self, symbol, spot):
        self._spot = spot
        self._spot_lbl.setText(f"{spot:,.2f}")

    def set_price_df(self, df):
        self._price_df = df

    def on_filing_updated(self, symbol: str, fv):
        if symbol != self._symbol:
            return
        self._fv = fv
        self._refresh_filing()

    def _refresh_filing(self):
        from modules.setup_classifier import (
            classify_setup_v3,
            OptionsSignalState,
            MomentumState,
            SETUP_COLORS,
        )

        fv, ctx, ps = self._fv, self._ctx, self._ps
        badge_c = {"BULLISH": EM, "BEARISH": RED}.get(
            getattr(fv, "badge_color", None), MUTED
        )

        # Strip
        if fv is None:
            self._filing_badge.set_badge("NO FILING", "NEUTRAL")
            self._filing_detail_lbl.setText("No actionable filing in past 72h")
            self._filing_detail_lbl.setStyleSheet(f"color: {MUTED};")
            self._conviction_lbl.setText("○○○○○○○○○○")
            self._filing_recency_lbl.setText("")
            self._filing_setup_lbl.setText("SETUP: —")
            self._filing_subject_lbl.setText("")
            self._filing_deal_lbl.setText("")
            self._filing_conv_badge.set_badge("CONV —", "NEUTRAL")
            return

        self._filing_badge.set_badge(fv.badge_text, fv.badge_color)
        self._filing_detail_lbl.setText(fv.detail_line)
        self._filing_detail_lbl.setStyleSheet(f"color: {badge_c};")
        h = fv.recency_h
        rec_str = (
            f"fresh {h:.1f}h"
            if h < 6
            else (f"recent {h:.1f}h" if h < 24 else f"{h:.0f}h ago")
        )
        self._filing_recency_lbl.setText(rec_str)
        self._filing_recency_lbl.setStyleSheet(
            f"color: {EM if h < 6 else (GOLD if h < 24 else MUTED)};"
        )
        filled = "●" * fv.conviction + "○" * (10 - fv.conviction)
        conv_c = EM if fv.conviction >= 7 else (GOLD if fv.conviction >= 4 else MUTED)
        self._conviction_lbl.setText(filled)
        self._conviction_lbl.setStyleSheet(f"color: {conv_c};")

        # Setup re-classification
        setup_label, setup_color = "—", MUTED
        if ctx and hasattr(ctx, "regime") and ps:
            try:
                opts = OptionsSignalState(
                    gex_regime=ctx.regime.regime,
                    gex_rising=getattr(ctx.gex, "gex_rising", False),
                    pcr=getattr(ctx.walls, "pcr_oi", 1.0) or 1.0,
                    pcr_trending="FLAT",
                    iv_skew="FLAT",
                    delta_bias="NEUTRAL",
                    call_oi_wall_pct=0.0,
                    pct_to_resistance=getattr(ctx.walls, "resistance_pct", None),
                    pcroi=getattr(ctx.walls, "pcr_oi", 1.0) or 1.0,
                )
                mom = MomentumState(
                    bb_position=ps.bb_position,
                    position_state=ps.position_state,
                    vol_state=ps.vol_state,
                    wr_phase=ps.wr_phase,
                    wr_value=ps.wr_value,
                    wr_in_momentum=ps.wr_in_momentum,
                )
                st, _ = classify_setup_v3(
                    viability_score=ctx.viability.score,
                    filing_variance=fv.variance,
                    filing_direction=fv.badge_color,
                    filing_conviction=fv.conviction,
                    filing_category=fv.category.value,
                    opts=opts,
                    mom=mom,
                )
                setup_label = st.value
                setup_color = SETUP_COLORS.get(st, MUTED)
            except Exception:
                setup_label = fv.badge_text.split()[0]

        self._setup_badge.set_badge(f"SETUP: {setup_label}", setup_label)
        self._filing_setup_lbl.setText(f"SETUP: {setup_label}")
        self._filing_setup_lbl.setStyleSheet(
            f"color: {setup_color}; font-weight: bold;"
        )
        self._filing_subject_lbl.setText(f"↳ {fv.raw_subject}")
        self._filing_deal_lbl.setText(fv.detail_line)
        self._filing_deal_lbl.setStyleSheet(f"color: {badge_c};")
        conv_text = f"CONV {fv.conviction}/10" + (" ✓" if fv.confirmed else "")
        self._filing_conv_badge.set_badge(
            conv_text,
            (
                "BULLISH"
                if fv.conviction >= 7
                else ("NEUTRAL" if fv.conviction >= 4 else "BEARISH")
            ),
        )

    def _update_chart(self):
        df = self._price_df
        if df is not None and not df.empty:
            self._chart.update_data(df, self._ctx, self._ps, self._symbol)

    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_badges(self):
        ps = self._ps
        if not ps:
            return
        self._badges["DAILY"].set_badge(f"DAILY: {ps.daily_bias}", ps.daily_bias)
        self._badges["VOL"].set_badge(f"VOL: {ps.vol_state}", ps.vol_state)
        st_map = {
            "RIDING_UPPER": "RIDING UPPER",
            "FIRST_DIP": "FIRST DIP",
            "MID_BAND_BROKEN": "MID BAND BROKEN",
            "CONSOLIDATING": "CONSOLIDATING",
        }
        st_key = {
            "RIDING_UPPER": "BULLISH",
            "FIRST_DIP": "LATE",
            "MID_BAND_BROKEN": "BEARISH",
            "CONSOLIDATING": "NEUTRAL",
        }
        self._badges["STATE"].set_badge(
            st_map.get(ps.position_state, ps.position_state),
            st_key.get(ps.position_state, "NEUTRAL"),
        )
        if ps.wr_value is not None:
            zone = "IN ZONE" if ps.wr_in_momentum else "NOT IN ZONE"
            self._badges["W%R"].set_badge(
                f"W%R {ps.wr_value:.1f} {zone}",
                "FRESH" if ps.wr_in_momentum else "BEARISH",
            )

    def _refresh_kpi(self):
        ps, ctx = self._ps, self._ctx
        is_etf = isinstance(ctx, ETFContext) if ctx else False
        if is_etf:
            self._kpi["PCR OI"].set_title("VSR")  # Volume Surge Ratio
            self._kpi["NET GEX"].set_title("VOL TREND")
            self._kpi["TO EXPIRY"].set_title("POC DIST")
        else:
            self._kpi["PCR OI"].set_title("PCR OI")
            self._kpi["NET GEX"].set_title("NET GEX")
            self._kpi["TO EXPIRY"].set_title("TO EXPIRY")

        if self._spot > 0:
            self._kpi["SPOT"].set_value(f"{self._spot:,.2f}", WHITE)
        if ps and ps.wr_value is not None:
            wc = wr_color(ps.wr_value)
            self._kpi["W%R(50)"].set_value(f"{ps.wr_value:.1f}", wc)
            self._kpi["W%R(50)"].set_delta(ps.wr_phase, wc)
        if ps:
            self._kpi["BB %B"].set_value(f"{ps.bb_pct:.2f}", WHITE)
            self._kpi["BB %B"].set_delta(
                ps.bb_position.replace("_", " ").title(), MUTED
            )
        if ps:
            bc = {"BULLISH": EM, "BEARISH": RED, "NEUTRAL": GOLD}.get(
                ps.daily_bias, MUTED
            )
            self._kpi["DAILY BIAS"].set_value(ps.daily_bias, bc)
            if ps.daily_bias_pct:
                self._kpi["DAILY BIAS"].set_delta(
                    f"{ps.daily_bias_pct:+.1f}% vs SMA", bc
                )
        if ctx and hasattr(ctx, "walls") and ctx.walls.pcr_oi:
            pcr = ctx.walls.pcr_oi
            self._kpi["PCR OI"].set_value(
                f"{pcr:.3f}", EM if pcr >= 1.1 else (RED if pcr < 0.7 else WHITE)
            )
            self._kpi["PCR OI"].set_delta(ctx.walls.pcr_sentiment, MUTED)
        if ctx and hasattr(ctx, "gex"):
            g = ctx.gex.net_gex
            self._kpi["NET GEX"].set_value(
                f"{g:+,.0f}M", EM if g < 0 else (RED if g > 0 else MUTED)
            )
            self._kpi["NET GEX"].set_delta(ctx.gex.regime, MUTED)
        if ctx and hasattr(ctx, "expiry") and ctx.expiry.days_remaining < 99:
            d = ctx.expiry.days_remaining
            self._kpi["TO EXPIRY"].set_value(
                f"{d}d", RED if d <= 2 else (GOLD if d <= 4 else WHITE)
            )
            self._kpi["TO EXPIRY"].set_delta(f"{ctx.expiry.pin_risk} pin", MUTED)
        if ps:
            vc = {"SQUEEZE": VIOLET, "EXPANDED": GOLD, "NORMAL": WHITE}.get(
                ps.vol_state, MUTED
            )
            self._kpi["VOL STATE"].set_value(ps.vol_state, vc)
            self._kpi["VOL STATE"].set_delta(f"{ps.bb_width_pctl:.0f}th pctl", MUTED)

    def _refresh_intel(self):
        ctx, ps = self._ctx, self._ps
        if ctx is None:
            return

        # ── ETFContext: no F&O panels — show viability + ETF badge only ──
        if not hasattr(ctx, "regime"):
            v = ctx.viability
            sc = score_color(v.score)
            self._pb.set_accent(sc)
            self._score.setText(str(v.score))
            self._score.setStyleSheet(f"color: {sc}; font-weight: bold;")
            self._vlabel.setText(v.label)
            self._vlabel.setStyleSheet(f"color: {sc}; font-weight: bold;")
            self._size_badge.set_badge(f"SIZE: {v.sizing}", v.sizing)
            self._regime_lbl.setText("REGIME: ETF — No options chain")
            self._regime_lbl.setStyleSheet(f"color: {BLUE}; font-weight: bold;")
            self._regime_cap.set_badge("ETF", "NEUTRAL")
            for lr in self._levels:
                lr.setVisible(False)
            return  # skip all F&O-specific panel logic below

        r = ctx.regime
        rc = {
            "TREND-FRIENDLY": EM,
            "PINNING": RED,
            "GEX POSITIVE": GOLD,
            "GEX NEG / EXPIRY RISK": GOLD,
            "NEUTRAL": GOLD,
        }.get(r.regime, MUTED)
        self._pa.set_accent(rc)
        self._regime_lbl.setText(f"REGIME: {r.regime}")
        self._regime_lbl.setStyleSheet(f"color: {rc}; font-weight: bold;")
        self._regime_cap.set_badge(f"MAX: {r.size_cap}", r.size_cap)
        self._regime_detail.setText(r.detail)
        tint = {
            "TREND-FRIENDLY": "16,185,129",
            "PINNING": "239,68,68",
            "GEX POSITIVE": "245,158,11",
            "GEX NEG / EXPIRY RISK": "245,158,11",
            "NEUTRAL": "245,158,11",
        }
        rgb = tint.get(r.regime, "100,116,139")
        self._regime_frame.setStyleSheet(
            f"background: rgba({rgb},0.06); border-left: 3px solid {rc}; "
            f"border-radius: 4px; padding: 4px 8px;"
        )

        v = ctx.viability
        self._wf_badge.set_badge(f"SIZE: {v.sizing}", v.sizing)
        _regime_badge_state = {
            "TREND-FRIENDLY": "BULLISH",
            "PINNING": "BEARISH",
            "GEX POSITIVE": "LATE",
            "GEX NEG / EXPIRY RISK": "LATE",
            "NEUTRAL": "NEUTRAL",
        }
        self._badges.get("REGIME", Badge()).set_badge(
            f"REGIME: {r.regime}", _regime_badge_state.get(r.regime, "NEUTRAL")
        )

        try:
            from modules.commentary import get_commentary

            comm = get_commentary(ctx, ps, self._symbol, self._price_df)
        except:
            comm = {}
        for key, ck in {
            "TREND & STATE": "bias_line",
            "GEX REGIME": "gex_line",
            "OI WALLS": "wall_line",
            "EXPIRY": "expiry_line",
            "WILLIAMS %R": "wr_line",
        }.items():
            if key in self._wf:
                self._wf[key].setText(comm.get(ck, ""))

        verdict = comm.get("verdict", "")
        sz = comm.get("sizing", v.sizing)
        vrgb = {"SKIP": "239,68,68", "HALF": "245,158,11"}.get(sz, "16,185,129")
        vc = {"SKIP": RED, "HALF": GOLD}.get(sz, EM)
        self._verdict.setText(verdict)
        self._verdict.setStyleSheet(
            f"background: rgba({vrgb},0.06); color: {WHITE}; "
            f"border-left: 3px solid {vc}; border-radius: 3px; padding: 5px 8px;"
        )

        # Panel B
        sc = score_color(v.score)
        self._pb.set_accent(sc)
        self._score.setText(str(v.score))
        self._score.setStyleSheet(f"color: {sc}; font-weight: bold;")
        self._vlabel.setText(v.label)
        self._vlabel.setStyleSheet(f"color: {sc}; font-weight: bold;")
        self._size_badge.set_badge(f"SIZE: {v.sizing}", v.sizing)

        while self._risk_container.count():
            w = self._risk_container.takeAt(0).widget()
            if w:
                w.deleteLater()
        for note in v.risk_notes[:3]:
            self._risk_container.addWidget(RiskNote(note))

        w = ctx.walls
        level_data = []
        if w.resistance:
            level_data.append(
                ("RESISTANCE", w.resistance, RED, f"{w.resistance_pct:+.1f}%")
            )
        if w.support:
            level_data.append(("SUPPORT", w.support, GREEN, f"{w.support_pct:+.1f}%"))
        if w.max_pain:
            level_data.append(("MAX PAIN", w.max_pain, GOLD, ""))
        if ctx.gex.hvl:
            level_data.append(("GEX HVL", ctx.gex.hvl, VIOLET, ""))
        if ps and ps.daily_sma:
            level_data.append(
                (
                    "DAILY SMA",
                    ps.daily_sma,
                    EM if ps.daily_bias == "BULLISH" else RED,
                    "",
                )
            )
        for i, lr in enumerate(self._levels):
            if i < len(level_data):
                lr.set_data(*level_data[i])
                lr.setVisible(True)
            else:
                lr.setVisible(False)

        if ctx.walls.pcr_oi:
            self._bot["PCR"].set_badge(f"PCR {ctx.walls.pcr_oi:.2f}", "NEUTRAL")
        self._bot["PIN"].set_badge(
            f"PIN {ctx.expiry.pin_risk}",
            {"HIGH": "BEARISH", "MODERATE": "LATE", "LOW": "BULLISH"}.get(
                ctx.expiry.pin_risk, "NEUTRAL"
            ),
        )
        self._bot["GEX"].set_badge(
            f"GEX {ctx.gex.regime.upper()}",
            {"Positive": "BEARISH", "Negative": "BULLISH", "Neutral": "NEUTRAL"}.get(
                ctx.gex.regime, "NEUTRAL"
            ),
        )
        if ctx.expiry.days_remaining < 99:
            self._bot["DTE"].set_badge(f"+{ctx.expiry.days_remaining}D", "NEUTRAL")

        for i, cr in enumerate(self._checks):
            if i < len(v.checklist):
                ci = v.checklist[i]
                cr.set_data(ci.status, ci.item, ci.detail, ci.implication)
                cr.setVisible(True)
            else:
                cr.setVisible(False)

    def _refresh_intel_etf(self, ctx) -> None:
        """Render all three intelligence panels for ETFContext."""
        from ui.theme import (
            EM,
            RED,
            GOLD,
            MUTED,
            WHITE,
            GREEN,
            BLUE,
            VIOLET,
            score_color,
        )
        from ui.components import RiskNote, ChecklistRow

        v = ctx.viability
        info = ctx.info
        vp = ctx.volume_profile
        ev = ctx.etf_volume
        et = ctx.etf_trend
        nav = ctx.etf_nav

        # ── KPI overrides ─────────────────────────────────────────────────────
        if ev:
            vsr_c = EM if ev.vsr >= 2.5 else (GOLD if ev.vsr >= 1.5 else MUTED)
            self._kpi["PCR OI"].set_value(f"{ev.vsr:.1f}x", vsr_c)
            self._kpi["PCR OI"].set_delta(f"{ev.surge_label} · {ev.trend}", vsr_c)
        else:
            self._kpi["PCR OI"].set_value("—", MUTED)
            self._kpi["PCR OI"].set_delta("VSR N/A", MUTED)

        if nav and nav.available:
            nav_c = (
                EM
                if nav.premium_pct <= -0.25
                else (RED if nav.premium_pct >= 1.0 else WHITE)
            )
            self._kpi["NET GEX"].set_value(f"{nav.premium_pct:+.2f}%", nav_c)
            self._kpi["NET GEX"].set_delta(nav.label, nav_c)
        else:
            self._kpi["NET GEX"].set_value("—", MUTED)
            self._kpi["NET GEX"].set_delta("NAV N/A", MUTED)

        aum_str = (
            f"₹{info.aum_cr/1000:.0f}kCr"
            if info.aum_cr >= 1000
            else f"₹{info.aum_cr:.0f}Cr"
        )
        self._kpi["TO EXPIRY"].set_value(aum_str, WHITE)
        self._kpi["TO EXPIRY"].set_delta(info.category.replace("_", " ").title(), MUTED)

        # ── Panel A — ETF Momentum + Workflow ─────────────────────────────────
        sc = score_color(v.score)
        self._pa.set_accent(sc)
        lbl_col = {"STRONG": EM, "MODERATE": GOLD, "WEAK": GOLD, "AVOID": RED}.get(
            v.label, MUTED
        )
        self._regime_lbl.setText(f"ETF MOMENTUM: {v.label}")
        self._regime_lbl.setStyleSheet(f"color: {lbl_col}; font-weight: bold;")
        self._regime_cap.set_badge(f"SIZE: {v.sizing}", v.sizing)
        self._regime_detail.setText(f"{info.name}  ·  {info.underlying}")
        tint = {
            "STRONG": "16,185,129",
            "MODERATE": "245,158,11",
            "WEAK": "245,158,11",
            "AVOID": "239,68,68",
        }.get(v.label, "100,116,139")
        self._regime_frame.setStyleSheet(
            f"background: rgba({tint},0.06); border-left: 3px solid {lbl_col}; "
            f"border-radius: 4px; padding: 4px 8px;"
        )

        self._wf_badge.set_badge(f"SIZE: {v.sizing}", v.sizing)
        if "REGIME" in self._badges:
            self._badges["REGIME"].set_badge(
                f"ETF: {v.label}",
                (
                    "BULLISH"
                    if v.label == "STRONG"
                    else "NEUTRAL" if v.label in ("MODERATE", "WEAK") else "BEARISH"
                ),
            )

        trend_line = (
            f"Daily {ctx.daily_bias}  ·  BB: {ctx.bb_state.replace('_', ' ')}  ·  "
            f"W%R {ctx.wr_value:.0f} ({'in zone' if ctx.wr_in_momentum else 'NOT in zone'})"
        )
        vol_line = (
            f"VSR {ev.vsr:.1f}x → {ev.surge_label}  ·  Volume {ev.trend}"
            if ev
            else "Volume data unavailable"
        )
        vwap_line = (
            f"{'Above' if et.above_vwap else 'Below'} VWAP ({et.pct_from_vwap:+.1f}%)"
            f"  ·  VWAP: {et.vwap:.2f}"
            if et and et.vwap > 0
            else "VWAP unavailable"
        )
        underlying_line = (
            f"{info.underlying}: {et.underlying_bias}  ({et.underlying_pct:+.1f}% 20d)"
            if et
            else "Underlying data unavailable"
        )
        nav_line = (
            f"iNAV: {nav.inav:.2f}  ·  Spot: {nav.spot:.2f}  ·  "
            f"{nav.label} ({nav.premium_pct:+.2f}%)"
            if nav and nav.available
            else "NAV unavailable — NSE iNAV not fetched"
        )
        wf_map = {
            "TREND & STATE": trend_line,
            "GEX REGIME": vol_line,
            "OI WALLS": vwap_line,
            "EXPIRY": underlying_line,
            "WILLIAMS %R": nav_line,
        }
        for key, line in wf_map.items():
            if key in self._wf:
                self._wf[key].setText(line)

        bd = v.breakdown
        breakdown_str = "  ·  ".join(f"{k} {pts}pt" for k, pts in bd.items())
        verdict_text = (
            f"Score {v.score}/100 → {v.label}  ·  SIZE {v.sizing}\n"
            f"{breakdown_str}\n"
            f"{'; '.join(v.reasons[:4])}"
        )
        v_rgb = {"SKIP": "239,68,68", "HALF": "245,158,11"}.get(v.sizing, "16,185,129")
        v_col = {"SKIP": RED, "HALF": GOLD}.get(v.sizing, EM)
        self._verdict.setText(verdict_text)
        self._verdict.setStyleSheet(
            f"background: rgba({v_rgb},0.06); color: {WHITE}; "
            f"border-left: 3px solid {v_col}; border-radius: 3px; padding: 5px 8px;"
        )

        # ── Panel B — Score + ETF Levels ──────────────────────────────────────
        self._pb.set_accent(sc)
        self._score.setText(str(v.score))
        self._score.setStyleSheet(f"color: {sc}; font-weight: bold;")
        self._vlabel.setText(v.label)
        self._vlabel.setStyleSheet(f"color: {sc}; font-weight: bold;")
        self._size_badge.set_badge(f"SIZE: {v.sizing}", v.sizing)

        while self._risk_container.count():
            w = self._risk_container.takeAt(0).widget()
            if w:
                w.deleteLater()
        for r in v.reasons[:3]:
            self._risk_container.addWidget(RiskNote(r))

        level_data = []
        if vp:
            level_data.append(("VAH (Resist)", vp.vah, RED, "Upper value area"))
            level_data.append(("POC", vp.poc, GOLD, "High-vol node"))
            level_data.append(("VAL (Support)", vp.val, GREEN, "Lower value area"))
        if et and et.vwap > 0:
            level_data.append(
                (
                    "VWAP",
                    et.vwap,
                    BLUE,
                    f"{'Above' if et.above_vwap else 'Below'} {et.pct_from_vwap:+.1f}%",
                )
            )
        if self._ps and self._ps.daily_sma:
            level_data.append(
                (
                    "DAILY SMA",
                    self._ps.daily_sma,
                    EM if self._ps.daily_bias == "BULLISH" else RED,
                    "",
                )
            )
        for i, lr in enumerate(self._levels):
            if i < len(level_data):
                lr.set_data(*level_data[i])
                lr.setVisible(True)
            else:
                lr.setVisible(False)

        # Bottom badges: VSR / SURGE / UNDERLYING / AUM
        if ev:
            self._bot["PCR"].set_badge(f"VSR {ev.vsr:.1f}x", "NEUTRAL")
            self._bot["PIN"].set_badge(
                ev.surge_label,
                {
                    "SURGE": "BULLISH",
                    "ELEVATED": "BULLISH",
                    "NORMAL": "NEUTRAL",
                    "DRY": "BEARISH",
                }.get(ev.surge_label, "NEUTRAL"),
            )
        if et:
            self._bot["GEX"].set_badge(
                et.underlying_bias[:4],
                {"BULLISH": "BULLISH", "BEARISH": "BEARISH", "NEUTRAL": "NEUTRAL"}.get(
                    et.underlying_bias, "NEUTRAL"
                ),
            )
        aum_badge = (
            f"AUM {info.aum_cr/1000:.0f}kCr"
            if info.aum_cr >= 1000
            else f"AUM {info.aum_cr:.0f}Cr"
        )
        self._bot["DTE"].set_badge(aum_badge, "NEUTRAL")

        # ── Panel C — ETF Checklist ───────────────────────────────────────────
        for i, cr in enumerate(self._checks):
            if i < len(v.checklist):
                ci = v.checklist[i]
                cr.set_data(ci.status, ci.item, ci.detail, ci.implication)
                cr.setVisible(True)
            else:
                cr.setVisible(False)
