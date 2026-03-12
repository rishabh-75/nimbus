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
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
)
from PyQt6.QtGui import QFont
from ui.theme import (
    BG, SURFACE, S2, BORDER, EM, RED, GOLD, MUTED, WHITE, GREEN, BLUE,
    VIOLET, FONT_MONO, FONT_UI, score_color, wr_color,
)
from ui.chart_widget import NimbusChart
from ui.components import KPITile, PanelFrame, Badge, ChecklistRow, LevelRow, RiskNote
from modules.analytics import OptionsContext
from modules.indicators import PriceSignals

logger = logging.getLogger(__name__)

def _h(t, c=MUTED, s=7):
    l = QLabel(t.upper()); l.setFont(QFont(FONT_UI, s))
    l.setStyleSheet(f"color: {c}; letter-spacing: 0.8px;"); return l

def _t(t="", c=WHITE, s=8, mono=False, bold=False):
    l = QLabel(t); f = QFont(FONT_MONO if mono else FONT_UI, s); f.setBold(bold)
    l.setFont(f); l.setStyleSheet(f"color: {c};"); l.setWordWrap(True); return l

def _sep():
    s = QFrame(); s.setFixedHeight(1); s.setStyleSheet(f"background: {BORDER};"); return s

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
    return sa


class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._symbol = ""
        self._spot = 0.0
        self._price_df: Optional[pd.DataFrame] = None
        self._ps: Optional[PriceSignals] = None
        self._ctx: Optional[OptionsContext] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 4)
        root.setSpacing(5)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QFrame(); hdr.setFixedHeight(30)
        hdr.setStyleSheet(f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(12, 0, 12, 0)
        self._sym = _t("--", WHITE, 12, mono=True, bold=True); hl.addWidget(self._sym)
        hl.addSpacing(8)
        self._spot_lbl = _t("", WHITE, 11, mono=True); hl.addWidget(self._spot_lbl)
        hl.addSpacing(6)
        self._chg = _t("", MUTED, 9, mono=True); hl.addWidget(self._chg)
        hl.addStretch(1)
        self._ts = _t("", MUTED, 7, mono=True); hl.addWidget(self._ts)
        root.addWidget(hdr)

        # ── Badge row ─────────────────────────────────────────────────────────
        br = QHBoxLayout(); br.setSpacing(4)
        self._badges = {}
        for k in ("DAILY", "VOL", "STATE", "W%R", "REGIME"):
            b = Badge("--"); br.addWidget(b); self._badges[k] = b
        br.addStretch(1)
        root.addLayout(br)

        # ── Chart (stretch=1) ─────────────────────────────────────────────────
        self._chart = NimbusChart()
        root.addWidget(self._chart, stretch=1)

        # ── KPI tiles ─────────────────────────────────────────────────────────
        kr = QHBoxLayout(); kr.setSpacing(5)
        self._kpi = {}
        for k in ("SPOT", "W%R(50)", "BB %B", "DAILY BIAS",
                   "PCR OI", "NET GEX", "TO EXPIRY", "VOL STATE"):
            t = KPITile(k); kr.addWidget(t); self._kpi[k] = t
        root.addLayout(kr)

        # ── Intelligence row: each panel individually scrollable ──────────────
        intel = QHBoxLayout(); intel.setSpacing(5)

        pa_widget = QWidget(); self._build_panel_a(pa_widget)
        intel.addWidget(_make_scroll_panel(pa_widget, 260), 42)

        pb_widget = QWidget(); self._build_panel_b(pb_widget)
        intel.addWidget(_make_scroll_panel(pb_widget, 260), 28)

        pc_widget = QWidget(); self._build_panel_c(pc_widget)
        intel.addWidget(_make_scroll_panel(pc_widget, 260), 30)

        root.addLayout(intel)

    # ── Panel A: Regime + Workflow ────────────────────────────────────────────
    def _build_panel_a(self, container):
        self._pa = PanelFrame("", accent=EM)
        pa = self._pa.content_layout(); pa.setSpacing(4)

        # Regime tile — NO fixedHeight so name wraps naturally
        rt = QFrame()
        rt.setStyleSheet(
            f"background: rgba(16,185,129,0.06); border-left: 3px solid {EM}; "
            f"border-radius: 4px; padding: 4px 8px;")
        self._regime_frame = rt
        rl = QVBoxLayout(rt); rl.setContentsMargins(4, 2, 4, 2); rl.setSpacing(2)
        rh = QHBoxLayout()
        self._regime_lbl = _t("REGIME: --", EM, 8, bold=True)
        self._regime_lbl.setWordWrap(True)
        rh.addWidget(self._regime_lbl, stretch=1)
        self._regime_cap = Badge("--"); rh.addWidget(self._regime_cap)
        rl.addLayout(rh)
        self._regime_detail = _t("", MUTED, 7)
        rl.addWidget(self._regime_detail)
        pa.addWidget(rt)

        pa.addWidget(_sep())

        wh = QHBoxLayout()
        wh.addWidget(_h("Workflow Analysis")); wh.addStretch(1)
        self._wf_badge = Badge("--"); wh.addWidget(self._wf_badge)
        pa.addLayout(wh)

        self._wf = {}
        for key in ("TREND & STATE", "GEX REGIME", "OI WALLS", "EXPIRY", "WILLIAMS %R"):
            pa.addWidget(_h(key, MUTED, 6))
            body = _t("", WHITE, 7); pa.addWidget(body); self._wf[key] = body

        self._verdict = QLabel("")
        self._verdict.setFont(QFont(FONT_UI, 7)); self._verdict.setWordWrap(True)
        self._verdict.setStyleSheet(
            f"background: rgba(16,185,129,0.06); color: {WHITE}; "
            f"border-left: 3px solid {EM}; border-radius: 3px; padding: 5px 8px;")
        pa.addWidget(self._verdict)

        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._pa)

    # ── Panel B: Viability + Levels ───────────────────────────────────────────
    def _build_panel_b(self, container):
        self._pb = PanelFrame("Trade Viability", accent=GOLD)
        pb = self._pb.content_layout(); pb.setSpacing(4)

        # Score row — no fixedWidth, natural alignment
        sr = QHBoxLayout(); sr.setSpacing(6)
        self._score = _t("--", MUTED, 24, mono=True, bold=True)
        sr.addWidget(self._score)

        sv = QVBoxLayout(); sv.setSpacing(0)
        self._vlabel = _t("", MUTED, 8, bold=True)
        sv.addWidget(self._vlabel)
        sv.addWidget(_t("/100", MUTED, 6))
        sr.addLayout(sv)
        sr.addStretch(1)
        self._size_badge = Badge("--"); sr.addWidget(self._size_badge)
        pb.addLayout(sr)

        self._risk_container = QVBoxLayout(); self._risk_container.setSpacing(2)
        pb.addLayout(self._risk_container)
        pb.addWidget(_sep())
        pb.addWidget(_h("Key Levels"))

        self._levels: list[LevelRow] = []
        for _ in range(5):
            lr = LevelRow(); lr.setVisible(False); pb.addWidget(lr); self._levels.append(lr)

        bb = QHBoxLayout(); bb.setSpacing(3)
        self._bot = {}
        for k in ("PCR", "PIN", "GEX", "DTE"):
            b = Badge("--"); b.setFixedHeight(16); bb.addWidget(b); self._bot[k] = b
        bb.addStretch(1); pb.addLayout(bb)

        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._pb)

    # ── Panel C: Checklist ────────────────────────────────────────────────────
    def _build_panel_c(self, container):
        self._pc = PanelFrame("Pre-Trade Checklist", accent=BORDER)
        pc = self._pc.content_layout(); pc.setSpacing(2)
        self._checks: list[ChecklistRow] = []
        for _ in range(10):
            cr = ChecklistRow(); cr.setVisible(False); pc.addWidget(cr); self._checks.append(cr)

        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._pc)

    # ══════════════════════════════════════════════════════════════════════════
    # DATA SLOTS (same as before — compact)
    # ══════════════════════════════════════════════════════════════════════════
    def on_price_updated(self, symbol, df):
        self._symbol = symbol; self._price_df = df; self._sym.setText(symbol)
        if df is not None and not df.empty:
            last = df.iloc[-1]; self._spot = float(last["Close"])
            self._spot_lbl.setText(f"{self._spot:,.2f}")
            op = float(last["Open"])
            if op > 0:
                ch = self._spot - op; p = ch / op * 100; c = EM if ch >= 0 else RED
                self._chg.setText(f"{ch:+,.2f} ({p:+.2f}%)"); self._chg.setStyleSheet(f"color: {c};")
            try:
                ts = df.index[-1]
                if hasattr(ts, 'strftime'): self._ts.setText(ts.strftime("%d %b %H:%M IST"))
            except: pass
        self._update_chart()

    def on_ps_updated(self, symbol, ps):
        self._ps = ps; self._refresh_kpi(); self._refresh_badges(); self._update_chart()

    def on_context_updated(self, symbol, ctx):
        self._ctx = ctx; self._refresh_kpi(); self._refresh_intel(); self._update_chart()

    def on_spot_updated(self, symbol, spot):
        self._spot = spot; self._spot_lbl.setText(f"{spot:,.2f}")

    def set_price_df(self, df): self._price_df = df

    def _update_chart(self):
        df = self._price_df
        if df is not None and not df.empty:
            self._chart.update_data(df, self._ctx, self._ps, self._symbol)

    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_badges(self):
        ps = self._ps
        if not ps: return
        self._badges["DAILY"].set_badge(f"DAILY: {ps.daily_bias}", ps.daily_bias)
        self._badges["VOL"].set_badge(f"VOL: {ps.vol_state}", ps.vol_state)
        st_map = {"RIDING_UPPER": "RIDING UPPER", "FIRST_DIP": "FIRST DIP",
                   "MID_BAND_BROKEN": "MID BAND BROKEN", "CONSOLIDATING": "CONSOLIDATING"}
        st_key = {"RIDING_UPPER": "BULLISH", "FIRST_DIP": "LATE",
                   "MID_BAND_BROKEN": "BEARISH", "CONSOLIDATING": "NEUTRAL"}
        self._badges["STATE"].set_badge(st_map.get(ps.position_state, ps.position_state),
            st_key.get(ps.position_state, "NEUTRAL"))
        if ps.wr_value is not None:
            zone = "IN ZONE" if ps.wr_in_momentum else "NOT IN ZONE"
            self._badges["W%R"].set_badge(f"W%R {ps.wr_value:.1f} {zone}",
                "FRESH" if ps.wr_in_momentum else "BEARISH")

    def _refresh_kpi(self):
        ps, ctx = self._ps, self._ctx
        if self._spot > 0: self._kpi["SPOT"].set_value(f"{self._spot:,.2f}", WHITE)
        if ps and ps.wr_value is not None:
            wc = wr_color(ps.wr_value)
            self._kpi["W%R(50)"].set_value(f"{ps.wr_value:.1f}", wc)
            self._kpi["W%R(50)"].set_delta(ps.wr_phase, wc)
        if ps:
            self._kpi["BB %B"].set_value(f"{ps.bb_pct:.2f}", WHITE)
            self._kpi["BB %B"].set_delta(ps.bb_position.replace("_", " ").title(), MUTED)
        if ps:
            bc = {"BULLISH": EM, "BEARISH": RED, "NEUTRAL": GOLD}.get(ps.daily_bias, MUTED)
            self._kpi["DAILY BIAS"].set_value(ps.daily_bias, bc)
            if ps.daily_bias_pct: self._kpi["DAILY BIAS"].set_delta(f"{ps.daily_bias_pct:+.1f}% vs SMA", bc)
        if ctx and ctx.walls.pcr_oi:
            pcr = ctx.walls.pcr_oi
            self._kpi["PCR OI"].set_value(f"{pcr:.3f}",
                EM if pcr >= 1.1 else (RED if pcr < 0.7 else WHITE))
            self._kpi["PCR OI"].set_delta(ctx.walls.pcr_sentiment, MUTED)
        if ctx:
            g = ctx.gex.net_gex
            self._kpi["NET GEX"].set_value(f"{g:+,.0f}M", EM if g < 0 else (RED if g > 0 else MUTED))
            self._kpi["NET GEX"].set_delta(ctx.gex.regime, MUTED)
        if ctx and ctx.expiry.days_remaining < 99:
            d = ctx.expiry.days_remaining
            self._kpi["TO EXPIRY"].set_value(f"{d}d", RED if d <= 2 else (GOLD if d <= 4 else WHITE))
            self._kpi["TO EXPIRY"].set_delta(f"{ctx.expiry.pin_risk} pin", MUTED)
        if ps:
            vc = {"SQUEEZE": VIOLET, "EXPANDED": GOLD, "NORMAL": WHITE}.get(ps.vol_state, MUTED)
            self._kpi["VOL STATE"].set_value(ps.vol_state, vc)
            self._kpi["VOL STATE"].set_delta(f"{ps.bb_width_pctl:.0f}th pctl", MUTED)

    def _refresh_intel(self):
        ctx, ps = self._ctx, self._ps
        if ctx is None: return

        r = ctx.regime
        rc = {"TREND-FRIENDLY": EM, "PINNING": RED, "NEUTRAL": GOLD}.get(r.regime, MUTED)
        self._pa.set_accent(rc)
        self._regime_lbl.setText(f"REGIME: {r.regime}")
        self._regime_lbl.setStyleSheet(f"color: {rc}; font-weight: bold;")
        self._regime_cap.set_badge(f"MAX: {r.size_cap}", r.size_cap)
        self._regime_detail.setText(r.detail)
        tint = {"TREND-FRIENDLY": "16,185,129", "PINNING": "239,68,68", "NEUTRAL": "245,158,11"}
        rgb = tint.get(r.regime, "100,116,139")
        self._regime_frame.setStyleSheet(
            f"background: rgba({rgb},0.06); border-left: 3px solid {rc}; "
            f"border-radius: 4px; padding: 4px 8px;")

        v = ctx.viability
        self._wf_badge.set_badge(f"SIZE: {v.sizing}", v.sizing)
        self._badges.get("REGIME", Badge()).set_badge(f"REGIME: {r.regime}",
            "BULLISH" if r.regime == "TREND-FRIENDLY" else ("BEARISH" if r.regime == "PINNING" else "NEUTRAL"))

        try:
            from modules.commentary import get_commentary
            comm = get_commentary(ctx, ps, self._symbol, self._price_df)
        except: comm = {}
        for key, ck in {"TREND & STATE": "bias_line", "GEX REGIME": "gex_line",
                         "OI WALLS": "wall_line", "EXPIRY": "expiry_line",
                         "WILLIAMS %R": "wr_line"}.items():
            if key in self._wf: self._wf[key].setText(comm.get(ck, ""))

        verdict = comm.get("verdict", "")
        sz = comm.get("sizing", v.sizing)
        vrgb = {"SKIP": "239,68,68", "HALF": "245,158,11"}.get(sz, "16,185,129")
        vc = {"SKIP": RED, "HALF": GOLD}.get(sz, EM)
        self._verdict.setText(verdict)
        self._verdict.setStyleSheet(
            f"background: rgba({vrgb},0.06); color: {WHITE}; "
            f"border-left: 3px solid {vc}; border-radius: 3px; padding: 5px 8px;")

        # Panel B
        sc = score_color(v.score)
        self._pb.set_accent(sc)
        self._score.setText(str(v.score)); self._score.setStyleSheet(f"color: {sc}; font-weight: bold;")
        self._vlabel.setText(v.label); self._vlabel.setStyleSheet(f"color: {sc}; font-weight: bold;")
        self._size_badge.set_badge(f"SIZE: {v.sizing}", v.sizing)

        while self._risk_container.count():
            w = self._risk_container.takeAt(0).widget()
            if w: w.deleteLater()
        for note in v.risk_notes[:3]: self._risk_container.addWidget(RiskNote(note))

        w = ctx.walls
        level_data = []
        if w.resistance: level_data.append(("RESISTANCE", w.resistance, RED, f"{w.resistance_pct:+.1f}%"))
        if w.support: level_data.append(("SUPPORT", w.support, GREEN, f"{w.support_pct:+.1f}%"))
        if w.max_pain: level_data.append(("MAX PAIN", w.max_pain, GOLD, ""))
        if ctx.gex.hvl: level_data.append(("GEX HVL", ctx.gex.hvl, VIOLET, ""))
        if ps and ps.daily_sma:
            level_data.append(("DAILY SMA", ps.daily_sma, EM if ps.daily_bias == "BULLISH" else RED, ""))
        for i, lr in enumerate(self._levels):
            if i < len(level_data): lr.set_data(*level_data[i]); lr.setVisible(True)
            else: lr.setVisible(False)

        if ctx.walls.pcr_oi: self._bot["PCR"].set_badge(f"PCR {ctx.walls.pcr_oi:.2f}", "NEUTRAL")
        self._bot["PIN"].set_badge(f"PIN {ctx.expiry.pin_risk}",
            {"HIGH": "BEARISH", "MODERATE": "LATE", "LOW": "BULLISH"}.get(ctx.expiry.pin_risk, "NEUTRAL"))
        self._bot["GEX"].set_badge(f"GEX {ctx.gex.regime.upper()}",
            {"Positive": "BEARISH", "Negative": "BULLISH", "Neutral": "NEUTRAL"}.get(ctx.gex.regime, "NEUTRAL"))
        if ctx.expiry.days_remaining < 99:
            self._bot["DTE"].set_badge(f"+{ctx.expiry.days_remaining}D", "NEUTRAL")

        for i, cr in enumerate(self._checks):
            if i < len(v.checklist):
                ci = v.checklist[i]
                cr.set_data(ci.status, ci.item, ci.detail, ci.implication); cr.setVisible(True)
            else: cr.setVisible(False)
