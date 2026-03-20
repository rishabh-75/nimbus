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

    # allow horizontal scroll when content is wider than viewport
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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


def _build_dual_checklist(dm_sig, ctx=None):
    """Build checklist for unified mean-reversion signal."""
    checks = []

    # 1. WR(30) oversold
    if dm_sig.wr_30 < -50:
        checks.append(("pass", "WR(30) Deep", f"WR={dm_sig.wr_30:.0f}", "Strong bounce expected"))
    elif dm_sig.wr_30 < -30:
        checks.append(("pass", "WR(30) Zone", f"WR={dm_sig.wr_30:.0f}", "Mean reversion zone"))
    else:
        checks.append(("fail", "WR(30) Wait", f"WR={dm_sig.wr_30:.0f}", "Not oversold"))

    # 2. Below SMA
    if not dm_sig.above_sma and dm_sig.pct_from_sma < -3:
        checks.append(("pass", "Below SMA", f"{dm_sig.pct_from_sma:+.1f}%", "Deep pullback"))
    elif not dm_sig.above_sma:
        checks.append(("pass", "Below SMA", f"{dm_sig.pct_from_sma:+.1f}%", "Pullback active"))
    else:
        checks.append(("fail", "Above SMA", f"{dm_sig.pct_from_sma:+.1f}%", "Not a pullback"))

    # 3. MFI
    if dm_sig.mfi >= 50:
        checks.append(("pass", "MFI Strong", f"MFI={dm_sig.mfi:.0f}", "Accumulation"))
    elif dm_sig.mfi >= 30:
        checks.append(("pass", "MFI OK", f"MFI={dm_sig.mfi:.0f}", "Acceptable flow"))
    else:
        checks.append(("fail", "MFI Weak", f"MFI={dm_sig.mfi:.0f}", "No buying pressure"))

    # 4. Drawdown from high
    if dm_sig.dd_from_high <= -10:
        checks.append(("pass", "Deep Drawdown", f"{dm_sig.dd_from_high:.1f}%", "Capitulation"))
    elif dm_sig.dd_from_high <= -5:
        checks.append(("pass", "Moderate DD", f"{dm_sig.dd_from_high:.1f}%", "Meaningful pullback"))
    else:
        checks.append(("warn", "Shallow DD", f"{dm_sig.dd_from_high:.1f}%", "Minor dip"))

    # 5. Red streak
    if dm_sig.red_streak >= 4:
        checks.append(("pass", "Exhaustion", f"{dm_sig.red_streak}d red", "Selling spent"))
    elif dm_sig.red_streak >= 2:
        checks.append(("pass", "Red Streak", f"{dm_sig.red_streak}d red", "Sellers tiring"))
    else:
        checks.append(("warn", "No Streak", f"{dm_sig.red_streak}d", "No exhaustion signal"))

    # 6. Options
    if dm_sig.options_overlay > 3:
        checks.append(("pass", "Options ✓", dm_sig.options_detail, f"+{dm_sig.options_overlay}"))
    elif dm_sig.options_overlay < -3:
        checks.append(("fail", "Options ✗", dm_sig.options_detail, f"{dm_sig.options_overlay}"))

    # 7. Filing
    if dm_sig.is_trap:
        checks.append(("fail", "⚠ TRAP", dm_sig.filing_detail, "Bearish filing"))
    elif dm_sig.filing_overlay > 3:
        checks.append(("pass", "Filing ✓", dm_sig.filing_detail, f"+{dm_sig.filing_overlay}"))

    # 8. Entry verdict
    if dm_sig.tier == "PRIMARY":
        checks.append(("pass", "▶ PRIMARY", dm_sig.entry_reason, "FULL size, 5% PT"))
    elif dm_sig.tier == "SECONDARY":
        checks.append(("pass", "▷ SECONDARY", dm_sig.entry_reason, "HALF size, 5% PT"))
    else:
        missing = []
        if dm_sig.wr_30 >= -30: missing.append("WR oversold")
        if dm_sig.above_sma: missing.append("below SMA")
        if dm_sig.mfi < 30: missing.append("MFI > 30")
        checks.append(("fail", "NO ENTRY", f"Missing: {', '.join(missing)}" if missing else "Not met", "Wait"))

    return checks


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
        for k in ("TIER", "WR", "MFI", "SMA", "DD", "REGIME"):
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
            "WR(30)",
            "MFI",
            "vs SMA",
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

        # ── Dual-Mode Signal Strip ────────────────────────────────────────────
        dmstrip = QFrame()
        dmstrip.setFixedHeight(24)
        dmstrip.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 3px;"
        )
        dl = QHBoxLayout(dmstrip)
        dl.setContentsMargins(6, 0, 6, 0)
        dl.setSpacing(4)

        self._dm_mode_badge = Badge("MODE —")
        dl.addWidget(self._dm_mode_badge)

        self._dm_score_lbl = _t("—", MUTED, 9, mono=True, bold=True)
        dl.addWidget(self._dm_score_lbl)

        self._dm_label_badge = Badge("—")
        dl.addWidget(self._dm_label_badge)

        self._dm_sizing_badge = Badge("SIZE —")
        dl.addWidget(self._dm_sizing_badge)

        self._dm_entry_lbl = _t("", MUTED, 7)
        dl.addWidget(self._dm_entry_lbl, stretch=1)

        self._dm_wr_lbl = _t("", MUTED, 7, mono=True)
        dl.addWidget(self._dm_wr_lbl)

        self._dm_adx_lbl = _t("", MUTED, 7, mono=True)
        dl.addWidget(self._dm_adx_lbl)

        root.addWidget(dmstrip)

        # ── Verdict strip ─────────────────────────────────────────────────
        vstrip = QFrame()
        vstrip.setFixedHeight(28)
        vstrip.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 3px;"
        )
        vl = QHBoxLayout(vstrip)
        vl.setContentsMargins(8, 0, 8, 0)
        vl.setSpacing(6)

        vl.addWidget(_t("VERDICT", MUTED, 7, mono=True, bold=True))

        self._verdict_action = _t("—", WHITE, 10, mono=True, bold=True)
        vl.addWidget(self._verdict_action)

        self._verdict_detail = _t("", MUTED, 7)
        self._verdict_detail.setWordWrap(False)
        vl.addWidget(self._verdict_detail, stretch=1)

        self._verdict_exit = _t("", MUTED, 7, mono=True)
        self._verdict_exit.setWordWrap(False)
        self._verdict_exit.setMinimumWidth(120)
        vl.addWidget(self._verdict_exit)

        root.addWidget(vstrip)

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
        self._wf_titles = {}  # NEW: store heading labels so we can rename for ETFs
        for key in (
            "TREND & STATE",
            "GEX REGIME",
            "OI WALLS",
            "EXPIRY",
            "WILLIAMS %R",
            "MFI FLOW",
        ):
            title_lbl = _h(key, MUTED, 6)
            pa.addWidget(title_lbl)
            body = _t("", WHITE, 7)
            pa.addWidget(body)
            self._wf[key] = body
            self._wf_titles[key] = title_lbl

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
        for _ in range(7):
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
        for _ in range(12):
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

    def on_dual_mode_updated(self, symbol: str, dm_sig):
        """Update dual-mode strip, KPI tiles, Panel B score, and Panel C checklist."""
        if symbol != self._symbol:
            return

        from modules.dual_mode import DualModeSignal
        if not isinstance(dm_sig, DualModeSignal):
            return
        if not dm_sig.data_sufficient:
            return

        # ── Header badges (dual-mode-aware) ──────────────────────────────
        self._refresh_badges_dual(dm_sig)

        # ── Dual-mode strip ───────────────────────────────────────────────
        tier_style = {"PRIMARY": "BULLISH", "SECONDARY": "LATE", "NONE": "NEUTRAL"}.get(dm_sig.tier, "NEUTRAL")
        self._dm_mode_badge.set_badge(
            {"PRIMARY": "▶ PRIMARY", "SECONDARY": "▷ SECONDARY", "NONE": "MEAN REV"}.get(dm_sig.tier, "MR"),
            tier_style,
        )

        sc = dm_sig.dual_score
        from ui.theme import score_color
        sc_c = score_color(sc)
        self._dm_score_lbl.setText(f"{sc}")
        self._dm_score_lbl.setStyleSheet(f"color: {sc_c}; font-weight: bold;")

        label_style = {
            "STRONG": "BULLISH", "GOOD": "BULLISH",
            "WATCH": "LATE", "AVOID": "BEARISH",
        }.get(dm_sig.dual_label, "NEUTRAL")
        self._dm_label_badge.set_badge(dm_sig.dual_label, label_style)
        self._dm_sizing_badge.set_badge(f"SIZE: {dm_sig.dual_sizing}", dm_sig.dual_sizing)

        if dm_sig.entry_triggered:
            self._dm_entry_lbl.setText(f"▶ {dm_sig.entry_reason}")
            self._dm_entry_lbl.setStyleSheet(f"color: {EM};")
        else:
            parts = []
            if dm_sig.wr_30 >= -30: parts.append(f"WR={dm_sig.wr_30:.0f}")
            if dm_sig.above_sma: parts.append("above SMA")
            if dm_sig.mfi < 30: parts.append(f"MFI={dm_sig.mfi:.0f}")
            self._dm_entry_lbl.setText(
                f"Waiting: {', '.join(parts)}" if parts else "Conditions not met"
            )
            self._dm_entry_lbl.setStyleSheet(f"color: {MUTED};")

        # WR and MFI on strip
        wr_c = EM if dm_sig.wr_30 < -50 else (GREEN if dm_sig.wr_30 < -30 else MUTED)
        self._dm_wr_lbl.setText(f"WR(30)={dm_sig.wr_30:.0f}")
        self._dm_wr_lbl.setStyleSheet(f"color: {wr_c};")

        mfi_c = EM if dm_sig.mfi >= 50 else (GREEN if dm_sig.mfi >= 30 else RED)
        self._dm_adx_lbl.setText(f"MFI={dm_sig.mfi:.0f}")
        self._dm_adx_lbl.setStyleSheet(f"color: {mfi_c};")

        # ── Verdict strip ─────────────────────────────────────────────────
        self._update_verdict_strip(dm_sig)

        # ── KPI tiles ─────────────────────────────────────────────────────
        wr_c = EM if dm_sig.wr_30 < -50 else (GREEN if dm_sig.wr_30 < -30 else MUTED)
        self._kpi["WR(30)"].set_value(f"{dm_sig.wr_30:.0f}", wr_c)
        zone = "DEEP" if dm_sig.wr_30 < -50 else ("IN ZONE" if dm_sig.wr_30 < -30 else "—")
        self._kpi["WR(30)"].set_delta(zone, wr_c)

        mfi_c = EM if dm_sig.mfi >= 50 else (GREEN if dm_sig.mfi >= 30 else RED)
        self._kpi["MFI"].set_value(f"{dm_sig.mfi:.0f}", mfi_c)
        mfi_d = "Strong" if dm_sig.mfi >= 50 else ("OK" if dm_sig.mfi >= 30 else "Weak")
        self._kpi["MFI"].set_delta(mfi_d, mfi_c)

        sma_c = RED if not dm_sig.above_sma else EM
        bias = "BELOW" if not dm_sig.above_sma else "ABOVE"
        self._kpi["vs SMA"].set_value(bias, sma_c)
        self._kpi["vs SMA"].set_delta(f"{dm_sig.pct_from_sma:+.1f}% vs SMA", sma_c)

        # ── Panel B: Dual-mode score ─────────────────────────────────────
        self._pb.set_accent(sc_c)
        self._score.setText(str(sc))
        self._score.setStyleSheet(f"color: {sc_c}; font-weight: bold;")
        self._vlabel.setText(dm_sig.dual_label)
        self._vlabel.setStyleSheet(f"color: {sc_c}; font-weight: bold;")
        self._size_badge.set_badge(f"SIZE: {dm_sig.dual_sizing}", dm_sig.dual_sizing)

        # Risk notes
        while self._risk_container.count():
            w = self._risk_container.takeAt(0).widget()
            if w:
                w.deleteLater()
        from ui.components import RiskNote
        risk_notes = []
        if dm_sig.is_trap:
            risk_notes.append("⚠ TRAP: Bearish filing vs pullback setup")
        if dm_sig.options_overlay < -3:
            risk_notes.append(f"Options bearish ({dm_sig.options_detail})")
        if dm_sig.mfi < 30:
            risk_notes.append(f"MFI={dm_sig.mfi:.0f} — weak money flow")
        if dm_sig.above_sma:
            risk_notes.append("Above SMA — not a pullback")
        if dm_sig.vol_ratio < 0.5:
            risk_notes.append(f"Dry volume ({dm_sig.vol_ratio:.1f}x avg)")
        for note in risk_notes[:3]:
            self._risk_container.addWidget(RiskNote(note))

        # Key levels: always-available price levels + options when present
        all_levels = []

        # Slot 0: Daily SMA (always available from dual-mode)
        if dm_sig.sma_20 > 0:
            sma_lbl_c = EM if dm_sig.above_sma else RED
            all_levels.append(("DAILY SMA", dm_sig.sma_20, sma_lbl_c, f"{dm_sig.pct_from_sma:+.1f}%"))

        # Price-derived levels from BB bands (always available)
        ps = self._ps
        if ps:
            if getattr(ps, "upper", 0) > 0:
                pct = (dm_sig.close / ps.upper - 1) * 100 if dm_sig.close > 0 else 0
                all_levels.append(("BB UPPER", round(ps.upper, 2), GOLD, f"{pct:+.1f}%"))
            if getattr(ps, "lower", 0) > 0:
                pct = (dm_sig.close / ps.lower - 1) * 100 if dm_sig.close > 0 else 0
                all_levels.append(("BB LOWER", round(ps.lower, 2), GREEN, f"{pct:+.1f}%"))

        # Options-derived levels (when available — market hours only)
        ctx = self._ctx
        if ctx and hasattr(ctx, "walls"):
            w = ctx.walls
            if getattr(w, "resistance", None):
                all_levels.append(("OI RESIST", w.resistance, RED, f"{w.resistance_pct:+.1f}%"))
            if getattr(w, "support", None):
                all_levels.append(("OI SUPPORT", w.support, GREEN, f"{w.support_pct:+.1f}%"))
            if getattr(w, "max_pain", None):
                all_levels.append(("MAX PAIN", w.max_pain, GOLD, ""))
        if ctx and hasattr(ctx, "gex") and getattr(ctx.gex, "hvl", None):
            all_levels.append(("GEX HVL", ctx.gex.hvl, VIOLET, ""))

        for i, lr in enumerate(self._levels):
            if i < len(all_levels):
                lr.set_data(*all_levels[i])
                lr.setVisible(True)
            else:
                lr.setVisible(False)

        # Bottom badges — only show real options data
        has_real_opts = (
            ctx and hasattr(ctx, "walls")
            and getattr(ctx.walls, "pcr_oi", None) is not None
            and ctx.walls.pcr_oi != 1.0
        )
        if has_real_opts:
            self._bot["PCR"].set_badge(f"PCR {ctx.walls.pcr_oi:.2f}", "NEUTRAL")
        else:
            self._bot["PCR"].set_badge("PCR —", "NEUTRAL")
        if has_real_opts and hasattr(ctx, "gex"):
            self._bot["GEX"].set_badge(
                f"GEX {ctx.gex.regime.upper()}",
                {"Positive": "BEARISH", "Negative": "BULLISH", "Neutral": "NEUTRAL"}.get(
                    ctx.gex.regime, "NEUTRAL"
                ),
            )

        # ── Panel A: Regime tile ──────────────────────────────────────────
        tier_name = {"PRIMARY": "Primary Entry", "SECONDARY": "Secondary Entry", "NONE": "No Entry"}.get(dm_sig.tier, "—")
        self._regime_lbl.setText(f"MEAN REVERSION: {tier_name}")
        self._regime_lbl.setStyleSheet(f"color: {sc_c}; font-weight: bold;")
        self._regime_cap.set_badge(f"MAX: {dm_sig.dual_sizing}", dm_sig.dual_sizing)

        detail = (
            f"WR(30)={dm_sig.wr_30:.0f} | SMA {dm_sig.pct_from_sma:+.1f}% | "
            f"MFI={dm_sig.mfi:.0f} | DD={dm_sig.dd_from_high:.1f}% | "
            f"{dm_sig.red_streak}d red | Vol={dm_sig.vol_ratio:.1f}x"
        )
        self._regime_detail.setText(detail)

        frame_c = sc_c if dm_sig.dual_score >= 60 else GOLD if dm_sig.dual_score >= 45 else RED
        self._regime_frame.setStyleSheet(
            f"background: rgba(16,185,129,0.06); border-left: 3px solid {frame_c}; "
            f"border-radius: 4px; padding: 4px 8px;"
        )
        self._wf_badge.set_badge(f"SIZE: {dm_sig.dual_sizing}", dm_sig.dual_sizing)

        # Workflow lines
        if "TREND & STATE" in self._wf:
            self._wf["TREND & STATE"].setText(
                f"Mean reversion · WR(30)={dm_sig.wr_30:.0f} · "
                f"{dm_sig.pct_from_sma:+.1f}% vs SMA · MFI={dm_sig.mfi:.0f}"
            )
        if "WILLIAMS %R" in self._wf:
            if dm_sig.wr_30 < -50:
                self._wf["WILLIAMS %R"].setText(
                    f"WR(30) {dm_sig.wr_30:.0f} → Deeply oversold → Strong bounce expected"
                )
            elif dm_sig.wr_30 < -30:
                self._wf["WILLIAMS %R"].setText(
                    f"WR(30) {dm_sig.wr_30:.0f} → In mean-reversion zone → Pullback entry"
                )
            else:
                self._wf["WILLIAMS %R"].setText(
                    f"WR(30) {dm_sig.wr_30:.0f} → Not oversold → Wait for pullback"
                )
        if "MFI FLOW" in self._wf:
            if dm_sig.mfi >= 50:
                self._wf["MFI FLOW"].setText(
                    f"MFI {dm_sig.mfi:.0f} → Strong accumulation during pullback"
                )
            elif dm_sig.mfi >= 30:
                self._wf["MFI FLOW"].setText(
                    f"MFI {dm_sig.mfi:.0f} → Acceptable money flow"
                )
            else:
                self._wf["MFI FLOW"].setText(
                    f"MFI {dm_sig.mfi:.0f} → Weak — no buying pressure in this dip"
                )

        # Verdict
        if dm_sig.tier == "PRIMARY":
            vrgb = "16,185,129"; vc = EM
            verdict = (
                f"PRIMARY ENTRY: {dm_sig.entry_reason}. "
                f"Size: FULL. Exit: +5% PT or BBW contraction. Max hold: 25d."
            )
        elif dm_sig.tier == "SECONDARY":
            vrgb = "245,158,11"; vc = GOLD
            verdict = (
                f"SECONDARY ENTRY: {dm_sig.entry_reason}. "
                f"Size: HALF. Exit: +5% PT or BBW contraction. Max hold: 25d."
            )
        elif dm_sig.core_met:
            vrgb = "245,158,11"; vc = GOLD
            missing = []
            if dm_sig.dd_from_high > -5: missing.append(f"DD={dm_sig.dd_from_high:.1f}% (need <-5%)")
            if dm_sig.red_streak < 2: missing.append(f"{dm_sig.red_streak}d red (need ≥2)")
            if dm_sig.vol_ratio < 0.5: missing.append(f"Vol={dm_sig.vol_ratio:.1f}x (need ≥0.5)")
            verdict = (
                f"Core conditions met (WR+SMA+MFI). "
                f"Needs for PRIMARY: {'; '.join(missing)}. SECONDARY entry available."
            )
        else:
            vrgb = "239,68,68"; vc = RED
            if dm_sig.is_trap:
                verdict = "⚠ TRAP: Bearish filing conflicts with setup. SKIP."
            else:
                missing = []
                if dm_sig.wr_30 >= -30: missing.append(f"WR={dm_sig.wr_30:.0f} (need <-30)")
                if dm_sig.above_sma: missing.append("above SMA")
                if dm_sig.mfi < 30: missing.append(f"MFI={dm_sig.mfi:.0f} (need >30)")
                verdict = f"No entry. {'; '.join(missing)}. Wait for pullback."

        self._verdict.setText(verdict)
        self._verdict.setStyleSheet(
            f"background: rgba({vrgb},0.06); color: {WHITE}; "
            f"border-left: 3px solid {vc}; border-radius: 3px; padding: 5px 8px;"
        )

        # ── Panel C: Dual-mode checklist ─────────────────────────────────
        checks = _build_dual_checklist(dm_sig, ctx)
        for i, cr in enumerate(self._checks):
            if i < len(checks):
                cr.set_data(*checks[i])
                cr.setVisible(True)
            else:
                cr.setVisible(False)

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
            self._conviction_lbl.setText("-")
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
        """Legacy badge refresh from PriceSignals — will be overwritten by dual-mode."""
        ps = self._ps
        if not ps:
            return
        # These are placeholders until dual-mode updates arrive
        bias = ps.daily_bias
        self._badges["SMA"].set_badge(
            f"{bias} {ps.daily_bias_pct:+.1f}%" if ps.daily_bias_pct else bias,
            {"BULLISH": "BULLISH", "BEARISH": "BEARISH", "NEUTRAL": "NEUTRAL"}.get(bias, "NEUTRAL"),
        )

    def _update_verdict_strip(self, dm_sig):
        """Populate the verdict strip with actionable guidance."""
        tier = dm_sig.tier
        sc = dm_sig.dual_score
        mfi = dm_sig.mfi
        wr = dm_sig.wr_30
        dd = dm_sig.dd_from_high

        # Action text + color
        if sc <= 30:
            action = "⚠ AVOID"
            ac = RED
            detail = "TRAP detected or setup too weak"
            exit_txt = ""
        elif tier == "PRIMARY":
            action = "▶ BUY FULL" if sc >= 75 else "▶ BUY FULL"
            ac = GREEN
            detail = (
                f"Score {sc} {'STRONG' if sc >= 75 else 'GOOD'} — "
                f"all conditions met — WR={wr:.0f} MFI={mfi:.0f} DD={dd:.1f}%"
            )
            exit_txt = "PT +5% · BBW · 25d"
        elif tier == "SECONDARY":
            if sc >= 75:
                action = "▷ BUY HALF"
                detail = f"Score {sc} STRONG — core met, needs DD/streak for FULL"
            elif sc >= 60:
                action = "▷ BUY HALF"
                detail = f"Score {sc} GOOD — core conditions met"
            else:
                action = "▷ HALF"
                detail = f"Score {sc} — marginal, manage risk"
            ac = GOLD
            exit_txt = "PT +5% · BBW · 25d"
        else:
            # No entry — explain what to watch
            if mfi < 30:
                if wr < -50 and dd < -10:
                    action = "WATCH"
                    ac = BLUE
                    detail = f"Capitulation — WR={wr:.0f} DD={dd:.1f}% — wait for MFI to cross 30"
                else:
                    action = "WAIT"
                    ac = GOLD
                    detail = f"MFI={mfi:.0f} too weak — watch for accumulation signal"
            elif wr >= -30:
                action = "SKIP"
                ac = MUTED
                detail = f"WR={wr:.0f} — not oversold, wait for pullback"
            else:
                action = "FORMING"
                ac = GOLD
                detail = f"Score {sc} — setup building, watch for entry trigger"
            exit_txt = ""

        self._verdict_action.setText(action)
        self._verdict_action.setStyleSheet(f"color: {ac}; font-weight: bold;")
        self._verdict_detail.setText(detail)
        self._verdict_detail.setStyleSheet(f"color: {WHITE};")
        self._verdict_exit.setText(exit_txt)
        self._verdict_exit.setStyleSheet(f"color: {MUTED};")

    def _refresh_badges_dual(self, dm_sig):
        """Update header badges from unified mean-reversion signal."""
        # Tier
        tier = dm_sig.tier
        tier_style = {"PRIMARY": "FRESH", "SECONDARY": "LATE", "NONE": "NEUTRAL"}.get(tier, "NEUTRAL")
        self._badges["TIER"].set_badge(
            {"PRIMARY": "PRIMARY", "SECONDARY": "SECONDARY", "NONE": "NO ENTRY"}.get(tier, tier),
            tier_style,
        )

        # WR(30)
        wr = dm_sig.wr_30
        if wr < -50:
            self._badges["WR"].set_badge(f"WR(30) {wr:.0f} DEEP", "FRESH")
        elif wr < -30:
            self._badges["WR"].set_badge(f"WR(30) {wr:.0f} ZONE", "BULLISH")
        else:
            self._badges["WR"].set_badge(f"WR(30) {wr:.0f}", "NEUTRAL")

        # MFI
        mfi = dm_sig.mfi
        if mfi >= 50:
            self._badges["MFI"].set_badge(f"MFI {mfi:.0f} STRONG", "BULLISH")
        elif mfi >= 30:
            self._badges["MFI"].set_badge(f"MFI {mfi:.0f} OK", "FRESH")
        else:
            self._badges["MFI"].set_badge(f"MFI {mfi:.0f} WEAK", "BEARISH")

        # SMA — below is GOOD for mean reversion
        if not dm_sig.above_sma:
            self._badges["SMA"].set_badge(f"BELOW SMA {dm_sig.pct_from_sma:+.1f}%", "BULLISH")
        else:
            self._badges["SMA"].set_badge(f"ABOVE SMA {dm_sig.pct_from_sma:+.1f}%", "BEARISH")

        # DD from high
        dd = dm_sig.dd_from_high
        streak = dm_sig.red_streak
        if dd <= -10:
            self._badges["DD"].set_badge(f"DD {dd:.0f}% {streak}d red", "FRESH")
        elif dd <= -5:
            self._badges["DD"].set_badge(f"DD {dd:.0f}% {streak}d red", "BULLISH")
        else:
            self._badges["DD"].set_badge(f"DD {dd:.0f}%", "NEUTRAL")

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
            self._kpi["WR(30)"].set_value(f"{ps.wr_value:.1f}", wc)
            self._kpi["WR(30)"].set_delta(ps.wr_phase, wc)
        if ps:
            # MFI tile placeholder (will be overwritten by dual-mode)
            if ps.mfi_value is not None:
                mc = EM if ps.mfi_value >= 50 else (GREEN if ps.mfi_value >= 30 else RED)
                self._kpi["MFI"].set_value(f"{ps.mfi_value:.0f}", mc)
            bc = {"BULLISH": EM, "BEARISH": RED, "NEUTRAL": GOLD}.get(
                ps.daily_bias, MUTED
            )
            self._kpi["vs SMA"].set_value(ps.daily_bias, bc)
            if ps.daily_bias_pct:
                self._kpi["vs SMA"].set_delta(
                    f"{ps.daily_bias_pct:+.1f}% vs SMA", bc
                )
        # Options-derived KPIs — show real or cached data, indicate staleness
        has_real_opts = (
            ctx and hasattr(ctx, "walls")
            and getattr(ctx.walls, "pcr_oi", None) is not None
            and ctx.walls.pcr_oi != 1.0  # default/empty value
        )
        if has_real_opts:
            pcr = ctx.walls.pcr_oi
            self._kpi["PCR OI"].set_value(
                f"{pcr:.3f}", EM if pcr >= 1.1 else (RED if pcr < 0.7 else WHITE)
            )
            sentiment = getattr(ctx.walls, "pcr_sentiment", "")
            self._kpi["PCR OI"].set_delta(sentiment, MUTED)
        else:
            self._kpi["PCR OI"].set_value("—", MUTED)
            self._kpi["PCR OI"].set_delta("no data", MUTED)

        if has_real_opts and hasattr(ctx, "gex"):
            g = ctx.gex.net_gex
            self._kpi["NET GEX"].set_value(
                f"{g:+,.0f}M", EM if g < 0 else (RED if g > 0 else MUTED)
            )
            self._kpi["NET GEX"].set_delta(ctx.gex.regime, MUTED)
        else:
            self._kpi["NET GEX"].set_value("—", MUTED)
            self._kpi["NET GEX"].set_delta("", MUTED)
        if has_real_opts and ctx and hasattr(ctx, "expiry") and ctx.expiry.days_remaining < 99:
            d = ctx.expiry.days_remaining
            self._kpi["TO EXPIRY"].set_value(
                f"{d}d", RED if d <= 2 else (GOLD if d <= 4 else WHITE)
            )
            self._kpi["TO EXPIRY"].set_delta(f"{ctx.expiry.pin_risk} pin", MUTED)
        else:
            self._kpi["TO EXPIRY"].set_value("—", MUTED)
            self._kpi["TO EXPIRY"].set_delta("", MUTED)
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
        except Exception as _e:
            logger.exception("get_commentary failed: %s", _e)  # ← shows real error
            comm = {}
        for key, ck in {
            "TREND & STATE": "bias_line",
            "GEX REGIME": "gex_line",
            "OI WALLS": "wall_line",
            "EXPIRY": "expiry_line",
            "WILLIAMS %R": "wr_line",
            "MFI FLOW": "mfi_line",
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

        # ── ETF-specific workflow headings ───────────────────────────────────
        if hasattr(self, "_wf_titles"):
            title_map = {
                "TREND & STATE": "TREND & STATE",
                "GEX REGIME": "VOLUME & VSR",
                "OI WALLS": "VWAP & POC",
                "EXPIRY": "UNDERLYING",
                "WILLIAMS %R": "NAV & PREMIUM",
            }
            for old, new in title_map.items():
                if old in self._wf_titles:
                    self._wf_titles[old].setText(new.upper())

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

        # MFI
        ps = self._ps

        if ps and ps.mfi_value is not None:
            mv = ps.mfi_value
            if not ps.mfi_reliable:
                mfi_line = f"MFI {mv:.0f} — volume too thin, signal unreliable"
            elif ps.mfi_diverge:
                mfi_line = (
                    f"MFI {mv:.0f} ⚠ BEARISH DIVERGENCE — "
                    f"price near high but money leaving → SIZE capped HALF"
                )
            else:
                desc = {
                    "STRONG": "strong buying pressure — volume confirming upside",
                    "RISING": "money flowing in — volume supporting price",
                    "NEUTRAL": "neutral money flow — no volume edge",
                    "FALLING": "money flow softening — watch for continuation",
                    "WEAK": "heavy selling pressure — distribution in progress",
                }.get(ps.mfi_state, "")
                mfi_line = f"MFI {mv:.0f} {ps.mfi_state} — {desc}"
        else:
            mfi_line = "MFI — volume data unavailable"

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
            "MFI FLOW": mfi_line,
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
            level_data.append(("VAH", vp.vah, RED, "val area"))
            level_data.append(("POC", vp.poc, GOLD, "vol node"))
            level_data.append(("VAL", vp.val, GREEN, "val area"))
        if et and et.vwap > 0:
            level_data.append(
                (
                    "VWAP",
                    et.vwap,
                    BLUE,
                    f"{et.pct_from_vwap:+.1f}%",
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
