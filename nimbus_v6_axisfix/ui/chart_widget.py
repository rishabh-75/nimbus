"""
ui/chart_widget.py — TradingView-quality pyqtgraph chart.
Fixes: Y-axis level tags (not floating text), consistent axis widths, VWAP.
"""
from __future__ import annotations
import datetime, logging
from typing import Optional
import numpy as np, pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPicture, QColor, QFont
from modules.analytics import OptionsContext
from modules.indicators import PriceSignals
from ui.theme import (BG, SURFACE, BORDER, EM, RED, GOLD, VIOLET,
                       WHITE, MUTED, GREEN, UP, DOWN, BB_LINE, BLUE, FONT_MONO)

logger = logging.getLogger(__name__)

_UP_BRUSH   = pg.mkBrush(UP)
_DOWN_BRUSH = pg.mkBrush(DOWN)
_BB_UPPER   = pg.mkPen('#C8C8D4', width=1.8)
_BB_LOWER   = pg.mkPen('#C8C8D4', width=1.8)
_BB_MID     = pg.mkPen(color='#F59E0B', width=1.2, style=Qt.PenStyle.DashLine)
_BB_FILL    = pg.mkBrush(200, 200, 210, 18)
_WR_PEN     = pg.mkPen(EM, width=1.5)
_VWAP_PEN   = pg.mkPen('#E2E8F0', width=1.0, style=Qt.PenStyle.DotLine)
_CROSS_PEN  = pg.mkPen(BORDER, width=0.6, style=Qt.PenStyle.DashLine)
_VOL_UP     = pg.mkBrush(38, 166, 154, 180)
_VOL_DOWN   = pg.mkBrush(239, 83, 80, 180)

_WALL_COLORS = {
    "R": RED, "S": GREEN, "MP": GOLD, "HVL": VIOLET, "SMA": MUTED,
}
_Y_AXIS_W = 80  # consistent Y-axis width for price panel


class DateAxisItem(pg.AxisItem):
    def __init__(self, timestamps=None, *a, **kw):
        super().__init__(*a, **kw)
        self._ts = timestamps or []
    def set_timestamps(self, ts): self._ts = ts
    def tickStrings(self, values, scale, spacing):
        r, prev = [], None
        for v in values:
            i = int(round(v))
            if 0 <= i < len(self._ts):
                t = self._ts[i]
                if isinstance(t, pd.Timestamp): t = t.to_pydatetime()
                d = t.strftime("%d %b")
                r.append(d if d != prev else t.strftime("%H:%M"))
                prev = d
            else: r.append("")
        return r


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        super().__init__()
        self.picture = QPicture()
        p = QPainter(self.picture)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = 0.28
        for i, (_, row) in enumerate(data.iterrows()):
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            if pd.isna(o) or pd.isna(c): continue
            color = QColor(UP) if c >= o else QColor(DOWN)
            p.setPen(pg.mkPen(color, width=0.8))
            p.drawLine(QtCore.QPointF(i, l), QtCore.QPointF(i, h))
            p.setBrush(pg.mkBrush(color))
            bt, bb = max(o, c), min(o, c)
            if bt - bb < 0.001:
                p.drawLine(QtCore.QPointF(i-w, c), QtCore.QPointF(i+w, c))
            else:
                p.drawRect(QtCore.QRectF(i-w, bb, 2*w, bt-bb))
        p.end()
    def paint(self, p, *a): p.drawPicture(0, 0, self.picture)
    def boundingRect(self): return QtCore.QRectF(self.picture.boundingRect())


def filter_market_hours(df):
    if df is None or df.empty: return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None: idx_ist = idx.tz_convert("Asia/Kolkata")
    else:
        try: idx_ist = idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except: idx_ist = idx
    hours = idx_ist.hour if hasattr(idx_ist, 'hour') else None
    if hours is None or hours.min() == hours.max(): return df
    return df[(idx_ist.weekday < 5) & (idx_ist.hour >= 9) & (idx_ist.hour < 16)].copy()


def _to_ist(df):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None: idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    else: idx = idx + pd.Timedelta(hours=5, minutes=30)
    return idx.to_pydatetime().tolist()


def _compute_vwap(df):
    if "Volume" not in df.columns or df["Volume"].sum() == 0: return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0)
    dates = pd.DatetimeIndex(df.index).date
    vwap = pd.Series(index=df.index, dtype=float)
    for d in pd.unique(dates):
        m = dates == d
        cv = (tp[m] * vol[m]).cumsum()
        vwap[m] = cv / vol[m].cumsum().replace(0, np.nan)
    return vwap.values


class NimbusChart(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, show=False)
        self.setBackground(BG)
        self._price_plot = self._vol_plot = self._wr_plot = None
        self._date_axis = None
        self._crosshair_v = None
        self._crosshair_h = []
        self._timestamps = []
        self._df = None
        self._show_vwap = True
        self._last_ctx = self._last_ps = self._last_sym = None
        self._init_layout()

    def _init_layout(self):
        self.ci.layout.setSpacing(3)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self._date_axis = DateAxisItem([], orientation="bottom")

        self._price_plot = self.addPlot(row=0, col=0)
        self._cfg(self._price_plot, show_x=False, y_width=_Y_AXIS_W)

        self._vol_plot = self.addPlot(row=1, col=0)
        self._cfg(self._vol_plot, show_x=False, y_width=50)

        self._wr_plot = self.addPlot(row=2, col=0, axisItems={"bottom": self._date_axis})
        self._cfg(self._wr_plot, show_x=True, y_width=50)
        self._wr_plot.setYRange(-100, 0, padding=0.02)
        self._wr_plot.getAxis("right").setTicks([
            [(-20, "-20"), (-50, "-50"), (-80, "-80")],
            [(0, "0"), (-100, "-100")],
        ])

        for p in (self._vol_plot, self._wr_plot): p.setXLink(self._price_plot)
        self.ci.layout.setRowStretchFactor(0, 63)
        self.ci.layout.setRowStretchFactor(1, 12)
        self.ci.layout.setRowStretchFactor(2, 25)

        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=_CROSS_PEN)
        self._crosshair_v.setVisible(False)
        self._price_plot.addItem(self._crosshair_v, ignoreBounds=True)
        for plot in (self._price_plot, self._vol_plot, self._wr_plot):
            h = pg.InfiniteLine(angle=0, movable=False, pen=_CROSS_PEN)
            h.setVisible(False); plot.addItem(h, ignoreBounds=True)
            self._crosshair_h.append(h)
        self._price_plot.scene().sigMouseMoved.connect(self._on_mouse)

    def _cfg(self, plot, show_x=False, y_width=60):
        plot.hideButtons(); plot.setMenuEnabled(False)
        plot.showGrid(x=True, y=True, alpha=0.08)
        for a in ("left", "right", "top", "bottom"):
            ax = plot.getAxis(a)
            ax.setPen(pg.mkPen(BORDER, width=0.5))
            ax.setTextPen(pg.mkPen(MUTED))
            ax.setStyle(tickLength=-4, tickTextOffset=4)
        plot.showAxis("right"); plot.hideAxis("left")
        plot.getAxis("right").setWidth(y_width)
        if not show_x: plot.hideAxis("bottom")
        plot.hideAxis("top")
        plot.getViewBox().setDefaultPadding(0.01)

    def _on_mouse(self, pos):
        for i, plot in enumerate((self._price_plot, self._vol_plot, self._wr_plot)):
            if plot.sceneBoundingRect().contains(pos):
                mp = plot.vb.mapSceneToView(pos)
                self._crosshair_v.setPos(mp.x()); self._crosshair_v.setVisible(True)
                self._crosshair_h[i].setPos(mp.y()); self._crosshair_h[i].setVisible(True)
                for j, h in enumerate(self._crosshair_h):
                    if j != i: h.setVisible(False)
                return
        self._crosshair_v.setVisible(False)
        for h in self._crosshair_h: h.setVisible(False)

    def update_data(self, price_df, ctx=None, ps=None, symbol="NIFTY"):
        if price_df is None or price_df.empty: return
        self._last_ctx, self._last_ps, self._last_sym = ctx, ps, symbol
        df = filter_market_hours(price_df)
        if df.empty: df = price_df
        self._df = df; n = len(df); xs = np.arange(n, dtype=float)
        self._timestamps = _to_ist(df)
        self._date_axis.set_timestamps(self._timestamps)

        for plot in (self._price_plot, self._vol_plot, self._wr_plot): plot.clear()
        # Re-add crosshairs
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=_CROSS_PEN)
        self._crosshair_v.setVisible(False)
        self._price_plot.addItem(self._crosshair_v, ignoreBounds=True)
        self._crosshair_h = []
        for plot in (self._price_plot, self._vol_plot, self._wr_plot):
            h = pg.InfiniteLine(angle=0, movable=False, pen=_CROSS_PEN)
            h.setVisible(False); plot.addItem(h, ignoreBounds=True)
            self._crosshair_h.append(h)

        # ── Candles + BB ──────────────────────────────────────────────────────
        self._price_plot.addItem(CandlestickItem(df))
        if "BB_Upper" in df.columns:
            u = df["BB_Upper"].ffill().values.astype(float)
            l = df["BB_Lower"].ffill().values.astype(float)
            m = df["BB_Mid"].ffill().values.astype(float)
            uc = pg.PlotDataItem(xs, u, pen=_BB_UPPER)
            lc = pg.PlotDataItem(xs, l, pen=_BB_LOWER)
            self._price_plot.addItem(pg.FillBetweenItem(uc, lc, brush=_BB_FILL))
            self._price_plot.addItem(uc); self._price_plot.addItem(lc)
            self._price_plot.addItem(pg.PlotDataItem(xs, m, pen=_BB_MID))

        # VWAP
        if self._show_vwap:
            vwap = _compute_vwap(df)
            if vwap is not None:
                v_ok = ~np.isnan(vwap)
                if v_ok.any():
                    self._price_plot.addItem(pg.PlotDataItem(
                        xs[v_ok], vwap[v_ok], pen=_VWAP_PEN, connect="finite"))

        # Daily SMA
        if ps and ps.daily_sma:
            sc = EM if ps.daily_bias == "BULLISH" else (RED if ps.daily_bias == "BEARISH" else GOLD)
            self._price_plot.addItem(pg.InfiniteLine(
                pos=ps.daily_sma, angle=0, movable=False,
                pen=pg.mkPen(sc, width=0.9, style=Qt.PenStyle.DashLine)))

        # ── Wall lines + right-margin level labels ─────────────────────────────
        # DO NOT use setTicks() on price axis — it kills auto price ticks.
        # Instead, place small colored TextItems anchored at the right edge.
        if ctx:
            levels = [
                (ctx.walls.resistance, RED, Qt.PenStyle.DashLine, "R"),
                (ctx.walls.support, GREEN, Qt.PenStyle.DashLine, "S"),
                (ctx.walls.max_pain, GOLD, Qt.PenStyle.DotLine, "MP"),
                (ctx.gex.hvl, VIOLET, Qt.PenStyle.DashDotLine, "HVL"),
            ]
            for yv, c, dash, prefix in levels:
                if yv is None: continue
                self._price_plot.addItem(pg.InfiniteLine(
                    pos=yv, angle=0, movable=False,
                    pen=pg.mkPen(c, width=0.8, style=dash)))
                # Small label at right edge of data
                lbl = pg.TextItem(
                    f" {prefix} {yv:,.0f} ", color=c, anchor=(0, 0.5),
                    fill=pg.mkBrush(QColor(BG)),
                    border=pg.mkPen(c, width=0.5))
                lbl.setFont(QFont(FONT_MONO, 7))
                lbl.setPos(n + 0.5, yv)
                self._price_plot.addItem(lbl)

        # Position state badge
        if ps and ps.position_state not in ("UNKNOWN", None):
            labels = {"RIDING_UPPER": "RIDING UPPER", "FIRST_DIP": "FIRST DIP",
                      "MID_BAND_BROKEN": "MID-BAND BROKEN", "CONSOLIDATING": "CONSOLIDATING"}
            colors = {"RIDING_UPPER": EM, "FIRST_DIP": GOLD, "MID_BAND_BROKEN": RED}
            col = colors.get(ps.position_state, MUTED)
            lbl = pg.TextItem(f" {labels.get(ps.position_state, '')} ", color=col,
                              anchor=(0, 0), border=pg.mkPen(col, width=1), fill=pg.mkBrush(BG))
            lbl.setFont(QFont(FONT_MONO, 7))
            if "High" in df.columns: lbl.setPos(0, float(df["High"].max()))
            self._price_plot.addItem(lbl)

        # ── Volume ────────────────────────────────────────────────────────────
        has_vol = False
        if "Volume" in df.columns:
            vols = df["Volume"].fillna(0).values.astype(float)
            if vols.sum() > 0:
                has_vol = True
                br = [_VOL_UP if c >= o else _VOL_DOWN
                      for c, o in zip(df["Close"].values, df["Open"].values)]
                self._vol_plot.addItem(pg.BarGraphItem(
                    x=xs, height=vols, width=0.7, brushes=br, pen=pg.mkPen(None)))
                self._vol_plot.enableAutoRange(axis='y')
        if has_vol:
            self.ci.layout.setRowStretchFactor(1, 12)
            self._vol_plot.setMaximumHeight(16777215)
        else:
            self.ci.layout.setRowStretchFactor(1, 0)
            self._vol_plot.setMaximumHeight(0)

        # ── W%R ───────────────────────────────────────────────────────────────
        if "WR" in df.columns:
            wr = df["WR"].ffill().values.astype(float)
            ok = ~np.isnan(wr)
            if ok.any():
                self._wr_plot.addItem(pg.PlotDataItem(xs[ok], wr[ok], pen=_WR_PEN))
                # Zone fills
                ct = pg.PlotDataItem(xs, np.clip(wr, -20, 0))
                cb = pg.PlotDataItem(xs, np.full(n, -20.0))
                self._wr_plot.addItem(pg.FillBetweenItem(ct, cb, brush=pg.mkBrush(16, 185, 129, 20)))
                ct2 = pg.PlotDataItem(xs, np.full(n, -80.0))
                cb2 = pg.PlotDataItem(xs, np.clip(wr, -100, -80))
                self._wr_plot.addItem(pg.FillBetweenItem(ct2, cb2, brush=pg.mkBrush(239, 68, 68, 15)))
                for yv, c, w in [(-20, EM, 0.6), (-50, BORDER, 0.4), (-80, RED, 0.6)]:
                    self._wr_plot.addItem(pg.InfiniteLine(
                        pos=yv, angle=0, movable=False,
                        pen=pg.mkPen(c, width=w, style=Qt.PenStyle.DotLine)))
                # Current WR
                lw = float(wr[ok][-1])
                from ui.theme import wr_color
                wrc = wr_color(lw)
                self._wr_plot.addItem(pg.InfiniteLine(
                    pos=lw, angle=0, movable=False,
                    pen=pg.mkPen(wrc, width=0.8, style=Qt.PenStyle.DashLine)))
                # Override WR Y-axis ticks to include current value
                self._wr_plot.getAxis("right").setTicks([
                    [(-20, "-20"), (-80, "-80"), (lw, f"{lw:.1f}")],
                    [(-50, "-50")],
                ])

        self._price_plot.setXRange(-1, n+2, padding=0.01)
        self._wr_plot.setYRange(-100, 0, padding=0.02)
        if "High" in df.columns and "Low" in df.columns:
            self._price_plot.setYRange(
                float(df["Low"].min())*0.997, float(df["High"].max())*1.003, padding=0.01)
