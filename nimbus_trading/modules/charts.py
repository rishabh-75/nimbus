"""
modules/charts.py  —  NIMBUS Emerald Slate
==========================================
Root-cause fixes vs broken versions:
  1. Index → ISO string BEFORE any plotly call (eliminates 1970 x-axis / ns epoch bug)
  2. BB fill uses polygon method (upper + reversed lower, toself)
     — NOT tonexty (breaks when candlestick trace is between them)
  3. Wall shapes use row=1, col=1 explicitly
  4. Wall annotations use yref="y1" (not "y") to pin to candle panel
  5. WR hlines use yref="y3" string form (not add_hline which can't specify subplot)
"""

from __future__ import annotations
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.analytics import OptionsContext

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#080c10"
SURFACE = "#0d1117"
BORDER = "#1e2937"
EM = "#10b981"
RED = "#ef4444"
GOLD = "#f59e0b"
VIOLET = "#8b5cf6"
WHITE = "#e2e8f0"
MUTED = "#64748b"
UP = "#26a69a"
DOWN = "#ef5350"

BB_LINE = "rgba(200,200,220,0.60)"  # neutral grey — TradingView style
BB_FILL = "rgba(200,200,220,0.05)"
BB_MID = "rgba(200,200,220,0.30)"


def _to_str_index(df: pd.DataFrame) -> list:
    """
    Convert any DatetimeIndex variant to ISO strings before passing to Plotly.
    Handles: tz-aware, tz-naive, datetime64[ns], datetime64[s], DatetimeIndex.
    This eliminates the 1970-epoch bug caused by nanosecond timestamps.
    """
    try:
        idx = pd.DatetimeIndex(df.index)
        # tz_convert(None) = convert to UTC and strip tz (safe for tz-aware)
        # For tz-naive, this is a no-op
        if idx.tzinfo is not None:
            idx = idx.tz_convert(None)
        return idx.strftime("%Y-%m-%d %H:%M").tolist()
    except Exception:
        # Last resort: stringify whatever we have
        return [str(x)[:16] for x in df.index]


def main_chart(
    price_df: pd.DataFrame,
    ctx: Optional[OptionsContext] = None,
    symbol: str = "NIFTY",
) -> go.Figure:

    if price_df is None or price_df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=BG,
            plot_bgcolor=SURFACE,
            height=640,
            margin=dict(l=10, r=20, t=32, b=10),
            annotations=[
                dict(
                    text="No price data — click Refresh",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color=MUTED, size=13),
                )
            ],
        )
        return fig

    df = price_df.copy()
    xs = _to_str_index(df)  # ← ALL traces share this string x-axis

    has_vol = "Volume" in df.columns and df["Volume"].fillna(0).sum() > 0
    has_wr = "WR" in df.columns
    has_bb = "BB_Upper" in df.columns

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.63, 0.14, 0.23],
        shared_xaxes=True,
        vertical_spacing=0.008,
    )

    # ── Row 1 · Candlesticks ──────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=xs,
            open=df["Open"].tolist(),
            high=df["High"].tolist(),
            low=df["Low"].tolist(),
            close=df["Close"].tolist(),
            increasing=dict(line=dict(color=UP, width=0.8), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=0.8), fillcolor=DOWN),
            name="Price",
            hovertext=[
                f"O:{r['Open']:.1f}  H:{r['High']:.1f}  L:{r['Low']:.1f}  C:{r['Close']:.1f}"
                for _, r in df.iterrows()
            ],
            hoverinfo="text+x",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ── Row 1 · Bollinger Bands (neutral grey, polygon fill) ─────────────────
    if has_bb:
        upper = df["BB_Upper"].ffill().tolist()
        lower = df["BB_Lower"].ffill().tolist()
        mid = df["BB_Mid"].ffill().tolist()

        # Filled band: upper path forward + lower path reversed = closed polygon
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=BB_FILL,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # Upper band line
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=upper,
                line=dict(color=BB_LINE, width=1.2),
                hovertemplate="BB Up: %{y:.1f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # Lower band line
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=lower,
                line=dict(color=BB_LINE, width=1.2),
                hovertemplate="BB Lo: %{y:.1f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # Mid line
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=mid,
                line=dict(color=BB_MID, width=0.9, dash="dot"),
                hovertemplate="BB Mid: %{y:.1f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ── Row 1 · OI wall lines ─────────────────────────────────────────────────
    if ctx is not None:
        _add_wall_lines(fig, ctx)

    # ── Row 2 · Volume ────────────────────────────────────────────────────────
    if has_vol:
        closes = df["Close"].values
        opens = df["Open"].values
        vol_colors = [
            "rgba(38,166,154,0.50)" if c >= o else "rgba(239,83,80,0.50)"
            for c, o in zip(closes, opens)
        ]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=df["Volume"].fillna(0).tolist(),
                marker_color=vol_colors,
                marker_line_width=0,
                hovertemplate="Vol: %{y:,.0f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=2, col=1)

    # ── Row 3 · Williams %R ───────────────────────────────────────────────────
    if has_wr:
        wr_vals = df["WR"].ffill().tolist()

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=wr_vals,
                line=dict(color=EM, width=1.6),
                hovertemplate="W%%R: %{y:.1f}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Momentum zone fill: shade above -20
        above = [max(v, -20) for v in wr_vals]
        fill_y = above + [-20] * len(xs)
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=fill_y,
                fill="toself",
                fillcolor="rgba(16,185,129,0.10)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Reference lines on WR panel (yref="y3" = third subplot y-axis)
        for y_val, color in [(-20, EM), (-50, BORDER), (-80, RED)]:
            fig.add_shape(
                type="line",
                xref="x3",
                yref="y3",
                x0=xs[0],
                x1=xs[-1],
                y0=y_val,
                y1=y_val,
                line=dict(color=color, width=0.8, dash="dash"),
            )
        # Labels for WR thresholds
        for y_val, color, lbl in [(-20, EM, "-20"), (-80, RED, "-80")]:
            fig.add_annotation(
                xref="paper",
                yref="y3",
                x=1.01,
                y=y_val,
                text=lbl,
                showarrow=False,
                font=dict(color=color, size=8, family="JetBrains Mono,monospace"),
                xanchor="left",
                yanchor="middle",
            )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    xax = dict(
        gridcolor=BORDER,
        color=MUTED,
        tickfont=dict(size=8, color=MUTED),
        rangeslider=dict(visible=False),
        showgrid=True,
        showline=False,
        # Don't set type="date" — we're passing strings
    )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font=dict(family="'JetBrains Mono',monospace", color=WHITE, size=10),
        margin=dict(l=10, r=118, t=32, b=8),
        showlegend=False,
        height=660,
        title=dict(
            text=(
                f"<b style='color:{WHITE}'>{symbol}</b>"
                f"<span style='color:{MUTED}'>"
                f" · 4H · BB(20,1σ) · W%R(50) · Vol</span>"
            ),
            font=dict(size=11),
            x=0.01,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER,
            font=dict(color=WHITE, size=10),
        ),
        xaxis=xax,
        xaxis2=xax,
        xaxis3=xax,
        yaxis=dict(
            gridcolor=BORDER,
            color=WHITE,
            tickformat=",.0f",
            tickfont=dict(size=9),
            side="right",
            showline=False,
        ),
        yaxis2=dict(
            gridcolor="rgba(0,0,0,0)",
            color=MUTED,
            tickformat=".2s",
            tickfont=dict(size=7, color=MUTED),
            side="right",
            showgrid=False,
            showline=False,
        ),
        yaxis3=dict(
            gridcolor=BORDER,
            color=MUTED,
            tickformat=".0f",
            tickfont=dict(size=8, color=MUTED),
            side="right",
            range=[-105, 5],
            tickvals=[-20, -50, -80],
            showline=False,
        ),
    )
    return fig


def _add_wall_lines(fig: go.Figure, ctx: OptionsContext) -> None:
    """
    Horizontal OI wall / max pain / GEX HVL lines on the candle panel (row 1).
    Shapes: row=1, col=1 to pin to first subplot.
    Annotations: yref="y1" to pin to first subplot y-axis.
    """
    levels = [
        (
            ctx.walls.resistance,
            RED,
            "dash",
            f"R  {ctx.walls.resistance:,.0f}" if ctx.walls.resistance else None,
        ),
        (
            ctx.walls.support,
            EM,
            "dash",
            f"S  {ctx.walls.support:,.0f}" if ctx.walls.support else None,
        ),
        (
            ctx.walls.max_pain,
            GOLD,
            "dot",
            f"MP {ctx.walls.max_pain:,.0f}" if ctx.walls.max_pain else None,
        ),
        (
            ctx.gex.hvl,
            VIOLET,
            "dashdot",
            f"HVL {ctx.gex.hvl:,.0f}" if ctx.gex.hvl else None,
        ),
    ]
    for y_val, color, dash, label in levels:
        if y_val is None or label is None:
            continue
        fig.add_shape(
            type="line",
            xref="paper",
            yref="y1",  # y1 = first subplot axis
            x0=0,
            x1=1,
            y0=y_val,
            y1=y_val,
            line=dict(color=color, width=1.1, dash=dash),
        )
        fig.add_annotation(
            xref="paper",
            yref="y1",  # y1 = first subplot axis
            x=1.01,
            y=y_val,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=color, size=9, family="JetBrains Mono,monospace"),
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(8,12,16,0.9)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )


def gex_expiry_bar(ctx: OptionsContext) -> go.Figure:
    data = ctx.gex.by_expiry
    fig = go.Figure()

    if not data:
        fig.add_annotation(
            text="No GEX data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=10),
        )
    else:
        labels = [d[0] for d in data]
        vals = [d[2] for d in data]
        pcts = [d[3] for d in data]
        colors = [
            "rgba(38,166,154,0.75)" if v >= 0 else "rgba(239,68,68,0.75)" for v in vals
        ]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                marker_line_width=0,
                text=[f"{p:.1f}%" for p in pcts],
                textposition="outside",
                textfont=dict(color=WHITE, size=8),
                hovertemplate="<b>%{x}</b><br>%{y:,.0f}M<extra></extra>",
            )
        )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font=dict(family="monospace", color=WHITE, size=9),
        margin=dict(l=8, r=8, t=24, b=8),
        height=150,
        title=dict(text="GEX by Expiry", font=dict(size=9, color=MUTED), x=0.01),
        xaxis=dict(gridcolor=BORDER, color=MUTED, tickfont=dict(size=8)),
        yaxis=dict(
            gridcolor=BORDER,
            color=MUTED,
            tickformat=",.3s",
            zerolinecolor=BORDER,
            tickfont=dict(size=8),
        ),
        bargap=0.2,
    )
    return fig
