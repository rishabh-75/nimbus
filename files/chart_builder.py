"""
ChartBuilder — All Plotly chart functions for the NSE Trading Signal Dashboard.
Dark theme throughout. Uses add_shape + add_annotation instead of add_vline
to avoid Plotly xref enumeration bugs across versions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

# ─── theme constants ──────────────────────────────────────────────────────────

BG       = "#0d0d1a"
CARD_BG  = "#14142a"
GRID     = "#1e1e2e"
GREEN    = "#00e676"
RED      = "#ff5252"
GOLD     = "#ffd740"
CYAN     = "#00e5ff"
ORANGE   = "#ff6d00"
PURPLE   = "#ce93d8"
WHITE    = "#e0e0e0"
MUTED    = "#9e9e9e"


# ─── private helpers ──────────────────────────────────────────────────────────

def _base_layout(**kwargs) -> dict:
    layout = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=WHITE, family="Inter, Segoe UI, sans-serif"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=WHITE),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=WHITE),
        legend=dict(
            bgcolor=CARD_BG, bordercolor=GRID, borderwidth=1,
            font=dict(color=WHITE),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=GRID, font=dict(color=WHITE)),
    )
    layout.update(kwargs)
    return layout


def _vline(fig, x, color, dash="dash", width=2.0, label="", label_y=0.97):
    """
    Vertical line via add_shape (xref='x', yref='paper') + add_annotation.
    Avoids the add_vline xref='paper domain' ValueError in some Plotly builds.
    """
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=x, x1=x,
        y0=0.0, y1=1.0,
        line=dict(color=color, width=width, dash=dash),
        layer="above",
    )
    if label:
        fig.add_annotation(
            xref="x", yref="paper",
            x=x, y=label_y,
            text=label,
            showarrow=False,
            font=dict(color=color, size=10),
            bgcolor=BG,
            borderpad=2,
            xanchor="center",
        )


def _hline(fig, y, color, dash="dot", width=1.5, label=""):
    """Horizontal reference line via add_shape."""
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0.0, x1=1.0,
        y0=y, y1=y,
        line=dict(color=color, width=width, dash=dash),
        layer="above",
    )
    if label:
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.02, y=y,
            text=label,
            showarrow=False,
            font=dict(color=color, size=10),
            xanchor="left",
        )


def _empty_fig(msg, height=280):
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=MUTED, size=14),
    )
    fig.update_layout(**_base_layout(height=height))
    return fig


# ─── 1. Main OI Wall Chart ────────────────────────────────────────────────────

def oi_wall_chart(
    walls_df: pd.DataFrame,
    call_walls: pd.DataFrame,
    put_walls: pd.DataFrame,
    cmp: Optional[float] = None,
    max_pain: Optional[float] = None,
    title: str = "Options Wall — Open Interest by Strike",
) -> go.Figure:

    fig = go.Figure()

    # Call OI bars (green, upward)
    fig.add_trace(go.Bar(
        x=walls_df["Strike"],
        y=walls_df["Call_OI"],
        name="Call OI",
        marker_color=GREEN,
        opacity=0.85,
        customdata=walls_df["PCR_OI"],
        hovertemplate=(
            "<b>Strike %{x}</b><br>"
            "Call OI: %{y:,.0f}<br>"
            "PCR: %{customdata:.3f}<extra></extra>"
        ),
    ))

    # Put OI bars (red, inverted)
    fig.add_trace(go.Bar(
        x=walls_df["Strike"],
        y=-walls_df["Put_OI"],
        name="Put OI",
        marker_color=RED,
        opacity=0.85,
        customdata=walls_df["Put_OI"],
        hovertemplate=(
            "<b>Strike %{x}</b><br>"
            "Put OI: %{customdata:,.0f}<extra></extra>"
        ),
    ))

    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=16, color=WHITE)),
        barmode="overlay",
        xaxis_title="Strike Price",
        yaxis=dict(
            title="Open Interest",
            gridcolor=GRID,
            zerolinecolor=WHITE,
            zerolinewidth=1,
            tickformat=",",
            color=WHITE,
        ),
        height=420,
    ))

    # Wall lines drawn AFTER layout is set so xref='x' is valid
    for _, row in call_walls.iterrows():
        _vline(fig, row["Strike"], GREEN, dash="dash", width=2,
               label=f"CW {row['Strike']:.0f}", label_y=0.97)

    for _, row in put_walls.iterrows():
        _vline(fig, row["Strike"], RED, dash="dash", width=2,
               label=f"PW {row['Strike']:.0f}", label_y=0.03)

    if cmp is not None:
        _vline(fig, cmp, GOLD, dash="solid", width=2.5,
               label=f"CMP {cmp:.0f}", label_y=0.88)

    if max_pain is not None:
        _vline(fig, max_pain, CYAN, dash="dot", width=2,
               label=f"MaxPain {max_pain:.0f}", label_y=0.12)

    return fig


# ─── 2. PCR by Strike ────────────────────────────────────────────────────────

def pcr_by_strike_chart(walls_df: pd.DataFrame) -> go.Figure:
    df = walls_df.copy()
    colors = df["PCR_OI"].apply(
        lambda v: GREEN if v < 0.7 else (ORANGE if v < 1.3 else RED)
    )

    fig = go.Figure(go.Bar(
        x=df["Strike"],
        y=df["PCR_OI"],
        marker_color=colors,
        hovertemplate="<b>Strike %{x}</b><br>PCR OI: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title="PCR OI by Strike",
        xaxis_title="Strike",
        yaxis_title="Put/Call OI Ratio",
        height=300,
    ))

    _hline(fig, 0.7, GREEN, label="0.7 Bullish")
    _hline(fig, 1.0, WHITE, label="1.0 Neutral")
    _hline(fig, 1.3, RED,   label="1.3 Bearish")

    return fig


# ─── 3. IV Smile ─────────────────────────────────────────────────────────────

def iv_smile_chart(walls_df: pd.DataFrame) -> go.Figure:
    df = walls_df.sort_values("Strike")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Strike"], y=df["Call_IV"],
        name="Call IV",
        line=dict(color=GREEN, width=2),
        mode="lines+markers",
        hovertemplate="<b>Strike %{x}</b><br>Call IV: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Strike"], y=df["Put_IV"],
        name="Put IV",
        line=dict(color=RED, width=2),
        mode="lines+markers",
        fill="tonexty",
        fillcolor="rgba(255,82,82,0.08)",
        hovertemplate="<b>Strike %{x}</b><br>Put IV: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title="IV Smile",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        height=300,
    ))
    return fig


# ─── 4. Total OI Area Chart ──────────────────────────────────────────────────

def total_oi_area_chart(walls_df: pd.DataFrame) -> go.Figure:
    df = walls_df.sort_values("Strike")

    fig = go.Figure(go.Scatter(
        x=df["Strike"], y=df["Total_OI"],
        fill="tozeroy",
        fillcolor="rgba(0,230,118,0.12)",
        line=dict(color=GREEN, width=2),
        hovertemplate="<b>Strike %{x}</b><br>Total OI: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="Total OI by Strike",
        xaxis_title="Strike",
        yaxis_title="Total Open Interest",
        height=280,
    ))
    return fig


# ─── 5. Wall Strength % Grouped Bar ──────────────────────────────────────────

def wall_strength_chart(walls_df: pd.DataFrame) -> go.Figure:
    df = walls_df.sort_values("Strike")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Strike"], y=df["Call_Strength_%"],
        name="Call Strength %",
        marker_color=GREEN, opacity=0.85,
        hovertemplate="Strike %{x}<br>Call Strength: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["Strike"], y=df["Put_Strength_%"],
        name="Put Strength %",
        marker_color=RED, opacity=0.85,
        hovertemplate="Strike %{x}<br>Put Strength: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title="Wall Strength % Distribution",
        barmode="group",
        xaxis_title="Strike",
        yaxis_title="Strength (%)",
        height=280,
    ))
    return fig


# ─── 6. OI Wall + Deal Overlay ───────────────────────────────────────────────

def oi_deal_overlay_chart(
    walls_df: pd.DataFrame,
    call_walls: pd.DataFrame,
    put_walls: pd.DataFrame,
    matched_df: Optional[pd.DataFrame] = None,
    zones_df: Optional[pd.DataFrame] = None,
    cmp: Optional[float] = None,
    max_pain: Optional[float] = None,
) -> go.Figure:

    fig = oi_wall_chart(
        walls_df, call_walls, put_walls, cmp, max_pain,
        title="OI Wall + Institutional Deal Overlay",
    )

    if matched_df is not None and not matched_df.empty:
        buys  = matched_df[matched_df["Buy/Sell"] == "BUY"].copy()
        sells = matched_df[matched_df["Buy/Sell"] == "SELL"].copy()
        max_qty = matched_df["Quantity Traded"].max() or 1

        def _sz(df_):
            return (df_["Quantity Traded"] / max_qty * 25 + 8).clip(upper=40)

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["Trade Price/Wght. Avg. Price"],
                y=[0] * len(buys),
                mode="markers",
                marker=dict(
                    symbol="triangle-up", size=_sz(buys),
                    color=GREEN, opacity=0.9,
                    line=dict(color=WHITE, width=0.5),
                ),
                name="BUY Deals",
                text=buys["Client Name"],
                customdata=buys["Quantity Traded"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "BUY @ %{x:.2f}<br>"
                    "Qty: %{customdata:,.0f}<extra></extra>"
                ),
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["Trade Price/Wght. Avg. Price"],
                y=[0] * len(sells),
                mode="markers",
                marker=dict(
                    symbol="triangle-down", size=_sz(sells),
                    color=RED, opacity=0.9,
                    line=dict(color=WHITE, width=0.5),
                ),
                name="SELL Deals",
                text=sells["Client Name"],
                customdata=sells["Quantity Traded"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "SELL @ %{x:.2f}<br>"
                    "Qty: %{customdata:,.0f}<extra></extra>"
                ),
            ))

    if zones_df is not None and not zones_df.empty:
        for _, z in zones_df.iterrows():
            color  = GOLD if z["Zone_Type"] == "ACCUMULATION" else ORANGE
            zlabel = "ACC" if z["Zone_Type"] == "ACCUMULATION" else "DIST"
            _vline(fig, z["Strike"], color, dash="dot", width=1.5,
                   label=f"{zlabel} {z['Strike']:.0f}", label_y=0.55)

    return fig


# ─── 7. Signal Score Scatter ─────────────────────────────────────────────────

def signal_score_scatter(matched_df: pd.DataFrame) -> go.Figure:
    if matched_df is None or matched_df.empty:
        return _empty_fig("No matched deals to display")

    df = matched_df.copy()
    df["_color"] = df["Buy/Sell"].map({"BUY": GREEN, "SELL": RED})
    df["_label"] = df.apply(
        lambda r: (r["Client Name"][:15] + "…") if r["Score"] >= 60 else "", axis=1
    )
    max_qty = df["Quantity Traded"].max() or 1

    fig = go.Figure(go.Scatter(
        x=df["Trade Price/Wght. Avg. Price"],
        y=df["Score"],
        mode="markers+text",
        marker=dict(
            size=(df["Quantity Traded"] / max_qty * 30 + 8).clip(upper=50),
            color=df["_color"],
            opacity=0.85,
            line=dict(color=WHITE, width=0.5),
        ),
        text=df["_label"],
        textposition="top center",
        textfont=dict(color=WHITE, size=9),
        customdata=df[["Client Name", "Buy/Sell", "Quantity Traded", "Entity"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]} @ %{x:.2f}<br>"
            "Score: %{y}<br>"
            "Qty: %{customdata[2]:,.0f}<br>"
            "Entity: %{customdata[3]}<extra></extra>"
        ),
    ))

    fig.update_layout(**_base_layout(
        title="Signal Score — Deal Price vs Score",
        xaxis_title="Deal Price",
        yaxis=dict(
            title="Signal Score (0–100)",
            range=[0, 108],
            gridcolor=GRID, color=WHITE,
        ),
        height=350,
    ))

    _hline(fig, 60, GOLD, label="High Conviction (60)")
    _hline(fig, 80, RED,  label="Very High (80)")

    return fig


# ─── 8. Net Buy/Sell per Wall Level ──────────────────────────────────────────

def net_qty_bar_chart(aggregated_df: pd.DataFrame) -> go.Figure:
    if aggregated_df is None or aggregated_df.empty:
        return _empty_fig("No aggregated deal data")

    df = aggregated_df.sort_values("Net_Qty")
    qty_cr = df["Net_Qty"] / 1e7
    colors = [GREEN if v >= 0 else RED for v in qty_cr]

    fig = go.Figure(go.Bar(
        y=df["Strike"].astype(str),
        x=qty_cr,
        orientation="h",
        marker_color=colors,
        text=[f"₹{abs(v):.2f}Cr" for v in qty_cr],
        textposition="outside",
        textfont=dict(color=WHITE, size=10),
        hovertemplate="Strike %{y}<br>Net: %{x:.2f} Cr<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title="Net Institutional Buy/Sell per Wall Level (₹Cr)",
        xaxis_title="Net Qty (Cr)",
        yaxis_title="Strike",
        height=350,
    ))

    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=0, x1=0, y0=0.0, y1=1.0,
        line=dict(color=WHITE, width=1),
    )

    return fig


# ─── 9. Composite Signal Gauge ───────────────────────────────────────────────

def composite_gauge(score: float, label: str) -> go.Figure:
    color      = GREEN if score > 30 else (RED if score < -30 else GOLD)
    normalized = (score + 100) / 2   # remap –100..+100 → 0..100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=normalized,
        delta={"reference": 50, "valueformat": ".1f"},
        number={"font": {"size": 32, "color": WHITE}},
        title={"text": label, "font": {"size": 14, "color": WHITE}},
        gauge=dict(
            axis=dict(
                range=[0, 100], tickwidth=1,
                tickcolor=WHITE, tickfont=dict(color=WHITE),
            ),
            bar=dict(color=color, thickness=0.3),
            bgcolor=CARD_BG,
            borderwidth=2,
            bordercolor=GRID,
            steps=[
                {"range": [0,  33], "color": "rgba(255,82,82,0.25)"},
                {"range": [33, 67], "color": "rgba(255,215,64,0.15)"},
                {"range": [67,100], "color": "rgba(0,230,118,0.25)"},
            ],
            threshold=dict(
                line=dict(color=WHITE, width=3),
                thickness=0.8,
                value=normalized,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        margin=dict(l=20, r=20, t=60, b=20),
        height=260,
    )
    return fig


# ─── 10. OI Change Chart (Historical) ───────────────────────────────────────

def oi_change_chart(today_df: pd.DataFrame, yesterday_df: pd.DataFrame) -> go.Figure:
    merged = today_df[["Strike", "Total_OI"]].merge(
        yesterday_df[["Strike", "Total_OI"]].rename(columns={"Total_OI": "Prior_OI"}),
        on="Strike", how="inner",
    )
    merged["OI_Change"] = merged["Total_OI"] - merged["Prior_OI"]
    colors = [GREEN if v >= 0 else RED for v in merged["OI_Change"]]

    fig = go.Figure(go.Bar(
        x=merged["Strike"],
        y=merged["OI_Change"],
        marker_color=colors,
        hovertemplate="Strike %{x}<br>OI Δ: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="OI Change (Today vs Yesterday)",
        xaxis_title="Strike",
        yaxis_title="OI Delta",
        height=300,
    ))
    _hline(fig, 0, WHITE, dash="solid", width=1)
    return fig


# ─── 11. PCR Trend Line ──────────────────────────────────────────────────────

def pcr_trend_chart(pcr_series: pd.Series, dates: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=dates, y=pcr_series,
        mode="lines+markers",
        line=dict(color=PURPLE, width=2),
        marker=dict(color=PURPLE, size=6),
        hovertemplate="%{x}<br>PCR: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="PCR OI Trend",
        xaxis_title="Date",
        yaxis_title="PCR OI",
        height=280,
    ))
    _hline(fig, 0.7, GREEN, label="0.7 Bullish")
    _hline(fig, 1.3, RED,   label="1.3 Bearish")
    return fig


# ─── 12. Max Pain Migration ──────────────────────────────────────────────────

def max_pain_migration_chart(max_pains: list, dates: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=dates, y=max_pains,
        mode="lines+markers",
        line=dict(color=CYAN, width=2),
        marker=dict(color=CYAN, size=7, symbol="diamond"),
        hovertemplate="%{x}<br>Max Pain: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="Max Pain Migration",
        xaxis_title="Date",
        yaxis_title="Max Pain Strike",
        height=280,
    ))
    return fig


# ─── 13. Trade Setup Visualiser ──────────────────────────────────────────────

def trade_setup_chart(setup, walls_df: pd.DataFrame, cmp: Optional[float] = None) -> go.Figure:
    """
    Horizontal price-level chart showing entry zone, targets, and stop loss
    on top of the OI distribution for context.
    """
    fig = go.Figure()

    # OI context (faint bars)
    fig.add_trace(go.Bar(
        x=walls_df["Call_OI"], y=walls_df["Strike"],
        orientation="h", name="Call OI",
        marker_color=GREEN, opacity=0.18,
        hovertemplate="Strike %{y}<br>Call OI: %{x:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=-walls_df["Put_OI"], y=walls_df["Strike"],
        orientation="h", name="Put OI",
        marker_color=RED, opacity=0.18,
        customdata=walls_df["Put_OI"],
        hovertemplate="Strike %{y}<br>Put OI: %{customdata:,.0f}<extra></extra>",
    ))

    color = GREEN if setup.direction == "LONG" else RED

    def _hband(y_val, label, clr, dash="solid", width=2):
        fig.add_shape(
            type="line", xref="paper", yref="y",
            x0=0.0, x1=1.0, y0=y_val, y1=y_val,
            line=dict(color=clr, width=width, dash=dash),
        )
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.02, y=y_val,
            text=label, showarrow=False,
            font=dict(color=clr, size=11),
            xanchor="left",
        )

    _hband(setup.stop_loss, f"SL  {setup.stop_loss:.2f}", RED, dash="dot", width=2)
    fig.add_shape(
        type="rect", xref="paper", yref="y",
        x0=0, x1=1,
        y0=setup.entry_low, y1=setup.entry_high,
        fillcolor=f"rgba({'0,230,118' if setup.direction=='LONG' else '255,82,82'},0.15)",
        line=dict(width=0),
    )
    fig.add_annotation(
        xref="paper", yref="y",
        x=0.5, y=(setup.entry_low + setup.entry_high) / 2,
        text=f"ENTRY {setup.entry_low:.2f}\u2013{setup.entry_high:.2f}",
        showarrow=False,
        font=dict(color=color, size=11),
        xanchor="center",
    )
    _hband(setup.target1, f"T1  {setup.target1:.2f}", GOLD,  dash="dash", width=2)
    _hband(setup.target2, f"T2  {setup.target2:.2f}", GREEN, dash="dash", width=1.5)
    if cmp is not None:
        _hband(cmp, f"CMP {cmp:.2f}", WHITE, dash="solid", width=1.5)

    direction_arrow = "\u25b2 LONG" if setup.direction == "LONG" else "\u25bc SHORT"
    fig.update_layout(**_base_layout(
        title=f"{setup.symbol} \u2014 {direction_arrow} | {setup.strategy} | R:R {setup.rr_ratio:.1f}x",
        xaxis=dict(title="Open Interest", gridcolor=GRID, color=WHITE),
        yaxis=dict(title="Strike / Price", gridcolor=GRID, color=WHITE, tickformat=".0f"),
        barmode="overlay",
        height=400,
    ))
    return fig


# ─── 14. Conviction Bar Chart ────────────────────────────────────────────────

def conviction_bar_chart(setups_df: pd.DataFrame) -> go.Figure:
    if setups_df is None or setups_df.empty:
        return _empty_fig("No trade setups generated")

    df = setups_df.sort_values("Conviction", ascending=True)
    colors = [GREEN if d == "LONG" else RED for d in df["Direction"]]
    labels = df.apply(lambda r: f"#{r['ID']} {r['Direction']} {r['Strategy']}", axis=1)

    fig = go.Figure(go.Bar(
        y=labels, x=df["Conviction"],
        orientation="h",
        marker_color=colors,
        text=df["Conviction"].astype(str),
        textposition="outside",
        textfont=dict(color=WHITE, size=11),
        customdata=df[["R:R", "Move %", "Timeframe"]].values,
        hovertemplate=(
            "%{y}<br>Conviction: %{x}<br>"
            "R:R: %{customdata[0]}<br>"
            "Move: %{customdata[1]}<br>"
            "TF: %{customdata[2]}<extra></extra>"
        ),
    ))

    fig.update_layout(**_base_layout(
        title="Trade Setups \u2014 Conviction Ranking",
        xaxis=dict(title="Conviction Score (0\u2013100)", range=[0, 115], gridcolor=GRID, color=WHITE),
        yaxis=dict(gridcolor=GRID, color=WHITE),
        height=max(280, len(df) * 52 + 60),
    ))
    _hline(fig, 60, GOLD,  label="High (60)")
    _hline(fig, 80, GREEN, label="Very High (80)")
    return fig


# ─── 15. Risk/Reward Scatter ─────────────────────────────────────────────────

def rr_scatter_chart(setups_df: pd.DataFrame) -> go.Figure:
    if setups_df is None or setups_df.empty:
        return _empty_fig("No trade setups")

    df = setups_df.copy()
    colors = [GREEN if d == "LONG" else RED for d in df["Direction"]]
    risk_vals = df["Risk %"].str.replace("%", "").astype(float)

    fig = go.Figure(go.Scatter(
        x=risk_vals,
        y=df["R:R"],
        mode="markers+text",
        marker=dict(
            size=df["Conviction"] / 4 + 8,
            color=colors, opacity=0.85,
            line=dict(color=WHITE, width=0.5),
        ),
        text=df.apply(lambda r: f"#{r['ID']} {r['Strategy'][:10]}", axis=1),
        textposition="top center",
        textfont=dict(color=WHITE, size=9),
        customdata=df[["Direction", "Conviction", "Timeframe"]].values,
        hovertemplate=(
            "%{text}<br>%{customdata[0]}<br>"
            "Risk: %{x:.2f}%<br>R:R: %{y:.2f}x<br>"
            "Conviction: %{customdata[1]}<br>TF: %{customdata[2]}<extra></extra>"
        ),
    ))

    _hline(fig, 2.0, GOLD, label="R:R 2.0x")
    fig.update_layout(**_base_layout(
        title="Risk % vs R:R Ratio  (bubble size = conviction)",
        xaxis=dict(title="Risk % (to stop loss)", gridcolor=GRID, color=WHITE),
        yaxis=dict(title="Risk:Reward Ratio",    gridcolor=GRID, color=WHITE),
        height=350,
    ))
    return fig


# ─── 13. Trade Setup Visualiser ──────────────────────────────────────────────

def trade_setup_chart(setup, walls_df: pd.DataFrame, cmp: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=walls_df["Call_OI"], y=walls_df["Strike"],
        orientation="h", name="Call OI",
        marker_color=GREEN, opacity=0.18,
        hovertemplate="Strike %{y}<br>Call OI: %{x:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=-walls_df["Put_OI"], y=walls_df["Strike"],
        orientation="h", name="Put OI",
        marker_color=RED, opacity=0.18,
        customdata=walls_df["Put_OI"],
        hovertemplate="Strike %{y}<br>Put OI: %{customdata:,.0f}<extra></extra>",
    ))
    color = GREEN if setup.direction == "LONG" else RED

    def _hband(y_val, label, clr, dash="solid", width=2):
        fig.add_shape(type="line", xref="paper", yref="y",
                      x0=0.0, x1=1.0, y0=y_val, y1=y_val,
                      line=dict(color=clr, width=width, dash=dash))
        fig.add_annotation(xref="paper", yref="y", x=1.02, y=y_val,
                           text=label, showarrow=False,
                           font=dict(color=clr, size=11), xanchor="left")

    _hband(setup.stop_loss, f"SL  {setup.stop_loss:.2f}", RED, dash="dot", width=2)
    fig.add_shape(type="rect", xref="paper", yref="y",
                  x0=0, x1=1, y0=setup.entry_low, y1=setup.entry_high,
                  fillcolor=f"rgba({'0,230,118' if setup.direction=='LONG' else '255,82,82'},0.15)",
                  line=dict(width=0))
    fig.add_annotation(xref="paper", yref="y",
                       x=0.5, y=(setup.entry_low + setup.entry_high) / 2,
                       text=f"ENTRY {setup.entry_low:.2f}-{setup.entry_high:.2f}",
                       showarrow=False, font=dict(color=color, size=11), xanchor="center")
    _hband(setup.target1, f"T1  {setup.target1:.2f}", GOLD,  dash="dash", width=2)
    _hband(setup.target2, f"T2  {setup.target2:.2f}", GREEN, dash="dash", width=1.5)
    if cmp is not None:
        _hband(cmp, f"CMP {cmp:.2f}", WHITE, dash="solid", width=1.5)

    arrow = "LONG" if setup.direction == "LONG" else "SHORT"
    fig.update_layout(**_base_layout(
        title=f"{setup.symbol} | {arrow} | {setup.strategy} | R:R {setup.rr_ratio:.1f}x",
        xaxis=dict(title="Open Interest", gridcolor=GRID, color=WHITE),
        yaxis=dict(title="Price Level", gridcolor=GRID, color=WHITE, tickformat=".0f"),
        barmode="overlay", height=400,
    ))
    return fig


# ─── 14. Conviction Bar Chart ────────────────────────────────────────────────

def conviction_bar_chart(setups_df: pd.DataFrame) -> go.Figure:
    if setups_df is None or setups_df.empty:
        return _empty_fig("No trade setups generated")
    df = setups_df.sort_values("Conviction", ascending=True)
    colors = [GREEN if d == "LONG" else RED for d in df["Direction"]]
    labels = df.apply(lambda r: f"#{r['ID']} {r['Direction']} {r['Strategy']}", axis=1)
    fig = go.Figure(go.Bar(
        y=labels, x=df["Conviction"], orientation="h",
        marker_color=colors,
        text=df["Conviction"].astype(str),
        textposition="outside", textfont=dict(color=WHITE, size=11),
        customdata=df[["R:R", "Move %", "Timeframe"]].values,
        hovertemplate="%{y}<br>Conviction: %{x}<br>R:R: %{customdata[0]}<br>Move: %{customdata[1]}<br>TF: %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="Trade Setups - Conviction Ranking",
        xaxis=dict(title="Conviction Score (0-100)", range=[0, 115], gridcolor=GRID, color=WHITE),
        yaxis=dict(gridcolor=GRID, color=WHITE),
        height=max(280, len(df) * 52 + 60),
    ))
    _hline(fig, 60, GOLD,  label="High (60)")
    _hline(fig, 80, GREEN, label="Very High (80)")
    return fig


# ─── 15. Risk/Reward Scatter ─────────────────────────────────────────────────

def rr_scatter_chart(setups_df: pd.DataFrame) -> go.Figure:
    if setups_df is None or setups_df.empty:
        return _empty_fig("No trade setups")
    df = setups_df.copy()
    colors = [GREEN if d == "LONG" else RED for d in df["Direction"]]
    risk_vals = df["Risk %"].str.replace("%", "").astype(float)
    fig = go.Figure(go.Scatter(
        x=risk_vals, y=df["R:R"],
        mode="markers+text",
        marker=dict(size=df["Conviction"] / 4 + 8, color=colors, opacity=0.85,
                    line=dict(color=WHITE, width=0.5)),
        text=df.apply(lambda r: f"#{r['ID']} {r['Strategy'][:10]}", axis=1),
        textposition="top center", textfont=dict(color=WHITE, size=9),
        customdata=df[["Direction", "Conviction", "Timeframe"]].values,
        hovertemplate="%{text}<br>%{customdata[0]}<br>Risk: %{x:.2f}%<br>R:R: %{y:.2f}x<br>Conviction: %{customdata[1]}<br>TF: %{customdata[2]}<extra></extra>",
    ))
    _hline(fig, 2.0, GOLD, label="R:R 2.0x")
    fig.update_layout(**_base_layout(
        title="Risk % vs R:R Ratio  (bubble = conviction)",
        xaxis=dict(title="Risk % (to stop loss)", gridcolor=GRID, color=WHITE),
        yaxis=dict(title="Risk:Reward Ratio", gridcolor=GRID, color=WHITE),
        height=350,
    ))
    return fig
