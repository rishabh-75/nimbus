"""
OptionsWallCalculator — Cross-expiry OI aggregation & wall identification
for NSE options chains.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class OptionsWallCalculator:
    """
    Aggregates options chain data across all expiry dates and identifies
    structural OI walls, PCR levels, IV skew, max pain, and key levels.
    """

    REQUIRED_COLS = {"Strike", "Expiry", "OptionType", "OpenInterest", "Volume", "IV", "LTP"}

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self.raw = df.copy()
        self.raw["Strike"] = pd.to_numeric(self.raw["Strike"], errors="coerce")
        self.raw["OpenInterest"] = pd.to_numeric(self.raw["OpenInterest"], errors="coerce").fillna(0)
        self.raw["Volume"] = pd.to_numeric(self.raw["Volume"], errors="coerce").fillna(0)
        self.raw["IV"] = pd.to_numeric(self.raw["IV"], errors="coerce").fillna(0)
        self.raw["LTP"] = pd.to_numeric(self.raw["LTP"], errors="coerce").fillna(0)
        self.raw["OptionType"] = self.raw["OptionType"].str.upper().str.strip()
        self.consolidated: Optional[pd.DataFrame] = None

    # ─── validation ───────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame):
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Options chain missing columns: {missing}")

    # ─── core aggregation ────────────────────────────────────────────────────

    def consolidate_walls(self) -> pd.DataFrame:
        """
        Aggregate OI / Volume / IV across ALL expiries per strike.
        Returns consolidated_walls DataFrame.
        """
        calls = self.raw[self.raw["OptionType"] == "CE"].copy()
        puts = self.raw[self.raw["OptionType"] == "PE"].copy()

        def agg(grp: pd.DataFrame, prefix: str) -> pd.DataFrame:
            out = grp.groupby("Strike").agg(
                OI=("OpenInterest", "sum"),
                Volume=("Volume", "sum"),
                IV=("IV", lambda x: np.average(x, weights=grp.loc[x.index, "OpenInterest"].clip(lower=1))),
                LTP=("LTP", "last"),
                Expiry_Count=("Expiry", "nunique"),
                Expiries=("Expiry", lambda x: ", ".join(sorted(x.astype(str).unique()))),
            ).reset_index()
            out.columns = ["Strike"] + [f"{prefix}_{c}" for c in ["OI", "Volume", "IV", "LTP", "Expiry_Count", "Expiries"]]
            return out

        c_agg = agg(calls, "Call")
        p_agg = agg(puts, "Put")

        merged = pd.merge(c_agg, p_agg, on="Strike", how="outer").fillna(0)

        # Keep string columns as string
        for col in ["Call_Expiries", "Put_Expiries"]:
            if col in merged.columns:
                merged[col] = merged[col].astype(str)

        merged["Total_OI"] = merged["Call_OI"] + merged["Put_OI"]
        merged["PCR_OI"] = np.where(merged["Call_OI"] > 0, merged["Put_OI"] / merged["Call_OI"], 0)

        total_call_oi = merged["Call_OI"].sum()
        total_put_oi = merged["Put_OI"].sum()
        merged["Call_Strength_%"] = np.where(total_call_oi > 0, merged["Call_OI"] / total_call_oi * 100, 0)
        merged["Put_Strength_%"] = np.where(total_put_oi > 0, merged["Put_OI"] / total_put_oi * 100, 0)

        # Unified expiry info
        merged["Expiry_Count"] = merged[["Call_Expiry_Count", "Put_Expiry_Count"]].max(axis=1).astype(int)
        merged["Expiries"] = merged.apply(
            lambda r: r["Call_Expiries"] if r["Call_Expiries"] != "0" else r["Put_Expiries"], axis=1
        )

        # Rename for compatibility
        merged = merged.rename(columns={
            "Call_OI": "Call_OI",
            "Put_OI": "Put_OI",
            "Call_Volume": "Call_Volume",
            "Put_Volume": "Put_Volume",
            "Call_IV": "Call_IV",
            "Put_IV": "Put_IV",
            "Call_LTP": "Call_LTP",
            "Put_LTP": "Put_LTP",
        })

        merged = merged.sort_values("Strike").reset_index(drop=True)
        self.consolidated = merged
        return merged

    # ─── wall identification ─────────────────────────────────────────────────

    def identify_walls(self, pct: float = 75) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (call_walls, put_walls) above the given OI percentile.
        """
        if self.consolidated is None:
            self.consolidate_walls()
        df = self.consolidated

        call_threshold = np.percentile(df["Call_OI"], pct)
        put_threshold = np.percentile(df["Put_OI"], pct)

        call_walls = df[df["Call_OI"] >= call_threshold].copy()
        put_walls = df[df["Put_OI"] >= put_threshold].copy()

        call_walls["Wall_Type"] = "Call Wall"
        put_walls["Wall_Type"] = "Put Wall"

        return call_walls, put_walls

    # ─── PCR analysis ────────────────────────────────────────────────────────

    def analyze_pcr(self) -> dict:
        """
        Returns PCR OI, PCR Volume, and sentiment label.
        """
        if self.consolidated is None:
            self.consolidate_walls()
        df = self.consolidated

        total_call_oi = df["Call_OI"].sum()
        total_put_oi = df["Put_OI"].sum()
        total_call_vol = df["Call_Volume"].sum()
        total_put_vol = df["Put_Volume"].sum()

        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0

        if pcr_oi < 0.7:
            sentiment = "Bullish"
        elif pcr_oi < 1.0:
            sentiment = "Mildly Bullish"
        elif pcr_oi < 1.3:
            sentiment = "Mildly Bearish"
        else:
            sentiment = "Bearish"

        return {
            "pcr_oi": round(pcr_oi, 3),
            "pcr_volume": round(pcr_vol, 3),
            "sentiment": sentiment,
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
        }

    # ─── IV skew ─────────────────────────────────────────────────────────────

    def analyze_iv_skew(self) -> dict:
        """
        Weighted average IV for calls vs puts and skew interpretation.
        """
        if self.consolidated is None:
            self.consolidate_walls()
        df = self.consolidated

        # OI-weighted average IV
        call_iv = np.average(df["Call_IV"], weights=df["Call_OI"].clip(lower=1)) if df["Call_OI"].sum() > 0 else 0
        put_iv = np.average(df["Put_IV"], weights=df["Put_OI"].clip(lower=1)) if df["Put_OI"].sum() > 0 else 0

        skew = put_iv - call_iv  # positive = put skew (bearish hedging demand)

        if skew > 3:
            interpretation = "Put IV premium — protective hedging, mild upside bias"
            direction = "bullish"
        elif skew < -3:
            interpretation = "Call IV premium — call buying / distribution pressure"
            direction = "bearish"
        else:
            interpretation = "Neutral IV skew — balanced expectations"
            direction = "neutral"

        return {
            "call_iv": round(call_iv, 2),
            "put_iv": round(put_iv, 2),
            "skew": round(skew, 2),
            "interpretation": interpretation,
            "direction": direction,
        }

    # ─── max pain ─────────────────────────────────────────────────────────────

    def calculate_max_pain(self) -> float:
        """
        Max pain = strike where total option writers lose the least (OI-weighted).
        """
        if self.consolidated is None:
            self.consolidate_walls()
        df = self.consolidated
        strikes = df["Strike"].values

        pain = {}
        for s in strikes:
            # ITM calls at this expiry: strikes above s — writers pay (s - K) * OI
            call_loss = ((strikes - s).clip(min=0) * df["Call_OI"].values).sum()
            # ITM puts: strikes below s
            put_loss = ((s - strikes).clip(min=0) * df["Put_OI"].values).sum()
            pain[s] = call_loss + put_loss

        max_pain_strike = min(pain, key=pain.get)
        return float(max_pain_strike)

    # ─── key levels ──────────────────────────────────────────────────────────

    def identify_key_levels(self, cmp: Optional[float] = None, pct: float = 75) -> dict:
        """
        Identify support, resistance, and consolidation zones.
        """
        call_walls, put_walls = self.identify_walls(pct=pct)

        # Highest put wall = primary support (floor)
        primary_support = put_walls["Strike"].max() if not put_walls.empty else None
        # Highest call wall (by OI) = primary resistance (ceiling)
        if not call_walls.empty:
            primary_resistance = call_walls.loc[call_walls["Call_OI"].idxmax(), "Strike"]
        else:
            primary_resistance = None

        # Consolidation zone: range between top put & top call walls
        consolidation_low = primary_support
        consolidation_high = primary_resistance

        levels = {
            "primary_support": primary_support,
            "primary_resistance": primary_resistance,
            "consolidation_low": consolidation_low,
            "consolidation_high": consolidation_high,
            "call_walls": call_walls["Strike"].tolist(),
            "put_walls": put_walls["Strike"].tolist(),
        }

        if cmp is not None:
            levels["cmp"] = cmp
            if primary_support and cmp > primary_support:
                levels["cmp_vs_support"] = "above_support"
            elif primary_support:
                levels["cmp_vs_support"] = "below_support"
            if primary_resistance and cmp < primary_resistance:
                levels["cmp_vs_resistance"] = "below_resistance"
            elif primary_resistance:
                levels["cmp_vs_resistance"] = "above_resistance"

        return levels
