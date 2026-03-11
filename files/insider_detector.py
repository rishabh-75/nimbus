"""
InsiderWallDetector — Matches NSE bulk/block deals to options wall strikes
and scores them for institutional conviction signals.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Optional


# ─── entity classification keywords ──────────────────────────────────────────

FII_KEYWORDS = [
    "fii",
    "fpi",
    "foreign",
    "overseas",
    "mauritius",
    "singapore",
    "cayman",
    "llp",
    "inc",
    "sarl",
    "gmbh",
    "limited liability",
    "pte ltd",
    "bv",
    "nv",
    "global",
    "international",
    "capital partners",
    "investment fund",
]

DII_KEYWORDS = [
    "dii",
    "mutual fund",
    "mf",
    "amc",
    "asset management",
    "hdfc mf",
    "sbi mf",
    "lic",
    "nippon",
    "icici pru",
    "axis mf",
    "kotak mf",
    "insurance",
    "pension",
    "epf",
    "nps",
    "uti",
    "birla",
]

PROMOTER_KEYWORDS = [
    "promoter",
    "director",
    "md",
    "ceo",
    "cfo",
    "chairman",
    "managing",
    "whole time",
    "wholetime",
    "key managerial",
    "kmp",
    "insider",
    "related party",
    "group company",
    "holding company",
    "subsidiary",
]

BROKER_KEYWORDS = [
    "securities",
    "broking",
    "broker",
    "zerodha",
    "icici securities",
    "hdfc securities",
    "iifl",
    "motilal",
    "kotak securities",
    "sharekhan",
    "angel",
    "5paisa",
    "upstox",
    "groww",
]


class InsiderWallDetector:
    """
    Detects institutional / insider activity at options wall levels.
    """

    DEAL_REQUIRED_COLS = {
        "Date",
        "Symbol",
        "Client Name",
        "Buy/Sell",
        "Quantity Traded",
        "Trade Price/Wght. Avg. Price",
    }

    def __init__(
        self,
        consolidated_walls: pd.DataFrame,
        deals_df: pd.DataFrame,
        proximity_pct: float = 1.5,
    ):
        self.walls = consolidated_walls.copy()
        self.deals = self._prepare_deals(deals_df)
        self.proximity_pct = proximity_pct
        self.matched: Optional[pd.DataFrame] = None
        self.zones: Optional[pd.DataFrame] = None

    # ─── preparation ─────────────────────────────────────────────────────────

    def _prepare_deals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalise column names
        df.columns = [c.strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["Quantity Traded"] = pd.to_numeric(
            df["Quantity Traded"].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
        df["Trade Price/Wght. Avg. Price"] = pd.to_numeric(
            df["Trade Price/Wght. Avg. Price"].astype(str).str.replace(",", ""),
            errors="coerce",
        ).fillna(0)
        df["Client Name"] = df["Client Name"].fillna("Unknown").str.strip()
        df["Buy/Sell"] = df["Buy/Sell"].str.upper().str.strip()
        df["Entity"] = df["Client Name"].apply(self._classify_entity)
        return df

    # ─── entity classifier ───────────────────────────────────────────────────

    @staticmethod
    def _classify_entity(name: str) -> str:
        n = name.lower()
        for kw in PROMOTER_KEYWORDS:
            if kw in n:
                return "Promoter/Insider"
        for kw in DII_KEYWORDS:
            if kw in n:
                return "DII/MF"
        for kw in FII_KEYWORDS:
            if kw in n:
                return "FII/FPI"
        for kw in BROKER_KEYWORDS:
            if kw in n:
                return "Institutional Broker"
        return "Retail/Unknown"

    # ─── proximity matching ──────────────────────────────────────────────────

    def match_deals_to_walls(self) -> pd.DataFrame:
        """
        For each deal, find the nearest wall strike within proximity_pct.
        Score each match on 5 axes.
        """
        walls = self.walls
        deals = self.deals.copy()

        records = []
        for _, deal in deals.iterrows():
            price = deal["Trade Price/Wght. Avg. Price"]
            if price <= 0:
                continue
            # Find nearest strike
            diffs = (walls["Strike"] - price).abs()
            idx = diffs.idxmin()
            nearest_strike = walls.loc[idx, "Strike"]
            pct_diff = abs(nearest_strike - price) / price * 100

            if pct_diff > self.proximity_pct:
                continue  # outside proximity tolerance

            row = deal.to_dict()
            row["Nearest_Strike"] = nearest_strike
            row["Proximity_%"] = round(pct_diff, 3)
            row["Wall_Call_OI"] = walls.loc[idx, "Call_OI"]
            row["Wall_Put_OI"] = walls.loc[idx, "Put_OI"]
            row["Wall_PCR"] = walls.loc[idx, "PCR_OI"]
            row["Wall_Call_IV"] = walls.loc[idx, "Call_IV"]
            row["Wall_Put_IV"] = walls.loc[idx, "Put_IV"]

            # Determine wall type hit
            if row["Wall_Put_OI"] > row["Wall_Call_OI"]:
                row["Wall_Type_Hit"] = "Put Wall"
            elif row["Wall_Call_OI"] > row["Wall_Put_OI"]:
                row["Wall_Type_Hit"] = "Call Wall"
            else:
                row["Wall_Type_Hit"] = "Neutral"

            row["Score"] = self._score_deal(row)
            row["Signal_Strength"] = self._signal_strength(row["Score"])
            row["Interpretation"] = self._interpret_deal(row)
            records.append(row)

        MATCHED_COLS = [
            "Date",
            "Symbol",
            "Client Name",
            "Buy/Sell",
            "Quantity Traded",
            "Trade Price/Wght. Avg. Price",
            "Remarks",
            "Entity",
            "Nearest_Strike",
            "Proximity_%",
            "Wall_Call_OI",
            "Wall_Put_OI",
            "Wall_PCR",
            "Wall_Call_IV",
            "Wall_Put_IV",
            "Wall_Type_Hit",
            "Score",
            "Signal_Strength",
            "Interpretation",
        ]
        if records:
            self.matched = pd.DataFrame(records)
            for col in MATCHED_COLS:
                if col not in self.matched.columns:
                    self.matched[col] = None
        else:
            self.matched = pd.DataFrame(columns=MATCHED_COLS)
        return self.matched

    # ─── scoring ─────────────────────────────────────────────────────────────

    def _score_deal(self, deal: dict) -> int:
        score = 0

        # 1. Proximity (30 pts) — closer = higher
        prox = deal.get("Proximity_%", self.proximity_pct)
        score += max(0, int(30 * (1 - prox / self.proximity_pct)))

        # 2. Wall OI size (20 pts)
        total_oi = deal.get("Wall_Call_OI", 0) + deal.get("Wall_Put_OI", 0)
        wall_oi_all = self.walls["Total_OI"].max() if "Total_OI" in self.walls else 1
        score += int(20 * min(total_oi / max(wall_oi_all, 1), 1))

        # 3. Deal size (20 pts) — normalised to top 1% of deals
        qty = deal.get("Quantity Traded", 0)
        price = deal.get("Trade Price/Wght. Avg. Price", 1)
        deal_value_cr = qty * price / 1e7  # crores
        score += int(min(deal_value_cr / 10, 1) * 20)  # cap at 10 Cr for full score

        # 4. Entity quality (20 pts)
        entity_scores = {
            "Promoter/Insider": 20,
            "FII/FPI": 18,
            "DII/MF": 16,
            "Institutional Broker": 10,
            "Retail/Unknown": 0,
        }
        score += entity_scores.get(deal.get("Entity", "Retail/Unknown"), 0)

        # 5. Directional alignment (10 pts)
        side = deal.get("Buy/Sell", "")
        wall_type = deal.get("Wall_Type_Hit", "")
        if (side == "BUY" and wall_type == "Put Wall") or (
            side == "SELL" and wall_type == "Call Wall"
        ):
            score += 10  # aligned with wall direction

        return min(score, 100)

    @staticmethod
    def _signal_strength(score: int) -> str:
        if score >= 80:
            return "🔴 VERY HIGH"
        elif score >= 60:
            return "🟠 HIGH"
        elif score >= 40:
            return "🟡 MODERATE"
        else:
            return "⚪ LOW"

    @staticmethod
    def _interpret_deal(deal: dict) -> str:
        side = deal.get("Buy/Sell", "")
        wall_type = deal.get("Wall_Type_Hit", "")
        entity = deal.get("Entity", "Unknown")
        strike = deal.get("Nearest_Strike", 0)
        score = deal.get("Score", 0)

        if side == "BUY" and wall_type == "Put Wall" and score >= 60:
            return (
                f"{entity} ACCUMULATING at Put Wall {strike:.0f} — bullish confirmation"
            )
        elif side == "SELL" and wall_type == "Call Wall" and score >= 60:
            return f"{entity} DISTRIBUTING at Call Wall {strike:.0f} — bearish confirmation"
        elif side == "BUY" and score >= 60:
            return f"{entity} buying near {strike:.0f} — watch for breakout"
        elif side == "SELL" and score >= 60:
            return f"{entity} selling near {strike:.0f} — watch for breakdown"
        else:
            return f"{entity} activity near {strike:.0f} — low conviction"

    # ─── zone detection ──────────────────────────────────────────────────────

    def detect_zones(self) -> pd.DataFrame:
        """
        Classify each wall strike as ACCUMULATION, DISTRIBUTION, or MIXED.
        """
        if self.matched is None:
            self.match_deals_to_walls()

        if self.matched.empty or "Score" not in self.matched.columns:
            self.zones = pd.DataFrame()
            return self.zones

        aggregated = self.aggregate_by_level()

        zones = []
        for _, row in aggregated.iterrows():
            net_qty = row.get("Net_Qty", 0)
            conviction = row.get("Avg_Score", 0)
            strike = row["Strike"]
            wall_row = self.walls[self.walls["Strike"] == strike]
            wall_type = (
                "Put Wall"
                if not wall_row.empty
                and wall_row.iloc[0]["Put_OI"] > wall_row.iloc[0]["Call_OI"]
                else "Call Wall"
            )

            if net_qty > 0 and wall_type == "Put Wall":
                zone_type = "ACCUMULATION"
            elif net_qty < 0 and wall_type == "Call Wall":
                zone_type = "DISTRIBUTION"
            else:
                zone_type = "MIXED"

            zones.append(
                {
                    "Strike": strike,
                    "Zone_Type": zone_type,
                    "Net_Direction": "BUY" if net_qty >= 0 else "SELL",
                    "Net_Qty": abs(net_qty),
                    "Wall_OI": row.get("Total_OI", 0),
                    "PCR": row.get("PCR", 0),
                    "Avg_Score": round(conviction, 1),
                    "Conviction": self._signal_strength(int(conviction)),
                    "Interpretation": self._zone_interpretation(
                        zone_type, strike, conviction
                    ),
                }
            )

        self.zones = (
            pd.DataFrame(zones)
            .sort_values("Avg_Score", ascending=False)
            .reset_index(drop=True)
        )
        return self.zones

    @staticmethod
    def _zone_interpretation(zone_type: str, strike: float, score: float) -> str:
        if zone_type == "ACCUMULATION":
            return f"Institutional buying at {strike:.0f} — potential support floor (score {score:.0f})"
        elif zone_type == "DISTRIBUTION":
            return f"Institutional selling at {strike:.0f} — potential resistance ceiling (score {score:.0f})"
        else:
            return (
                f"Mixed signals at {strike:.0f} — wait for clarity (score {score:.0f})"
            )

    # ─── aggregation ─────────────────────────────────────────────────────────

    def aggregate_by_level(self) -> pd.DataFrame:
        """
        Net buy/sell qty and conviction per wall level.
        """
        if self.matched is None:
            self.match_deals_to_walls()

        if self.matched.empty or "Score" not in self.matched.columns:
            return pd.DataFrame()

        matched = self.matched.copy()
        matched["Signed_Qty"] = matched.apply(
            lambda r: (
                r["Quantity Traded"]
                if r["Buy/Sell"] == "BUY"
                else -r["Quantity Traded"]
            ),
            axis=1,
        )

        agg = (
            matched.groupby("Nearest_Strike")
            .agg(
                Net_Qty=("Signed_Qty", "sum"),
                Avg_Score=("Score", "mean"),
                Deal_Count=("Score", "count"),
                Entities=("Entity", lambda x: ", ".join(x.unique())),
            )
            .reset_index()
        )
        agg.rename(columns={"Nearest_Strike": "Strike"}, inplace=True)

        # Merge wall OI info
        wall_oi = self.walls[["Strike", "Total_OI", "PCR_OI"]].rename(
            columns={"Total_OI": "Total_OI", "PCR_OI": "PCR"}
        )
        agg = agg.merge(wall_oi, on="Strike", how="left").fillna(0)
        return agg

    # ─── summary stats ───────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        if self.matched is None:
            self.match_deals_to_walls()
        if self.zones is None:
            self.detect_zones()

        total_deals = len(self.matched)

        # Guard against empty or column-less DataFrame
        has_score = "Score" in self.matched.columns and not self.matched.empty
        has_entity = "Entity" in self.matched.columns and not self.matched.empty

        at_wall = (
            int(self.matched[self.matched["Score"] >= 40]["Score"].count())
            if has_score
            else 0
        )
        high_conviction = (
            int(self.matched[self.matched["Score"] >= 60]["Score"].count())
            if has_score
            else 0
        )
        top_entity = (
            self.matched["Entity"].value_counts().index[0] if has_entity else "N/A"
        )

        acc_zones = (
            len(self.zones[self.zones["Zone_Type"] == "ACCUMULATION"])
            if not self.zones.empty
            else 0
        )
        dist_zones = (
            len(self.zones[self.zones["Zone_Type"] == "DISTRIBUTION"])
            if not self.zones.empty
            else 0
        )

        net_fii_dii = 0
        if not self.matched.empty and "Entity" in self.matched.columns:
            inst = self.matched[
                self.matched["Entity"].isin(["FII/FPI", "DII/MF"])
            ].copy()
            if (
                not inst.empty
                and "Buy/Sell" in inst.columns
                and "Quantity Traded" in inst.columns
            ):
                inst["Signed_Qty"] = inst.apply(
                    lambda r: (
                        r["Quantity Traded"]
                        if r["Buy/Sell"] == "BUY"
                        else -r["Quantity Traded"]
                    ),
                    axis=1,
                )
                net_fii_dii = inst["Signed_Qty"].sum()

        return {
            "total_deals": total_deals,
            "at_wall_levels": at_wall,
            "high_conviction": high_conviction,
            "top_entity": top_entity,
            "accumulation_zones": acc_zones,
            "distribution_zones": dist_zones,
            "net_fii_dii_qty": net_fii_dii,
        }
