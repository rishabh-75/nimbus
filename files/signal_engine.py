"""
SignalEngine — Composite signal generation from options walls + institutional activity.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


class SignalEngine:
    """
    Generates directional trading signals from options wall and
    institutional activity data.
    """

    # ─── options signal ───────────────────────────────────────────────────────

    @staticmethod
    def compute_options_signal(
        walls_df: pd.DataFrame,
        pcr: float,
        iv_skew: dict,
        max_pain: float,
        cmp: Optional[float] = None,
    ) -> dict:
        """
        Returns score -100 to +100 (positive = bullish) with component breakdown.
        """
        score = 0
        components = []

        # 1. PCR OI contribution
        if pcr < 0.7:
            s = 30
            desc = f"PCR OI = {pcr:.2f} — strongly bullish (call-heavy)"
        elif pcr < 1.0:
            s = 15
            desc = f"PCR OI = {pcr:.2f} — mildly bullish"
        elif pcr < 1.3:
            s = -15
            desc = f"PCR OI = {pcr:.2f} — mildly bearish"
        else:
            s = -30
            desc = f"PCR OI = {pcr:.2f} — strongly bearish (put-heavy)"
        score += s
        components.append({"component": "PCR OI", "score": s, "desc": desc})

        # 2. CMP vs Max Pain
        if cmp is not None and max_pain > 0:
            if cmp < max_pain:
                s = 20
                desc = f"CMP {cmp:.0f} below Max Pain {max_pain:.0f} — upward gravitational pull"
            else:
                s = -20
                desc = f"CMP {cmp:.0f} above Max Pain {max_pain:.0f} — downward gravitational pull"
            score += s
            components.append({"component": "Max Pain", "score": s, "desc": desc})

        # 3. IV Skew
        skew = iv_skew.get("skew", 0)
        put_iv = iv_skew.get("put_iv", 0)
        call_iv = iv_skew.get("call_iv", 0)
        if skew > 3:
            s = 15
            desc = f"Put IV ({put_iv:.1f}) > Call IV ({call_iv:.1f}) — protective hedging, upside bias"
        elif skew < -3:
            s = -15
            desc = f"Call IV ({call_iv:.1f}) > Put IV ({put_iv:.1f}) — call buying / distribution"
        else:
            s = 0
            desc = f"IV skew neutral ({skew:+.1f})"
        score += s
        components.append({"component": "IV Skew", "score": s, "desc": desc})

        # 4. Wall position vs CMP
        if cmp is not None and not walls_df.empty:
            put_walls = walls_df[walls_df["Put_OI"] > walls_df["Call_OI"]]
            call_walls = walls_df[walls_df["Call_OI"] > walls_df["Put_OI"]]

            if not put_walls.empty:
                highest_put_wall = put_walls["Strike"].max()
                if highest_put_wall < cmp:
                    s = 15
                    desc = f"Highest Put Wall {highest_put_wall:.0f} below CMP — floor support active"
                else:
                    s = -10
                    desc = f"Put Wall {highest_put_wall:.0f} above CMP — potential resistance"
                score += s
                components.append({"component": "Put Wall Position", "score": s, "desc": desc})

            if not call_walls.empty:
                top_call_wall = call_walls.loc[call_walls["Call_OI"].idxmax(), "Strike"]
                if top_call_wall > cmp:
                    s = -15
                    desc = f"Call Wall {top_call_wall:.0f} above CMP — ceiling resistance active"
                else:
                    s = 10
                    desc = f"CMP cleared Call Wall {top_call_wall:.0f} — breakout strength"
                score += s
                components.append({"component": "Call Wall Position", "score": s, "desc": desc})

        score = max(-100, min(100, score))
        label = SignalEngine._score_to_label(score)

        return {
            "score": score,
            "label": label,
            "components": components,
            "dominant_component": max(components, key=lambda x: abs(x["score"])) if components else {},
        }

    # ─── institutional signal ─────────────────────────────────────────────────

    @staticmethod
    def compute_institutional_signal(
        zones_df: pd.DataFrame,
        matched_df: pd.DataFrame,
    ) -> dict:
        """
        Returns score -100 to +100 with component breakdown.
        """
        score = 0
        components = []

        if zones_df is None or zones_df.empty:
            return {
                "score": 0,
                "label": "Neutral",
                "components": [],
                "dominant_component": {},
            }

        # 1. Accumulation zones
        acc = zones_df[
            (zones_df["Zone_Type"] == "ACCUMULATION") & (zones_df["Avg_Score"] >= 60)
        ]
        acc_count = min(len(acc), 3)
        s = acc_count * 15
        desc = f"{acc_count} high-conviction Accumulation zone(s) detected"
        score += s
        components.append({"component": "Accumulation Zones", "score": s, "desc": desc})

        # 2. Distribution zones
        dist = zones_df[
            (zones_df["Zone_Type"] == "DISTRIBUTION") & (zones_df["Avg_Score"] >= 60)
        ]
        dist_count = min(len(dist), 3)
        s = -(dist_count * 15)
        desc = f"{dist_count} high-conviction Distribution zone(s) detected"
        score += s
        components.append({"component": "Distribution Zones", "score": s, "desc": desc})

        if matched_df is None or matched_df.empty:
            score = max(-100, min(100, score))
            return {
                "score": score,
                "label": SignalEngine._score_to_label(score),
                "components": components,
                "dominant_component": max(components, key=lambda x: abs(x["score"])) if components else {},
            }

        # 3. Promoter/Insider activity
        insider = matched_df[matched_df["Entity"] == "Promoter/Insider"]
        if not insider.empty:
            net_insider = (
                insider.apply(
                    lambda r: r["Quantity Traded"] if r["Buy/Sell"] == "BUY" else -r["Quantity Traded"],
                    axis=1,
                ).sum()
            )
            s = 20 if net_insider > 0 else -20
            desc = f"Promoter/Insider net {'BUY' if net_insider > 0 else 'SELL'}: {abs(net_insider):,.0f} shares"
            score += s
            components.append({"component": "Insider Activity", "score": s, "desc": desc})

        # 4. Net FII + DII
        inst = matched_df[matched_df["Entity"].isin(["FII/FPI", "DII/MF"])].copy()
        if not inst.empty:
            inst["Signed"] = inst.apply(
                lambda r: r["Quantity Traded"] if r["Buy/Sell"] == "BUY" else -r["Quantity Traded"], axis=1
            )
            net = inst["Signed"].sum()
            s = 10 if net > 0 else -10
            desc = f"Net FII+DII: {'positive' if net >= 0 else 'negative'} ({abs(net)/1e6:.2f}M shares)"
            score += s
            components.append({"component": "FII/DII Net Flow", "score": s, "desc": desc})

        # 5. Very high conviction institutional buy
        vhigh = matched_df[(matched_df["Score"] >= 80) & (matched_df["Buy/Sell"] == "BUY")]
        if not vhigh.empty:
            s = 15
            top = vhigh.iloc[0]
            desc = f"Score ≥80 institutional BUY: {top['Client Name']} @ {top['Trade Price/Wght. Avg. Price']:.0f}"
            score += s
            components.append({"component": "High Conviction Buy", "score": s, "desc": desc})

        score = max(-100, min(100, score))
        return {
            "score": score,
            "label": SignalEngine._score_to_label(score),
            "components": components,
            "dominant_component": max(components, key=lambda x: abs(x["score"])) if components else {},
        }

    # ─── composite ───────────────────────────────────────────────────────────

    @staticmethod
    def composite_signal(
        options_score: float,
        institutional_score: float,
        iv_skew_score: float,
    ) -> dict:
        """
        Weighted blend of the three signal axes.
        Returns final score, label, confidence, and colour.
        """
        weighted = (
            options_score * 0.40
            + institutional_score * 0.40
            + iv_skew_score * 0.20
        )
        weighted = max(-100, min(100, weighted))

        label = SignalEngine._score_to_label(weighted)
        confidence = int(abs(weighted))

        if weighted > 60:
            colour = "#00e676"
            emoji = "🟢"
        elif weighted > 20:
            colour = "#69f0ae"
            emoji = "🟢"
        elif weighted > -20:
            colour = "#ffd740"
            emoji = "🟡"
        elif weighted > -60:
            colour = "#ff7043"
            emoji = "🔴"
        else:
            colour = "#ff5252"
            emoji = "🔴"

        return {
            "score": round(weighted, 1),
            "label": label,
            "confidence": confidence,
            "colour": colour,
            "emoji": emoji,
        }

    # ─── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score > 30:
            return "BULLISH"
        elif score > 10:
            return "Mildly Bullish"
        elif score >= -10:
            return "NEUTRAL"
        elif score >= -30:
            return "Mildly Bearish"
        else:
            return "BEARISH"

    @staticmethod
    def iv_skew_score(skew: float) -> float:
        """Convert raw IV skew to -100/+100 score component."""
        if skew > 5:
            return 50
        elif skew > 3:
            return 30
        elif skew > 0:
            return 10
        elif skew > -3:
            return -10
        elif skew > -5:
            return -30
        else:
            return -50

    # ─── master card text ─────────────────────────────────────────────────────

    @staticmethod
    def build_master_card(
        composite: dict,
        options_signal: dict,
        institutional_signal: dict,
        pcr: float,
        max_pain: float,
        cmp: Optional[float],
        summary: dict,
    ) -> dict:
        """
        Build the master signal card content for Tab 3.
        """
        bullets = []

        # Bullet 1: PCR
        pcr_sentiment = "bullish" if pcr < 1 else "bearish"
        bullets.append(f"PCR OI = {pcr:.2f} — {pcr_sentiment} bias")

        # Bullet 2: Max Pain
        if cmp and max_pain:
            diff = cmp - max_pain
            direction = "above" if diff > 0 else "below"
            bullets.append(
                f"Max Pain ({max_pain:.0f}) is {abs(diff):.0f} pts {direction} CMP ({cmp:.0f})"
            )

        # Bullet 3: Institutional
        acc = summary.get("accumulation_zones", 0)
        dist = summary.get("distribution_zones", 0)
        top_entity = summary.get("top_entity", "N/A")
        net = summary.get("net_fii_dii_qty", 0)
        net_label = f"net bought {abs(net)/1e6:.1f}M" if net >= 0 else f"net sold {abs(net)/1e6:.1f}M"
        bullets.append(
            f"{top_entity} {net_label} shares; {acc} acc. zone(s), {dist} dist. zone(s)"
        )

        # Key reason
        opt_dom = options_signal.get("dominant_component", {})
        key_reason = opt_dom.get("desc", "No dominant signal") if opt_dom else "Insufficient data"

        return {
            "emoji": composite["emoji"],
            "label": composite["label"],
            "confidence": composite["confidence"],
            "key_reason": key_reason,
            "bullets": bullets,
            "colour": composite["colour"],
        }
