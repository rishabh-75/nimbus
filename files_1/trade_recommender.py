"""
TradeRecommender — Generates ranked, actionable trade setups from options wall
structure, institutional activity, and composite signal data.

Each recommendation includes:
  - Direction (LONG / SHORT)
  - Strategy type (Breakout / Bounce / Range / Momentum / Reversal)
  - Timeframe (Intraday / Swing 2–5d / Positional 1–2w)
  - Entry zone (low–high band)
  - Target 1, Target 2
  - Stop Loss
  - Risk:Reward ratio
  - Conviction score (0–100)
  - Reasoning bullets
  - Invalidation condition
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── data classes ─────────────────────────────────────────────────────────────

@dataclass
class TradeSetup:
    id: int
    symbol: str
    direction: str              # LONG | SHORT
    strategy: str               # Breakout | Bounce | Range | Momentum | Reversal
    timeframe: str              # Intraday | Swing | Positional
    entry_low: float
    entry_high: float
    target1: float
    target2: float
    stop_loss: float
    rr_ratio: float             # risk:reward (target1)
    conviction: int             # 0–100
    signal_source: str          # Options | Institutional | Combined
    reasoning: list[str]        # 3–5 bullet points
    invalidation: str           # condition that kills the trade
    expected_move_pct: float    # % move to target1
    risk_pct: float             # % risk to stop loss

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_zone"] = f"{self.entry_low:.2f} – {self.entry_high:.2f}"
        d["reasoning_text"] = " | ".join(self.reasoning)
        return d


# ─── recommender ─────────────────────────────────────────────────────────────

class TradeRecommender:
    """
    Generates trade setups by combining:
      1. Options wall support / resistance levels
      2. Institutional zone direction (accumulation / distribution)
      3. PCR / IV skew bias
      4. Composite signal direction + confidence
    """

    ENTRY_BUFFER_PCT  = 0.003   # 0.3% buffer around entry levels
    MIN_RR            = 1.5     # minimum risk:reward to include a setup
    MAX_SETUPS        = 8       # cap on number of setups returned

    def __init__(
        self,
        symbol: str,
        walls_df: pd.DataFrame,
        call_walls: pd.DataFrame,
        put_walls: pd.DataFrame,
        zones_df: Optional[pd.DataFrame],
        matched_df: Optional[pd.DataFrame],
        pcr_data: dict,
        iv_skew: dict,
        max_pain: float,
        key_levels: dict,
        composite: dict,
        options_signal: dict,
        institutional_signal: dict,
        cmp: Optional[float] = None,
    ):
        self.symbol         = symbol.upper()
        self.walls          = walls_df
        self.call_walls     = call_walls
        self.put_walls      = put_walls
        self.zones          = zones_df if zones_df is not None else pd.DataFrame()
        self.matched        = matched_df if matched_df is not None else pd.DataFrame()
        self.pcr            = pcr_data
        self.iv_skew        = iv_skew
        self.max_pain       = max_pain
        self.key_levels     = key_levels
        self.composite      = composite
        self.options_signal = options_signal
        self.inst_signal    = institutional_signal
        self.cmp            = cmp
        self._id_counter    = 0
        self._setups: list[TradeSetup] = []

    # ─── public entry point ──────────────────────────────────────────────────

    def generate(self) -> list[TradeSetup]:
        """Run all strategy generators and return ranked setups."""
        self._setups = []

        if self.cmp is None or self.walls.empty:
            return []

        self._strategy_put_wall_bounce()
        self._strategy_call_wall_short()
        self._strategy_max_pain_magnet()
        self._strategy_breakout_above_call_wall()
        self._strategy_breakdown_below_put_wall()
        self._strategy_institutional_accumulation()
        self._strategy_institutional_distribution()
        self._strategy_range_trade()

        # Filter minimum R:R and sort by conviction
        valid = [s for s in self._setups if s.rr_ratio >= self.MIN_RR]
        valid.sort(key=lambda x: x.conviction, reverse=True)
        return valid[: self.MAX_SETUPS]

    def to_dataframe(self) -> pd.DataFrame:
        setups = self.generate()
        if not setups:
            return pd.DataFrame()
        rows = []
        for s in setups:
            rows.append({
                "ID":          s.id,
                "Direction":   s.direction,
                "Strategy":    s.strategy,
                "Timeframe":   s.timeframe,
                "Entry Zone":  f"{s.entry_low:.2f}–{s.entry_high:.2f}",
                "Target 1":    round(s.target1, 2),
                "Target 2":    round(s.target2, 2),
                "Stop Loss":   round(s.stop_loss, 2),
                "R:R":         round(s.rr_ratio, 2),
                "Conviction":  s.conviction,
                "Signal":      s.signal_source,
                "Move %":      f"{s.expected_move_pct:+.2f}%",
                "Risk %":      f"{s.risk_pct:.2f}%",
                "Reasoning":   " | ".join(s.reasoning),
                "Invalidation": s.invalidation,
            })
        return pd.DataFrame(rows)

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _buf(self, price: float) -> tuple[float, float]:
        """Return (low, high) entry band around a price."""
        return price * (1 - self.ENTRY_BUFFER_PCT), price * (1 + self.ENTRY_BUFFER_PCT)

    def _rr(self, entry: float, target: float, stop: float) -> float:
        """Calculate risk:reward using mid-entry."""
        reward = abs(target - entry)
        risk   = abs(entry - stop)
        return round(reward / risk, 2) if risk > 0 else 0.0

    def _expected_move(self, entry: float, target: float) -> float:
        return round((target - entry) / entry * 100, 2)

    def _risk_pct(self, entry: float, stop: float) -> float:
        return round(abs(entry - stop) / entry * 100, 2)

    def _composite_score(self) -> float:
        return self.composite.get("score", 0)

    def _composite_label(self) -> str:
        return self.composite.get("label", "NEUTRAL")

    def _top_put_wall_strike(self) -> Optional[float]:
        if self.put_walls.empty:
            return None
        return float(self.put_walls.loc[self.put_walls["Put_OI"].idxmax(), "Strike"])

    def _top_call_wall_strike(self) -> Optional[float]:
        if self.call_walls.empty:
            return None
        return float(self.call_walls.loc[self.call_walls["Call_OI"].idxmax(), "Strike"])

    def _next_put_wall_below_cmp(self) -> Optional[float]:
        """Highest put wall strike that is still below CMP."""
        if self.put_walls.empty or self.cmp is None:
            return None
        below = self.put_walls[self.put_walls["Strike"] < self.cmp]
        return float(below["Strike"].max()) if not below.empty else None

    def _next_call_wall_above_cmp(self) -> Optional[float]:
        """Lowest call wall strike that is above CMP."""
        if self.call_walls.empty or self.cmp is None:
            return None
        above = self.call_walls[self.call_walls["Strike"] > self.cmp]
        return float(above["Strike"].min()) if not above.empty else None

    def _strike_step(self) -> float:
        """Infer the strike spacing from the walls dataframe."""
        strikes = sorted(self.walls["Strike"].unique())
        if len(strikes) < 2:
            return 10.0
        diffs = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
        return float(np.median(diffs))

    def _pcr_bullish(self) -> bool:
        return self.pcr.get("pcr_oi", 1.0) < 1.0

    def _iv_bullish(self) -> bool:
        return self.iv_skew.get("direction", "neutral") == "bullish"

    # ─── Strategy 1: Put Wall Bounce (LONG) ──────────────────────────────────

    def _strategy_put_wall_bounce(self):
        """
        CMP is near a high-OI put wall → expect a bounce long.
        Best when PCR < 1, IV skew bullish, institutional accumulation present.
        """
        support = self._next_put_wall_below_cmp()
        if support is None or self.cmp is None:
            return

        distance_pct = (self.cmp - support) / self.cmp * 100
        if distance_pct > 3.0:   # too far from support
            return

        entry_low, entry_high = self._buf(support)
        entry_mid = (entry_low + entry_high) / 2
        step  = self._strike_step()
        tgt1  = support + 2 * step
        tgt2  = support + 4 * step
        stop  = support - step * 0.8

        rr = self._rr(entry_mid, tgt1, stop)
        if rr < self.MIN_RR:
            return

        # Conviction scoring
        conv = 40
        if self._pcr_bullish():           conv += 15
        if self._iv_bullish():             conv += 10
        if self._composite_score() > 20:   conv += 15
        if distance_pct < 1.0:            conv += 10  # very close to wall
        # Institutional accumulation at this level?
        if not self.zones.empty:
            acc = self.zones[
                (self.zones["Zone_Type"] == "ACCUMULATION") &
                (abs(self.zones["Strike"] - support) <= step)
            ]
            if not acc.empty:
                conv += min(int(acc["Avg_Score"].max() / 5), 15)

        conv = min(conv, 100)

        reasoning = [
            f"Put Wall at {support:.0f} has highest PE OI — max pain floor for option writers",
            f"CMP {self.cmp:.0f} is only {distance_pct:.1f}% above support — low-risk entry",
            f"PCR OI = {self.pcr.get('pcr_oi',0):.2f} — {'bullish bias' if self._pcr_bullish() else 'watch for breakdown'}",
        ]
        if self._iv_bullish():
            reasoning.append(f"Put IV > Call IV (+{self.iv_skew.get('skew',0):.1f}) — protective hedging demand supports floor")
        if not self.zones.empty and not self.zones[
            (self.zones["Zone_Type"] == "ACCUMULATION") & (abs(self.zones["Strike"] - support) <= step)
        ].empty:
            reasoning.append(f"Institutional ACCUMULATION detected at {support:.0f} — confirms support thesis")

        self._setups.append(TradeSetup(
            id=self._next_id(),
            symbol=self.symbol,
            direction="LONG",
            strategy="Bounce",
            timeframe="Intraday" if distance_pct < 0.8 else "Swing",
            entry_low=entry_low, entry_high=entry_high,
            target1=tgt1, target2=tgt2,
            stop_loss=stop,
            rr_ratio=rr,
            conviction=conv,
            signal_source="Options + Institutional",
            reasoning=reasoning,
            invalidation=f"Close below {stop:.2f} on 15-min candle invalidates the put wall floor",
            expected_move_pct=self._expected_move(entry_mid, tgt1),
            risk_pct=self._risk_pct(entry_mid, stop),
        ))

    # ─── Strategy 2: Call Wall Short ─────────────────────────────────────────

    def _strategy_call_wall_short(self):
        """
        CMP approaching a high-OI call wall from below → fade the resistance.
        """
        resistance = self._next_call_wall_above_cmp()
        if resistance is None or self.cmp is None:
            return

        distance_pct = (resistance - self.cmp) / self.cmp * 100
        if distance_pct > 3.0:
            return

        entry_low, entry_high = self._buf(resistance)
        entry_mid = (entry_low + entry_high) / 2
        step  = self._strike_step()
        tgt1  = resistance - 2 * step
        tgt2  = resistance - 4 * step
        stop  = resistance + step * 0.8

        rr = self._rr(entry_mid, tgt1, stop)
        if rr < self.MIN_RR:
            return

        conv = 40
        if not self._pcr_bullish():        conv += 15
        if not self._iv_bullish():          conv += 10
        if self._composite_score() < -20:   conv += 15
        if distance_pct < 1.0:             conv += 10
        if not self.zones.empty:
            dist = self.zones[
                (self.zones["Zone_Type"] == "DISTRIBUTION") &
                (abs(self.zones["Strike"] - resistance) <= step)
            ]
            if not dist.empty:
                conv += min(int(dist["Avg_Score"].max() / 5), 15)

        conv = min(conv, 100)

        reasoning = [
            f"Call Wall at {resistance:.0f} — massive CE OI creates a ceiling for writers",
            f"CMP {self.cmp:.0f} is {distance_pct:.1f}% below resistance — ideal short zone",
            f"PCR = {self.pcr.get('pcr_oi',0):.2f} — {'high put accumulation, supports bear thesis' if not self._pcr_bullish() else 'watch for call wall breach'}",
        ]
        if not self._iv_bullish():
            reasoning.append(f"Call IV > Put IV ({self.iv_skew.get('skew',0):.1f}) — call buying / distribution pressure near resistance")
        if not self.zones.empty and not self.zones[
            (self.zones["Zone_Type"] == "DISTRIBUTION") & (abs(self.zones["Strike"] - resistance) <= step)
        ].empty:
            reasoning.append(f"Institutional DISTRIBUTION at {resistance:.0f} — confirms selling interest")

        self._setups.append(TradeSetup(
            id=self._next_id(),
            symbol=self.symbol,
            direction="SHORT",
            strategy="Fade Resistance",
            timeframe="Intraday" if distance_pct < 0.8 else "Swing",
            entry_low=entry_low, entry_high=entry_high,
            target1=tgt1, target2=tgt2,
            stop_loss=stop,
            rr_ratio=rr,
            conviction=conv,
            signal_source="Options",
            reasoning=reasoning,
            invalidation=f"Hourly close above {stop:.2f} (call wall breach) invalidates the short",
            expected_move_pct=self._expected_move(entry_mid, tgt1),
            risk_pct=self._risk_pct(entry_mid, stop),
        ))

    # ─── Strategy 3: Max Pain Magnet ─────────────────────────────────────────

    def _strategy_max_pain_magnet(self):
        """
        If CMP is far from max pain, it tends to gravitate toward it near expiry.
        Generate a mean-reversion trade toward max pain.
        """
        if self.cmp is None or self.max_pain <= 0:
            return

        diff       = self.max_pain - self.cmp
        diff_pct   = abs(diff) / self.cmp * 100
        step       = self._strike_step()

        if diff_pct < 1.5:   # already near max pain — no edge
            return
        if diff_pct > 8.0:   # too far — magnet effect too weak intraday
            return

        direction  = "LONG" if diff > 0 else "SHORT"
        entry_mid  = self.cmp
        entry_low, entry_high = self._buf(entry_mid)

        if direction == "LONG":
            tgt1  = self.cmp + step
            tgt2  = self.max_pain
            stop  = self.cmp - step * 0.8
        else:
            tgt1  = self.cmp - step
            tgt2  = self.max_pain
            stop  = self.cmp + step * 0.8

        rr = self._rr(entry_mid, tgt1, stop)
        if rr < self.MIN_RR:
            return

        conv = 35
        if diff_pct > 3.0:   conv += 15    # larger gap = stronger pull
        if diff_pct > 5.0:   conv += 10
        # Composite alignment
        if direction == "LONG"  and self._composite_score() > 10: conv += 15
        if direction == "SHORT" and self._composite_score() < -10: conv += 15
        conv = min(conv, 100)

        reasoning = [
            f"Max Pain at {self.max_pain:.0f} — CMP ({self.cmp:.0f}) is {diff_pct:.1f}% {'below' if diff > 0 else 'above'}",
            "Option writers profit maximally at max pain — gravitational pull near expiry",
            f"Expected mean-reversion move: {direction} toward {self.max_pain:.0f}",
            "Best played on expiry day or 1–2 days prior",
        ]

        self._setups.append(TradeSetup(
            id=self._next_id(),
            symbol=self.symbol,
            direction=direction,
            strategy="Max Pain Magnet",
            timeframe="Intraday",
            entry_low=entry_low, entry_high=entry_high,
            target1=tgt1, target2=tgt2,
            stop_loss=stop,
            rr_ratio=rr,
            conviction=conv,
            signal_source="Options",
            reasoning=reasoning,
            invalidation=f"Ignore if more than 3 days to expiry or if OI shifts max pain",
            expected_move_pct=self._expected_move(entry_mid, tgt1),
            risk_pct=self._risk_pct(entry_mid, stop),
        ))

    # ─── Strategy 4: Breakout above Call Wall ────────────────────────────────

    def _strategy_breakout_above_call_wall(self):
        """
        If CMP has just cleared the primary call wall, look for a breakout continuation.
        """
        top_call = self._top_call_wall_strike()
        if top_call is None or self.cmp is None:
            return

        gap_pct = (self.cmp - top_call) / top_call * 100
        if not (0 < gap_pct <= 1.5):   # CMP must be just above the call wall
            return

        step  = self._strike_step()
        entry_low, entry_high = top_call, self.cmp * 1.002
        entry_mid = (entry_low + entry_high) / 2
        tgt1  = self.cmp + 2 * step
        tgt2  = self.cmp + 4 * step
        stop  = top_call - step * 0.5   # just below the broken wall

        rr = self._rr(entry_mid, tgt1, stop)
        if rr < self.MIN_RR:
            return

        conv = 45
        if self._pcr_bullish():           conv += 10
        if self._composite_score() > 30:  conv += 20
        # Volume confirmation from deals
        if not self.matched.empty:
            near_wall = self.matched[
                (abs(self.matched["Nearest_Strike"] - top_call) <= step) &
                (self.matched["Buy/Sell"] == "BUY")
            ]
            if not near_wall.empty:
                conv += min(int(near_wall["Score"].max() / 10), 15)

        conv = min(conv, 100)

        reasoning = [
            f"CMP {self.cmp:.0f} has cleared the Call Wall at {top_call:.0f} — resistance flips to support",
            f"Breakout above major OI wall is a high-probability continuation signal",
            f"Bullish composite score {self._composite_score():+.0f} confirms momentum",
            f"New support floor: {top_call:.0f} (re-entry on any pullback to this level)",
        ]

        self._setups.append(TradeSetup(
            id=self._next_id(),
            symbol=self.symbol,
            direction="LONG",
            strategy="Breakout",
            timeframe="Swing",
            entry_low=entry_low, entry_high=entry_high,
            target1=tgt1, target2=tgt2,
            stop_loss=stop,
            rr_ratio=rr,
            conviction=conv,
            signal_source="Options + Composite",
            reasoning=reasoning,
            invalidation=f"Hourly close back below {top_call:.2f} — false breakout, exit immediately",
            expected_move_pct=self._expected_move(entry_mid, tgt1),
            risk_pct=self._risk_pct(entry_mid, stop),
        ))

    # ─── Strategy 5: Breakdown below Put Wall ────────────────────────────────

    def _strategy_breakdown_below_put_wall(self):
        """
        CMP has just broken below the primary put wall — bearish continuation.
        """
        top_put = self._top_put_wall_strike()
        if top_put is None or self.cmp is None:
            return

        gap_pct = (top_put - self.cmp) / top_put * 100
        if not (0 < gap_pct <= 1.5):
            return

        step  = self._strike_step()
        entry_low, entry_high = self.cmp * 0.998, top_put
        entry_mid = (entry_low + entry_high) / 2
        tgt1  = self.cmp - 2 * step
        tgt2  = self.cmp - 4 * step
        stop  = top_put + step * 0.5

        rr = self._rr(entry_mid, tgt1, stop)
        if rr < self.MIN_RR:
            return

        conv = 45
        if not self._pcr_bullish():         conv += 10
        if self._composite_score() < -30:   conv += 20
        if not self.matched.empty:
            near_wall = self.matched[
                (abs(self.matched["Nearest_Strike"] - top_put) <= step) &
                (self.matched["Buy/Sell"] == "SELL")
            ]
            if not near_wall.empty:
                conv += min(int(near_wall["Score"].max() / 10), 15)

        conv = min(conv, 100)

        reasoning = [
            f"CMP {self.cmp:.0f} has broken below Put Wall {top_put:.0f} — support has failed",
            "Put wall breakdown signals call writers winning — bearish momentum",
            f"Bearish composite score {self._composite_score():+.0f} confirms direction",
            f"Old support {top_put:.0f} now becomes resistance",
        ]

        self._setups.append(TradeSetup(
            id=self._next_id(),
            symbol=self.symbol,
            direction="SHORT",
            strategy="Breakdown",
            timeframe="Swing",
            entry_low=entry_low, entry_high=entry_high,
            target1=tgt1, target2=tgt2,
            stop_loss=stop,
            rr_ratio=rr,
            conviction=conv,
            signal_source="Options + Composite",
            reasoning=reasoning,
            invalidation=f"Hourly close back above {top_put:.2f} — false breakdown, exit",
            expected_move_pct=self._expected_move(entry_mid, tgt1),
            risk_pct=self._risk_pct(entry_mid, stop),
        ))

    # ─── Strategy 6: Institutional Accumulation Long ─────────────────────────

    def _strategy_institutional_accumulation(self):
        """
        High-conviction institutional BUY at a put wall → multi-day long.
        """
        if self.zones.empty or self.cmp is None:
            return

        acc_zones = self.zones[
            (self.zones["Zone_Type"] == "ACCUMULATION") &
            (self.zones["Avg_Score"] >= 60)
        ].sort_values("Avg_Score", ascending=False)

        if acc_zones.empty:
            return

        best = acc_zones.iloc[0]
        strike = best["Strike"]

        # Only trade if we're reasonably close to the accumulation zone
        if abs(self.cmp - strike) / self.cmp > 0.05:
            return

        step   = self._strike_step()
        entry_low, entry_high = self._buf(strike * 1.001)
        entry_mid = (entry_low + entry_high) / 2
        tgt1   = strike + 3 * step
        tgt2   = strike + 6 * step
        stop   = strike - 1.5 * step

        rr = self._rr(entry_mid, tgt1, stop)
        if rr < self.MIN_RR:
            return

        conv = int(min(best["Avg_Score"] * 0.9, 95))

        # Top entity involved
        top_entity = "Institutional"
        if not self.matched.empty:
            nearby = self.matched[
                (abs(self.matched["Nearest_Strike"] - strike) <= step) &
                (self.matched["Buy/Sell"] == "BUY")
            ]
            if not nearby.empty:
                top_entity = nearby.sort_values("Score", ascending=False).iloc[0]["Client Name"]

        reasoning = [
            f"ACCUMULATION zone at {strike:.0f} — {top_entity} buying confirmed",
            f"Institutional conviction score: {best['Avg_Score']:.0f}/100 — strong signal",
            f"Put Wall at same level — options writers defending this floor",
            f"Net qty: {best.get('Net_Qty', 0):,.0f} shares bought — demand absorption",
        ]
        if self._pcr_bullish():
            reasoning.append(f"PCR {self.pcr.get('pcr_oi',0):.2f} confirms bullish positioning")

        self._setups.append(TradeSetup(
            id=self._next_id(),
            symbol=self.symbol,
            direction="LONG",
            strategy="Institutional Accumulation",
            timeframe="Positional",
            entry_low=entry_low, entry_high=entry_high,
            target1=tgt1, target2=tgt2,
            stop_loss=stop,
            rr_ratio=rr,
            conviction=conv,
            signal_source="Institutional",
            reasoning=reasoning,
            invalidation=f"Large SELL deal from same entity OR close below {stop:.2f}",
            expected_move_pct=self._expected_move(entry_mid, tgt1),
            risk_pct=self._risk_pct(entry_mid, stop),
        ))

    # ─── Strategy 7: Institutional Distribution Short ────────────────────────

    def _strategy_institutional_distribution(self):
        """
        High-conviction institutional SELL at a call wall → multi-day short.
        """
        if self.zones.empty or self.cmp is None:
            return

        dist_zones = self.zones[
            (self.zones["Zone_Type"] == "DISTRIBUTION") &
            (self.zones["Avg_Score"] >= 60)
        ].sort_values("Avg_Score", ascending=False)

        if dist_zones.empty:
            return

        best   = dist_zones.iloc[0]
        strike = best["Strike"]

        if abs(self.cmp - strike) / self.cmp > 0.05:
            return

        step   = self._strike_step()
        entry_low, entry_high = self._buf(strike * 0.999)
        entry_mid = (entry_low + entry_high) / 2
        tgt1   = strike - 3 * step
        tgt2   = strike - 6 * step
        stop   = strike + 1.5 * step

        rr = self._rr(entry_mid, tgt1, stop)
        if rr < self.MIN_RR:
            return

        conv = int(min(best["Avg_Score"] * 0.9, 95))

        top_entity = "Institutional"
        if not self.matched.empty:
            nearby = self.matched[
                (abs(self.matched["Nearest_Strike"] - strike) <= step) &
                (self.matched["Buy/Sell"] == "SELL")
            ]
            if not nearby.empty:
                top_entity = nearby.sort_values("Score", ascending=False).iloc[0]["Client Name"]

        reasoning = [
            f"DISTRIBUTION zone at {strike:.0f} — {top_entity} selling confirmed",
            f"Institutional conviction score: {best['Avg_Score']:.0f}/100 — strong signal",
            f"Call Wall at same level — options writers defending this ceiling",
            f"Net qty: {best.get('Net_Qty', 0):,.0f} shares sold — supply overhang",
        ]

        self._setups.append(TradeSetup(
            id=self._next_id(),
            symbol=self.symbol,
            direction="SHORT",
            strategy="Institutional Distribution",
            timeframe="Positional",
            entry_low=entry_low, entry_high=entry_high,
            target1=tgt1, target2=tgt2,
            stop_loss=stop,
            rr_ratio=rr,
            conviction=conv,
            signal_source="Institutional",
            reasoning=reasoning,
            invalidation=f"Large BUY deal reversal OR close above {stop:.2f}",
            expected_move_pct=self._expected_move(entry_mid, tgt1),
            risk_pct=self._risk_pct(entry_mid, stop),
        ))

    # ─── Strategy 8: Range Trade ─────────────────────────────────────────────

    def _strategy_range_trade(self):
        """
        When CMP is inside put wall support + call wall resistance,
        generate both a range buy near support and range sell near resistance.
        """
        support    = self.key_levels.get("primary_support")
        resistance = self.key_levels.get("primary_resistance")
        if support is None or resistance is None or self.cmp is None:
            return

        rng_size = resistance - support
        if rng_size <= 0:
            return

        rng_pct = rng_size / self.cmp * 100
        # Range must be meaningful but not too wide
        if rng_pct < 1.5 or rng_pct > 8.0:
            return

        # CMP must be within the range
        if not (support <= self.cmp <= resistance):
            return

        step = self._strike_step()

        # Range BUY leg — near support
        if (self.cmp - support) / self.cmp < 0.025:
            el, eh  = self._buf(support)
            em      = (el + eh) / 2
            t1      = support + rng_size * 0.5
            t2      = resistance * 0.99
            sl      = support - step * 0.8
            rr      = self._rr(em, t1, sl)
            conv    = 40
            if self._pcr_bullish(): conv += 15
            if self._composite_score() >= 0: conv += 10
            conv = min(conv, 85)

            if rr >= self.MIN_RR:
                self._setups.append(TradeSetup(
                    id=self._next_id(),
                    symbol=self.symbol,
                    direction="LONG",
                    strategy="Range Buy",
                    timeframe="Intraday",
                    entry_low=el, entry_high=eh,
                    target1=t1, target2=t2,
                    stop_loss=sl,
                    rr_ratio=rr,
                    conviction=conv,
                    signal_source="Options",
                    reasoning=[
                        f"Range defined: {support:.0f} (Put Wall) → {resistance:.0f} (Call Wall)",
                        f"CMP {self.cmp:.0f} near range bottom — favourable entry",
                        f"Range width: {rng_pct:.1f}% — good intraday oscillation potential",
                        "Buy bottom of range, target mid / top",
                    ],
                    invalidation=f"Range breaks below {sl:.2f} — momentum shift, exit",
                    expected_move_pct=self._expected_move(em, t1),
                    risk_pct=self._risk_pct(em, sl),
                ))

        # Range SELL leg — near resistance
        if (resistance - self.cmp) / self.cmp < 0.025:
            el, eh  = self._buf(resistance)
            em      = (el + eh) / 2
            t1      = resistance - rng_size * 0.5
            t2      = support * 1.01
            sl      = resistance + step * 0.8
            rr      = self._rr(em, t1, sl)
            conv    = 40
            if not self._pcr_bullish(): conv += 15
            if self._composite_score() <= 0: conv += 10
            conv = min(conv, 85)

            if rr >= self.MIN_RR:
                self._setups.append(TradeSetup(
                    id=self._next_id(),
                    symbol=self.symbol,
                    direction="SHORT",
                    strategy="Range Sell",
                    timeframe="Intraday",
                    entry_low=el, entry_high=eh,
                    target1=t1, target2=t2,
                    stop_loss=sl,
                    rr_ratio=rr,
                    conviction=conv,
                    signal_source="Options",
                    reasoning=[
                        f"Range defined: {support:.0f} (Put Wall) → {resistance:.0f} (Call Wall)",
                        f"CMP {self.cmp:.0f} near range top — short entry",
                        f"Range width: {rng_pct:.1f}% — sell top, target mid / bottom",
                        "Bounded by heavy call OI — writers will defend ceiling",
                    ],
                    invalidation=f"Range breaks above {sl:.2f} — breakout, cover short",
                    expected_move_pct=self._expected_move(em, t1),
                    risk_pct=self._risk_pct(em, sl),
                ))
