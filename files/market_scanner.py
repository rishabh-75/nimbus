"""
MarketScanner — Runs the full pipeline (options walls + institutional + signal engine
+ trade recommender) across a watchlist of symbols and returns a ranked
cross-market opportunity table.
"""

from __future__ import annotations
import time
import logging
import datetime
import traceback
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from options_wall import OptionsWallCalculator
from insider_detector import InsiderWallDetector
from signal_engine import SignalEngine
from data_manager import DataManager
from trade_recommender import TradeRecommender, TradeSetup

logger = logging.getLogger(__name__)


# ─── per-symbol scan result ───────────────────────────────────────────────────


@dataclass
class SymbolScanResult:
    symbol: str
    status: str  # "ok" | "no_data" | "error"
    error_msg: str = ""

    # Wall summary
    primary_support: Optional[float] = None
    primary_resistance: Optional[float] = None
    pcr: Optional[float] = None
    sentiment: str = ""
    max_pain: Optional[float] = None
    iv_skew_dir: str = ""
    cmp: Optional[float] = None

    # Signal
    composite_score: float = 0.0
    composite_label: str = "NEUTRAL"
    composite_emoji: str = "⚪"

    # Deals
    total_deals: int = 0
    high_conviction_deals: int = 0
    top_entity: str = ""
    accumulation_zones: int = 0
    distribution_zones: int = 0

    # Best setup from recommender
    best_setup: Optional[TradeSetup] = None
    total_setups: int = 0
    best_rr: float = 0.0

    # Meta
    scanned_at: Optional[float] = None
    is_demo: bool = False


# ─── scanner ──────────────────────────────────────────────────────────────────


class MarketScanner:
    """
    Iterates over a list of symbols, runs the full analysis pipeline for each,
    and returns ranked scan results + a flat DataFrame of all setups.
    """

    def __init__(
        self,
        watchlist: list[str],
        data_manager: DataManager,
        date: Optional[datetime.date] = None,
        wall_pct: float = 75,
        proximity_pct: float = 1.5,
        min_setup_score: int = 40,
        min_rr: float = 1.5,
    ):
        self.watchlist = [s.upper().strip() for s in watchlist if s.strip()]
        self.dm = data_manager
        self.date = date or datetime.date.today()
        self.wall_pct = wall_pct
        self.proximity_pct = proximity_pct
        self.min_setup_score = min_setup_score
        self.min_rr = min_rr

        self.results: list[SymbolScanResult] = []
        self.all_setups_df: pd.DataFrame = pd.DataFrame()

    # ─── main scan ────────────────────────────────────────────────────────────

    def scan(self, progress_callback=None) -> list[SymbolScanResult]:
        """
        Run full pipeline for each symbol.
        progress_callback(symbol, idx, total) — optional UI hook.
        """
        self.results = []
        all_setup_rows = []

        for idx, symbol in enumerate(self.watchlist):
            if progress_callback:
                progress_callback(symbol, idx, len(self.watchlist))

            result = self._scan_symbol(symbol)
            self.results.append(result)

            # Collect setups into flat frame
            if result.best_setup is not None and result.status == "ok":
                rec = TradeRecommender(
                    symbol=symbol,
                    walls_df=self._last_walls,
                    call_walls=self._last_call_walls,
                    put_walls=self._last_put_walls,
                    zones_df=self._last_zones,
                    matched_df=self._last_matched,
                    pcr_data=self._last_pcr,
                    iv_skew=self._last_iv_skew,
                    max_pain=self._last_max_pain,
                    key_levels=self._last_key_levels,
                    composite=self._last_composite,
                    options_signal=self._last_opt_signal,
                    institutional_signal=self._last_inst_signal,
                    cmp=result.cmp,
                )
                setups = rec.generate()
                for s in setups:
                    all_setup_rows.append(
                        {
                            "Symbol": symbol,
                            "ID": s.id,
                            "Direction": s.direction,
                            "Strategy": s.strategy,
                            "Timeframe": s.timeframe,
                            "Entry Zone": f"{s.entry_low:.2f}–{s.entry_high:.2f}",
                            "Target 1": round(s.target1, 2),
                            "Target 2": round(s.target2, 2),
                            "Stop Loss": round(s.stop_loss, 2),
                            "R:R": round(s.rr_ratio, 2),
                            "Conviction": s.conviction,
                            "Move %": f"{s.expected_move_pct:+.2f}%",
                            "Risk %": f"{s.risk_pct:.2f}%",
                            "Signal": s.signal_source,
                            "Reasoning": " | ".join(s.reasoning),
                            "Invalidation": s.invalidation,
                            "PCR": round(result.pcr or 0, 3),
                            "Composite": round(result.composite_score, 1),
                            "CMP": result.cmp,
                        }
                    )

        self.all_setups_df = (
            pd.DataFrame(all_setup_rows)
            .sort_values(["Conviction", "R:R"], ascending=False)
            .reset_index(drop=True)
            if all_setup_rows
            else pd.DataFrame()
        )

        # Sort results: status=ok first, then by best conviction desc
        self.results.sort(
            key=lambda r: (
                0 if r.status == "ok" else 1,
                -(r.best_setup.conviction if r.best_setup else 0),
            )
        )
        return self.results

    # ─── single symbol pipeline ───────────────────────────────────────────────

    # We store the last-computed intermediates so scan() can re-use them
    # for the full setup generation pass without re-running everything twice.
    _last_walls = None
    _last_call_walls = None
    _last_put_walls = None
    _last_zones = None
    _last_matched = None
    _last_pcr = {}
    _last_iv_skew = {}
    _last_max_pain = 0.0
    _last_key_levels = {}
    _last_composite = {}
    _last_opt_signal = {}
    _last_inst_signal = {}

    def _scan_symbol(self, symbol: str) -> SymbolScanResult:
        result = SymbolScanResult(
            symbol=symbol, status="no_data", scanned_at=time.time()
        )
        try:
            # 1. Options chain
            options_df, is_demo = self.dm.load_options_chain(symbol, self.date)
            result.is_demo = is_demo

            if options_df is None or options_df.empty:
                result.status = "no_data"
                return result

            # 2. Walls
            calc = OptionsWallCalculator(options_df)
            walls_df = calc.consolidate_walls()
            call_walls, put_walls = calc.identify_walls(pct=self.wall_pct)
            pcr_data = calc.analyze_pcr()
            iv_skew = calc.analyze_iv_skew()
            max_pain = calc.calculate_max_pain()
            cmp = self.dm.get_cmp(symbol)
            key_levels = calc.identify_key_levels(cmp=cmp, pct=self.wall_pct)

            # 3. Deals
            deals_df, _ = self.dm.load_deals(symbol=symbol)
            detector = InsiderWallDetector(
                walls_df, deals_df, proximity_pct=self.proximity_pct
            )
            matched_df = detector.match_deals_to_walls()
            zones_df = detector.detect_zones()
            deal_summary = detector.get_summary()

            # 4. Signals
            engine = SignalEngine()
            opt_signal = engine.compute_options_signal(
                walls_df, pcr_data["pcr_oi"], iv_skew, max_pain, cmp
            )
            inst_signal = engine.compute_institutional_signal(zones_df, matched_df)
            iv_score = engine.iv_skew_score(iv_skew.get("skew", 0))
            composite = engine.composite_signal(
                opt_signal["score"], inst_signal["score"], iv_score
            )

            # 5. Quick recommender pass (just count + best setup)
            rec = TradeRecommender(
                symbol=symbol,
                walls_df=walls_df,
                call_walls=call_walls,
                put_walls=put_walls,
                zones_df=zones_df,
                matched_df=matched_df,
                pcr_data=pcr_data,
                iv_skew=iv_skew,
                max_pain=max_pain,
                key_levels=key_levels,
                composite=composite,
                options_signal=opt_signal,
                institutional_signal=inst_signal,
                cmp=cmp,
            )
            setups = rec.generate()

            # Stash intermediates for reuse
            self.__class__._last_walls = walls_df
            self.__class__._last_call_walls = call_walls
            self.__class__._last_put_walls = put_walls
            self.__class__._last_zones = zones_df
            self.__class__._last_matched = matched_df
            self.__class__._last_pcr = pcr_data
            self.__class__._last_iv_skew = iv_skew
            self.__class__._last_max_pain = max_pain
            self.__class__._last_key_levels = key_levels
            self.__class__._last_composite = composite
            self.__class__._last_opt_signal = opt_signal
            self.__class__._last_inst_signal = inst_signal

            # Populate result
            result.status = "ok"
            result.primary_support = key_levels.get("primary_support")
            result.primary_resistance = key_levels.get("primary_resistance")
            result.pcr = pcr_data.get("pcr_oi")
            result.sentiment = pcr_data.get("sentiment", "")
            result.max_pain = max_pain
            result.iv_skew_dir = iv_skew.get("direction", "neutral")
            result.cmp = cmp
            result.composite_score = composite.get("score", 0)
            result.composite_label = composite.get("label", "NEUTRAL")
            result.composite_emoji = composite.get("emoji", "⚪")
            result.total_deals = deal_summary.get("total_deals", 0)
            result.high_conviction_deals = deal_summary.get("high_conviction", 0)
            result.top_entity = deal_summary.get("top_entity", "")
            result.accumulation_zones = deal_summary.get("accumulation_zones", 0)
            result.distribution_zones = deal_summary.get("distribution_zones", 0)
            result.total_setups = len(setups)
            result.best_setup = setups[0] if setups else None
            result.best_rr = setups[0].rr_ratio if setups else 0.0

        except Exception as e:
            result.status = "error"
            result.error_msg = str(e)
            logger.warning(f"Scanner error for {symbol}: {e}\n{traceback.format_exc()}")

        return result

    # ─── derived views ────────────────────────────────────────────────────────

    def summary_dataframe(self) -> pd.DataFrame:
        """One row per symbol — the market overview grid."""
        rows = []
        for r in self.results:
            if r.status != "ok":
                rows.append(
                    {
                        "Symbol": r.symbol,
                        "Status": "⚠️ " + r.status,
                        "Signal": "—",
                        "PCR": "—",
                        "Setups": 0,
                        "Best Setup": "—",
                        "Best R:R": "—",
                        "Conviction": 0,
                        "Acc Zones": 0,
                        "Dist Zones": 0,
                        "Support": "—",
                        "Resistance": "—",
                        "Max Pain": "—",
                        "IV Skew": "—",
                        "Demo": r.is_demo,
                    }
                )
                continue

            bs = r.best_setup
            rows.append(
                {
                    "Symbol": r.symbol,
                    "Status": "✅ ok",
                    "Signal": f"{r.composite_emoji} {r.composite_label}",
                    "PCR": round(r.pcr, 3) if r.pcr else "—",
                    "Setups": r.total_setups,
                    "Best Setup": f"{bs.direction} {bs.strategy}" if bs else "None",
                    "Best R:R": round(r.best_rr, 2) if r.best_rr else "—",
                    "Conviction": bs.conviction if bs else 0,
                    "Acc Zones": r.accumulation_zones,
                    "Dist Zones": r.distribution_zones,
                    "Support": (
                        round(r.primary_support, 2) if r.primary_support else "—"
                    ),
                    "Resistance": (
                        round(r.primary_resistance, 2) if r.primary_resistance else "—"
                    ),
                    "Max Pain": round(r.max_pain, 2) if r.max_pain else "—",
                    "IV Skew": r.iv_skew_dir.title(),
                    "Demo": r.is_demo,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty and "Conviction" in df.columns:
            df = df.sort_values("Conviction", ascending=False).reset_index(drop=True)
        return df

    def top_setups(self, n: int = 10) -> pd.DataFrame:
        """Top N setups across all symbols, ranked by conviction."""
        if self.all_setups_df.empty:
            return pd.DataFrame()
        return self.all_setups_df.head(n)

    def symbols_with_setups(self) -> list[str]:
        """Symbols that have at least one valid setup."""
        return [
            r.symbol for r in self.results if r.status == "ok" and r.total_setups > 0
        ]

    def symbols_by_signal(self, direction: str = "BULLISH") -> list[str]:
        """Symbols whose composite label contains the given direction string."""
        return [
            r.symbol
            for r in self.results
            if r.status == "ok" and direction.upper() in r.composite_label.upper()
        ]


# ─── default NSE watchlist ────────────────────────────────────────────────────

NSE_LARGE_CAP = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ITC",
    "SBIN",
    "BHARTIARTL",
    "KOTAKBANK",
    "LT",
    "AXISBANK",
    "ASIANPAINT",
    "MARUTI",
    "TITAN",
    "WIPRO",
    "ULTRACEMCO",
    "BAJFINANCE",
    "NESTLEIND",
    "HCLTECH",
]

NSE_MIDCAP = [
    "BANKBARODA",
    "PNB",
    "CANBK",
    "UNIONBANK",
    "IDFCFIRSTB",
    "INDUSINDBK",
    "FEDERALBNK",
    "RBLBANK",
    "YESBANK",
    "BANDHANBNK",
    "TATASTEEL",
    "JSWSTEEL",
    "HINDALCO",
    "VEDL",
    "NATIONALUM",
    "ONGC",
    "BPCL",
    "IOC",
    "GAIL",
    "PETRONET",
]

NSE_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]

DEFAULT_WATCHLIST = NSE_INDICES + NSE_LARGE_CAP[:10]
