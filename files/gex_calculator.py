"""
GEXCalculator — Gamma Exposure analysis for NSE options chains.

GEX (Gamma Exposure) measures how much the market maker's delta hedging
will amplify or dampen price moves at each strike:

    GEX_call = +Gamma × OI × LotSize × Spot²  (dealers short calls → long gamma)
    GEX_put  = -Gamma × OI × LotSize × Spot²  (dealers long puts  → short gamma)
    Net GEX  = GEX_call + GEX_put

Key levels derived from GEX:
  - Call Resistance  : strike with most positive GEX above spot (hard ceiling)
  - Put Support      : strike with most negative GEX below spot (hard floor)
  - HVL (High Volatility Level) : strike where cumulative GEX crosses zero
    - Above HVL → dealers long gamma → they DAMPEN moves (range bound)
    - Below HVL → dealers short gamma → they AMPLIFY moves (trending)
  - GEX Zero Cross   : same as HVL, key inflection

Per-expiry analysis surfaces which expiry is driving the most gamma pressure.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─── Black-Scholes gamma ──────────────────────────────────────────────────────


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes gamma for a European option.
    S=spot, K=strike, T=time to expiry (years), r=risk-free rate, sigma=IV (decimal).
    Returns 0 on any domain error.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        pdf_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        return pdf_d1 / (S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def _days_to_expiry(expiry_str: str) -> float:
    """Parse NSE expiry date string and return calendar days as float."""
    import datetime

    today = datetime.date.today()
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            exp_date = datetime.datetime.strptime(str(expiry_str).strip(), fmt).date()
            dte = (exp_date - today).days
            return max(dte, 0) / 365.0
        except ValueError:
            continue
    return 30 / 365.0  # fallback: 1 month


# ─── result containers ────────────────────────────────────────────────────────


@dataclass
class ExpiryGEX:
    expiry: str
    dte_days: int
    strikes_df: pd.DataFrame  # Strike | GEX_Call | GEX_Put | Net_GEX | ...
    total_gex: float  # sum of abs(net_gex) — "GEX expiring"
    net_gex: float  # signed net
    call_resistance: Optional[float]
    put_support: Optional[float]
    hvl: Optional[float]  # High Volatility Level
    gex_pct_of_total: float = 0.0  # share of total chain GEX


@dataclass
class GEXResult:
    symbol: str
    spot: float
    risk_free: float
    lot_size: int
    expiries: list[ExpiryGEX]
    all_strikes: pd.DataFrame  # aggregated across all expiries

    # Convenience accessors
    # "live" = DTE > 0 (not already expired today)
    @property
    def _live(self) -> list:
        """Expiries with DTE > 0, sorted nearest first. Falls back to all if none alive."""
        live = [e for e in self.expiries if e.dte_days > 0]
        return live if live else self.expiries

    @property
    def first_expiry(self) -> Optional["ExpiryGEX"]:
        """Nearest expiry that hasn't expired yet."""
        return self._live[0] if self._live else None

    @property
    def next_expiry(self) -> Optional["ExpiryGEX"]:
        """Second nearest live expiry."""
        live = self._live
        return live[1] if len(live) > 1 else None

    @property
    def highest_gex_expiry(self) -> Optional["ExpiryGEX"]:
        """Live expiry with the largest absolute GEX."""
        if not self._live:
            return None
        return max(self._live, key=lambda e: abs(e.total_gex))

    @property
    def second_highest_gex_expiry(self) -> Optional["ExpiryGEX"]:
        """Live expiry with 2nd largest absolute GEX."""
        live = self._live
        if len(live) < 2:
            return None
        ranked = sorted(live, key=lambda e: abs(e.total_gex), reverse=True)
        return ranked[1]

    @property
    def overall_hvl(self) -> Optional[float]:
        return self.all_strikes.attrs.get("hvl")

    @property
    def overall_call_resistance(self) -> Optional[float]:
        return self.all_strikes.attrs.get("call_resistance")

    @property
    def overall_put_support(self) -> Optional[float]:
        return self.all_strikes.attrs.get("put_support")


# ─── lot size lookup ──────────────────────────────────────────────────────────

NSE_LOT_SIZES: dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 15,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 75,
    "NIFTYNXT50": 25,
    "SENSEX": 10,
    "BANKEX": 15,
    "RELIANCE": 250,
    "TCS": 150,
    "INFY": 300,
    "HDFCBANK": 550,
    "ICICIBANK": 700,
    "SBIN": 1500,
    "AXISBANK": 625,
    "KOTAKBANK": 400,
    "LT": 175,
    "BAJFINANCE": 125,
    "BHARTIARTL": 475,
    "MARUTI": 100,
    "TITAN": 375,
    "WIPRO": 1500,
    "ITC": 3200,
    "HINDUNILVR": 300,
    "ASIANPAINT": 300,
    "ULTRACEMCO": 100,
    "NESTLEIND": 50,
    "HCLTECH": 700,
    "TATASTEEL": 5500,
    "JSWSTEEL": 1350,
    "HINDALCO": 2150,
    "ONGC": 3850,
    "BPCL": 1800,
    "IOC": 4750,
}

DEFAULT_LOT_SIZE = 500


# ─── main calculator ──────────────────────────────────────────────────────────


class GEXCalculator:
    """
    Computes Gamma Exposure from an options chain DataFrame.

    Required columns: Strike, Expiry, OptionType (CE/PE), OpenInterest, IV
    Optional:         LTP (used only for fallback IV estimation)

    Usage:
        calc = GEXCalculator(options_df, symbol="NIFTY", spot=22000)
        result = calc.compute()
        # result.first_expiry, result.highest_gex_expiry, result.all_strikes, ...
    """

    RISK_FREE = 0.065  # RBI repo rate proxy

    def __init__(
        self,
        options_df: pd.DataFrame,
        symbol: str,
        spot: Optional[float] = None,
        lot_size: Optional[int] = None,
        risk_free: float = RISK_FREE,
        min_iv: float = 0.01,  # floor IV to avoid BS blowup
        max_dte_days: int = 120,  # ignore LEAPs / very far expiries
    ):
        self.options_df = options_df.copy()
        self.symbol = symbol.upper()
        self.spot = spot or self._infer_spot()
        self.lot_size = lot_size or NSE_LOT_SIZES.get(self.symbol, DEFAULT_LOT_SIZE)
        self.risk_free = risk_free
        self.min_iv = min_iv
        self.max_dte = max_dte_days

    def _infer_spot(self) -> float:
        """Estimate spot from UnderlyingValue column or ATM strike."""
        df = self.options_df
        if "UnderlyingValue" in df.columns:
            vals = pd.to_numeric(df["UnderlyingValue"], errors="coerce").dropna()
            if not vals.empty and vals.max() > 0:
                return float(vals.median())
        # Fallback: weighted average of ATM strikes by OI
        if "OpenInterest" in df.columns and "Strike" in df.columns:
            strikes = pd.to_numeric(df["Strike"], errors="coerce").dropna()
            oi = pd.to_numeric(df["OpenInterest"], errors="coerce").fillna(0)
            if oi.sum() > 0:
                return float((strikes * oi).sum() / oi.sum())
        return 0.0

    def compute(self) -> GEXResult:
        df = self._prepare()
        if df.empty:
            return GEXResult(
                symbol=self.symbol,
                spot=self.spot,
                risk_free=self.risk_free,
                lot_size=self.lot_size,
                expiries=[],
                all_strikes=pd.DataFrame(),
            )

        expiry_results: list[ExpiryGEX] = []
        total_abs_gex = 0.0

        # Process each expiry independently
        for expiry, grp in df.groupby("Expiry", sort=False):
            dte_years = _days_to_expiry(str(expiry))
            dte_days = int(dte_years * 365)
            if dte_days > self.max_dte:
                continue

            strike_gex = self._compute_strike_gex(grp, dte_years)
            if strike_gex.empty:
                continue

            total_abs = float(strike_gex["Net_GEX"].abs().sum())
            net = float(strike_gex["Net_GEX"].sum())
            total_abs_gex += total_abs

            cr, ps, hvl = self._key_levels(strike_gex)

            expiry_results.append(
                ExpiryGEX(
                    expiry=str(expiry),
                    dte_days=dte_days,
                    strikes_df=strike_gex,
                    total_gex=total_abs,
                    net_gex=net,
                    call_resistance=cr,
                    put_support=ps,
                    hvl=hvl,
                )
            )

        # Sort by DTE ascending
        expiry_results.sort(key=lambda e: e.dte_days)

        # Compute pct_of_total for each expiry
        for e in expiry_results:
            e.gex_pct_of_total = (
                (e.total_gex / total_abs_gex * 100) if total_abs_gex > 0 else 0.0
            )

        # Aggregate all expiries into one DataFrame
        all_strikes = self._aggregate_all(expiry_results)

        return GEXResult(
            symbol=self.symbol,
            spot=self.spot,
            risk_free=self.risk_free,
            lot_size=self.lot_size,
            expiries=expiry_results,
            all_strikes=all_strikes,
        )

    # ── internal helpers ───────────────────────────────────────────────────────

    def _prepare(self) -> pd.DataFrame:
        df = self.options_df.copy()
        for col in ["Strike", "OpenInterest", "IV"]:
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Strike", "OpenInterest"])
        df["IV"] = df["IV"].fillna(0)
        # Convert % IV to decimal if > 1 (NSE gives IV as 14.5, not 0.145)
        df.loc[df["IV"] > 1, "IV"] = df.loc[df["IV"] > 1, "IV"] / 100.0
        df["IV"] = df["IV"].clip(lower=self.min_iv)
        df = df[df["OpenInterest"] > 0]

        if "OptionType" not in df.columns:
            return pd.DataFrame()
        df["OptionType"] = df["OptionType"].str.upper().str.strip()
        df = df[df["OptionType"].isin(["CE", "PE"])]

        if "Expiry" not in df.columns:
            df["Expiry"] = "Unknown"

        return df.reset_index(drop=True)

    def _compute_strike_gex(self, grp: pd.DataFrame, T: float) -> pd.DataFrame:
        """Compute net GEX per strike for one expiry group."""
        S = self.spot
        r = self.risk_free
        L = self.lot_size

        rows = []
        for strike, sub in grp.groupby("Strike"):
            ce = sub[sub["OptionType"] == "CE"]
            pe = sub[sub["OptionType"] == "PE"]

            ce_oi = float(ce["OpenInterest"].sum()) if not ce.empty else 0
            pe_oi = float(pe["OpenInterest"].sum()) if not pe.empty else 0
            ce_iv = float(ce["IV"].mean()) if not ce.empty else self.min_iv
            pe_iv = float(pe["IV"].mean()) if not pe.empty else self.min_iv

            ce_gamma = _bs_gamma(S, float(strike), T, r, ce_iv)
            pe_gamma = _bs_gamma(S, float(strike), T, r, pe_iv)

            # GEX convention: dealers short calls (long gamma) → positive
            #                 dealers long puts  (short gamma) → negative
            # ÷1e6 gives M-range values (avoids G/T prefix on axis)
            gex_call = +ce_gamma * ce_oi * L * S * S / 1e6
            gex_put = -pe_gamma * pe_oi * L * S * S / 1e6

            rows.append(
                {
                    "Strike": float(strike),
                    "GEX_Call": gex_call,
                    "GEX_Put": gex_put,
                    "Net_GEX": gex_call + gex_put,
                    "CE_OI": ce_oi,
                    "PE_OI": pe_oi,
                    "CE_IV": ce_iv * 100,
                    "PE_IV": pe_iv * 100,
                    "CE_Gamma": ce_gamma,
                    "PE_Gamma": pe_gamma,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
        # Cumulative GEX profile (sorted by strike desc → top to bottom like the chart)
        df["Cumulative_GEX"] = df["Net_GEX"].cumsum()
        return df

    def _key_levels(
        self, strike_df: pd.DataFrame
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Derive Call Resistance, Put Support, and HVL from a per-strike GEX frame.

        Key levels are anchored to OI (same as OptionsWallCalculator) so the
        GEX tab and Wall Overview always agree on which strike is support/resistance.
        GEX magnitude is used only for HVL (which is a gamma-specific concept).
        """
        if strike_df.empty or self.spot <= 0:
            return None, None, None

        S = self.spot

        # Call Resistance: highest CE_OI wall at or above spot  (matches Wall Overview)
        cr = None
        if "CE_OI" in strike_df.columns:
            above = strike_df[strike_df["Strike"] >= S]
            pool = above if not above.empty else strike_df
            cr = float(pool.loc[pool["CE_OI"].idxmax(), "Strike"])

        # Put Support: highest PE_OI wall at or below spot  (matches Wall Overview)
        ps = None
        if "PE_OI" in strike_df.columns:
            below = strike_df[strike_df["Strike"] <= S]
            pool = below if not below.empty else strike_df
            ps = float(pool.loc[pool["PE_OI"].idxmax(), "Strike"])

        # HVL: where cumulative Net_GEX (sorted strike ascending) crosses zero
        # Sort ascending, find first zero-crossing
        df_asc = strike_df.sort_values("Strike")
        cumsum = df_asc["Net_GEX"].cumsum().values
        strikes = df_asc["Strike"].values
        hvl = None
        for i in range(1, len(cumsum)):
            if cumsum[i - 1] * cumsum[i] <= 0:
                # Linear interpolation between bracketing strikes
                w = abs(cumsum[i - 1]) / (abs(cumsum[i - 1]) + abs(cumsum[i]) + 1e-12)
                hvl = float(strikes[i - 1] * (1 - w) + strikes[i] * w)
                break

        return cr, ps, hvl

    def _aggregate_all(self, expiries: list[ExpiryGEX]) -> pd.DataFrame:
        """Sum GEX across all expiries per strike."""
        if not expiries:
            return pd.DataFrame()

        frames = [e.strikes_df.assign(Expiry=e.expiry) for e in expiries]
        combined = pd.concat(frames, ignore_index=True)

        agg = (
            combined.groupby("Strike")
            .agg(
                GEX_Call=("GEX_Call", "sum"),
                GEX_Put=("GEX_Put", "sum"),
                Net_GEX=("Net_GEX", "sum"),
                CE_OI=("CE_OI", "sum"),
                PE_OI=("PE_OI", "sum"),
            )
            .reset_index()
            .sort_values("Strike")
        )
        agg["Cumulative_GEX"] = agg["Net_GEX"].cumsum()

        # Attach key levels as DataFrame attrs for easy access
        cr, ps, hvl = self._key_levels(agg)
        agg.attrs["call_resistance"] = cr
        agg.attrs["put_support"] = ps
        agg.attrs["hvl"] = hvl

        return agg
