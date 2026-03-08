"""
DataManager — File watching, NSE URL download, OHLCV parsing, and demo data generation.
"""

from __future__ import annotations
import io
import os
import re
import time
import logging
import datetime
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─── NSE download URLs ────────────────────────────────────────────────────────

NSE_BULK_URL = "https://archives.nseindia.com/content/equities/bulk.csv"
NSE_BLOCK_URL = "https://archives.nseindia.com/content/equities/block.csv"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com",
    "Accept-Language": "en-US,en;q=0.9",
}


class DataManager:
    """
    Manages all data I/O: options CSVs, deal CSVs, price CSVs,
    NSE bulk/block downloads, and demo-mode synthetic data.
    """

    def __init__(
        self,
        data_folder: str = "./data",
        options_folder: str = "./data/options",
        deals_folder: str = "./data/deals",
        price_folder: str = "./data/price",
    ):
        self.data_folder = Path(data_folder)
        self.options_folder = Path(options_folder)
        self.deals_folder = Path(deals_folder)
        self.price_folder = Path(price_folder)

        for d in [self.options_folder, self.deals_folder, self.price_folder]:
            d.mkdir(parents=True, exist_ok=True)

        self._options_cache: dict[str, Tuple[pd.DataFrame, float]] = {}
        self._deals_cache: Optional[Tuple[pd.DataFrame, float]] = None
        self._price_cache: dict[str, pd.DataFrame] = {}

        self.last_options_ts: Optional[float] = None
        self.last_deals_ts: Optional[float] = None
        self.demo_mode: bool = False

    # ─── options chain ────────────────────────────────────────────────────────

    def scan_options_files(self, symbol: str, date: Optional[datetime.date] = None) -> list[Path]:
        """Return matching options CSV files for a symbol, sorted newest first."""
        pattern = re.compile(
            rf"^{re.escape(symbol.upper())}_options_(\d{{8}})\.csv$", re.IGNORECASE
        )
        files = []
        for f in self.options_folder.iterdir():
            m = pattern.match(f.name)
            if m:
                file_date = datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
                if date is None or file_date == date:
                    files.append((file_date, f))
        files.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in files]

    def load_options_chain(
        self,
        symbol: str,
        date: Optional[datetime.date] = None,
        uploaded_file=None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Load options chain from uploaded file, disk, or generate demo data.
        Returns (df, is_demo).
        """
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            self.last_options_ts = time.time()
            return self._normalise_options(df), False

        files = self.scan_options_files(symbol, date)
        if files:
            latest = files[0]
            df = pd.read_csv(latest)
            self.last_options_ts = time.time()
            return self._normalise_options(df), False

        # Demo mode
        self.demo_mode = True
        return self._generate_demo_options(symbol), True

    def _normalise_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to canonical names if needed."""
        col_map = {
            "strike": "Strike",
            "strike_price": "Strike",
            "expiry": "Expiry",
            "expiry_date": "Expiry",
            "option_type": "OptionType",
            "type": "OptionType",
            "open_interest": "OpenInterest",
            "oi": "OpenInterest",
            "volume": "Volume",
            "vol": "Volume",
            "implied_volatility": "IV",
            "iv": "IV",
            "ltp": "LTP",
            "last_price": "LTP",
            "close": "LTP",
        }
        df = df.rename(columns={c: col_map[c.lower()] for c in df.columns if c.lower() in col_map})
        return df

    # ─── deals data ──────────────────────────────────────────────────────────

    def load_deals(
        self,
        symbol: Optional[str] = None,
        uploaded_file=None,
        download: bool = False,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Load bulk/block deal data. Returns (df, is_demo).
        """
        frames = []

        if uploaded_file is not None:
            frames.append(pd.read_csv(uploaded_file))

        if download:
            for url, label in [(NSE_BULK_URL, "bulk"), (NSE_BLOCK_URL, "block")]:
                try:
                    resp = requests.get(url, headers=NSE_HEADERS, timeout=15)
                    resp.raise_for_status()
                    frame = pd.read_csv(io.StringIO(resp.text))
                    frame["Deal_Type"] = label
                    frames.append(frame)
                    logger.info(f"Downloaded {label} deals from NSE")
                except Exception as e:
                    logger.warning(f"Failed to download {label} deals: {e}")

        # Check local deal files
        for f in sorted(self.deals_folder.iterdir()):
            if f.suffix.lower() == ".csv":
                try:
                    frame = pd.read_csv(f)
                    frames.append(frame)
                except Exception as e:
                    logger.warning(f"Could not read {f}: {e}")

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = self._normalise_deals(df)
            if symbol:
                df = df[df["Symbol"].str.upper() == symbol.upper()]
            self.last_deals_ts = time.time()
            self.demo_mode = False
            return df, False

        # Demo
        self.demo_mode = True
        return self._generate_demo_deals(symbol or "SBIN"), True

    def _normalise_deals(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "date": "Date",
            "symbol": "Symbol",
            "scrip name": "Symbol",
            "client name": "Client Name",
            "buy/sell": "Buy/Sell",
            "quantity traded": "Quantity Traded",
            "qty": "Quantity Traded",
            "trade price/wght. avg. price": "Trade Price/Wght. Avg. Price",
            "price": "Trade Price/Wght. Avg. Price",
            "wght. avg. price": "Trade Price/Wght. Avg. Price",
            "remarks": "Remarks",
        }
        df = df.rename(columns={c: col_map.get(c.lower().strip(), c) for c in df.columns})
        for col in ["Date", "Symbol", "Client Name", "Buy/Sell", "Quantity Traded", "Trade Price/Wght. Avg. Price"]:
            if col not in df.columns:
                df[col] = None
        return df

    # ─── price data ──────────────────────────────────────────────────────────

    def load_price(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load OHLCV price data for symbol from price folder."""
        candidates = list(self.price_folder.glob(f"{symbol.upper()}_price.csv"))
        if not candidates:
            return None
        df = pd.read_csv(candidates[0], parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def get_cmp(self, symbol: str) -> Optional[float]:
        """Return the most recent closing price for the symbol."""
        df = self.load_price(symbol)
        if df is not None and not df.empty and "Close" in df.columns:
            return float(df["Close"].iloc[-1])
        return None

    # ─── historical options ──────────────────────────────────────────────────

    def load_all_options_history(self, symbol: str, max_days: int = 10) -> dict[datetime.date, pd.DataFrame]:
        """Load all available options CSVs for a symbol (up to max_days)."""
        files = self.scan_options_files(symbol)[:max_days]
        history = {}
        for f in files:
            m = re.search(r"_(\d{8})\.csv$", f.name)
            if m:
                d = datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
                history[d] = pd.read_csv(f)
        return history

    # ─── demo data generation ─────────────────────────────────────────────────

    def _generate_demo_options(self, symbol: str = "SBIN") -> pd.DataFrame:
        """
        Synthetic options chain for demo mode.
        Strikes 500–700 (step 10), 3 expiries with realistic OI distribution.
        """
        rng = np.random.default_rng(42)
        strikes = np.arange(500, 710, 10)
        atm = 600
        today = datetime.date.today()
        expiries = [
            (today + datetime.timedelta(weeks=1)).strftime("%d-%b-%Y"),
            (today + datetime.timedelta(weeks=4)).strftime("%d-%b-%Y"),
            (today + datetime.timedelta(weeks=8)).strftime("%d-%b-%Y"),
        ]

        rows = []
        for expiry in expiries:
            for strike in strikes:
                moneyness = (strike - atm) / atm

                for opt_type in ["CE", "PE"]:
                    # Gaussian OI distribution peaked at ATM
                    base_oi = int(1e6 * rng.exponential(0.5) * np.exp(-4 * moneyness**2))
                    if opt_type == "CE":
                        base_oi = int(base_oi * (1 + 0.5 * moneyness))  # more OI above ATM for calls
                    else:
                        base_oi = int(base_oi * (1 - 0.5 * moneyness))  # more OI below ATM for puts

                    base_oi = max(base_oi, 0)

                    # Spike at specific strikes to simulate walls
                    if strike in [580, 620] and opt_type == "PE":
                        base_oi = int(base_oi * rng.uniform(4, 6))
                    if strike in [640, 660] and opt_type == "CE":
                        base_oi = int(base_oi * rng.uniform(4, 6))

                    vol = int(base_oi * rng.uniform(0.05, 0.3))

                    # Black-Scholes-inspired IV
                    sigma = 0.20 + 0.05 * abs(moneyness) + rng.normal(0, 0.01)
                    if opt_type == "PE":
                        sigma += 0.02  # put skew

                    # LTP approximation (intrinsic + time value)
                    itm_intrinsic = max(atm - strike, 0) if opt_type == "PE" else max(strike - atm, 0)
                    time_val = atm * sigma * 0.2
                    ltp = max(itm_intrinsic + time_val * rng.uniform(0.8, 1.2), 0.05)

                    rows.append({
                        "Strike": strike,
                        "Expiry": expiry,
                        "OptionType": opt_type,
                        "OpenInterest": base_oi,
                        "Volume": vol,
                        "IV": round(sigma * 100, 2),
                        "LTP": round(ltp, 2),
                        "Symbol": symbol,
                    })

        return pd.DataFrame(rows)

    def _generate_demo_deals(self, symbol: str = "SBIN") -> pd.DataFrame:
        """
        Synthetic bulk/block deals near wall strikes for demo mode.
        """
        rng = np.random.default_rng(99)
        today = datetime.date.today()
        wall_strikes = [580, 590, 620, 640, 660]

        entities = [
            ("HDFC MUTUAL FUND", "BUY"),
            ("SBI LIFE INSURANCE CO LTD", "BUY"),
            ("ICICI PRUDENTIAL MF", "BUY"),
            ("NOMURA SINGAPORE LIMITED", "SELL"),
            ("BLACKROCK GLOBAL FUNDS FPI", "BUY"),
            ("PROMOTER GROUP ENTITY", "BUY"),
            ("KOTAK SECURITIES LTD", "SELL"),
            ("FRANKLIN TEMPLETON FPI", "BUY"),
        ]

        rows = []
        for strike in wall_strikes:
            for _ in range(rng.integers(1, 4)):
                entity, default_side = entities[rng.integers(0, len(entities))]
                side = default_side if rng.random() > 0.3 else ("SELL" if default_side == "BUY" else "BUY")
                price = strike + rng.uniform(-3, 3)
                qty = int(rng.uniform(100_000, 5_000_000))
                rows.append({
                    "Date": today.strftime("%d-%b-%Y"),
                    "Symbol": symbol,
                    "Client Name": entity,
                    "Buy/Sell": side,
                    "Quantity Traded": qty,
                    "Trade Price/Wght. Avg. Price": round(price, 2),
                    "Remarks": "BLOCK" if rng.random() > 0.5 else "BULK",
                })

        return pd.DataFrame(rows)

    # ─── market hours ─────────────────────────────────────────────────────────

    @staticmethod
    def is_market_open() -> bool:
        """Check if NSE is currently open (09:15–15:30 IST, Mon–Fri)."""
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.datetime.now(ist)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    @staticmethod
    def freshness_badge(ts: Optional[float]) -> str:
        if ts is None:
            return "🔴 No Data"
        age = time.time() - ts
        if age < 300:
            return "🟢 Live (<5 min)"
        elif age < 1800:
            return f"🟡 {int(age/60)} min old"
        else:
            return f"🔴 Stale ({int(age/3600)}h old)"
