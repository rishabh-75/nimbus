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

# Option chain v3 API (working as of 2025)
NSE_OC_CONTRACT_INFO = (
    "https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}"
)
NSE_OC_V3_URL = "https://www.nseindia.com/api/option-chain-v3?type={oc_type}&symbol={symbol}&expiry={expiry}"
NSE_OC_REFERER = "https://www.nseindia.com/option-chain"

# Index symbols → type=Indices; everything else → type=Equities
NSE_INDEX_SYMBOLS = {
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
    "SENSEX",
}

NSE_HEADERS = {
    "Host": "www.nseindia.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": NSE_OC_REFERER,
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
        self.last_download_errors: list = []
        self.last_download_success: list = []
        self.last_deals_ts: Optional[float] = None
        self.demo_mode: bool = False

    # ─── options chain ────────────────────────────────────────────────────────

    def scan_options_files(
        self, symbol: str, date: Optional[datetime.date] = None
    ) -> list[Path]:
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
        auto_download: bool = True,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Load options chain — priority order:
          1. Uploaded file (sidebar uploader)
          2. Matching CSV on disk (./data/options/)
          3. Auto-download from NSE API (if auto_download=True)
          4. Demo data (synthetic fallback)
        Returns (df, is_demo).
        """
        # 1. Uploaded file
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            self.last_options_ts = time.time()
            self.last_download_errors = []
            return self._normalise_options(df), False

        # 2. Disk — any date if no date specified, exact date if specified
        files = self.scan_options_files(symbol, date)
        if not files and date is not None:
            # Also try without date filter (use latest available)
            files = self.scan_options_files(symbol, None)
        if files:
            latest = files[0]
            df = pd.read_csv(latest)
            self.last_options_ts = time.time()
            self.last_download_errors = []
            return self._normalise_options(df), False

        # 3. Auto-download from NSE
        if auto_download:
            df, msg = self.download_options_chain(symbol)
            if df is not None and not df.empty:
                return df, False

        # 4. Demo fallback
        self.demo_mode = True
        return self._generate_demo_options(symbol), True

    # ── single symbol ─────────────────────────────────────────────────────────

    def download_options_chain(self, symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Download the complete options chain for `symbol` across ALL expiries.

        Flow (proven working pattern):
          1. GET /option-chain page with allow_redirects=False → grabs session cookies
          2. GET /api/option-chain-contract-info → full expiry date list
          3. GET /api/option-chain-v3 for EVERY expiry (sequential, rate-limited)
          4. Merge all expiries → save to ./data/options/{SYMBOL}_options_{DATE}.csv

        Returns (DataFrame, status_message).
        """
        symbol = symbol.upper().strip()
        self.last_download_errors = []
        self.last_download_success = []

        try:
            session, header, cookies = self._make_nse_session()
        except Exception as e:
            err = f"NSE session init failed: {e}"
            self.last_download_errors.append(err)
            return None, err

        # Step 2 — expiry list
        try:
            expiries = self._get_expiry_dates(session, header, cookies, symbol)
        except Exception as e:
            err = f"Could not fetch expiry list for {symbol}: {e}"
            self.last_download_errors.append(err)
            return None, err

        if not expiries:
            err = f"NSE returned no expiry dates for {symbol}"
            self.last_download_errors.append(err)
            return None, err

        chain_type = "Indices" if symbol in NSE_INDEX_SYMBOLS else "Equities"
        logger.info(f"{symbol} ({chain_type}): {len(expiries)} expiries")

        # Step 3 — fetch each expiry (sequential + 0.4 s delay = ~150 req/min, well within NSE limit)
        all_rows: list[dict] = []
        failed: list[str] = []
        for expiry in expiries:
            try:
                raw = self._fetch_chain_for_expiry(
                    session, header, cookies, symbol, expiry, chain_type
                )
                rows = self._rows_from_chain(raw, symbol, expiry)
                all_rows.extend(rows)
                logger.debug(f"  {symbol} {expiry}: {len(rows)} legs")
            except Exception as e:
                failed.append(f"{expiry}: {e}")
                logger.warning(f"  Failed {symbol} {expiry}: {e}")
            time.sleep(0.4)  # polite rate limit

        if not all_rows:
            err = f"No data parsed for {symbol}. Failures: {failed[:3]}"
            self.last_download_errors.append(err)
            return None, err

        df = self._finalise_oc_df(all_rows)

        today_str = datetime.date.today().strftime("%Y%m%d")
        save_path = self.options_folder / f"{symbol}_options_{today_str}.csv"
        df.to_csv(save_path, index=False)

        msg = (
            f"{symbol}: {len(df):,} rows, {len(expiries)-len(failed)}/{len(expiries)} expiries"
            + (f" ({len(failed)} failed)" if failed else "")
            + f" → {save_path.name}"
        )
        self.last_download_success.append(msg)
        self.last_options_ts = time.time()
        logger.info(msg)
        return df, msg

    # ── watchlist bulk download (async via ThreadPoolExecutor) ────────────────

    def download_watchlist_chains(
        self,
        symbols: list[str],
        max_workers: int = 3,
        inter_symbol_delay: float = 1.5,
        progress_callback=None,
    ) -> dict[str, Tuple[Optional[pd.DataFrame], str]]:
        """
        Download options chains for a list of symbols concurrently.

        Uses ThreadPoolExecutor with `max_workers` parallel threads.
        Each thread has its own NSE session (separate cookie jar).
        A per-symbol delay staggers session creation to avoid simultaneous
        cookie requests triggering NSE rate limiting.

        Args:
            symbols:              List of NSE symbols.
            max_workers:          Parallel threads (keep ≤ 4 to stay under NSE rate limit).
            inter_symbol_delay:   Seconds to stagger thread start times.
            progress_callback:    Optional fn(symbol, status, completed, total).

        Returns:
            dict {symbol: (DataFrame | None, status_message)}
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        symbols = [s.upper().strip() for s in symbols if s.strip()]
        total = len(symbols)
        results = {}
        lock = threading.Lock()
        completed = [0]

        def _worker(
            sym: str, stagger_idx: int
        ) -> Tuple[str, Optional[pd.DataFrame], str]:
            # Stagger session creation so not all threads hammer NSE simultaneously
            time.sleep(stagger_idx * inter_symbol_delay)
            dm = DataManager(
                options_folder=str(self.options_folder),
                deals_folder=str(self.deals_folder),
                price_folder=str(self.price_folder),
            )
            df, msg = dm.download_options_chain(sym)
            return sym, df, msg

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, sym, i): sym for i, sym in enumerate(symbols)
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    sym_out, df, msg = future.result()
                    results[sym_out] = (df, msg)
                    status = "ok" if df is not None else "error"
                except Exception as e:
                    results[sym] = (None, str(e))
                    status = "error"

                with lock:
                    completed[0] += 1
                if progress_callback:
                    progress_callback(sym, status, completed[0], total)

        # Merge errors/successes into self for UI display
        self.last_download_errors = [
            msg for _, (df, msg) in results.items() if df is None
        ]
        self.last_download_success = [
            msg for _, (df, msg) in results.items() if df is not None
        ]
        return results

    # ── static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _finalise_oc_df(rows: list[dict]) -> pd.DataFrame:
        """Convert list of row dicts to typed DataFrame."""
        df = pd.DataFrame(rows)
        for col in [
            "Strike",
            "OpenInterest",
            "OI_Change",
            "Volume",
            "IV",
            "LTP",
            "BidQty",
            "AskQty",
            "Change",
            "PctChange",
            "UnderlyingValue",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    # ─── working NSE session (Firefox headers + allow_redirects=False) ──────────

    def _make_nse_session(self) -> tuple:
        """
        Build a requests.Session with NSE cookies.
        Key: use allow_redirects=False on the Referer page — this is what makes
        NSE hand over the real session cookie instead of a bot-detection redirect.
        Returns (session, cookies_dict).
        """
        session = requests.Session()
        header = {
            "Host": "www.nseindia.com",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) "
                "Gecko/20100101 Firefox/82.0"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Referer": "https://www.nseindia.com/option-chain",
        }
        resp = session.get(
            header["Referer"],
            headers=header,
            allow_redirects=False,
            timeout=15,
        )
        cookies = dict(resp.cookies)
        return session, header, cookies

    def _get_expiry_dates(
        self, session, header: dict, cookies: dict, symbol: str
    ) -> list[str]:
        """
        Fetch all expiry dates for a symbol via the contract-info endpoint.
        Returns list of expiry date strings e.g. ["27-Mar-2025", "03-Apr-2025", ...]
        """
        url = (
            f"https://www.nseindia.com/api/option-chain-contract-info"
            f"?symbol={symbol}"
        )
        resp = session.get(
            url,
            headers=header,
            cookies=cookies,
            allow_redirects=False,
            timeout=15,
        )
        data = resp.json()
        return data.get("expiryDates", [])

    def _fetch_chain_for_expiry(
        self,
        session,
        header: dict,
        cookies: dict,
        symbol: str,
        expiry: str,
        chain_type: str = "Indices",
    ) -> list[dict]:
        """
        Fetch option chain rows for one symbol + expiry via option-chain-v3.
        chain_type: "Indices" for index symbols, "Equities" for stocks.
        Returns raw list of row dicts.
        """
        url = (
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type={chain_type}&symbol={symbol}&expiry={expiry}"
        )
        resp = session.get(
            url,
            headers=header,
            cookies=cookies,
            allow_redirects=False,
            timeout=20,
        )
        data = resp.json()
        return data.get("data", []) or data.get("records", {}).get("data", []) or []

    @staticmethod
    def _rows_from_chain(raw_rows: list, symbol: str, expiry: str) -> list[dict]:
        """Parse raw option-chain-v3 rows into canonical flat dicts."""
        rows = []
        for item in raw_rows:
            strike = item.get("strikePrice", 0)
            for opt_type in ("CE", "PE"):
                leg = item.get(opt_type)
                if not leg:
                    continue
                rows.append(
                    {
                        "Symbol": symbol,
                        "Strike": strike,
                        "Expiry": expiry,
                        "OptionType": opt_type,
                        "OpenInterest": leg.get("openInterest", 0),
                        "OI_Change": leg.get("changeinOpenInterest", 0),
                        "Volume": leg.get("totalTradedVolume", 0),
                        "IV": leg.get("impliedVolatility", 0),
                        "LTP": leg.get("lastPrice", 0),
                        "BidQty": leg.get("bidQty", 0),
                        "AskQty": leg.get("askQty", 0),
                        "Change": leg.get("change", 0),
                        "PctChange": leg.get("pChange", 0),
                        "UnderlyingValue": item.get("PE", {}).get("underlyingValue", 0)
                        or item.get("CE", {}).get("underlyingValue", 0),
                    }
                )
        return rows

    def _normalise_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept multiple CSV formats and normalise to canonical columns:
        Strike | Expiry | OptionType | OpenInterest | Volume | IV | LTP | Symbol

        Supported formats:
          A. Canonical (already correct column names)
          B. NSE Bhavcopy wide format (CE.OI, PE.OI columns per row)
          C. NSE option chain page download (wide with CE/PE prefixed columns)
          D. Generic lowercase rename
        """
        # --- Format B/C: NSE wide format where each row = one strike,
        #     columns like "CALLS_OI", "PUTS_OI" or "CE OI", "PE OI"
        col_lower = {c.lower().strip(): c for c in df.columns}
        wide_indicators = [
            k
            for k in col_lower
            if any(
                x in k
                for x in [
                    "calls_oi",
                    "puts_oi",
                    "ce oi",
                    "pe oi",
                    "call oi",
                    "put oi",
                    "ce_oi",
                    "pe_oi",
                ]
            )
        ]
        if wide_indicators:
            return self._parse_nse_wide_format(df)

        # --- Format D: Generic lowercase rename
        col_map = {
            "strike": "Strike",
            "strike price": "Strike",
            "strike_price": "Strike",
            "strikeprice": "Strike",
            "expiry": "Expiry",
            "expiry date": "Expiry",
            "expiry_date": "Expiry",
            "expirydate": "Expiry",
            "option_type": "OptionType",
            "option type": "OptionType",
            "type": "OptionType",
            "optiontype": "OptionType",
            "instrument": "OptionType",
            "open_interest": "OpenInterest",
            "openinterest": "OpenInterest",
            "oi": "OpenInterest",
            "open interest": "OpenInterest",
            "chng in oi": "OI_Change",
            "change in oi": "OI_Change",
            "volume": "Volume",
            "vol": "Volume",
            "traded volume": "Volume",
            "totaltradesvolume": "Volume",
            "tot trd vol": "Volume",
            "implied_volatility": "IV",
            "impliedvolatility": "IV",
            "iv": "IV",
            "impl. vlt. (%)": "IV",
            "ltp": "LTP",
            "last_price": "LTP",
            "last price": "LTP",
            "close": "LTP",
            "closing price": "LTP",
        }
        df = df.rename(
            columns={
                c: col_map[c.lower().strip()]
                for c in df.columns
                if c.lower().strip() in col_map
            }
        )

        # Normalise OptionType values
        if "OptionType" in df.columns:
            df["OptionType"] = df["OptionType"].astype(str).str.upper().str.strip()
            df["OptionType"] = df["OptionType"].replace(
                {
                    "CALL": "CE",
                    "PUT": "PE",
                    "C": "CE",
                    "P": "PE",
                    "OPTIDX": "CE",  # sometimes NSE bhavcopy has instrument type
                }
            )
            # If still not CE/PE try extracting from instrument name like "NIFTY26MAR24C22000"
            mask = ~df["OptionType"].isin(["CE", "PE"])
            if mask.any() and "Strike" in df.columns:
                df.loc[
                    mask & df["OptionType"].str.contains("C", na=False), "OptionType"
                ] = "CE"
                df.loc[
                    mask & df["OptionType"].str.contains("P", na=False), "OptionType"
                ] = "PE"

        # Numeric coercions
        for col in ["Strike", "OpenInterest", "Volume", "IV", "LTP"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def _parse_nse_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert NSE option-chain wide CSV (one row per strike, CE+PE side by side)
        into long format (one row per strike+type).

        NSE website Download CSV columns look like:
        Expiry Date | Strike Price | CE OI | CE Chng OI | CE Volume | CE IV | CE LTP | ... |
        PE LTP | PE IV | PE Volume | PE Chng OI | PE OI | ...
        """
        # Normalise column headers
        df.columns = [str(c).strip() for c in df.columns]
        col_lower = {c.lower(): c for c in df.columns}

        # Find strike and expiry
        strike_col = next((col_lower[k] for k in col_lower if "strike" in k), None)
        expiry_col = next((col_lower[k] for k in col_lower if "expiry" in k), None)

        if strike_col is None:
            raise ValueError("Cannot find Strike column in wide-format NSE CSV")

        def _find(keywords):
            for kw in keywords:
                for k, orig in col_lower.items():
                    if kw in k:
                        return orig
            return None

        # CE side
        ce_oi = _find(["ce oi", "calls_oi", "call oi", "ce_oi"])
        ce_vol = _find(["ce volume", "calls_vol", "ce_vol", "ce vol"])
        ce_iv = _find(["ce iv", "ce impl", "calls_iv", "ce_iv"])
        ce_ltp = _find(["ce ltp", "ce last", "calls_ltp", "ce_ltp"])

        # PE side
        pe_oi = _find(["pe oi", "puts_oi", "put oi", "pe_oi"])
        pe_vol = _find(["pe volume", "puts_vol", "pe_vol", "pe vol"])
        pe_iv = _find(["pe iv", "pe impl", "puts_iv", "pe_iv"])
        pe_ltp = _find(["pe ltp", "pe last", "puts_ltp", "pe_ltp"])

        rows = []
        for _, row in df.iterrows():
            strike = row[strike_col]
            expiry = row[expiry_col] if expiry_col else ""
            for opt_type, oi_c, vol_c, iv_c, ltp_c in [
                ("CE", ce_oi, ce_vol, ce_iv, ce_ltp),
                ("PE", pe_oi, pe_vol, pe_iv, pe_ltp),
            ]:
                rows.append(
                    {
                        "Strike": pd.to_numeric(strike, errors="coerce"),
                        "Expiry": expiry,
                        "OptionType": opt_type,
                        "OpenInterest": (
                            pd.to_numeric(row[oi_c], errors="coerce") if oi_c else 0
                        ),
                        "Volume": (
                            pd.to_numeric(row[vol_c], errors="coerce") if vol_c else 0
                        ),
                        "IV": pd.to_numeric(row[iv_c], errors="coerce") if iv_c else 0,
                        "LTP": (
                            pd.to_numeric(row[ltp_c], errors="coerce") if ltp_c else 0
                        ),
                    }
                )
        result = pd.DataFrame(rows).fillna(0)
        return result

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
            self.last_download_errors = []
            self.last_download_success = []

            for url, label in [(NSE_BULK_URL, "bulk"), (NSE_BLOCK_URL, "block")]:
                result = self._try_nse_download(url, label)
                if result is not None:
                    frames.append(result)
                    # Save to disk for offline reuse
                    save_path = (
                        self.deals_folder
                        / f"nse_{label}_{datetime.date.today().strftime('%Y%m%d')}.csv"
                    )
                    try:
                        result.to_csv(save_path, index=False)
                    except Exception:
                        pass

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
        df = df.rename(
            columns={c: col_map.get(c.lower().strip(), c) for c in df.columns}
        )
        for col in [
            "Date",
            "Symbol",
            "Client Name",
            "Buy/Sell",
            "Quantity Traded",
            "Trade Price/Wght. Avg. Price",
        ]:
            if col not in df.columns:
                df[col] = None
        return df

    def _try_nse_download(self, url: str, label: str) -> Optional[pd.DataFrame]:
        """
        Try multiple strategies to download an NSE CSV.
        NSE uses Cloudflare + cookie-based bot protection, so we try:
          1. Full browser session (homepage → data URL)
          2. Direct with archive-specific headers
          3. Legacy URL variant
        Returns DataFrame on success, None on failure (error stored in self.last_download_errors).
        """
        strategies = [
            self._nse_strategy_session,
            self._nse_strategy_direct,
            self._nse_strategy_archive,
        ]
        for strategy in strategies:
            try:
                df = strategy(url, label)
                if df is not None and not df.empty:
                    df["Deal_Type"] = label
                    self.last_download_success.append(
                        f"{label}: {len(df)} rows downloaded"
                    )
                    logger.info(
                        f"NSE {label} downloaded via {strategy.__name__}: {len(df)} rows"
                    )
                    return df
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed for {label}: {e}")
                continue

        msg = (
            f"NSE {label} download blocked. NSE requires a browser session. "
            f"Download manually: {url}"
        )
        self.last_download_errors.append(msg)
        logger.warning(msg)
        return None

    def _nse_strategy_session(self, url: str, label: str) -> Optional[pd.DataFrame]:
        """Strategy 1: prime session via homepage first (gets cookies)."""
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com", timeout=12)
        time.sleep(1.5)
        session.get("https://www.nseindia.com/market-data/bulk-block-deals", timeout=10)
        time.sleep(1.0)
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return self._parse_nse_csv(resp, label)

    def _nse_strategy_direct(self, url: str, label: str) -> Optional[pd.DataFrame]:
        """Strategy 2: direct download with archive-specific headers."""
        headers = {
            **NSE_HEADERS,
            "Host": "archives.nseindia.com",
            "Origin": "https://www.nseindia.com",
            "Referer": "https://www.nseindia.com/market-data/bulk-block-deals",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-site",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        resp = requests.get(url, headers=headers, timeout=25)
        resp.raise_for_status()
        return self._parse_nse_csv(resp, label)

    def _nse_strategy_archive(self, url: str, label: str) -> Optional[pd.DataFrame]:
        """Strategy 3: dated archive URL for today."""
        today = datetime.date.today()
        dated_url = (
            f"https://archives.nseindia.com/content/equities/"
            f"{label}deals{today.strftime('%d%m%Y')}.csv"
        )
        resp = requests.get(dated_url, headers=NSE_HEADERS, timeout=20)
        resp.raise_for_status()
        return self._parse_nse_csv(resp, label)

    @staticmethod
    def _parse_nse_csv(resp: requests.Response, label: str) -> Optional[pd.DataFrame]:
        """Parse response text as CSV; validate it looks like deal data."""
        text = resp.text.strip()
        if not text or len(text) < 30:
            raise ValueError(f"Empty response body for {label}")
        # NSE sometimes returns HTML error pages
        if text.lstrip().startswith("<"):
            raise ValueError(f"Got HTML instead of CSV (bot block) for {label}")
        df = pd.read_csv(io.StringIO(text))
        if df.empty or len(df.columns) < 3:
            raise ValueError(f"CSV too sparse for {label}")
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

    def get_cmp(
        self, symbol: str, options_df: Optional[pd.DataFrame] = None
    ) -> Optional[float]:
        """
        Return current market price for the symbol.
        Priority:
          1. Price CSV (most accurate)
          2. UnderlyingValue column in options chain (set by NSE in live data)
          3. OI-weighted midpoint of ATM strikes (fallback for demo/uploaded data)
        """
        # 1. Price file
        df = self.load_price(symbol)
        if df is not None and not df.empty and "Close" in df.columns:
            return float(df["Close"].iloc[-1])

        # 2. UnderlyingValue from options chain
        if options_df is not None and not options_df.empty:
            if "UnderlyingValue" in options_df.columns:
                vals = pd.to_numeric(
                    options_df["UnderlyingValue"], errors="coerce"
                ).dropna()
                vals = vals[vals > 0]
                if not vals.empty:
                    return float(vals.median())

            # 3. OI-weighted midpoint of strikes
            if "Strike" in options_df.columns and "OpenInterest" in options_df.columns:
                strikes = pd.to_numeric(options_df["Strike"], errors="coerce")
                oi = pd.to_numeric(options_df["OpenInterest"], errors="coerce").fillna(
                    0
                )
                total = oi.sum()
                if total > 0:
                    return float((strikes * oi).sum() / total)

        return None

    # ─── historical options ──────────────────────────────────────────────────

    def load_all_options_history(
        self, symbol: str, max_days: int = 10
    ) -> dict[datetime.date, pd.DataFrame]:
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
                    base_oi = int(
                        1e6 * rng.exponential(0.5) * np.exp(-4 * moneyness**2)
                    )
                    if opt_type == "CE":
                        base_oi = int(
                            base_oi * (1 + 0.5 * moneyness)
                        )  # more OI above ATM for calls
                    else:
                        base_oi = int(
                            base_oi * (1 - 0.5 * moneyness)
                        )  # more OI below ATM for puts

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
                    itm_intrinsic = (
                        max(atm - strike, 0)
                        if opt_type == "PE"
                        else max(strike - atm, 0)
                    )
                    time_val = atm * sigma * 0.2
                    ltp = max(itm_intrinsic + time_val * rng.uniform(0.8, 1.2), 0.05)

                    rows.append(
                        {
                            "Strike": strike,
                            "Expiry": expiry,
                            "OptionType": opt_type,
                            "OpenInterest": base_oi,
                            "Volume": vol,
                            "IV": round(sigma * 100, 2),
                            "LTP": round(ltp, 2),
                            "Symbol": symbol,
                        }
                    )

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
                side = (
                    default_side
                    if rng.random() > 0.3
                    else ("SELL" if default_side == "BUY" else "BUY")
                )
                price = strike + rng.uniform(-3, 3)
                qty = int(rng.uniform(100_000, 5_000_000))
                rows.append(
                    {
                        "Date": today.strftime("%d-%b-%Y"),
                        "Symbol": symbol,
                        "Client Name": entity,
                        "Buy/Sell": side,
                        "Quantity Traded": qty,
                        "Trade Price/Wght. Avg. Price": round(price, 2),
                        "Remarks": "BLOCK" if rng.random() > 0.5 else "BULK",
                    }
                )

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
