"""
modules/sector_map.py
─────────────────────
Sector ticker registry for NIMBUS Market Context tab.

Tickers:
  ^CNX* / ^NSE*  — NSE index tickers (yfinance confirmed)
  *.NS           — NSE ETF tickers (yfinance confirmed)

Nifty 500 benchmark: MONIFTY500.NS (MO ETF) → fallback ^NSEI
"""

from __future__ import annotations

# ── Market benchmark ──────────────────────────────────────────────────────────
MARKET_TICKER = "MONIFTY500.NS"  # Motilal Oswal Nifty 500 ETF
MARKET_FALLBACK = "^NSEI"  # Nifty 50 fallback

# ── Sector map: yfinance_ticker → (Display Name, category) ───────────────────
# Category is used for future grouping/colour overrides.
# Ordered by logical grouping — sorted by RS at runtime.
SECTOR_MAP: dict[str, tuple[str, str]] = {
    # Financials
    "^NSEBANK": ("Nifty Bank", "financials"),
    "^CNXPSUBANK": ("PSU Bank", "financials"),
    "PVTBANIETF.NS": ("Private Bank", "financials"),
    # Technology
    "^CNXIT": ("IT", "technology"),
    # Healthcare
    "^CNXPHARMA": ("Pharma", "healthcare"),
    "PHARMABEES": ("PHARMABEES", "healthcare"),
    # Materials
    "^CNXMETAL": ("Metal", "materials"),
    # Consumer
    "^CNXFMCG": ("FMCG", "consumer"),
    "CONSUMBEES.NS": ("Consumption", "consumer"),
    # Auto
    "^CNXAUTO": ("Auto", "auto"),
    # Energy / PSU
    "^CNXENERGY": ("Energy", "energy"),
    "CPSEETF.NS": ("CPSE", "psu"),
    "OILIETF.NS": ("Oil & Gas", "energy"),
    # Infrastructure
    "^CNXINFRA": ("Infra", "infra"),
    "^CNXREALTY": ("Realty", "realty"),
    # Media
    "^CNXMEDIA": ("Media", "media"),
    # Thematic
    "MODEFENCE.NS": ("MODEFENCE", "defence"),
    "METALIETF.NS": ("METALIETF", "materials"),
    # Commodity
    "GOLDBEES.NS": ("Gold", "COMMODITY"),
    "SILVERBEES.NS": ("Silver", "COMMODITY"),
    # INTERNATIONAL
    "MON100.NS": ("MON100", "International"),
    "MON100.NS": ("MON100", "International"),
    "MAFANG.NS": ("MAFANG", "International"),
    "HNGSNGBEES.NS": ("HNGSNGBEES", "International"),
    "MAHKTECH.NS": ("MAHKTECH", "International"),
}

# Convenience exports (legacy — kept for signal_engine.py compatibility)
SECTOR_TICKERS = list(SECTOR_MAP.keys())
SECTOR_NAMES = {t: v[0] for t, v in SECTOR_MAP.items()}

# ══════════════════════════════════════════════════════════════════════════════
# STOCK SEGMENT CLASSIFICATION (data-driven from param sweep)
#
# Two behavioral populations on NSE:
#   INSTITUTIONAL: institutional-flow-driven, trend-persistent
#     → Mode A: Early Momentum (WR(20), -50, ADX)
#   CYCLICAL: commodity/cycle-driven, mean-reverting
#     → Mode B: Mean Reversion (WR(30), -30, below SMA)
#
# Classification based on:
#   - Sector characteristics (banking, IT = institutional; metals, pharma = cyclical)
#   - Market cap (mega-cap tends institutional)
#   - Sweep results showing distinct optimal parameters per group
# ══════════════════════════════════════════════════════════════════════════════

SEGMENT_INSTITUTIONAL = {
    # Banking & Finance
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
    "BAJFINANCE", "BAJAJFINSV", "INDUSINDBK", "HDFCLIFE", "SBILIFE",
    # IT (institutional flow heavy)
    "TCS", "INFY",
    # Mega-cap conglomerates
    "RELIANCE", "BHARTIARTL", "ADANIENT", "ADANIPORTS",
    # Indices
    "NIFTY", "BANKNIFTY", "FINNIFTY",
}

SEGMENT_CYCLICAL = {
    # Metals
    "TATASTEEL", "JSWSTEEL", "HINDALCO",
    # IT (mid-tier, more cyclical)
    "WIPRO", "TECHM", "HCLTECH",
    # Pharma
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP",
    # Auto
    "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "M&M",
    # PSU / Infra
    "LT", "NTPC", "POWERGRID", "COALINDIA", "ONGC", "BPCL",
    # Consumer
    "ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    "TITAN", "ASIANPAINT",
    # Cement
    "ULTRACEMCO", "GRASIM",
    # Chemicals / New Age
    "UPL", "ETERNAL",
}


def get_segment(symbol: str) -> str:
    """
    Return 'INSTITUTIONAL' or 'CYCLICAL' for a given symbol.
    Default: CYCLICAL (mean reversion is the safer assumption).
    """
    sym = symbol.strip().upper().replace(".NS", "")
    if sym in SEGMENT_INSTITUTIONAL:
        return "INSTITUTIONAL"
    if sym in SEGMENT_CYCLICAL:
        return "CYCLICAL"
    # Default to cyclical — mean reversion has more robust sample sizes
    return "CYCLICAL"


# Legacy export for signal_engine.py
SECTOR_INDEX = {"DEFAULT": MARKET_TICKER}
