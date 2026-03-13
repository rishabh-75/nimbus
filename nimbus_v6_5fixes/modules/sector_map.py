"""
modules/sector_map.py
─────────────────────
Sector-to-index mapping for relative strength computation.
"""

SECTOR_INDEX = {
    # Financials / Banking
    "HDFCBANK": "^NSEBANK", "ICICIBANK": "^NSEBANK",
    "SBIN": "^NSEBANK", "KOTAKBANK": "^NSEBANK",
    "AXISBANK": "^NSEBANK", "INDUSINDBK": "^NSEBANK",
    "BAJFINANCE": "^NSEBANK", "BAJAJFINSV": "^NSEBANK",

    # IT
    "TCS": "NIFTYIT.NS", "INFY": "NIFTYIT.NS",
    "WIPRO": "NIFTYIT.NS", "HCLTECH": "NIFTYIT.NS",
    "TECHM": "NIFTYIT.NS",

    # Metals / Mining
    "COALINDIA": "^CNXMETAL", "HINDALCO": "^CNXMETAL",
    "JSWSTEEL": "^CNXMETAL", "TATASTEEL": "^CNXMETAL",
    "VEDL": "^CNXMETAL", "NMDC": "^CNXMETAL",

    # Defense / PSE
    "BEL": "^CNXPSE", "HAL": "^CNXPSE", "BHEL": "^CNXPSE",
    "RVNL": "^CNXPSE", "IRFC": "^CNXPSE", "RECLTD": "^CNXPSE",
    "PFC": "^CNXPSE",

    # Pharma
    "SUNPHARMA": "^CNXPHARMA", "DRREDDY": "^CNXPHARMA",
    "CIPLA": "^CNXPHARMA", "DIVISLAB": "^CNXPHARMA",

    # Auto
    "MARUTI": "^CNXAUTO", "TATAMOTORS": "^CNXAUTO",
    "M&M": "^CNXAUTO", "BAJAJ-AUTO": "^CNXAUTO",
    "EICHERMOT": "^CNXAUTO",

    # Energy / Oil & Gas
    "RELIANCE": "^CNXENERGY", "ONGC": "^CNXENERGY",
    "NTPC": "^CNXENERGY", "POWERGRID": "^CNXENERGY",
    "BPCL": "^CNXENERGY",

    # FMCG
    "HINDUNILVR": "^CNXFMCG", "ITC": "^CNXFMCG",
    "NESTLEIND": "^CNXFMCG", "BRITANNIA": "^CNXFMCG",

    # Default
    "DEFAULT": "^CNX100",
}

SECTOR_NAMES = {
    "^NSEBANK":    "Banking",
    "NIFTYIT.NS":  "IT",
    "^CNXMETAL":   "Metals",
    "^CNXPSE":     "PSE/Defense",
    "^CNXPHARMA":  "Pharma",
    "^CNXAUTO":    "Auto",
    "^CNXENERGY":  "Energy",
    "^CNXFMCG":    "FMCG",
    "^CNX100":     "NIFTY 100",
}

SECTOR_TICKERS = list(set(v for k, v in SECTOR_INDEX.items() if k != "DEFAULT"))
IT_TICKERS = ["NIFTYIT.NS", "^CNXIT"]  # fallback order
MARKET_TICKER = "^CRSLDX"
MARKET_FALLBACK = "^CNX100"
