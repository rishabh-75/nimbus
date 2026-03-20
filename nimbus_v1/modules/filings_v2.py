"""
modules/filings_v2.py
──────────────────────
NSE Reg-30 filing intelligence layer for NIMBUS.

Fix log vs live filings_v2.py
  FIX-NOISE  : _NOISE_PATTERNS pre-filter — trading window (Reg 30(5)), ESOP
               vesting, AGM notices are suppressed before classification.
               Confirmed against 3 live DMART subjects (14-Mar-2026).
  FIX-COMP   : 7 new COMPLIANCE patterns: fraud, default, arrest, insolvency,
               search-and-seizure, attachment (NSE Fraud/Default/Arrest category)
  FIX-PROMO  : 9 new PROMOTER patterns for SEBI SAST Reg 28 encumbrance
               language. SEBI replaced "pledge" with "encumbrance" in 2011;
               modern NSE subjects never say "pledge invocation".
  FIX-ORDER  : 8 new ORDER_WIN patterns. "receipt" (noun) ≠ "receiv" (verb
               stem) — NCC "receipt of Major Orders" was silently missed.
               Anchored: NCC Bharat Net LoA 25-Mar-2025 → +3.8% t+1.
  FIX-API    : fetch_announcements() replaces fetch_filings(). Correct endpoint:
                 /api/corporate-announcements?index=equities&symbol=
               (old /api/corp-info?...&subject=announcements returned empty)
  FIX-FIELD  : Text = f"{desc} {subject}".strip() — NSE alternates which field
               carries classifiable text; combining catches both cases.

Design contracts (unchanged)
  - FilingVariance.variance is a DISPLAY ANNOTATION, never modifies viability.score
  - CORP_ACTION TARGET stocks are ALWAYS suppressed (already at offer premium)
  - NIFTY 500 membership gates ALL output; fail-open if cache is empty
  - conviction 0-10 flows into classify_setup_v3 for EVENT_PLAY / PRE_BREAKOUT
  - badge_color "BULLISH" | "BEARISH" | "NONE" used by scanner.py as filing_direction

Validated: 73/73 test cases (tests/test_filings_v2.py)
"""

from __future__ import annotations

import datetime
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── NSE session ────────────────────────────────────────────────────────────────
_NSE_BASE = "https://www.nseindia.com"
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json, text/plain, */*",
}

# ── NIFTY 500 cache (primary: live API, 7-day TTL; fallback: CSV; fail-open) ──
_NIFTY500_CACHE: set[str] = set()
_NIFTY500_TS: float = 0.0
_NIFTY500_TTL: float = 7 * 24 * 3600


def get_nifty500() -> set[str]:
    global _NIFTY500_CACHE, _NIFTY500_TS
    if _NIFTY500_CACHE and time.time() - _NIFTY500_TS < _NIFTY500_TTL:
        return _NIFTY500_CACHE
    try:
        sess = requests.Session()
        sess.get(_NSE_BASE, headers=_NSE_HEADERS, timeout=8)
        r = sess.get(
            f"{_NSE_BASE}/api/equity-stockIndices?index=NIFTY%20500",
            headers=_NSE_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        syms = {row["symbol"] for row in r.json().get("data", []) if "symbol" in row}
        if syms:
            _NIFTY500_CACHE, _NIFTY500_TS = syms, time.time()
            logger.info("[filings_v2] NIFTY500 (API): %d symbols", len(syms))
            return _NIFTY500_CACHE
    except Exception as exc:
        logger.warning("[filings_v2] NIFTY500 API failed (%s) — trying CSV", exc)
    try:
        import io, urllib.request

        req = urllib.request.Request(
            "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
        col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if col:
            syms = {str(s).strip().upper() for s in df[col].dropna()}
            if syms:
                _NIFTY500_CACHE, _NIFTY500_TS = syms, time.time()
                logger.info("[filings_v2] NIFTY500 (CSV): %d symbols", len(syms))
    except Exception as exc:
        logger.warning("[filings_v2] NIFTY500 CSV failed (%s) — fail-open", exc)
    return _NIFTY500_CACHE


def is_nifty500(symbol: str) -> bool:
    cache = get_nifty500()
    return (not cache) or (symbol.strip().upper() in cache)


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS + DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════


class FilingCategory(Enum):
    COMPLIANCE = "COMPLIANCE"
    CORP_ACTION = "CORP_ACTION"
    PROMOTER = "PROMOTER"
    BUYBACK = "BUYBACK"
    EARNINGS = "EARNINGS"
    DIVIDEND = "DIVIDEND"
    ORDER_WIN = "ORDER_WIN"
    BOARD_OUTCOME = "BOARD_OUTCOME"
    INVESTOR_MEET = "INVESTOR_MEET"
    OTHER = "OTHER"


RISK_CATS: set[FilingCategory] = {FilingCategory.COMPLIANCE, FilingCategory.PROMOTER}

_BADGE_COLORS: dict[FilingCategory, str] = {
    FilingCategory.COMPLIANCE: "BEARISH",
    FilingCategory.PROMOTER: "BEARISH",
    FilingCategory.ORDER_WIN: "BULLISH",
    FilingCategory.CORP_ACTION: "BULLISH",
    FilingCategory.BUYBACK: "BULLISH",
    FilingCategory.DIVIDEND: "BULLISH",
    FilingCategory.BOARD_OUTCOME: "NEUTRAL",
    FilingCategory.INVESTOR_MEET: "NEUTRAL",
    FilingCategory.EARNINGS: "NEUTRAL",
    FilingCategory.OTHER: "NONE",
}


@dataclass
class DealAssessment:
    """3D institutional deal conviction output."""

    net_buy_cr: float = 0.0  # net institutional buy, Rs Cr
    inst_count: int = 0  # distinct buying institutions
    mktcap_pct: float = 0.0  # deal as % of market cap
    adv_multiple: float = 0.0  # deal vs 20-day ADV
    conviction: int = 0  # 0-10 composite score
    size_bonus: int = 0  # 0-4 added to variance
    detail: str = ""  # human-readable for tooltip


@dataclass
class FilingVariance:
    """Filing annotation emitted to scanner and dashboard."""

    variance: int  # display delta; NEVER modifies viability.score
    badge_text: str  # e.g. "ORDER WIN +10"
    badge_color: str  # "BULLISH" | "BEARISH" | "NEUTRAL" | "NONE"
    detail_line: str  # one-line for tooltip/panel
    category: FilingCategory
    recency_h: float  # hours since filing
    confirmed: bool  # True if institutional deal confirmed
    raw_subject: str  # original subject[:80]
    conviction: int = 0  # 0-10; RISK_CATS fixed at 9 (severity proxy)


# ══════════════════════════════════════════════════════════════════════════════
# FIX-NOISE: PRE-FILTER
# Suppressed before classification. High-volume boilerplate that carries
# zero informational content. Confirmed live against DMART (14-Mar-2026).
# ══════════════════════════════════════════════════════════════════════════════

_NOISE_PATTERNS: list[str] = [
    # Trading window (Regulation 30(5)) — confirmed DMART 14-Mar-2026
    r"trading\s*window",
    r"regulation\s*30\s*\(5\)",
    r"closure\s*of\s*trading",
    # ESOP / benefit trust — confirmed DMART 14-Mar-2026
    r"vesting\s*of\s*stock\s*option",
    r"employee\s*stock\s*option.*vest",
    r"esop.*grant|grant.*esop",
    r"employee\s*benefit\s*trust",
    # Routine AGM/EGM notice (zero price impact)
    r"notice\s*of\s*(annual|extra.?ordinary)\s*general\s*meeting",
    # Loss of share certificate (Regulation 39)
    r"loss\s*of\s*share\s*certificate",
    # Newspaper publication compliance (Regulation 47)
    r"regulation\s*47.*newspaper|newspaper.*regulation\s*47",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)


def _is_noise(text: str) -> bool:
    return bool(_NOISE_RE.search(text))


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER PATTERNS
# Order matters — first match wins.
# FIX-1 (carried forward): bare "sebi" → sebi + action word
# FIX-2 (carried forward): EARNINGS before DIVIDEND
# FIX-COMP  : fraud, insolvency, arrest, attachment
# FIX-PROMO : Reg 28 encumbrance language
# FIX-ORDER : receipt (noun) form, LOA acronym, major orders
# ══════════════════════════════════════════════════════════════════════════════

_PATTERNS: list[tuple[FilingCategory, list[str]]] = [
    (
        FilingCategory.COMPLIANCE,
        [
            r"sebi\s*(notice|action|order|penalty|show.cause|scn|interim|direction|adjudication|investigation|probe)",
            r"enforcement\s*directorate|ed\s*notice",
            r"adjudication\s*order|compounding|consent\s*order|settlement\s*order",
            r"prosecution|insider.?trading.?notice|front.?running",
            r"rbi\s*(penalty|notice|action|direction|directive|restrict|sanction)",
            r"nclt.*order|nclat.*order|tribunal.*order",
            r"regulatory\s*(action|notice|order)",
            r"show.cause\s*notice",
            r"penalty\s*impos",
            # FIX-COMP: NSE "Fraud/Default/Arrest" filings category
            r"\bfraud\b",
            r"default.*(?:payment|loan|debenture|ncd|bond)",
            r"arrest.*(?:promoter|director|officer|managing)",
            r"insolvency|bankruptcy|corporate\s*insolvency\s*resolution|\bcirp\b",
            r"search\s*and\s*seizure",
            r"attachment.*order|provisional\s*attachment",
        ],
    ),
    (
        FilingCategory.CORP_ACTION,
        [
            r"scheme\s*of\s*(arrangement|amalgamation|merger|reconstruction)",
            r"open\s*offer",
            r"acqui[sz]ition|takeover",
            r"merger|amalgamation|demerger",
            r"business\s*transfer|slump\s*sale",
            r"nclt.*scheme|scheme.*nclt",
            r"strategic\s*stake|stake\s*acqui",
            r"change\s*of\s*control",
            r"substantial\s*acquisition|sast\s*reg.?29",
            r"delisting",
        ],
    ),
    (
        FilingCategory.PROMOTER,
        [
            # Original patterns (carried forward)
            r"promoter.*pledge.*invok|pledge.*invok|invok.*pledge",
            r"promoter.*reclassif",
            r"promoter.*open\s*market\s*sale",
            r"creeping\s*acqui",
            r"inter.?se\s*transfer",
            r"promoter.*encumber",
            # FIX-PROMO: SEBI SAST 2011 replaced "pledge" with "encumbrance"
            r"encumbrance|encumber",
            r"regulation\s*28\b|reg\.?\s*28\b",  # SAST Reg 28 disclosure
            r"pledg(?:e|ing|ed)\s*(?:of\s*)?shares",
            r"shares?\s*(?:of\s*)?pledg",
            r"creation\s*of\s*(?:pledge|encumbrance)",
            r"revocation\s*of\s*(?:pledge|encumbrance)",
            r"promoter.*stake.*(?:declin|reduc|increas)",
        ],
    ),
    (
        FilingCategory.BUYBACK,
        [
            r"buy.?back|buyback",
            r"share\s*repurchase",
            r"tender\s*offer.*share",
        ],
    ),
    # FIX-2 (carried forward): EARNINGS must precede DIVIDEND in pattern list
    (
        FilingCategory.EARNINGS,
        [
            r"financial\s*results?",
            r"quarterly\s*results?",
            r"unaudited\s*results?",
            r"audited\s*results?",
            r"standalone\s*results?",
            r"consolidated\s*results?",
            r"q[1-4]\s*(fy|results?)",
            r"half.year(?:ly)?\s*results?",
            r"annual\s*results?",
            r"revenue.*profit|profit.loss.account",
        ],
    ),
    (
        FilingCategory.DIVIDEND,
        [
            r"dividend\s*(?:recommend|declar|approv|pay|interim|final|special)",
            r"interim\s*dividend",
            r"final\s*dividend",
            r"special\s*dividend",
            r"record\s*date.*dividend|dividend.*record\s*date",
        ],
    ),
    (
        FilingCategory.ORDER_WIN,
        [
            # Original verb-stem patterns (carried forward)
            r"order\s*(?:receiv|secur|award|win|bag|obtain)",
            r"(?:receiv|secur|win)\s*(?:an?\s*)?order",
            r"contract\s*(?:award|secur|win|receiv)",
            r"letter\s*of\s*(?:intent|award)\s*(?:receiv|secur|obtain)",
            r"loa\s*(?:receiv|award)",
            r"work\s*order\s*(?:receiv|award|secur)",
            r"purchase\s*order.*awarded|supply\s*order",
            r"defence\s*(?:order|contract|deal)",
            r"ministry.*order|government.*order|project.*awarded",
            r"order\s*inflow|order\s*book.*addition",
            # FIX-ORDER: "receipt" is a noun — NCC Bharat Net LoA 25-Mar-2025
            r"receipt\s*(?:of\s*)?(?:major\s*)?order",
            r"receipt\s*(?:of\s*)?(?:work\s*)?order",
            r"receipt\s*(?:of\s*)?(?:advance\s*)?work\s*order",
            r"major\s*orders?",
            r"disclosure\s*regard.*order",
            r"advance\s*(?:work\s*)?order",
            r"letter\s*of\s*award",  # LoA full form
            r"\bloa\b",  # LoA bare acronym (Cochin Shipyard style)
        ],
    ),
    (
        FilingCategory.BOARD_OUTCOME,
        [
            r"board\s*meeting.*outcome|outcome.*board\s*meeting",
            r"outcome\s*of\s*board|board\s*of\s*directors.*meeting",
            r"board\s*meeting\s*(?:held|conclud)",
            r"board\s*(?:approved|recommended|decided|resolved)",
        ],
    ),
    (
        FilingCategory.INVESTOR_MEET,
        [
            r"investor.meet|analyst.meet|earnings.call|conference.call",
            r"investor\s*day|analyst\s*day|roadshow",
        ],
    ),
]


def classify_announcement(text: str) -> FilingCategory:
    """Classify filing subject text. First pattern match wins."""
    s = text.lower().strip()
    for cat, patterns in _PATTERNS:
        for pat in patterns:
            if re.search(pat, s):
                return cat
    return FilingCategory.OTHER


# ══════════════════════════════════════════════════════════════════════════════
# 3D DEAL CONVICTION MODEL
# D1: Market cap % (0-4 pts)   — most important; context-aware
# D2: ADV multiple   (0-3 pts) — price pressure proxy
# D3: Institution count (0-3 pts) — consensus vs single desk
# Score 0-10 → size_bonus 0-4
# Falls back to absolute Rs Cr tiers when market_cap_cr = 0
# ══════════════════════════════════════════════════════════════════════════════


def _assess_deals(
    bulk_df: Optional[pd.DataFrame],
    symbol: str,
    market_cap_cr: float = 0.0,
    adv_cr: float = 0.0,
) -> DealAssessment:
    a = DealAssessment()
    if bulk_df is None or bulk_df.empty:
        return a

    df = bulk_df.copy()
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    sym_col = next((c for c in df.columns if "SYMBOL" in c), None)
    qty_col = next((c for c in df.columns if "QTY" in c or "QUANTITY" in c), None)
    price_col = next(
        (c for c in df.columns if "PRICE" in c or "TRADE_PRICE" in c), None
    )
    client_col = next((c for c in df.columns if "CLIENT" in c or "ENTITY" in c), None)
    trade_col = next(
        (c for c in df.columns if "TRADE_TYPE" in c or "BUY_SELL" in c), None
    )

    if not all([sym_col, qty_col, price_col]):
        return a

    sym_df = df[df[sym_col].str.upper() == symbol.upper()].copy()
    if sym_df.empty:
        return a

    sym_df["_val_cr"] = (
        pd.to_numeric(sym_df[qty_col], errors="coerce").fillna(0)
        * pd.to_numeric(sym_df[price_col], errors="coerce").fillna(0)
    ) / 1e7

    if trade_col:
        buys = sym_df[sym_df[trade_col].str.upper().str.contains("BUY", na=False)]
        sells = sym_df[sym_df[trade_col].str.upper().str.contains("SELL", na=False)]
        a.net_buy_cr = buys["_val_cr"].sum() - sells["_val_cr"].sum()
    else:
        a.net_buy_cr = sym_df["_val_cr"].sum()

    if client_col and trade_col:
        buyers = (
            sym_df[sym_df[trade_col].str.upper().str.contains("BUY", na=False)][
                client_col
            ]
            .dropna()
            .unique()
        )
        a.inst_count = len(buyers)
    elif client_col:
        a.inst_count = len(sym_df[client_col].dropna().unique())
    else:
        a.inst_count = 1 if a.net_buy_cr > 0 else 0

    if a.net_buy_cr <= 0:
        return a

    # D1: Market cap % (context-aware; falls back to absolute Rs Cr)
    d1 = 0
    if market_cap_cr > 0:
        pct = a.net_buy_cr / market_cap_cr * 100
        a.mktcap_pct = round(pct, 3)
        d1 = (
            4
            if pct >= 2.0
            else 3 if pct >= 1.0 else 2 if pct >= 0.5 else 1 if pct >= 0.1 else 0
        )
    else:
        d1 = (
            3
            if a.net_buy_cr >= 500
            else 2 if a.net_buy_cr >= 100 else 1 if a.net_buy_cr >= 25 else 0
        )

    # D2: ADV multiple
    d2 = 0
    if adv_cr > 0:
        mult = a.net_buy_cr / adv_cr
        a.adv_multiple = round(mult, 2)
        d2 = 3 if mult >= 3.0 else 2 if mult >= 1.0 else 1 if mult >= 0.5 else 0

    # D3: Institution count
    d3 = (
        3
        if a.inst_count >= 4
        else 2 if a.inst_count >= 2 else 1 if a.inst_count >= 1 else 0
    )

    raw = d1 + d2 + d3
    a.conviction = min(raw, 10)
    a.size_bonus = (
        4 if raw >= 8 else 3 if raw >= 6 else 2 if raw >= 4 else 1 if raw >= 2 else 0
    )
    a.detail = (
        f"₹{a.net_buy_cr:.0f}Cr net buy | {a.inst_count} inst | "
        f"mktcap {a.mktcap_pct:.2f}% | {a.adv_multiple:.1f}x ADV | "
        f"conviction {a.conviction}/10"
    )
    return a


def _read_bulk_block(
    symbol: str,
    db_path: str = "nimbus_data.db",
    window_days: int = 10,
) -> pd.DataFrame:
    """Read bulk_block_deals from SQLite → DataFrame."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT client_name, buy_sell, quantity, price "
            "FROM bulk_block_deals "
            "WHERE UPPER(symbol) = ? AND trade_date >= date('now', ?)",
            conn,
            params=(symbol.upper(), f"-{window_days} days"),
        )
        conn.close()
        if not df.empty:
            df = df.rename(
                columns={
                    "client_name": "CLIENT",
                    "buy_sell": "BUY_SELL",
                    "quantity": "QTY",
                    "price": "PRICE",
                }
            )
            df["SYMBOL"] = symbol.upper()
        return df
    except Exception as exc:
        logger.debug("[filings_v2] bulk_block read failed for %s: %s", symbol, exc)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# CORP_ACTION SUB-CLASSIFIER (5-signal voter)
# S1 Linguistics        always, 0 ms
# S2 SAST Reg29         live session, T+2 lag — definitive TARGET if fires
# S3 Open Offer         live session, T+5-15 lag — definitive TARGET if fires
# S4 Promoter flow      SQLite bulk_block, 0 ms
# S5 Price premium      PriceSignals, 0 ms
# TARGET → always suppressed (already at offer premium)
# ACQUIRER → emit only when confidence ≥ 1
# ══════════════════════════════════════════════════════════════════════════════


def _corp_action_role(
    subject: str,
    symbol: str,
    nse_session=None,
    bulk_df: Optional[pd.DataFrame] = None,
    ps=None,
) -> tuple[str, int]:
    s = subject.lower()
    acq_pts = tgt_pts = 0

    # S1 — Linguistics
    for pat in [
        r"acquir",
        r"merger",
        r"amalgamat",
        r"absorb",
        r"scheme\s*of\s*arrangement",
        r"business\s*transfer",
        r"binding\s*agreement",
        r"definitive\s*agreement",
    ]:
        if re.search(pat, s):
            acq_pts += 1
            break
    for pat in [
        r"open\s*offer",
        r"mandatory\s*open",
        r"delisting\s*proposal",
        r"takeover\s*of\s*company",
    ]:
        if re.search(pat, s):
            tgt_pts += 2
            break

    # S2 — SAST Reg29 (definitive; T+2 lag)
    if nse_session is not None:
        try:
            r = nse_session.get(
                f"{_NSE_BASE}/api/corporates-sast"
                f"?index=equities&symbol={symbol}&regType=SAST",
                headers=_NSE_HEADERS,
                timeout=8,
            )
            if r.ok and r.json().get("data"):
                return ("TARGET", 2)
        except Exception:
            pass

    # S3 — Open Offers (definitive; T+5-15 lag)
    if nse_session is not None:
        try:
            r = nse_session.get(
                f"{_NSE_BASE}/api/open-offers", headers=_NSE_HEADERS, timeout=8
            )
            if r.ok:
                offers = (
                    r.json() if isinstance(r.json(), list) else r.json().get("data", [])
                )
                if any(
                    str(o.get("symbol", "")).upper() == symbol.upper() for o in offers
                ):
                    return ("TARGET", 2)
        except Exception:
            pass

    # S4 — Promoter flow (bulk_block SQLite)
    if bulk_df is not None and not bulk_df.empty:
        df_s = bulk_df.copy()
        df_s.columns = [c.strip().upper().replace(" ", "_") for c in df_s.columns]
        sym_col = next((c for c in df_s.columns if "SYMBOL" in c), None)
        cl_col = next((c for c in df_s.columns if "CLIENT" in c or "ENTITY" in c), None)
        tr_col = next(
            (c for c in df_s.columns if "TRADE" in c or "BUY_SELL" in c), None
        )
        if sym_col and cl_col and tr_col:
            rows = df_s[df_s[sym_col].str.upper() == symbol.upper()]
            promo = rows[
                rows[cl_col]
                .str.lower()
                .str.contains(r"promoter|promoter group|pac", na=False)
            ]
            if not promo.empty:
                if not promo[
                    promo[tr_col].str.upper().str.contains("SELL", na=False)
                ].empty:
                    tgt_pts += 2
                if not promo[
                    promo[tr_col].str.upper().str.contains("BUY", na=False)
                ].empty:
                    acq_pts += 1

    # S5 — Price premium (PriceSignals)
    if ps is not None:
        try:
            if getattr(ps, "daily_sma", None) and ps.last_close > ps.daily_sma * 1.10:
                tgt_pts += 2
            elif getattr(ps, "daily_bias_pct", 0) < -1.5:
                acq_pts += 1
        except Exception:
            pass

    if tgt_pts > acq_pts:
        return ("TARGET", 1)
    if acq_pts >= 2:
        return ("ACQUIRER", 1)
    return ("UNKNOWN", 0)


# ══════════════════════════════════════════════════════════════════════════════
# VARIANCE CALCULATOR
# Returns (variance, badge_text, badge_color, detail_line)
# variance capped at [-12, +12]
# ══════════════════════════════════════════════════════════════════════════════


def _variance_for_category(
    cat: FilingCategory,
    subject: str,
    deal: DealAssessment,
) -> tuple[int, str, str, str]:
    b = deal.size_bonus

    if cat == FilingCategory.ORDER_WIN:
        v = min(6 + b, 12)
        d = (
            deal.detail
            if deal.conviction >= 3
            else "Order received — monitoring for inst. confirmation"
        )
        return (v, f"ORDER WIN +{v}", "BULLISH", d)

    if cat == FilingCategory.BUYBACK:
        v = min(5 + b, 10)
        note = f" | {deal.detail}" if deal.conviction >= 3 else ""
        return (v, f"BUYBACK +{v}", "BULLISH", f"Share buyback programme{note}")

    if cat == FilingCategory.DIVIDEND:
        v = min(2 + b, 6)
        return (
            v,
            f"DIVIDEND +{v}",
            "BULLISH",
            "Dividend declared — positive sentiment signal",
        )

    if cat == FilingCategory.BOARD_OUTCOME:
        return (
            2,
            "BOARD",
            "NEUTRAL",
            "Board meeting outcome — check for material decision",
        )

    if cat == FilingCategory.INVESTOR_MEET:
        return (
            1,
            "INV MEET",
            "NEUTRAL",
            "Analyst/investor event — constructive management tone",
        )

    if cat == FilingCategory.COMPLIANCE:
        return (
            -10,
            "SEBI/REG ⚠",
            "BEARISH",
            "Regulatory action — review exposure immediately",
        )

    if cat == FilingCategory.PROMOTER:
        return (
            -8,
            "PROMOTER ⚠",
            "BEARISH",
            "Promoter encumbrance/activity — assess direction",
        )

    # OTHER (shouldn't reach here — suppressed upstream, but safe fallback)
    return (1, "FILING", "NONE", f"Regulatory disclosure: {subject[:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# FIX-API + FIX-FIELD: NSE FETCH
# Correct endpoint: /api/corporate-announcements?index=equities&symbol=
# Combined text:    f"{desc} {subject}".strip()
# ══════════════════════════════════════════════════════════════════════════════


def make_nse_session() -> requests.Session:
    """Return a warmed NSE session (cookie handshake required before API calls)."""
    sess = requests.Session()
    sess.headers.update(_NSE_HEADERS)
    try:
        sess.get(_NSE_BASE, timeout=10)
    except Exception as exc:
        logger.warning("[filings_v2] NSE session warm failed: %s", exc)
    return sess


def fetch_announcements(
    symbol: str,
    sess: Optional[requests.Session] = None,
    max_records: int = 10,
) -> list[dict]:
    """
    Fetch recent Reg-30 corporate announcements for a symbol.

    FIX-API:   /api/corporate-announcements?index=equities&symbol=
               (old /api/corp-info?...&subject=announcements was wrong — returned empty)
    FIX-FIELD: Returns {"text": f"{desc} {subject}", "ts": ..., "attachment": ...}
               NSE alternates which field carries classifiable text; combining both
               ensures ORDER_WIN "Major Orders" subjects aren't missed.

    Returns [] on any failure — caller handles gracefully.
    """
    own_sess = sess is None
    if own_sess:
        sess = make_nse_session()

    url = f"{_NSE_BASE}/api/corporate-announcements?index=equities&symbol={symbol}"
    try:
        r = sess.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json()
        rows = raw if isinstance(raw, list) else raw.get("data", [])
    except Exception as exc:
        logger.warning("[filings_v2] fetch_announcements %s: %s", symbol, exc)
        return []

    results = []
    for row in rows[:max_records]:
        desc = str(row.get("desc", "") or "")
        subject = str(row.get("subject", "") or "")
        text = f"{desc} {subject}".strip()
        if not text:
            continue
        results.append(
            {
                "text": text,
                "ts": row.get("exchdisstime") or row.get("an_dt") or "",
                "attachment": row.get("attchmntFile", ""),
            }
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RECENCY THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
_FRESH_H = 6.0
_RECENT_H = 24.0
_STALE_H = 72.0


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════


def get_filing_variance(
    symbol: str,
    subject: Optional[str] = None,
    filing_ts: Optional[pd.Timestamp] = None,
    filings: Optional[list[dict]] = None,
    bulk_df: Optional[pd.DataFrame] = None,
    nse_session=None,
    ps=None,
    market_cap_cr: float = 0.0,
    adv_cr: float = 0.0,
    db_path: str = "nimbus_data.db",
) -> Optional[FilingVariance]:
    """
    Classify a Reg-30 filing and return FilingVariance or None.

    Two calling modes:
      A) Per-subject (FilingsWorker / SingleFilingWorker):
           get_filing_variance(symbol, subject=text, filing_ts=ts, ...)
      B) Legacy list mode:
           get_filing_variance(symbol, filings=[...], ...)

    Returns None when:
      - Symbol not in NIFTY 500
      - Noise filing (_NOISE_PATTERNS pre-filter)
      - Filing is stale (> 72h)
      - CORP_ACTION TARGET role (already at offer premium)
      - EARNINGS (no directional signal without beat/miss data)
      - OTHER with deal.conviction < 3
    """
    if not is_nifty500(symbol):
        logger.debug("[filings_v2] %s not in NIFTY 500 — skipped", symbol)
        return None

    # Mode B: resolve subject from filings list
    if subject is None:
        if filings is None:
            filings = fetch_announcements(symbol)
        if not filings:
            return None
        latest = filings[0]
        # FIX-FIELD: combine desc + subject
        desc = str(latest.get("desc", "") or "")
        subj = str(latest.get("subject", "") or "")
        text = str(
            latest.get("text", "") or ""
        )  # already combined if from fetch_announcements
        subject = text or f"{desc} {subj}".strip()
        if not subject:
            return None
        if filing_ts is None:
            raw_dt = (
                latest.get("ts")
                or latest.get("exchdisstime")
                or latest.get("an_dt")
                or ""
            )
            for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y"):
                try:
                    filing_ts = pd.Timestamp(
                        datetime.datetime.strptime(str(raw_dt).strip(), fmt)
                    )
                    break
                except (ValueError, AttributeError):
                    continue

    # FIX-NOISE: pre-filter (before any regex classification work)
    if _is_noise(subject):
        logger.debug("[filings_v2] %s noise filing suppressed: %.60s", symbol, subject)
        return None

    if adv_cr == 0.0 and ps is not None:
        adv_cr = getattr(ps, "adv_cr", 0.0)

    # Recency
    recency_h = 999.0
    if filing_ts is not None:
        try:
            recency_h = (
                pd.Timestamp.now() - pd.Timestamp(filing_ts)
            ).total_seconds() / 3600
        except Exception:
            pass

    if recency_h > _STALE_H:
        logger.debug(
            "[filings_v2] %s filing stale (%.1fh) — suppressed", symbol, recency_h
        )
        return None

    cat = classify_announcement(subject)

    if bulk_df is None:
        bulk_df = _read_bulk_block(symbol, db_path=db_path)
    deal = _assess_deals(bulk_df, symbol, market_cap_cr, adv_cr)

    # CORP_ACTION: sub-classify ACQUIRER vs TARGET
    if cat == FilingCategory.CORP_ACTION:
        role, confidence = _corp_action_role(
            subject, symbol, nse_session=nse_session, bulk_df=bulk_df, ps=ps
        )
        if role == "TARGET":
            logger.info("[filings_v2] %s CORP_ACTION TARGET — suppressed", symbol)
            return None
        if confidence < 1:
            logger.debug(
                "[filings_v2] %s CORP_ACTION low confidence — suppressed", symbol
            )
            return None
        v = min(4 + deal.size_bonus, 10)
        v = min(
            v + (2 if recency_h <= _FRESH_H else 1 if recency_h <= _RECENT_H else 0), 12
        )
        note = (
            f"Acquirer role confirmed. {deal.detail}"
            if deal.conviction >= 3
            else "Acquirer role (linguistics). Options quiet is EXPECTED for M&A timelines."
        )
        return FilingVariance(
            variance=v,
            badge_text=f"ACQUIRER +{v}",
            badge_color="BULLISH",
            detail_line=note,
            category=cat,
            recency_h=round(recency_h, 1),
            confirmed=(deal.conviction >= 3),
            raw_subject=subject[:80],
            conviction=int(deal.conviction),
        )

    # EARNINGS: always suppress (no directional signal without beat/miss)
    if cat == FilingCategory.EARNINGS:
        return None

    # OTHER: suppress unless institutional conviction confirms unusual interest
    if cat == FilingCategory.OTHER and deal.conviction < 3:
        return None

    variance, badge, color, detail = _variance_for_category(cat, subject, deal)
    if variance == 0:
        return None

    # Recency bonus (positive categories only)
    if variance > 0:
        rec_bonus = 2 if recency_h <= _FRESH_H else 1 if recency_h <= _RECENT_H else 0
        variance = min(variance + rec_bonus, 12)

    conviction = 9 if cat in RISK_CATS else int(deal.conviction)

    return FilingVariance(
        variance=variance,
        badge_text=badge,
        badge_color=color,
        detail_line=detail,
        category=cat,
        recency_h=round(recency_h, 1),
        confirmed=(deal.conviction >= 3),
        raw_subject=subject[:80],
        conviction=conviction,
    )
