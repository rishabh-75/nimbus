"""
modules/regime_filter.py
─────────────────────────
Market regime detection for NIMBUS.

Regime types:
  TRENDING  — momentum signals reliable, full sizing allowed
  RANGING   — momentum signals noisy, reduce sizing
  VOLATILE  — crisis/event, all sizing capped

Detection inputs:
  1. India VIX level (primary)
  2. NIFTY vs 20-SMA (secondary)
  3. BB width percentile of NIFTY (tertiary)

Regime affects:
  - Signal weight multipliers (trending amplifies WR/BB, ranging dampens)
  - Sizing caps
  - Exit sensitivity (tighter in volatile)

DOES NOT modify existing _viability() — applies as a post-scoring overlay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── VIX thresholds calibrated to NSE historical distribution ─────────────────
# India VIX: median ~13, 75th percentile ~17, 90th ~22
VIX_LOW = 13.0    # below → low vol, trending environment
VIX_MID = 18.0    # below → normal, mild range ok
VIX_HIGH = 24.0   # above → volatile, crisis-like


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str = "UNKNOWN"       # TRENDING | RANGING | VOLATILE | UNKNOWN
    vix: Optional[float] = None
    nifty_vs_sma: Optional[float] = None  # % above/below 20-SMA
    bb_width_pctl: Optional[float] = None  # NIFTY BB width percentile
    confidence: float = 0.0       # 0-1 confidence in classification

    # Scoring multipliers (applied to signal weights)
    momentum_mult: float = 1.0    # multiply WR/BB signal weights by this
    mean_revert_mult: float = 1.0 # multiply mean-reversion signals
    sizing_cap: str = "FULL"      # max allowed sizing
    exit_sensitivity: float = 1.0 # ATR multiplier (lower = tighter stops)

    reason: str = ""


def classify_regime(
    vix: Optional[float] = None,
    nifty_close: Optional[float] = None,
    nifty_sma20: Optional[float] = None,
    nifty_bb_width_pctl: Optional[float] = None,
) -> MarketRegime:
    """
    Classify current market regime from available inputs.

    Graceful degradation:
      - All inputs present → high confidence classification
      - VIX only → medium confidence
      - No VIX, SMA only → low confidence
      - Nothing → UNKNOWN with neutral multipliers
    """
    regime = MarketRegime()
    signals = []

    # ── VIX-based classification (primary) ────────────────────────────────
    if vix is not None:
        regime.vix = vix
        if vix < VIX_LOW:
            signals.append(("TRENDING", 0.5, f"VIX {vix:.1f} < {VIX_LOW}"))
        elif vix < VIX_MID:
            signals.append(("RANGING", 0.3, f"VIX {vix:.1f} normal range"))
        elif vix < VIX_HIGH:
            signals.append(("RANGING", 0.4, f"VIX {vix:.1f} elevated"))
        else:
            signals.append(("VOLATILE", 0.5, f"VIX {vix:.1f} > {VIX_HIGH}"))

    # ── Trend-SMA classification (secondary) ──────────────────────────────
    if nifty_close is not None and nifty_sma20 is not None and nifty_sma20 > 0:
        pct = (nifty_close / nifty_sma20 - 1) * 100
        regime.nifty_vs_sma = pct
        if pct > 1.5:
            signals.append(("TRENDING", 0.3, f"NIFTY {pct:+.1f}% above 20-SMA"))
        elif pct < -1.5:
            signals.append(("RANGING", 0.3, f"NIFTY {pct:+.1f}% below 20-SMA"))
        else:
            signals.append(("RANGING", 0.15, f"NIFTY near 20-SMA ({pct:+.1f}%)"))

    # ── BB width percentile (tertiary) ────────────────────────────────────
    if nifty_bb_width_pctl is not None:
        regime.bb_width_pctl = nifty_bb_width_pctl
        if nifty_bb_width_pctl < 20:
            signals.append(("TRENDING", 0.2, "NIFTY in BB squeeze"))
        elif nifty_bb_width_pctl > 80:
            signals.append(("VOLATILE", 0.2, "NIFTY BB expanded"))

    # ── Aggregate classification ──────────────────────────────────────────
    if not signals:
        return regime  # UNKNOWN with neutral multipliers

    # Weighted vote
    votes = {}
    for label, weight, _ in signals:
        votes[label] = votes.get(label, 0.0) + weight
    regime.regime = max(votes, key=votes.get)
    regime.confidence = min(1.0, max(votes.values()))
    regime.reason = " | ".join(reason for _, _, reason in signals)

    # ── Set multipliers based on regime ───────────────────────────────────
    if regime.regime == "TRENDING":
        regime.momentum_mult = 1.2      # amplify WR/BB signals
        regime.mean_revert_mult = 0.7   # dampen reversal signals
        regime.sizing_cap = "FULL"
        regime.exit_sensitivity = 1.0
    elif regime.regime == "RANGING":
        regime.momentum_mult = 0.7      # dampen WR/BB signals
        regime.mean_revert_mult = 1.2   # amplify reversal signals
        regime.sizing_cap = "HALF"
        regime.exit_sensitivity = 0.8   # tighter stops
    elif regime.regime == "VOLATILE":
        regime.momentum_mult = 0.5      # heavily dampen
        regime.mean_revert_mult = 0.5   # everything unreliable
        regime.sizing_cap = "SKIP"
        regime.exit_sensitivity = 0.6   # very tight stops

    logger.info(
        "Regime: %s (conf=%.2f) — mom_mult=%.1f sizing=%s | %s",
        regime.regime, regime.confidence, regime.momentum_mult,
        regime.sizing_cap, regime.reason,
    )
    return regime


def fetch_vix() -> Optional[float]:
    """Fetch current India VIX value. Returns None on failure."""
    try:
        import yfinance as yf
        vix = yf.download("^INDIAVIX", period="5d", interval="1d", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        if not vix.empty and "Close" in vix.columns:
            return float(vix["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.debug("VIX fetch failed: %s", exc)
    return None
