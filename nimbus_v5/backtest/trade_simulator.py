"""
backtest/trade_simulator.py
────────────────────────────
Simulates trades using NIMBUS entry/exit rules on historical data.

Entry rules (matching the live pipeline):
  - position_state == RIDING_UPPER
  - wr_in_momentum == True
  - daily_bias in (BULLISH, NEUTRAL)

Exit rules (the revamped exit intelligence):
  1. FIRST_DIP:        scale 50% at first dip below upper BB
  2. MID_BAND_BROKEN:  full exit when price closes below BB mid
  3. ATR trailing stop: exit if price drops > 2×ATR from peak
  4. Time exit:         exit if trade hasn't moved +1% in 10 bars
  5. Profit target:     take full profit at 2× BB width from entry
  6. MFI divergence:    reduce to 50% if mfi_diverge fires during trade

Output: list of Trade objects with entry/exit details, P&L, hold time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from modules.indicators import add_bollinger, add_williams_r, compute_price_signals

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TRADE RECORD
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    entry_signal: str  # what triggered entry
    sizing: str  # FULL / HALF

    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""

    hold_bars: int = 0
    pnl_pct: float = 0.0
    peak_price: float = 0.0
    max_drawdown_pct: float = 0.0
    max_runup_pct: float = 0.0

    # Signal state at entry (for analysis)
    wr_phase: str = ""
    vol_state: str = ""
    mfi_state: str = ""
    daily_bias: str = ""

    # Partial exits
    scaled_out_50: bool = False
    scale_out_bar: int = 0

    @property
    def closed(self) -> bool:
        return self.exit_date is not None

    @property
    def won(self) -> bool:
        return self.pnl_pct > 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "entry_price": round(self.entry_price, 2),
            "exit_date": self.exit_date,
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "exit_reason": self.exit_reason,
            "hold_bars": self.hold_bars,
            "pnl_pct": round(self.pnl_pct, 3),
            "max_dd_pct": round(self.max_drawdown_pct, 3),
            "max_runup_pct": round(self.max_runup_pct, 3),
            "wr_phase": self.wr_phase,
            "vol_state": self.vol_state,
            "mfi_state": self.mfi_state,
            "daily_bias": self.daily_bias,
            "entry_signal": self.entry_signal,
            "sizing": self.sizing,
            "scaled_out": self.scaled_out_50,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ATR CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ══════════════════════════════════════════════════════════════════════════════
# EXIT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Exit parameters (tunable)
ATR_STOP_MULT = 2.0         # trail stop at 2× ATR below peak
TIME_EXIT_BARS = 10          # max bars without +1% move
TIME_EXIT_MIN_MOVE = 1.0     # minimum % move to reset time clock
PROFIT_TARGET_BB_MULT = 2.0  # take profit at 2× BB width from entry
MAX_HOLD_BARS = 25           # absolute max holding period


def _check_exits(
    trade: Trade,
    bar_idx: int,
    close: float,
    bb_mid: float,
    bb_upper: float,
    bb_width: float,
    atr_val: float,
    ps,
    entry_bar: int,
) -> Optional[str]:
    """
    Check all exit conditions for an open trade.
    Returns exit_reason string or None if no exit triggered.

    Priority order (first match wins):
    1. Mid-band broken → full exit (momentum dead)
    2. ATR trail stop → stop loss (risk management)
    3. Profit target → take profit (BB width target)
    4. Time exit → stale trade (no conviction)
    5. Max hold → hard limit
    """
    bars_held = bar_idx - entry_bar
    pnl_pct = (close / trade.entry_price - 1) * 100
    peak_pnl = (trade.peak_price / trade.entry_price - 1) * 100

    # 1. MID_BAND_BROKEN — hard exit
    if close < bb_mid and bars_held >= 2:
        return "MID_BAND_BROKEN"

    # 2. ATR trailing stop
    if atr_val and atr_val > 0:
        trail_stop = trade.peak_price - ATR_STOP_MULT * atr_val
        if close < trail_stop and bars_held >= 3:
            return f"ATR_TRAIL_STOP ({ATR_STOP_MULT}x)"

    # 3. Profit target (2× BB width from entry)
    if bb_width and bb_width > 0:
        target = trade.entry_price * (1 + PROFIT_TARGET_BB_MULT * bb_width)
        if close >= target:
            return "PROFIT_TARGET"

    # 4. Time exit (no meaningful move in N bars)
    if bars_held >= TIME_EXIT_BARS and pnl_pct < TIME_EXIT_MIN_MOVE:
        return f"TIME_EXIT ({TIME_EXIT_BARS}bars, <{TIME_EXIT_MIN_MOVE}%)"

    # 5. Max hold
    if bars_held >= MAX_HOLD_BARS:
        return f"MAX_HOLD ({MAX_HOLD_BARS}bars)"

    return None


# ══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════


def simulate_trades(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    bb_period: int = 20,
    bb_std: float = 1.0,
    wr_period: int = 50,
    wr_thresh: float = -20.0,
    max_concurrent: int = 1,
    cooldown_bars: int = 3,
) -> list[Trade]:
    """
    Simulate trades on historical OHLCV using NIMBUS entry/exit rules.

    Walk-forward: at each bar, compute signals from history[:bar+1],
    then check entry/exit conditions. No lookahead.

    Args:
        max_concurrent: max open positions (1 for single-symbol)
        cooldown_bars: bars to wait after an exit before re-entering
    """
    warmup = max(bb_period, wr_period) + 30  # extra buffer for MFI
    if len(df) < warmup + 20:
        return []

    # Pre-compute indicators on full dataset for exit checks
    full = df.copy()
    full = add_bollinger(full, bb_period, bb_std)
    full = add_williams_r(full, wr_period)
    full["ATR_14"] = _atr(full, 14)

    trades: list[Trade] = []
    open_trade: Optional[Trade] = None
    last_exit_bar = -cooldown_bars - 1

    for bar in range(warmup, len(full)):
        close = float(full.iloc[bar]["Close"])
        bb_mid = float(full.iloc[bar].get("BB_Mid", close))
        bb_upper = float(full.iloc[bar].get("BB_Upper", close))
        bb_width_val = float(full.iloc[bar].get("BB_Width", 0))
        atr_val = float(full.iloc[bar].get("ATR_14", 0))

        # Compute signals using ONLY past data
        hist = full.iloc[: bar + 1]
        ps = compute_price_signals(hist, wr_thresh=wr_thresh)

        # ── EXIT CHECK ────────────────────────────────────────────────────
        if open_trade is not None:
            open_trade.hold_bars = bar - open_trade._entry_bar
            open_trade.peak_price = max(open_trade.peak_price, close)
            open_trade.max_runup_pct = max(
                open_trade.max_runup_pct,
                (close / open_trade.entry_price - 1) * 100,
            )
            dd = (close / open_trade.peak_price - 1) * 100
            open_trade.max_drawdown_pct = min(open_trade.max_drawdown_pct, dd)

            # Check FIRST_DIP partial exit
            if (
                not open_trade.scaled_out_50
                and ps.position_state == "FIRST_DIP"
                and open_trade.hold_bars >= 2
            ):
                open_trade.scaled_out_50 = True
                open_trade.scale_out_bar = bar

            # Check MFI divergence sizing reduction
            if ps.mfi_diverge and ps.mfi_reliable and open_trade.sizing == "FULL":
                open_trade.sizing = "HALF"

            exit_reason = _check_exits(
                open_trade, bar, close, bb_mid, bb_upper,
                bb_width_val, atr_val, ps, open_trade._entry_bar,
            )

            if exit_reason:
                open_trade.exit_date = full.index[bar]
                open_trade.exit_price = close
                open_trade.exit_reason = exit_reason
                open_trade.pnl_pct = (close / open_trade.entry_price - 1) * 100
                trades.append(open_trade)
                last_exit_bar = bar
                open_trade = None

        # ── ENTRY CHECK ───────────────────────────────────────────────────
        if open_trade is None and (bar - last_exit_bar) >= cooldown_bars:
            # NIMBUS entry conditions (exact match to live pipeline)
            entry_ok = (
                ps.position_state == "RIDING_UPPER"
                and ps.wr_in_momentum
                and ps.daily_bias in ("BULLISH", "NEUTRAL")
            )

            if entry_ok:
                sizing = "FULL"
                if ps.wr_phase == "LATE":
                    sizing = "HALF"
                if ps.vol_state == "EXPANDED":
                    sizing = "HALF"
                if ps.mfi_diverge and ps.mfi_reliable:
                    sizing = "HALF"

                entry_signal = (
                    f"WR={ps.wr_value:.0f} {ps.wr_phase} | "
                    f"BB={ps.position_state} | "
                    f"Bias={ps.daily_bias}"
                )

                t = Trade(
                    symbol=symbol,
                    entry_date=full.index[bar],
                    entry_price=close,
                    entry_signal=entry_signal,
                    sizing=sizing,
                    peak_price=close,
                    wr_phase=ps.wr_phase,
                    vol_state=ps.vol_state,
                    mfi_state=ps.mfi_state,
                    daily_bias=ps.daily_bias,
                )
                t._entry_bar = bar  # internal tracking
                open_trade = t

    # Close any remaining open trade at last bar
    if open_trade is not None:
        bar = len(full) - 1
        close = float(full.iloc[bar]["Close"])
        open_trade.exit_date = full.index[bar]
        open_trade.exit_price = close
        open_trade.exit_reason = "END_OF_DATA"
        open_trade.pnl_pct = (close / open_trade.entry_price - 1) * 100
        open_trade.hold_bars = bar - open_trade._entry_bar
        trades.append(open_trade)

    logger.info(
        "%s: %d trades simulated (%d wins, %d losses)",
        symbol, len(trades),
        sum(1 for t in trades if t.pnl_pct > 0),
        sum(1 for t in trades if t.pnl_pct <= 0),
    )
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SYMBOL SIMULATION
# ══════════════════════════════════════════════════════════════════════════════


def simulate_universe(
    universe: dict[str, pd.DataFrame],
    min_bars: int = 200,
    **kwargs,
) -> list[Trade]:
    """Run trade simulation across an entire symbol universe."""
    all_trades = []
    for sym, df in universe.items():
        if len(df) < min_bars:
            continue
        trades = simulate_trades(df, symbol=sym, **kwargs)
        all_trades.extend(trades)
    all_trades.sort(key=lambda t: t.entry_date)
    logger.info("Universe simulation: %d total trades", len(all_trades))
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════


def trade_summary(trades: list[Trade]) -> dict:
    """Compute aggregate performance metrics from a list of trades."""
    if not trades:
        return {"n_trades": 0}

    pnls = [t.pnl_pct for t in trades if t.closed]
    if not pnls:
        return {"n_trades": 0}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Compute equity curve
    equity = [100]
    for p in pnls:
        equity.append(equity[-1] * (1 + p / 100))
    eq = np.array(equity)
    peak_eq = np.maximum.accumulate(eq)
    drawdowns = (eq - peak_eq) / peak_eq * 100

    avg_hold = np.mean([t.hold_bars for t in trades if t.closed])

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "n_trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "avg_pnl": round(np.mean(pnls), 3),
        "med_pnl": round(np.median(pnls), 3),
        "total_pnl": round(sum(pnls), 2),
        "avg_win": round(np.mean(wins), 3) if wins else 0,
        "avg_loss": round(np.mean(losses), 3) if losses else 0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else 999,
        "sharpe": round(np.mean(pnls) / np.std(pnls) * np.sqrt(52), 2) if np.std(pnls) > 0 else 0,
        "max_drawdown": round(float(drawdowns.min()), 2),
        "avg_hold_bars": round(avg_hold, 1),
        "final_equity": round(eq[-1], 2),
        "exit_reasons": exit_reasons,
        "by_wr_phase": _group_metric(trades, "wr_phase"),
        "by_vol_state": _group_metric(trades, "vol_state"),
        "by_sizing": _group_metric(trades, "sizing"),
    }


def _group_metric(trades: list[Trade], attr: str) -> dict:
    """Group trades by attribute and compute per-group win rate + avg P&L."""
    groups = {}
    for t in trades:
        key = getattr(t, attr, "?")
        if key not in groups:
            groups[key] = []
        groups[key].append(t.pnl_pct)
    return {
        k: {
            "n": len(v),
            "win_pct": round(sum(1 for x in v if x > 0) / len(v) * 100, 1),
            "avg_pnl": round(np.mean(v), 3),
        }
        for k, v in groups.items()
    }
