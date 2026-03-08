"""
AlertManager — Real-time alert generation, session log, and notification dispatch.
"""

from __future__ import annotations
import time
import logging
import datetime
from typing import Optional
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages trading alerts: threshold checks, session log, toast notifications,
    and optional desktop notifications.
    """

    SESSION_KEY = "alert_log"
    PRIOR_WALLS_KEY = "prior_walls"
    PRIOR_PCR_KEY = "prior_pcr"
    PRIOR_SIGNAL_KEY = "prior_signal"
    PRIOR_MAX_PAIN_KEY = "prior_max_pain"

    def __init__(
        self,
        score_threshold: int = 70,
        oi_spike_pct: float = 20.0,
        pcr_thresholds: tuple = (0.7, 1.3),
    ):
        self.score_threshold = score_threshold
        self.oi_spike_pct = oi_spike_pct
        self.pcr_low, self.pcr_high = pcr_thresholds
        self._ensure_session_state()

    # ─── session state bootstrap ──────────────────────────────────────────────

    def _ensure_session_state(self):
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = []
        if self.PRIOR_WALLS_KEY not in st.session_state:
            st.session_state[self.PRIOR_WALLS_KEY] = None
        if self.PRIOR_PCR_KEY not in st.session_state:
            st.session_state[self.PRIOR_PCR_KEY] = None
        if self.PRIOR_SIGNAL_KEY not in st.session_state:
            st.session_state[self.PRIOR_SIGNAL_KEY] = None
        if self.PRIOR_MAX_PAIN_KEY not in st.session_state:
            st.session_state[self.PRIOR_MAX_PAIN_KEY] = None

    # ─── alert logging ────────────────────────────────────────────────────────

    def _log(self, message: str, level: str = "watch"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        emoji_map = {"buy": "🟢", "sell": "🔴", "watch": "🟡", "info": "ℹ️"}
        emoji = emoji_map.get(level, "🟡")
        entry = f"[{ts}] {emoji} {message}"
        st.session_state[self.SESSION_KEY].append(entry)
        logger.info(message)
        return entry

    def _toast(self, message: str, icon: str = "⚠️"):
        try:
            st.toast(f"{icon} {message}")
        except Exception:
            pass  # outside streamlit context

    # ─── threshold checks ────────────────────────────────────────────────────

    def check_deal_scores(self, matched_df: pd.DataFrame) -> list[str]:
        """Alert if any deal scores ≥ threshold."""
        alerts = []
        if matched_df is None or matched_df.empty:
            return alerts
        high = matched_df[matched_df["Score"] >= self.score_threshold]
        for _, row in high.iterrows():
            msg = (
                f"HIGH SCORE DEAL: {row['Client Name']} "
                f"{'BUY' if row['Buy/Sell'] == 'BUY' else 'SELL'} "
                f"score={row['Score']} at {row['Nearest_Strike']:.0f}"
            )
            level = "buy" if row["Buy/Sell"] == "BUY" else "sell"
            entry = self._log(msg, level)
            self._toast(msg, "🟢" if level == "buy" else "🔴")
            alerts.append(entry)
        return alerts

    def check_oi_spike(self, current_walls: pd.DataFrame) -> list[str]:
        """Alert if OI at any wall strike changes > spike_pct vs prior load."""
        alerts = []
        prior = st.session_state.get(self.PRIOR_WALLS_KEY)
        if prior is None or current_walls is None or current_walls.empty:
            st.session_state[self.PRIOR_WALLS_KEY] = (
                current_walls[["Strike", "Total_OI"]].copy() if current_walls is not None else None
            )
            return alerts

        merged = current_walls[["Strike", "Total_OI"]].merge(
            prior.rename(columns={"Total_OI": "Prior_OI"}), on="Strike", how="inner"
        )
        merged["Change_%"] = (merged["Total_OI"] - merged["Prior_OI"]) / merged["Prior_OI"].clip(lower=1) * 100

        spikes = merged[merged["Change_%"].abs() > self.oi_spike_pct]
        for _, row in spikes.iterrows():
            direction = "↑" if row["Change_%"] > 0 else "↓"
            msg = (
                f"OI SPIKE at {row['Strike']:.0f}: {direction}{abs(row['Change_%']):.1f}% change "
                f"({row['Prior_OI']:,.0f} → {row['Total_OI']:,.0f})"
            )
            entry = self._log(msg, "watch")
            self._toast(msg, "⚠️")
            alerts.append(entry)

        st.session_state[self.PRIOR_WALLS_KEY] = current_walls[["Strike", "Total_OI"]].copy()
        return alerts

    def check_pcr_crossover(self, current_pcr: float) -> list[str]:
        """Alert if PCR crosses the 0.7 or 1.3 thresholds."""
        alerts = []
        prior = st.session_state.get(self.PRIOR_PCR_KEY)
        if prior is not None:
            for threshold in [self.pcr_low, self.pcr_high]:
                crossed = (prior < threshold <= current_pcr) or (prior > threshold >= current_pcr)
                if crossed:
                    direction = "crossed UP" if current_pcr > prior else "crossed DOWN"
                    level = "buy" if current_pcr < threshold else "sell"
                    msg = f"PCR CROSSOVER: PCR {direction} {threshold} (now {current_pcr:.3f})"
                    entry = self._log(msg, level)
                    self._toast(msg, "📊")
                    alerts.append(entry)
        st.session_state[self.PRIOR_PCR_KEY] = current_pcr
        return alerts

    def check_max_pain_migration(self, current_max_pain: float, step: float = 10.0) -> list[str]:
        """Alert if Max Pain moves by more than 1 strike since morning."""
        alerts = []
        prior = st.session_state.get(self.PRIOR_MAX_PAIN_KEY)
        if prior is not None:
            diff = abs(current_max_pain - prior)
            if diff >= step:
                direction = "UP" if current_max_pain > prior else "DOWN"
                msg = (
                    f"MAX PAIN MIGRATION: moved {direction} by {diff:.0f} pts "
                    f"({prior:.0f} → {current_max_pain:.0f})"
                )
                entry = self._log(msg, "watch")
                self._toast(msg, "🎯")
                alerts.append(entry)
        st.session_state[self.PRIOR_MAX_PAIN_KEY] = current_max_pain
        return alerts

    def check_signal_flip(self, current_label: str) -> list[str]:
        """Alert if composite signal direction flips."""
        alerts = []
        prior = st.session_state.get(self.PRIOR_SIGNAL_KEY)

        def is_bullish(label):
            return "BULLISH" in label.upper() or "bullish" in label.lower()

        def is_bearish(label):
            return "BEARISH" in label.upper() or "bearish" in label.lower()

        if prior:
            flipped = (is_bullish(prior) and is_bearish(current_label)) or (
                is_bearish(prior) and is_bullish(current_label)
            )
            if flipped:
                msg = f"SIGNAL FLIP: {prior} → {current_label}"
                level = "buy" if is_bullish(current_label) else "sell"
                entry = self._log(msg, level)
                self._toast(msg, "🔄")
                alerts.append(entry)

        st.session_state[self.PRIOR_SIGNAL_KEY] = current_label
        return alerts

    def run_all_checks(
        self,
        matched_df: Optional[pd.DataFrame],
        walls_df: Optional[pd.DataFrame],
        pcr: Optional[float],
        max_pain: Optional[float],
        signal_label: Optional[str],
        strike_step: float = 10.0,
    ) -> list[str]:
        """Run all checks in one call; return combined alert list."""
        alerts = []
        if matched_df is not None:
            alerts += self.check_deal_scores(matched_df)
        if walls_df is not None:
            alerts += self.check_oi_spike(walls_df)
        if pcr is not None:
            alerts += self.check_pcr_crossover(pcr)
        if max_pain is not None:
            alerts += self.check_max_pain_migration(max_pain, step=strike_step)
        if signal_label is not None:
            alerts += self.check_signal_flip(signal_label)
        return alerts

    # ─── session log ─────────────────────────────────────────────────────────

    def get_log(self) -> list[str]:
        return st.session_state.get(self.SESSION_KEY, [])

    def clear_log(self):
        st.session_state[self.SESSION_KEY] = []

    def render_alert_log(self):
        """Render the alert log inside a Streamlit expander."""
        log = self.get_log()
        if not log:
            st.info("No alerts triggered in this session.")
            return
        for entry in reversed(log[-50:]):  # show last 50
            if "🟢" in entry:
                st.success(entry)
            elif "🔴" in entry:
                st.error(entry)
            else:
                st.warning(entry)

    # ─── optional desktop notification ───────────────────────────────────────

    @staticmethod
    def desktop_notify(title: str, message: str):
        try:
            from plyer import notification
            notification.notify(title=title, message=message, timeout=5)
        except Exception:
            pass
