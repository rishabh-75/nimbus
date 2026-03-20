"""
ui/kite_manager.py
──────────────────
Zerodha Kite integration for NIMBUS Qt — Phase 7.

KiteSessionManager:
    - Reads/writes data/.kite_session (JSON: api_key, api_secret, access_token)
    - Validates on startup via kite.profile()
    - Daily 09:00 IST pre-market check timer
    - Re-auth flow: open browser → user pastes request_token → generate_session()
    - Falls back to yfinance polling when Kite unavailable

TickerWorker:
    - QThread running KiteTicker WebSocket
    - Emits tick_received(list) for real-time price updates
    - Emits connected/disconnected signals

Credentials are NEVER hardcoded. Stored in data/.kite_session (gitignored).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
import webbrowser
from collections import deque
from typing import Optional

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

logger = logging.getLogger(__name__)

_SESSION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", ".kite_session"
)

# IST offset
_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

# Instrument token mapping for common NSE symbols
# (These need to be fetched from Kite API instruments list for production)
_KNOWN_TOKENS = {
    "NIFTY":     256265,
    "BANKNIFTY": 260105,
    "NIFTY":     256265,
}


def _kite_available() -> bool:
    """Check if kiteconnect package is installed."""
    try:
        import kiteconnect
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class KiteSessionManager(QObject):
    """
    Manages Kite session lifecycle: validate, re-auth, daily check.

    Signals:
        session_ready(access_token)  — valid session, ticker can connect
        session_invalid(error_msg)   — no valid session, fall back to yfinance
    """

    session_ready   = pyqtSignal(str)    # access_token
    session_invalid = pyqtSignal(str)    # error message

    def __init__(self, parent=None):
        super().__init__(parent)

        self._api_key:      Optional[str] = None
        self._api_secret:   Optional[str] = None
        self._access_token: Optional[str] = None
        self._kite = None

        # ── Daily 09:00 IST pre-market check ──────────────────────────────────
        self._daily_timer = QTimer(self)
        self._daily_timer.timeout.connect(self._daily_check)
        self._schedule_daily_check()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def validate_on_startup(self):
        """
        Called once at app startup. Reads session file, validates token.
        Emits session_ready or session_invalid.
        """
        if not _kite_available():
            logger.info("kiteconnect not installed — Kite features disabled")
            self.session_invalid.emit("kiteconnect not installed")
            return

        session = self._load_session()
        if not session:
            logger.info("No Kite session file — yfinance fallback")
            self.session_invalid.emit("No session file")
            return

        self._api_key    = session.get("api_key")
        self._api_secret = session.get("api_secret")
        self._access_token = session.get("access_token")

        if not self._api_key or not self._access_token:
            self.session_invalid.emit("Incomplete session file")
            return

        # Validate token
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self._api_key)
            self._kite.set_access_token(self._access_token)
            profile = self._kite.profile()
            logger.info(
                "Kite session valid: %s (%s)",
                profile.get("user_name", "?"), profile.get("user_id", "?"),
            )
            self.session_ready.emit(self._access_token)
        except Exception as exc:
            logger.warning("Kite token invalid: %s", exc)
            self._kite = None
            self.session_invalid.emit(str(exc))

    def start_reauth(self):
        """
        Open Kite login in system browser for re-authentication.
        Returns the login URL (caller should show a QInputDialog for request_token).
        """
        if not _kite_available() or not self._api_key:
            return None

        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=self._api_key)
        login_url = kite.login_url()
        webbrowser.open(login_url)
        logger.info("Kite login URL opened: %s", login_url)
        return login_url

    def complete_reauth(self, request_token: str) -> bool:
        """
        Complete re-auth with the request_token from browser callback.
        Returns True if successful.
        """
        if not _kite_available() or not self._api_key or not self._api_secret:
            return False

        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self._api_key)
            data = kite.generate_session(request_token, api_secret=self._api_secret)
            self._access_token = data["access_token"]
            kite.set_access_token(self._access_token)

            # Validate
            kite.profile()

            # Save updated session
            self._save_session({
                "api_key":      self._api_key,
                "api_secret":   self._api_secret,
                "access_token": self._access_token,
            })

            self._kite = kite
            logger.info("Kite re-auth successful")
            self.session_ready.emit(self._access_token)
            return True
        except Exception as exc:
            logger.error("Kite re-auth failed: %s", exc)
            self.session_invalid.emit(str(exc))
            return False

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    @property
    def access_token(self) -> Optional[str]:
        return self._access_token

    @property
    def is_valid(self) -> bool:
        return self._kite is not None and self._access_token is not None

    # ──────────────────────────────────────────────────────────────────────────
    # SESSION FILE I/O
    # ──────────────────────────────────────────────────────────────────────────

    def _load_session(self) -> Optional[dict]:
        try:
            if not os.path.exists(_SESSION_PATH):
                return None
            with open(_SESSION_PATH) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not read .kite_session: %s", exc)
            return None

    def _save_session(self, data: dict):
        try:
            os.makedirs(os.path.dirname(_SESSION_PATH), exist_ok=True)
            with open(_SESSION_PATH, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Kite session saved to %s", _SESSION_PATH)
        except Exception as exc:
            logger.error("Could not write .kite_session: %s", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # DAILY CHECK
    # ──────────────────────────────────────────────────────────────────────────

    def _schedule_daily_check(self):
        """Schedule timer to fire at 09:00 IST each day."""
        now = datetime.datetime.now(_IST)
        target = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if now >= target:
            target += datetime.timedelta(days=1)
        # Skip weekends
        while target.weekday() >= 5:
            target += datetime.timedelta(days=1)

        delay_ms = int((target - now).total_seconds() * 1000)
        delay_ms = max(delay_ms, 60000)  # at least 1 minute
        self._daily_timer.start(delay_ms)
        logger.info(
            "Kite daily check scheduled for %s (in %d min)",
            target.strftime("%Y-%m-%d %H:%M IST"), delay_ms // 60000,
        )

    def _daily_check(self):
        """09:00 IST pre-market session validation."""
        self._daily_timer.stop()
        logger.info("Kite daily pre-market check firing")
        self.validate_on_startup()
        self._schedule_daily_check()  # reschedule for next trading day


# ══════════════════════════════════════════════════════════════════════════════
# TICKER WORKER (WebSocket)
# ══════════════════════════════════════════════════════════════════════════════

class TickerWorker(QThread):
    """
    QThread running KiteTicker WebSocket for real-time ticks.

    Signals:
        tick_received(list)   — raw tick dicts from Kite
        connected()           — WebSocket connected
        disconnected(str)     — WebSocket disconnected with reason
    """

    tick_received  = pyqtSignal(list)
    connected      = pyqtSignal()
    disconnected   = pyqtSignal(str)

    def __init__(self, api_key: str, access_token: str,
                 instrument_tokens: list[int] = None, parent=None):
        super().__init__(parent)
        self._api_key      = api_key
        self._access_token = access_token
        self._tokens       = instrument_tokens or [256265]  # NIFTY by default
        self._kt           = None
        self._running       = True

    def run(self):
        if not _kite_available():
            self.disconnected.emit("kiteconnect not installed")
            return

        try:
            from kiteconnect import KiteTicker

            logger.info(
                "TickerWorker starting: %d instruments", len(self._tokens)
            )
            self._kt = KiteTicker(self._api_key, self._access_token)

            def on_ticks(ws, ticks):
                if self._running:
                    self.tick_received.emit(ticks)

            def on_connect(ws, response):
                logger.info("KiteTicker connected")
                ws.subscribe(self._tokens)
                ws.set_mode(ws.MODE_FULL, self._tokens)
                self.connected.emit()

            def on_close(ws, code, reason):
                logger.info("KiteTicker closed: %s %s", code, reason)
                if self._running:
                    self.disconnected.emit(str(reason or "Connection closed"))

            def on_error(ws, code, reason):
                logger.error("KiteTicker error: %s %s", code, reason)

            self._kt.on_ticks   = on_ticks
            self._kt.on_connect = on_connect
            self._kt.on_close   = on_close
            self._kt.on_error   = on_error

            self._kt.connect(threaded=False)  # blocks in this QThread

        except Exception as exc:
            logger.error("TickerWorker error: %s", exc)
            self.disconnected.emit(str(exc))

    def stop(self):
        """Gracefully stop the ticker."""
        self._running = False
        if self._kt:
            try:
                self._kt.close()
            except Exception:
                pass

    def update_tokens(self, tokens: list[int]):
        """Subscribe to new instrument tokens."""
        self._tokens = tokens
        if self._kt:
            try:
                self._kt.subscribe(tokens)
                self._kt.set_mode(self._kt.MODE_FULL, tokens)
            except Exception as exc:
                logger.warning("Could not update tokens: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# TICK AGGREGATOR (4H bar construction)
# ══════════════════════════════════════════════════════════════════════════════

class TickAggregator:
    """
    Accumulates ticks into 4H OHLCV bars.
    Emits bar_complete signal when a 4H boundary is crossed.

    4H boundaries for NSE (IST):
        09:15 → 13:15 (bar 1)
        13:15 → 15:30 (bar 2, truncated)
    """

    def __init__(self):
        self._ticks: deque = deque(maxlen=100000)
        self._current_bar: Optional[dict] = None
        self._bar_start: Optional[datetime.datetime] = None

    def add_tick(self, tick: dict) -> Optional[dict]:
        """
        Process a tick. Returns a completed 4H bar dict if boundary crossed,
        else None.
        """
        ltp = tick.get("last_price")
        ts  = tick.get("exchange_timestamp") or tick.get("timestamp")
        vol = tick.get("volume_traded", 0)

        if ltp is None or ts is None:
            return None

        if isinstance(ts, str):
            ts = datetime.datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_IST)

        self._ticks.append({"price": ltp, "ts": ts, "volume": vol})

        # Determine which 4H bucket this tick belongs to
        hour = ts.hour
        if hour < 13:
            bar_boundary = ts.replace(hour=13, minute=15, second=0, microsecond=0)
        else:
            bar_boundary = ts.replace(hour=15, minute=30, second=0, microsecond=0)

        completed_bar = None

        if self._current_bar is None:
            # Start a new bar
            self._current_bar = {
                "Open": ltp, "High": ltp, "Low": ltp, "Close": ltp,
                "Volume": vol,
            }
            self._bar_start = bar_boundary
        elif bar_boundary != self._bar_start:
            # Boundary crossed — complete the old bar
            completed_bar = dict(self._current_bar)
            completed_bar["timestamp"] = self._bar_start
            # Start new bar
            self._current_bar = {
                "Open": ltp, "High": ltp, "Low": ltp, "Close": ltp,
                "Volume": vol,
            }
            self._bar_start = bar_boundary
        else:
            # Same bar — update OHLCV
            self._current_bar["High"]   = max(self._current_bar["High"], ltp)
            self._current_bar["Low"]    = min(self._current_bar["Low"], ltp)
            self._current_bar["Close"]  = ltp
            self._current_bar["Volume"] = vol  # Kite gives cumulative volume

        return completed_bar

    @property
    def last_price(self) -> Optional[float]:
        if self._current_bar:
            return self._current_bar["Close"]
        return None
