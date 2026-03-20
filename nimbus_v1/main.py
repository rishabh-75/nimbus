"""
NIMBUS · Emerald Slate v5 · Qt Desktop Dashboard
─────────────────────────────────────────────────
Entry point: logging → pyqtgraph config → QApplication → MainWindow.

Logging is configured FIRST (before any other imports that might log).
pyqtgraph global config is set BEFORE any ui/ imports (roadmap §0.1).
"""

from __future__ import annotations

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# ── Logging (roadmap §0.3 — first 10 lines after imports) ────────────────────
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, "nimbus.log")

handler = RotatingFileHandler(_log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
logging.basicConfig(
    handlers=[handler, logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nimbus")
logger.info("NIMBUS starting — log path: %s", _log_path)

# ── pyqtgraph global config (BEFORE any ui/ imports) ─────────────────────────
import pyqtgraph as pg

pg.setConfigOptions(
    background="#0D1117",  # SURFACE
    foreground="#64748B",  # MUTED
    antialias=True,
)

# ── Qt imports ────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFontDatabase, QFont
from PyQt6.QtCore import Qt

# ── App imports (AFTER pyqtgraph config) ──────────────────────────────────────
from ui.main_window import MainWindow


def _load_fonts():
    """Load bundled JetBrains Mono from assets/ via QFontDatabase."""
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    loaded = 0
    for fname in ("JetBrainsMono-Regular.ttf", "JetBrainsMono-Bold.ttf"):
        path = os.path.join(assets_dir, fname)
        if os.path.exists(path):
            fid = QFontDatabase.addApplicationFont(path)
            if fid >= 0:
                families = QFontDatabase.applicationFontFamilies(fid)
                logger.info("Loaded font: %s → %s", fname, families)
                loaded += 1
            else:
                logger.warning("Failed to load font: %s", path)
        else:
            logger.info("Font file not found (optional): %s", path)
    return loaded


def main():
    # High-DPI scaling (Qt6 does this by default, but be explicit)
    app = QApplication(sys.argv)
    app.setApplicationName("NIMBUS")
    app.setOrganizationName("NIMBUS")

    # Load bundled fonts
    _load_fonts()

    # Verify JetBrains Mono is available (for the test label gate check)
    mono_available = "JetBrains Mono" in QFontDatabase.families()
    if mono_available:
        logger.info("JetBrains Mono: available ✓")
    else:
        logger.warning(
            "JetBrains Mono: NOT available — "
            "place .ttf files in assets/ for best rendering"
        )

    # Verify modules/ imports cleanly (gate 0.2)
    try:
        from modules import analytics, indicators, data, scanner

        logger.info("modules/ imported successfully ✓")
    except Exception as exc:
        logger.error("modules/ import failed: %s", exc)
        # Non-fatal — app still launches, but data pipeline won't work

    # Launch main window
    window = MainWindow()
    window.show()
    logger.info("MainWindow shown — Phase 0 gate open")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
