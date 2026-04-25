"""
agent/input/input_config.py — Input Source Configuration
=========================================================

Controls which video source feeds the ML pipeline.

Values:
    "MEET"    — Google Meet only. Webcam never starts.
    "WEBCAM"  — Webcam only. MeetBridge still runs (for /result endpoint) but
                frames from it are silently dropped.
    "AUTO"    — Meet preferred. Webcam starts immediately as fallback.
                If Meet frames arrive, webcam is stopped. If Meet goes silent
                for MEET_IDLE_TIMEOUT_SEC, webcam restarts (future).

Change this file to switch input mode without touching any other code.
Or pass --input-mode on the command line (overrides this file).
"""

# ── Primary setting ───────────────────────────────────────────────────────────
INPUT_MODE: str = "MEET"   # "MEET" | "WEBCAM" | "AUTO"

# ── AUTO-mode tuning ──────────────────────────────────────────────────────────
# How long to wait (seconds) after startup before deciding Meet is not active
# and starting the webcam fallback.  0 = start webcam immediately in AUTO mode.
AUTO_WEBCAM_START_DELAY_SEC: float = 0.0

# If Meet goes silent for this many seconds while webcam is stopped,
# the arbiter will restart the webcam.  Set to 0 to disable auto-restart.
MEET_IDLE_TIMEOUT_SEC: float = 0.0   # 0 = no auto-restart (conservative)

# ── Flask receiver port ────────────────────────────────────────────────────────
FLASK_PORT: int = 5001
FLASK_HOST: str = "127.0.0.1"
