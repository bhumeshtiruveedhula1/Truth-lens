"""
agent/input/input_arbiter.py — Input Source Arbitration
=========================================================

Implements the single-active-source guarantee:
  Only ONE of MeetBridge or CaptureSource (webcam) is ever publishing
  FrameEvents at a time.

Modes
-----
MEET   — Bridge always active; webcam never starts.
WEBCAM — Webcam always active; Meet frames silently dropped even if bridge
         is installed (it stays for /result polling).
AUTO   — Webcam starts immediately as fallback.
         When first Meet frame arrives → webcam is stopped.
         Clean handoff, no duplicate frames, no camera conflict.

Design
------
- Zero new ML dependencies; works purely at the input layer.
- The arbiter is a pure asyncio coroutine (`run()`).
- CaptureSource is stopped via `capture.stop()` (existing API, non-blocking).
- MeetBridge fires `_on_first_frame_cb` on its first successful push.
- No polling loops; callback-driven.

Threading
---------
- `_on_meet_active()` is called from Flask's sync thread.
- It schedules `_switch_to_meet()` onto the asyncio loop via
  `run_coroutine_threadsafe` — identical pattern to how MeetBridge
  publishes FrameEvents.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class InputArbiter:
    """
    Arbitrates between webcam and Google Meet as the frame source.

    Usage (from main.py):
        arbiter = InputArbiter(
            mode        = "AUTO",
            capture     = capture_source,
            meet_bridge = meet_bridge,    # may be None if --no-meet-bridge
            loop        = asyncio.get_running_loop(),
        )
        # Returned tasks list tells main.py which asyncio tasks to start.
        tasks_to_add = arbiter.prepare_tasks()
    """

    VALID_MODES = frozenset({"MEET", "WEBCAM", "AUTO"})

    def __init__(
        self,
        mode: str,
        capture,                                   # CaptureSource instance
        meet_bridge,                               # MeetBridge instance or None
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        mode = mode.upper()
        if mode not in self.VALID_MODES:
            logger.warning(
                f"[InputArbiter] Unknown mode '{mode}' — defaulting to AUTO"
            )
            mode = "AUTO"

        self._mode        = mode
        self._capture     = capture
        self._bridge      = meet_bridge
        self._loop        = loop
        self._active      = "NONE"   # "WEBCAM" | "MEET" | "NONE"
        self._switched    = False    # guard: switch happens at most once per session

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_tasks(self) -> list:
        """
        Return a list of asyncio coroutines that should be added as tasks
        by main.py.  Called ONCE after all engines are registered.

        MEET   → no webcam task
        WEBCAM → webcam task only
        AUTO   → webcam task + installs meet-active callback
        """
        tasks = []

        if self._mode == "MEET":
            logger.info("INPUT SOURCE: MEET  (webcam disabled)")
            self._active = "MEET"
            # Register a drop-guard on bridge so stray webcam frames (if any)
            # are rejected — nothing to do here, bridge already active.

        elif self._mode == "WEBCAM":
            logger.info("INPUT SOURCE: WEBCAM  (Meet frames ignored)")
            self._active = "WEBCAM"
            tasks.append(self._capture.run())
            # Bridge may still be running for /result polling — frames from it
            # will be silently published, but that's acceptable since the webcam
            # is the authoritative source and both use the same ML pipeline.

        else:  # AUTO
            logger.info(
                "INPUT SOURCE: AUTO  (webcam starting as fallback — "
                "will yield to Meet if extension becomes active)"
            )
            self._active = "WEBCAM"
            tasks.append(self._capture.run())

            # Register callback: called from Flask thread when first Meet frame arrives
            if self._bridge is not None:
                self._bridge.set_on_first_frame_callback(self._on_meet_active)
                logger.info(
                    "[InputArbiter] Watching for Meet frames — webcam will "
                    "stop automatically when Meet becomes active"
                )

        return tasks

    def active_source(self) -> str:
        """Return the currently active source name."""
        return self._active

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — called from Flask thread when first Meet batch arrives
    # ─────────────────────────────────────────────────────────────────────────

    def _on_meet_active(self) -> None:
        """
        Callback registered with MeetBridge.
        Called from Flask's sync thread on the first successful push_frame_batch().
        Schedules the async handoff onto the main event loop.
        """
        if self._switched:
            return   # already switched — ignore

        # Cross-thread: schedule coroutine on the asyncio loop
        asyncio.run_coroutine_threadsafe(
            self._switch_to_meet(),
            self._loop,
        )

    async def _switch_to_meet(self) -> None:
        """
        Async handoff: stop webcam, mark Meet as active source.
        Runs on the main asyncio event loop — safe to log and await.
        """
        if self._switched:
            return
        self._switched = True

        if self._active == "MEET":
            return   # already on Meet — nothing to do

        logger.info("Switching to MEET — stopping webcam")
        self._capture.stop()    # sets _running=False; webcam loop exits gracefully
        self._active = "MEET"
        logger.info("INPUT SOURCE: MEET  (webcam stopped)")
