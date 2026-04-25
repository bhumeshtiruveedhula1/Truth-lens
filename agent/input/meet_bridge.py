"""
agent/input/meet_bridge.py — Google Meet Frame Source Adapter
==============================================================
Bridges frames arriving from the external Flask receiver (PERSONA_SHIELD)
into the existing FrameEvent-based ML pipeline WITHOUT modifying any
existing module.

Contract:
  - Accepts a list of decoded BGR numpy frames (one batch from the extension)
  - Runs MediaPipe FaceMesh on each frame (same as CaptureSource does)
  - Builds a FrameEvent per frame using the exact same schema as CaptureSource
  - Publishes each event onto the shared event bus with ~33ms spacing
  - Does NOT touch GRU / CNN / Identity / Deepfake / Fusion / event bus internals

Threading model:
  Flask runs in a plain synchronous WSGI thread.
  The event bus uses asyncio (runs in the main asyncio event loop started by main.py).
  Bridge uses asyncio.run_coroutine_threadsafe() to safely schedule each
  bus.publish() call onto the existing loop without creating a new one.

Usage:
  # In main.py (once, after asyncio loop is up):
  from agent.input.meet_bridge import MeetBridge
  bridge = MeetBridge(session_id=session_id)
  bridge.set_loop(asyncio.get_event_loop())

  # In server.py (per request):
  from agent.input.meet_bridge import get_bridge
  get_bridge().push_frame_batch(decoded_frames)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from agent.event_bus import bus
from agent.events import FrameEvent

logger = logging.getLogger(__name__)

# ── Timing constants ──────────────────────────────────────────────────────────
# Space frames at ~30 fps to match webcam cadence.
# The sleep is done with time.sleep() (blocking) inside the thread-pool executor
# so the asyncio event loop is never stalled.
_FRAME_INTERVAL_SEC: float = 1.0 / 30.0   # ~33 ms between frames

# ── FaceMesh config — must match CaptureSource exactly ───────────────────────
_MAX_FACES              = 1
_REFINE_LANDMARKS       = True
_MIN_DETECTION_CONF     = 0.5
_MIN_TRACKING_CONF      = 0.5
_FACE_PAD_PX            = 20   # padding around landmark bbox for ROI crop

# ── Module-level singleton ────────────────────────────────────────────────────
_bridge_instance: Optional["MeetBridge"] = None


def get_bridge() -> "MeetBridge":
    """
    Return the global MeetBridge singleton.
    Raises RuntimeError if set_bridge() has not been called yet.
    """
    if _bridge_instance is None:
        raise RuntimeError(
            "[MeetBridge] Bridge not initialised. "
            "Call MeetBridge.install(session_id, loop) from main.py first."
        )
    return _bridge_instance


class MeetBridge:
    """
    Input adapter: receives decoded BGR frames from the Flask receiver and
    injects them into the existing FrameEvent pipeline.

    This class is the ONLY new component. Nothing downstream changes.
    """

    def __init__(self, session_id: str, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._session_id        = session_id
        self._loop              = loop
        self._frame_id          = 0
        self._ts_counter        = 0   # strictly-increasing counter fed to FrameEvent
        self._face_mesh         = None
        self._mesh_lock         = False
        self._fusion_engine     = None
        self._on_first_frame_cb = None
        self._first_frame_fired = False

        logger.info(f"[MeetBridge] created  session={session_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the running asyncio loop. Call from main.py after startup."""
        self._loop = loop
        logger.info("[MeetBridge] asyncio loop registered")

    def set_on_first_frame_callback(self, cb: "Callable[[], None]") -> None:
        """
        Register a zero-argument callable that fires the first time
        push_frame_batch() successfully processes frames.
        Used by InputArbiter to trigger webcam → Meet handoff.
        """
        self._on_first_frame_cb = cb

    def set_fusion_engine(self, fusion_engine) -> None:
        """
        Register the FusionEngine instance so get_latest_fusion_result()
        can read its latest_result dict from any thread (GIL-safe).
        Call from main.py after FusionEngine is constructed.
        """
        self._fusion_engine = fusion_engine
        logger.info("[MeetBridge] FusionEngine registered")

    def get_latest_fusion_result(self) -> dict:
        """
        Return a JSON-serialisable snapshot of the latest FusionEngine verdict.
        Thread-safe: dict assignment under CPython GIL is atomic.
        Called by server.py /result endpoint from Flask's sync thread.
        """
        if self._fusion_engine is None:
            return {
                "status":       "PIPELINE_NOT_READY",
                "final_status": None,
                "reason":       "FusionEngine not registered",
            }
        result = self._fusion_engine.latest_result
        # Return a shallow copy — avoids mutation races
        return dict(result)

    def push_frame_batch(self, frames: List[np.ndarray]) -> None:
        """
        Entry point called by Flask server on each incoming batch.

        Args:
            frames: list of BGR numpy arrays (decoded from JPEG).
                    Order must be temporal (oldest → newest).

        Behaviour:
            - Iterates frames sequentially (never parallel / batch)
            - Runs MediaPipe FaceMesh on each frame (off-thread via executor)
            - Builds a FrameEvent per frame
            - Schedules bus.publish(event) on the main asyncio loop
            - Sleeps _FRAME_INTERVAL_SEC between frames to simulate 30 fps
        """
        if not frames:
            logger.debug("[MeetBridge] push_frame_batch: empty list — skip")
            return

        if self._loop is None:
            logger.warning(
                "[MeetBridge] push_frame_batch called before loop registered — "
                "frames dropped. Call set_loop() from main.py."
            )
            return

        self._ensure_face_mesh()

        # ── Fire first-frame callback (once) ─────────────────────────────────
        # Called here, before the loop, so the arbiter can stop the webcam
        # before any FrameEvents from this batch enter the pipeline.
        if not self._first_frame_fired and self._on_first_frame_cb is not None:
            self._first_frame_fired = True
            logger.info("[MeetBridge] first Meet batch received — firing source callback")
            try:
                self._on_first_frame_cb()
            except Exception as exc:
                logger.warning(f"[MeetBridge] on_first_frame_cb error: {exc}")

        n = len(frames)
        logger.debug(f"[MeetBridge] push_frame_batch: {n} frames")

        for idx, bgr in enumerate(frames):
            if bgr is None or not isinstance(bgr, np.ndarray):
                logger.debug(f"[MeetBridge] frame {idx}: invalid — skipped")
                continue

            # Build FrameEvent — wrapped in try/except so a MediaPipe crash
            # on one frame does NOT abort the entire batch.
            try:
                event = self._build_frame_event(bgr)
            except Exception as exc:
                logger.warning(
                    f"[MeetBridge] frame {idx}: FaceMesh failed — skipped  ({exc})"
                )
                continue

            future = asyncio.run_coroutine_threadsafe(
                bus.publish(event),
                self._loop,
            )
            try:
                future.result(timeout=0.10)
            except Exception as exc:
                logger.warning(f"[MeetBridge] publish failed frame {idx}: {exc}")

            if idx < n - 1:
                time.sleep(_FRAME_INTERVAL_SEC)

        print(
            f"[MeetBridge] batch complete — {n} frames pushed  "
            f"session={self._session_id}  last_frame_id={self._frame_id}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — FaceMesh lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_face_mesh(self) -> None:
        """Lazily initialise FaceMesh on first use."""
        if self._face_mesh is not None:
            return
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces            = _MAX_FACES,
            refine_landmarks         = _REFINE_LANDMARKS,
            min_detection_confidence = _MIN_DETECTION_CONF,
            min_tracking_confidence  = _MIN_TRACKING_CONF,
            # CRITICAL: static_image_mode=True processes each frame independently.
            # This eliminates the "Packet timestamp mismatch" error that occurs
            # when frames arrive from Flask with non-monotonic wall-clock gaps.
            # Tracking mode (False) requires strictly increasing timestamps across
            # calls, which batched HTTP frames cannot guarantee.
            static_image_mode        = True,
        )
        logger.info("[MeetBridge] FaceMesh initialised (static_image_mode=True — no timestamp errors)")

    def close(self) -> None:
        """Release FaceMesh resources. Call on shutdown."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        logger.info("[MeetBridge] closed")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — FrameEvent construction (mirrors CaptureSource._process_frame)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_frame_event(self, bgr: np.ndarray) -> FrameEvent:
        """
        Run MediaPipe FaceMesh on a BGR frame and build a FrameEvent.

        Uses a monotonically-increasing _ts_counter as frame_id so that
        FrameEvents from consecutive batches always have distinct IDs —
        required by any downstream module that orders by frame_id.
        """
        self._frame_id   += 1
        self._ts_counter += 1       # strictly increasing, never reused
        h, w = bgr.shape[:2]

        # MediaPipe expects RGB; writeable=False avoids an unnecessary copy
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        # FaceMesh.process() is synchronous and NOT thread-safe.
        # Called here from the Flask thread — only one thread calls this
        # at a time because push_frame_batch() is synchronous.
        results = self._face_mesh.process(rgb)

        landmarks:      list                      = []
        face_bbox:      tuple                     = (0, 0, 0, 0)
        face_detected:  bool                      = False
        face_roi_bgr:   Optional[bytes]           = None
        face_roi_shape: Optional[tuple[int, int]] = None

        if results and results.multi_face_landmarks:
            face_detected = True
            lm            = results.multi_face_landmarks[0]
            landmarks     = [(p.x, p.y, p.z) for p in lm.landmark]

            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            face_bbox = (x1, y1, x2 - x1, y2 - y1)

            rx1 = max(0, x1 - _FACE_PAD_PX)
            ry1 = max(0, y1 - _FACE_PAD_PX)
            rx2 = min(w, x2 + _FACE_PAD_PX)
            ry2 = min(h, y2 + _FACE_PAD_PX)
            if rx2 > rx1 and ry2 > ry1:
                roi            = bgr[ry1:ry2, rx1:rx2]
                face_roi_shape = roi.shape[:2]
                face_roi_bgr   = roi.tobytes()

        return FrameEvent(
            session_id     = self._session_id,
            frame_id       = self._ts_counter,   # monotonic — never repeats
            face_landmarks = landmarks,
            face_bbox      = face_bbox,
            face_detected  = face_detected,
            frame_width    = w,
            frame_height   = h,
            face_roi_bgr   = face_roi_bgr,
            face_roi_shape = face_roi_shape,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Class-level factory — singleton management
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def install(cls, session_id: str, loop: asyncio.AbstractEventLoop) -> "MeetBridge":
        """
        Create and register the global singleton.
        Call once from main.py after the asyncio loop is running.

        Example (main.py):
            loop = asyncio.get_event_loop()
            bridge = MeetBridge.install(session_id=session_id, loop=loop)
        """
        global _bridge_instance
        _bridge_instance = cls(session_id=session_id, loop=loop)
        logger.info(
            f"[MeetBridge] installed  session={session_id}  "
            f"loop={id(loop):#x}"
        )
        return _bridge_instance
