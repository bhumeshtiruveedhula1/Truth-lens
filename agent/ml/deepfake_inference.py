"""
agent/ml/deepfake_inference.py -- Xception Deepfake Detection Engine
=====================================================================
Wraps deepfake_model.py (HongguLiu/Deepfake-Detection, Xception backbone)
as a non-blocking, throttled inference module that integrates with the
DeepShield event pipeline.

Design:
  - Subscribes to FusionEvent on the bus
  - Triggers inference ONLY when:
      (a) cnn_fake_probability > 0.6, OR
      (b) fusion_status == "WARNING" or "HIGH_RISK", OR
      (c) time since last run > 1.0 second
  - Minimum gap between consecutive runs: 300ms
  - Runs Xception forward pass in ThreadPoolExecutor (non-blocking)
  - Caches latest_result for DebugUI polling

Weights:
  Download df_c0_best.pkl from:
  https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8
  Place at: models/deepfake_xception.pkl

network/ package:
  Copied from Desktop/Deepfake-Detection/network/ into this project's
  agent/ml/deepfake_net/ to avoid path hacks at runtime.
  Alternatively, set DEEPFAKE_REPO env var to the repo root.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configurable paths ────────────────────────────────────────────────────────
_DEFAULT_WEIGHTS = Path("models/deepfake_xception.pkl")

# Repo root for deepfake_model.py + network/ package
# Priority: env var → sibling dir → Desktop default
_REPO_CANDIDATES = [
    os.environ.get("DEEPFAKE_REPO", ""),
    str(Path(__file__).parent / "deepfake_net"),          # bundled copy
    r"C:\Users\bhumeshjyothi\Desktop\Deepfake-Detection", # original location
]

# ── Throttle constants ────────────────────────────────────────────────────────
_MIN_GAP_SEC   = 0.30   # minimum 300ms between inference calls
_MAX_GAP_SEC   = 1.00   # force a run if >1s since last call
_CNN_TRIGGER   = 0.60   # CNN threshold to trigger
_FUSION_STATES = {"WARNING", "HIGH_RISK"}  # fusion states that trigger

# Raised from 0.50 — Xception produces mid-range scores for real faces;
# 0.75 requires high confidence before classifying as FAKE.
DEEPFAKE_THRESHOLD = 0.75

def _add_repo_to_path() -> bool:
    """Add deepfake repo root to sys.path so network/ can be imported."""
    for candidate in _REPO_CANDIDATES:
        if not candidate:
            continue
        p = Path(candidate)
        if (p / "network" / "models.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            logger.info(f"DeepfakeEngine: using network/ from {p}")
            return True
    return False


class DeepfakeInferenceEngine:
    """
    Xception-based deepfake detector.  Runs alongside GRU + CNN as a
    third visual signal, activated only when suspicion is already elevated.
    """

    def __init__(
        self,
        weights_path: str | Path = _DEFAULT_WEIGHTS,
        threshold:    float      = DEEPFAKE_THRESHOLD,  # 0.75 — calibrated
    ) -> None:
        self._weights_path = Path(weights_path)
        self._threshold    = threshold
        self._model        = None
        self._device       = None
        self._ready        = False

        # Throttle state
        self._last_run_time: float = 0.0
        self._frame_count:   int   = 0

        # Latest cached result
        self.latest_result: dict = {
            "status":               "LOADING",
            "deepfake_probability": 0.0,
            "deepfake_label":       "REAL",
        }

        # Last valid face ROI (bytes + shape) from FrameEvent
        self._last_roi_bytes: Optional[bytes]           = None
        self._last_roi_shape: Optional[tuple[int, int]] = None

        # Executor for off-thread inference
        self._executor = None  # None → default ThreadPoolExecutor

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load Xception checkpoint. Fails gracefully."""
        # Repo path must be on sys.path before importing deepfake_model
        if not _add_repo_to_path():
            print("DEEPFAKE: network/ package not found — "
                  "set DEEPFAKE_REPO env var or copy network/ to agent/ml/deepfake_net/")
            self.latest_result["status"] = "NO_NETWORK_PKG"
            return

        if not self._weights_path.exists():
            print(
                f"DEEPFAKE: weights not found at {self._weights_path}\n"
                "  Download df_c0_best.pkl from:\n"
                "  https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8\n"
                f"  Place at: {self._weights_path}"
            )
            self.latest_result["status"] = "NO_WEIGHTS"
            return

        try:
            import torch
            from deepfake_model import load_model  # uses network/ on sys.path

            # Device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"DEEPFAKE: loading model  device={self._device}")

            self._model = load_model(str(self._weights_path), device=self._device)
            self._ready = True
            self.latest_result["status"] = "READY"
            print(f"DEEPFAKE: model loaded OK  weights={self._weights_path}")

        except Exception as exc:
            print(f"DEEPFAKE: LOAD FAILED — {exc}")
            logger.error(f"DeepfakeInferenceEngine load failed: {exc}", exc_info=True)
            self.latest_result["status"] = "LOAD_ERROR"

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Event bus wiring
    # ─────────────────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to FusionEvent (trigger) and FrameEvent (ROI cache)."""
        try:
            from agent.event_bus import bus
            from agent.events import FusionEvent, FrameEvent

            bus.subscribe(FusionEvent, self._on_fusion_event)
            bus.subscribe(FrameEvent,  self._on_frame_event)
            print("DEEPFAKE: registered on event bus")
        except Exception as exc:
            print(f"DEEPFAKE: register FAILED — {exc}")

    async def _on_frame_event(self, event) -> None:
        """Cache the latest face ROI bytes for use at inference time."""
        if event.face_detected and event.face_roi_bgr is not None:
            self._last_roi_bytes = event.face_roi_bgr
            self._last_roi_shape = event.face_roi_shape

    async def _on_fusion_event(self, event) -> None:
        """
        Triggered at ~10 Hz by FusionEngine.
        Decides whether to fire Xception inference based on trigger conditions.
        """
        if not self._ready or self._model is None:
            return

        if self._last_roi_bytes is None:
            return

        now = time.monotonic()
        elapsed = now - self._last_run_time

        # ── Trigger conditions ────────────────────────────────────────────────
        cnn_high      = event.cnn_score > _CNN_TRIGGER
        fusion_alert  = event.final_status in _FUSION_STATES
        timeout       = elapsed > _MAX_GAP_SEC

        should_run = cnn_high or fusion_alert or timeout
        if not should_run:
            return

        # ── Throttle gate ─────────────────────────────────────────────────────
        if elapsed < _MIN_GAP_SEC:
            return

        self._last_run_time = now

        # ── Dispatch to thread pool ───────────────────────────────────────────
        roi_bytes = self._last_roi_bytes
        roi_shape = self._last_roi_shape
        try:
            loop = asyncio.get_running_loop()
            prob = await loop.run_in_executor(
                self._executor,
                self._infer,
                roi_bytes,
                roi_shape,
            )
        except Exception as exc:
            print(f"DEEPFAKE EXECUTOR ERROR: {exc}")
            return

        if prob is None:
            return  # keep last result

        label = "FAKE" if prob >= self._threshold else "REAL"
        print(f"DEEPFAKE RESULT: prob={prob:.4f}  label={label}  "
              f"trigger=cnn:{cnn_high} fusion:{fusion_alert} timeout:{timeout}")

        self.latest_result = {
            "status":               "READY",
            "deepfake_probability": round(prob, 4),
            "deepfake_label":       label,
        }

        # Publish DeepfakeEvent
        try:
            from agent.event_bus import bus
            from agent.events import DeepfakeEvent

            await bus.publish(DeepfakeEvent(
                session_id            = event.session_id,
                frame_id              = event.frame_id,
                deepfake_probability  = round(prob, 4),
                deepfake_label        = label,
                trigger_cnn           = cnn_high,
                trigger_fusion        = fusion_alert,
                trigger_timeout       = timeout,
            ))
        except Exception as exc:
            logger.debug(f"DEEPFAKE publish error: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Core inference (synchronous — runs in thread pool)
    # ─────────────────────────────────────────────────────────────────────────

    def _infer(
        self,
        roi_bytes: bytes,
        roi_shape: Optional[tuple[int, int]],
    ) -> Optional[float]:
        """
        Decode ROI bytes → reconstruct BGR ndarray → call deepfake_model.predict().
        Returns float probability or None on error.
        """
        try:
            from deepfake_model import predict

            if roi_shape is None:
                return None

            h, w = roi_shape
            arr = np.frombuffer(roi_bytes, dtype=np.uint8).reshape(h, w, 3)

            print(f"DEEPFAKE RUNNING INFERENCE  shape=({h},{w})")
            prob = predict(arr, self._model)
            print(f"DEEPFAKE RAW PROB: {prob:.4f}")
            return prob

        except Exception as exc:
            print(f"DEEPFAKE _INFER ERROR: {exc}")
            import traceback
            traceback.print_exc()
            return None
