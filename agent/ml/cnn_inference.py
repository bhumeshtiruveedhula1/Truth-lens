"""
agent/ml/cnn_inference.py -- CNN Visual Inference Engine
=========================================================
Runs MobileNetV2 on face ROI bytes from FrameEvent.
Non-blocking: inference runs on the asyncio thread via run_in_executor
so the camera loop is never stalled.

Fixes applied:
  - asyncio.get_event_loop() → asyncio.get_running_loop()  (Python 3.10+)
  - _infer() errors now print to stdout (visible in console)
  - Full diagnostic trace prints at every stage
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalisation (must match training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INFERENCE_EVERY_N = 3      # run CNN on every 3rd frame  (~10 Hz at 30 fps)
_THRESHOLD         = 0.50   # default; overridden by checkpoint if present


class CNNInferenceEngine:
    """
    Visual inference engine — parallel to GRUInferenceEngine.
    Zero coupling to GRU: reads only FrameEvent (face_roi_bgr).
    """

    def __init__(
        self,
        model_path: str | Path = "models/cnn_baseline.pt",
        threshold:  float      = _THRESHOLD,
    ) -> None:
        self._model_path = Path(model_path)
        self._threshold  = threshold
        self._device     = None
        self._model      = None
        self._ready      = False

        # Frame-skip counter
        self._frame_count: int = 0

        # Last valid inference result (reused for skipped frames)
        self.latest_result: dict = {
            "status":               "LOADING",
            "cnn_fake_probability": 0.0,
            "cnn_label":            "REAL",
        }

        # Executor for off-thread inference
        self._executor = None   # None → default asyncio ThreadPoolExecutor

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load CNN checkpoint.  CUDA → CPU fallback."""
        try:
            import torch
            from torchvision import models
            import torch.nn as nn

            # ── Device selection ─────────────────────────────────────────────
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                print("CNN: Using GPU (CUDA)")
            else:
                self._device = torch.device("cpu")
                print("CNN: Using CPU (no CUDA)")

            # ── Load checkpoint ──────────────────────────────────────────────
            if not self._model_path.exists():
                print(f"CNN: model not found at {self._model_path}")
                self.latest_result["status"] = "NO_MODEL"
                return

            ckpt = torch.load(
                str(self._model_path),
                map_location=self._device,
                weights_only=False,
            )

            if "best_threshold" in ckpt:
                self._threshold = float(ckpt["best_threshold"])

            # ── Rebuild MobileNetV2 ──────────────────────────────────────────
            m = models.mobilenet_v2(weights=None)
            in_f = m.classifier[1].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(in_f, 1),
            )
            m.load_state_dict(ckpt["model_state"])
            m.to(self._device)
            m.eval()
            self._model = m

            # ── Warm-up ───────────────────────────────────────────────────────
            dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32).to(self._device)
            with torch.no_grad():
                _ = self._model(dummy)

            self._ready = True
            self.latest_result["status"] = "READY"
            print(f"CNN: model loaded OK  device={self._device}  threshold={self._threshold:.2f}")

        except Exception as exc:
            print(f"CNN: LOAD FAILED — {exc}")
            logger.error(f"CNNInferenceEngine: load failed — {exc}", exc_info=True)
            self._ready = False
            self.latest_result["status"] = "LOAD_ERROR"

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Event bus wiring
    # ─────────────────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to FrameEvent on the bus."""
        try:
            from agent.event_bus import bus
            from agent.events import FrameEvent

            bus.subscribe(FrameEvent, self._on_frame_event)
            print("CNN: registered on event bus")
        except Exception as exc:
            print(f"CNN: register FAILED — {exc}")

    async def _on_frame_event(self, event) -> None:
        """
        Called at ~30 Hz.  Runs inference every N frames via executor.

        FIX: Use asyncio.get_running_loop() — required in Python 3.10+.
             asyncio.get_event_loop() is deprecated inside a running coroutine.
        """
        print(f"CNN FRAME RECEIVED fid={event.frame_id}")

        if not self._ready or self._model is None:
            print(f"CNN SKIP NOT READY fid={event.frame_id}")
            return

        # Skip frames without face
        if not event.face_detected:
            print(f"CNN SKIP NO FACE fid={event.frame_id}")
            return
        if event.face_roi_bgr is None:
            print(f"CNN SKIP ROI NONE fid={event.frame_id}")
            return

        print(f"CNN ROI PRESENT fid={event.frame_id}  "
              f"shape={event.face_roi_shape}  bytes={len(event.face_roi_bgr)}")

        self._frame_count += 1

        # Frame-skip
        if self._frame_count % _INFERENCE_EVERY_N != 0:
            print(f"CNN SKIP FRAME (throttle) count={self._frame_count}")
            return

        # ── KEY FIX: get_running_loop() not get_event_loop() ─────────────────
        try:
            loop = asyncio.get_running_loop()
            prob = await loop.run_in_executor(
                self._executor,
                self._infer,
                event.face_roi_bgr,
                event.face_roi_shape,
            )
        except Exception as exc:
            print(f"CNN EXECUTOR ERROR fid={event.frame_id}: {exc}")
            return

        if prob is None:
            print(f"CNN INFER RETURNED NONE fid={event.frame_id} (error in _infer)")
            return

        label = "FAKE" if prob >= self._threshold else "REAL"
        print(f"CNN RESULT UPDATED fid={event.frame_id}  prob={prob:.4f}  label={label}")

        self.latest_result = {
            "status":               "READY",
            "cnn_fake_probability": round(float(prob), 4),
            "cnn_label":            label,
        }

        # Publish CNNEvent
        try:
            from agent.event_bus import bus
            from agent.events import CNNEvent

            await bus.publish(CNNEvent(
                session_id            = event.session_id,
                frame_id              = event.frame_id,
                cnn_fake_probability  = float(prob),
                cnn_label             = label,
            ))
            print(f"CNN EVENT PUBLISHED fid={event.frame_id}")
        except Exception as exc:
            print(f"CNN PUBLISH ERROR: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Core inference (runs in thread pool — synchronous)
    # ─────────────────────────────────────────────────────────────────────────

    def _infer(
        self,
        roi_bytes: bytes,
        roi_shape: Optional[tuple[int, int]],
    ) -> Optional[float]:
        """
        Decode face ROI bytes → preprocess → forward pass.
        Returns sigmoid probability or None on error.
        """
        try:
            import torch
            import cv2

            print(f"CNN RUNNING INFERENCE  roi_shape={roi_shape}  bytes={len(roi_bytes)}")

            if roi_shape is None:
                print("CNN INFER FAIL: roi_shape is None")
                return None

            h, w = roi_shape
            arr = np.frombuffer(roi_bytes, dtype=np.uint8).reshape(h, w, 3)

            # Letterbox resize to 224x224
            scale   = 224 / max(h, w)
            nh, nw  = max(1, int(h * scale)), max(1, int(w * scale))
            resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas  = np.zeros((224, 224, 3), dtype=np.uint8)
            y0 = (224 - nh) // 2
            x0 = (224 - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized

            # BGR → RGB → float32 → CHW → normalize
            rgb    = canvas[:, :, ::-1].astype(np.float32) / 255.0
            rgb    = (rgb - _MEAN) / _STD
            chw    = rgb.transpose(2, 0, 1)
            tensor = torch.from_numpy(chw).unsqueeze(0).float().to(self._device)

            with torch.no_grad():
                logit = self._model(tensor)
                prob  = float(torch.sigmoid(logit).item())

            print(f"CNN RAW LOGIT: {logit.item():.4f}  PROB: {prob:.4f}")
            return prob

        except Exception as exc:
            print(f"CNN _INFER ERROR: {exc}")
            import traceback
            traceback.print_exc()
            return None
