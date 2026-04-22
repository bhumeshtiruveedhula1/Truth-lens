"""
provenance/compression.py — JPEG Compression Artifact Detector

Many real-time deepfakes work by compositing a generated face patch
onto a live video stream. This compositing almost always introduces
visible JPEG blocking artifacts at the face/background boundary,
or unnatural high-frequency noise in the face region.

Detection method:
  1. Extract face ROI from frame
  2. Compute blockiness via DCT (JPEG artifacts produce 8x8 grid patterns)
  3. Measure noise floor relative to natural face texture
"""

from __future__ import annotations
import collections
import logging
import time
from typing import Deque

import cv2
import numpy as np

from agent.events import FrameEvent, ProvenanceSignal
from agent.event_bus import bus

logger = logging.getLogger(__name__)

HISTORY_SEC = 10.0
PROCESS_EVERY_N_FRAMES = 5  # Don't process every frame (expensive)


class CompressionArtifactDetector:
    name = "compression_artifact"

    def __init__(self):
        self._frame_counter = 0
        self._anomaly_history: Deque[float] = collections.deque()
        self._ts_history: Deque[float] = collections.deque()
        self._registered = False

    def register(self) -> None:
        from agent.event_bus import bus
        from agent.events import FrameEvent
        bus.subscribe(FrameEvent, self.handle)
        self._registered = True

    async def handle(self, event: FrameEvent) -> None:
        self._frame_counter += 1
        if self._frame_counter % PROCESS_EVERY_N_FRAMES != 0:
            return
        if not event.face_detected or not event.face_roi_bgr or not event.face_roi_shape:
            return

        try:
            signal = self._analyze(event)
            if signal:
                await bus.publish(signal)
        except Exception as exc:
            logger.debug(
                "compression check skipped frame %s: %s",
                self._frame_counter,
                exc,
                exc_info=True,
            )

    def _analyze(self, event: FrameEvent) -> ProvenanceSignal | None:
        if not event.face_roi_bgr or not event.face_roi_shape:
            return None

        roi_h, roi_w = event.face_roi_shape

        face_roi_arr = np.frombuffer(event.face_roi_bgr, dtype=np.uint8)
        expected = roi_w * roi_h * 3
        if face_roi_arr.size != expected or roi_w < 32 or roi_h < 32:
            return None

        face_roi = face_roi_arr.reshape(roi_h, roi_w, 3)
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # --- Blockiness score via 8x8 DCT block analysis ---
        h_roi, w_roi = gray.shape
        block_scores = []

        for bx in range(0, w_roi - 8, 8):
            for by in range(0, h_roi - 8, 8):
                block = gray[by:by+8, bx:bx+8]
                dct = cv2.dct(block)
                # JPEG artifacts → high energy in specific coefficients
                energy_dc = abs(dct[0, 0])
                energy_ac = np.sum(np.abs(dct[1:, 1:]))
                if energy_dc > 0:
                    block_scores.append(energy_ac / (energy_dc + 1e-6))

        if not block_scores:
            return None

        blockiness = float(np.std(block_scores))

        # --- High-frequency noise (Laplacian variance) ---
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(np.var(laplacian))

        # --- Anomaly score computation ---
        # Normal face: blockiness std ~0.5-2.0, lap_var ~50-500
        # Composited deepfake face: spiky blockiness, unusual lap_var
        if blockiness > 5.0 or blockiness < 0.1:
            block_anomaly = min(1.0, abs(blockiness - 1.5) / 5.0)
        else:
            block_anomaly = 0.1

        if lap_var > 2000 or lap_var < 5:
            noise_anomaly = min(1.0, abs(lap_var - 200) / 2000)
        else:
            noise_anomaly = 0.1

        anomaly_score = 0.6 * block_anomaly + 0.4 * noise_anomaly
        anomaly_score = max(0.0, min(1.0, anomaly_score))

        now = time.time()
        self._anomaly_history.append(anomaly_score)
        self._ts_history.append(now)

        cutoff = now - HISTORY_SEC
        while self._ts_history and self._ts_history[0] < cutoff:
            self._ts_history.popleft()
            self._anomaly_history.popleft()

        smoothed = float(np.mean(self._anomaly_history))

        return ProvenanceSignal(
            session_id=event.session_id,
            check_name=self.name,
            anomaly_score=smoothed,
            evidence={
                "blockiness": round(blockiness, 4),
                "laplacian_variance": round(lap_var, 2),
                "raw_anomaly": round(anomaly_score, 4),
                "confidence": 0.75,
            },
        )
