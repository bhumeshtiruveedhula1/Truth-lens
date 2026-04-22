"""
provenance/temporal.py — Temporal Landmark Jitter Detector

Deepfake overlays often produce inter-frame landmark jitter that is
physically implausible for a human face. Real faces have smooth, 
continuous muscle movements. Deepfakes applied frame-by-frame create
micro-discontinuities in landmark positions.

This checker runs at a lower frequency (every 2 frames) and looks at
the smoothness of landmark trajectories across a short window.
"""

from __future__ import annotations
import collections
import logging
import time
from typing import Deque, Optional

import numpy as np

from agent.events import FrameEvent, ProvenanceSignal
from agent.event_bus import bus

logger = logging.getLogger(__name__)

# Structural face landmarks (nose, mouth corners, eye corners)
# sorted(set(...)) is critical: landmark order must be STABLE across frames.
# list(set()) does NOT guarantee order — causing cross-frame index mismatches.
STRUCTURAL_INDICES = sorted(set([
    1, 4, 5, 6, 195, 197, 168,   # nose
    61, 291, 78, 308,             # mouth corners
    33, 263, 133, 362,            # eye corners
]))

HISTORY_FRAMES = 15
PROCESS_EVERY_N = 2


class TemporalJitterDetector:
    name = "temporal_jitter"

    def __init__(self):
        self._landmark_history: Deque[list] = collections.deque(maxlen=HISTORY_FRAMES)
        self._frame_counter = 0
        self._anomaly_history: Deque[float] = collections.deque(maxlen=30)
        self._ts_history: Deque[float] = collections.deque(maxlen=30)

    def register(self) -> None:
        bus.subscribe(FrameEvent, self.handle)

    async def handle(self, event: FrameEvent) -> None:
        self._frame_counter += 1
        if self._frame_counter % PROCESS_EVERY_N != 0:
            return
        if not event.face_detected or not event.face_landmarks:
            return

        try:
            signal = self._analyze(event)
            if signal:
                await bus.publish(signal)
        except Exception as exc:
            logger.debug(
                "temporal jitter check skipped frame %s: %s",
                self._frame_counter,
                exc,
                exc_info=True,
            )

    def _analyze(self, event: FrameEvent) -> Optional[ProvenanceSignal]:
        lm = event.face_landmarks
        now = time.time()

        # Store structural landmarks for this frame
        structural = [(lm[i][0], lm[i][1]) for i in STRUCTURAL_INDICES if i < len(lm)]
        self._landmark_history.append(structural)

        if len(self._landmark_history) < 5:
            return None

        # Compute per-landmark acceleration (jitter = high 2nd derivative)
        history = list(self._landmark_history)
        n_points = min(len(h) for h in history)

        jitter_scores = []
        for pt_idx in range(n_points):
            positions = np.array([h[pt_idx] for h in history if pt_idx < len(h)])
            if len(positions) < 3:
                continue

            # 1st derivative (velocity)
            velocity = np.diff(positions, axis=0)
            # 2nd derivative (acceleration = jitter)
            acceleration = np.diff(velocity, axis=0)

            if len(acceleration) > 0:
                jitter = float(np.mean(np.abs(acceleration)))
                jitter_scores.append(jitter)

        if not jitter_scores:
            return None

        mean_jitter = float(np.mean(jitter_scores))
        jitter_std  = float(np.std(jitter_scores))

        # --- Anomaly scoring ---
        # Real faces: mean_jitter < 0.002 (normalized units), smooth
        # Deepfake: mean_jitter spikes or is unnaturally uniform
        if mean_jitter > 0.01:
            # High jitter → likely deepfake glitching
            anomaly = min(1.0, mean_jitter * 50)
        elif mean_jitter < 0.00001:
            # Zero jitter → likely frozen video
            anomaly = 0.6
        else:
            anomaly = 0.1  # Normal range

        self._anomaly_history.append(anomaly)
        self._ts_history.append(now)

        smoothed = float(np.mean(self._anomaly_history))

        return ProvenanceSignal(
            session_id=event.session_id,
            check_name=self.name,
            anomaly_score=smoothed,
            evidence={
                "mean_jitter": round(mean_jitter, 7),
                "jitter_std": round(jitter_std, 7),
                "frames_analyzed": len(history),
                "confidence": 0.8,
            },
        )
