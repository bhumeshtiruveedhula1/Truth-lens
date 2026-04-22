"""
liveness/micro_expr.py — Micro-Expression / Landmark Velocity Detector

Tracks frame-to-frame velocity of facial landmarks.
Real human faces have continuous micro-movements in muscles around the eyes,
mouth, and cheeks even when the person thinks they're keeping still.

Key deepfake tells:
  - Video replay: perfect stillness between frames (no micro-jitter)
  - DeepFaceLive: landmark velocity distribution is unnaturally smooth
  - AI generators: near-zero inter-frame velocity on non-expressive areas
"""

from __future__ import annotations
import collections
import time
from typing import Deque, Optional

import numpy as np

from agent.events import FrameEvent, LivenessSignal
from agent.liveness.base import SignalExtractor

# Key landmark regions for micro-expression analysis
# (avoid outer contour — focus on muscles)
MICRO_EXPR_INDICES = list(range(46, 55)) + list(range(276, 285))  # eyebrows
MICRO_EXPR_INDICES += [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]  # lips
MICRO_EXPR_INDICES += [159, 160, 161, 386, 387, 388]  # upper eyelids

HISTORY_SEC = 3.0
MIN_HISTORY_FRAMES = 10


class MicroExpressionDetector(SignalExtractor):
    name = "micro_expr"

    def __init__(self):
        self._prev_landmarks: Optional[list] = None
        self._velocity_history: Deque[float] = collections.deque()
        self._ts_history: Deque[float] = collections.deque()
        self._frame_count = 0

    def extract(self, event: FrameEvent) -> LivenessSignal | None:
        lm = event.face_landmarks
        now = time.time()
        self._frame_count += 1

        if self._prev_landmarks is None:
            self._prev_landmarks = lm
            return None

        # Compute per-landmark displacement for key micro-expression points
        velocities = []
        for idx in MICRO_EXPR_INDICES:
            if idx >= len(lm) or idx >= len(self._prev_landmarks):
                continue
            dx = lm[idx][0] - self._prev_landmarks[idx][0]
            dy = lm[idx][1] - self._prev_landmarks[idx][1]
            v = (dx**2 + dy**2) ** 0.5
            velocities.append(v)

        self._prev_landmarks = lm

        if not velocities:
            return None

        mean_velocity = float(np.mean(velocities))
        self._velocity_history.append(mean_velocity)
        self._ts_history.append(now)

        # Prune rolling window
        cutoff = now - HISTORY_SEC
        while self._ts_history and self._ts_history[0] < cutoff:
            self._ts_history.popleft()
            self._velocity_history.popleft()

        if len(self._velocity_history) < MIN_HISTORY_FRAMES:
            return LivenessSignal(
                session_id=event.session_id,
                extractor_name=self.name,
                value=mean_velocity,
                score=0.7,
                confidence=0.1,
            )

        vel_arr = np.array(list(self._velocity_history))
        mean_v = float(np.mean(vel_arr))
        std_v  = float(np.std(vel_arr))

        # --- Scoring ---
        # Human faces have mean micro-velocity > 0.0003 (normalized units)
        # and a natural standard deviation (not suspiciously constant)
        if mean_v < 0.00005:
            # Essentially frozen — very suspicious
            score = 0.05
        elif mean_v < 0.0003:
            score = mean_v / 0.0003 * 0.5
        else:
            score = 0.5 + min(0.5, mean_v * 500)

        # Penalize unnaturally constant velocity (no variance)
        if std_v < 0.00001 and mean_v > 0:
            score *= 0.7

        score = max(0.0, min(1.0, score))
        confidence = min(1.0, len(self._velocity_history) / 30.0)

        return LivenessSignal(
            session_id=event.session_id,
            extractor_name=self.name,
            value=mean_velocity,
            score=score,
            confidence=confidence,
            metadata={
                "mean_velocity": round(mean_v, 7),
                "velocity_std": round(std_v, 7),
            },
        )
