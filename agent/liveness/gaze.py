"""
liveness/gaze.py — Iris Gaze Tracker

Uses MediaPipe FaceMesh refined iris landmarks (requires refine_landmarks=True).
Tracks iris position relative to eye corners to estimate gaze direction.

Key deepfake tells:
  - Many deepfakes use the source video gaze with slight delay or misalignment
  - AI video generators (HeyGen, Synthesia) often have stiff, center-biased gaze
  - Real humans constantly make micro-saccades even when "looking straight ahead"

Iris landmark indices (MediaPipe refined):
  Right iris: 468 (center), 469 (right), 470 (top), 471 (left), 472 (bottom)
  Left iris:  473 (center), 474 (right), 475 (top), 476 (left), 477 (bottom)
"""

from __future__ import annotations
import collections
import time
from typing import Deque

import numpy as np

from agent.events import FrameEvent, LivenessSignal
from agent.liveness.base import SignalExtractor

# Iris center landmarks
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473

# Eye corner landmarks (for normalization)
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
LEFT_EYE_OUTER  = 263
LEFT_EYE_INNER  = 362

HISTORY_SEC = 5.0


def _iris_ratio(lm: list, iris_idx: int, outer_idx: int, inner_idx: int) -> float:
    """
    Normalized iris position: 0.0 = iris at outer corner, 1.0 = at inner corner.
    0.5 = centered (looking straight ahead).
    """
    iris_x = lm[iris_idx][0]
    outer_x = lm[outer_idx][0]
    inner_x = lm[inner_idx][0]
    span = abs(inner_x - outer_x) + 1e-6
    return (iris_x - outer_x) / span


class GazeTracker(SignalExtractor):
    name = "gaze"

    def __init__(self):
        self._gaze_history: Deque[float] = collections.deque()
        self._ts_history: Deque[float] = collections.deque()
        self._frame_count = 0

    def extract(self, event: FrameEvent) -> LivenessSignal | None:
        lm = event.face_landmarks

        # Need refined landmarks (index 468+ for iris)
        if len(lm) < 478:
            return None

        # Compute normalized gaze for both eyes
        r_ratio = _iris_ratio(lm, RIGHT_IRIS_CENTER, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
        l_ratio = _iris_ratio(lm, LEFT_IRIS_CENTER, LEFT_EYE_OUTER, LEFT_EYE_INNER)
        gaze = (r_ratio + l_ratio) / 2.0

        now = time.time()
        self._frame_count += 1
        self._gaze_history.append(gaze)
        self._ts_history.append(now)

        # Prune rolling window
        cutoff = now - HISTORY_SEC
        while self._ts_history and self._ts_history[0] < cutoff:
            self._ts_history.popleft()
            self._gaze_history.popleft()

        if len(self._gaze_history) < 10:
            return LivenessSignal(
                session_id=event.session_id,
                extractor_name=self.name,
                value=gaze,
                score=0.7,
                confidence=0.1,
                metadata={"gaze_ratio": round(gaze, 4)},
            )

        gaze_arr = np.array(list(self._gaze_history))
        variance = float(np.var(gaze_arr))
        mean_gaze = float(np.mean(gaze_arr))

        # --- Scoring logic ---
        # Human gaze has natural micro-saccades: variance > 0.001
        # AI gaze is often suspiciously centered (mean ~0.5) with low variance
        if variance < 0.0005:
            # Almost no gaze movement — highly suspicious
            score = 0.15
        elif variance < 0.002:
            score = 0.4 + variance * 100
        else:
            score = 1.0  # Natural variance

        # Slight penalty for unnaturally perfect center gaze
        if abs(mean_gaze - 0.5) < 0.03 and variance < 0.003:
            score *= 0.85  # Too perfectly centered = suspicious

        score = max(0.0, min(1.0, score))
        confidence = min(1.0, self._frame_count / 45.0)

        return LivenessSignal(
            session_id=event.session_id,
            extractor_name=self.name,
            value=gaze,
            score=score,
            confidence=confidence,
            metadata={
                "gaze_ratio": round(gaze, 4),
                "gaze_variance": round(variance, 6),
                "mean_gaze": round(mean_gaze, 4),
                "right_ratio": round(r_ratio, 4),
                "left_ratio": round(l_ratio, 4),
            },
        )
