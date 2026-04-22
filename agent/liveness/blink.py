"""
agent/liveness/blink.py

Blink detection utilities plus two adapters:
- BlinkModule for the lightweight liveness pipeline
- BlinkDetector for the event-bus based extractor expected by main.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from agent.events import FrameEvent, LivenessSignal
from agent.liveness.base import SignalExtractor


RIGHT_EYE_EAR_IDX: Tuple[int, ...] = (33, 159, 158, 133, 153, 145)
LEFT_EYE_EAR_IDX: Tuple[int, ...] = (362, 380, 374, 263, 386, 385)

BLINK_TIMEOUT_SEC = 4.0
HISTORY_SEC = 30.0

BlinkSignal = Dict[str, object]


def mediapipe_landmarks_to_dict(
    face_landmarks,
    frame_width: int,
    frame_height: int,
) -> Dict[int, Tuple[int, int]]:
    return {
        idx: (int(lm.x * frame_width), int(lm.y * frame_height))
        for idx, lm in enumerate(face_landmarks.landmark)
    }


def tuple_landmarks_to_dict(
    face_landmarks: list[tuple[float, float, float]],
    frame_width: int,
    frame_height: int,
) -> Dict[int, Tuple[int, int]]:
    return {
        idx: (int(lm[0] * frame_width), int(lm[1] * frame_height))
        for idx, lm in enumerate(face_landmarks)
    }


def compute_ear(
    landmarks: Dict[int, Tuple[int, int]],
    eye_idx: Tuple[int, ...],
) -> float:
    try:
        points = [np.array(landmarks[i], dtype=np.float64) for i in eye_idx]
    except KeyError:
        return 0.0

    vertical_a = np.linalg.norm(points[1] - points[5])
    vertical_b = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal < 1e-6:
        return 0.0

    return (vertical_a + vertical_b) / (2.0 * horizontal)


class EARSmoother:
    def __init__(self, window: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, ear: float) -> float:
        self._buf.append(ear)
        return float(np.mean(self._buf))

    def reset(self) -> None:
        self._buf.clear()


@dataclass
class BlinkState:
    ear_threshold: float = 0.28
    consec_frames: int = 3
    blink_count: int = field(default=0, init=False)
    frame_count: int = field(default=0, init=False)
    is_closed: bool = field(default=False, init=False)

    def update(self, ear: float) -> bool:
        blink_detected = False

        if ear < self.ear_threshold:
            self.frame_count += 1
            self.is_closed = True
        else:
            if self.is_closed and self.frame_count >= self.consec_frames:
                self.blink_count += 1
                blink_detected = True
            self.frame_count = 0
            self.is_closed = False

        return blink_detected

    def reset(self) -> None:
        self.blink_count = 0
        self.frame_count = 0
        self.is_closed = False


class _BlinkCore:
    def __init__(
        self,
        ear_threshold: float = 0.28,
        consec_frames: int = 3,
        smooth_window: int = 3,
    ) -> None:
        self._smoother = EARSmoother(window=smooth_window)
        self._state = BlinkState(
            ear_threshold=ear_threshold,
            consec_frames=consec_frames,
        )

    def process_landmark_dict(self, landmarks: Dict[int, Tuple[int, int]]) -> BlinkSignal:
        if not landmarks:
            self.reset()
            return {"blink_detected": False, "ear": 0.0}

        raw_ear = (
            compute_ear(landmarks, RIGHT_EYE_EAR_IDX)
            + compute_ear(landmarks, LEFT_EYE_EAR_IDX)
        ) / 2.0
        smooth_ear = self._smoother.update(raw_ear)
        blink_detected = self._state.update(smooth_ear)

        return {"blink_detected": blink_detected, "ear": raw_ear}

    @property
    def blink_count(self) -> int:
        return self._state.blink_count

    def reset(self) -> None:
        self._smoother.reset()
        self._state.reset()


class BlinkDetector(SignalExtractor):
    name = "blink"

    def __init__(
        self,
        ear_threshold: float = 0.28,
        consec_frames: int = 3,
        smooth_window: int = 3,
    ) -> None:
        self._core = _BlinkCore(
            ear_threshold=ear_threshold,
            consec_frames=consec_frames,
            smooth_window=smooth_window,
        )
        self._blink_timestamps: deque[float] = deque()
        self._last_blink_time: float = 0.0
        self._frame_count = 0

    def register(self) -> None:
        super().register()

    async def handle(self, event: FrameEvent) -> None:
        await super().handle(event)

    def extract(self, event: FrameEvent) -> LivenessSignal | None:
        if not event.face_landmarks:
            self._core.reset()
            return None

        landmarks = tuple_landmarks_to_dict(
            event.face_landmarks,
            event.frame_width,
            event.frame_height,
        )
        blink_signal = self._core.process_landmark_dict(landmarks)

        ear = float(blink_signal["ear"])
        blink_detected = bool(blink_signal["blink_detected"])
        now = event.timestamp
        self._frame_count += 1

        if blink_detected:
            self._last_blink_time = now
            self._blink_timestamps.append(now)
        elif self._last_blink_time == 0.0:
            self._last_blink_time = now

        cutoff = now - HISTORY_SEC
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        blink_rate = len(self._blink_timestamps) / HISTORY_SEC
        time_since_blink = now - self._last_blink_time

        if blink_rate > 0.08:
            rate_score = min(1.0, blink_rate / 0.25)
        else:
            rate_score = blink_rate / 0.08 if blink_rate > 0.0 else 0.0

        if time_since_blink > BLINK_TIMEOUT_SEC:
            timeout_penalty = max(
                0.0,
                1.0 - (time_since_blink - BLINK_TIMEOUT_SEC) / 10.0,
            )
        else:
            timeout_penalty = 1.0

        score = max(0.0, min(1.0, rate_score * timeout_penalty))
        confidence = min(1.0, self._frame_count / 45.0)
        print(f"[BLINK] EAR={round(ear,3)} blink={blink_detected}")

        return LivenessSignal(
            session_id=event.session_id,
            extractor_name=self.name,
            value=ear,
            score=score,
            confidence=confidence,
            metadata={
                "ear": round(ear, 4),
                "blink_detected": blink_detected,
                "blink_rate_per_sec": round(blink_rate, 4),
                "total_blinks": len(self._blink_timestamps),
                "time_since_blink": round(time_since_blink, 1),
            },
        )


class BlinkModule:
    def __init__(
        self,
        ear_threshold: float = 0.28,
        consec_frames: int = 3,
        smooth_window: int = 3,
    ) -> None:
        self._core = _BlinkCore(
            ear_threshold=ear_threshold,
            consec_frames=consec_frames,
            smooth_window=smooth_window,
        )

    def process(
        self,
        landmarks,
        w: int,
        h: int,
    ) -> BlinkSignal:
        if not landmarks:
            self._core.reset()
            return {"blink_detected": False, "ear": 0.0}

        landmark_dict = mediapipe_landmarks_to_dict(landmarks, w, h)
        return self._core.process_landmark_dict(landmark_dict)
