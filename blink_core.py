"""
blink_core.py
─────────────
Minimal, standalone blink-detection logic extracted from the
Eye-Blink-Detection-using-MediaPipe-and-OpenCV repo.

No camera code. No UI code. No CLI. No repo dependencies.
Input: a MediaPipe FaceMesh landmark dict  {id: (x, y)}  (pixel-space)
Output: EAR value, blink event, blink count.

Landmark indices follow MediaPipe's 468-point face mesh topology.
Compatible with mediapipe>=0.10.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────
# 1.  Landmark index constants
#     Source: blink_counter.py  →  RIGHT_EYE_EAR / LEFT_EYE_EAR
#     Only the 6-point EAR subset is needed (not the full 16-point ring).
# ─────────────────────────────────────────────────────────────────

# Layout for each eye:  [p1, p2, p3, p4, p5, p6]
#   p1 = outer corner, p4 = inner corner (horizontal axis)
#   p2, p3 = upper lid; p5, p6 = lower lid (vertical axes)
RIGHT_EYE_EAR_IDX: Tuple[int, ...] = (33, 159, 158, 133, 153, 145)
LEFT_EYE_EAR_IDX:  Tuple[int, ...] = (362, 380, 374, 263, 386, 385)


# ─────────────────────────────────────────────────────────────────
# 2.  EAR calculation
#     Source: BlinkCounter.eye_aspect_ratio()
#     Formula: EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 ‖p1-p4‖)
# ─────────────────────────────────────────────────────────────────

def compute_ear(
    landmarks: Dict[int, Tuple[int, int]],
    eye_idx: Tuple[int, ...],
) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for one eye.

    Args:
        landmarks: MediaPipe landmark dict {landmark_id: (x_px, y_px)}.
                   Must contain all indices in ``eye_idx``.
        eye_idx:   6-element tuple of landmark IDs in order
                   [p1, p2, p3, p4, p5, p6].

    Returns:
        EAR float.  Typical open-eye range ≈ 0.25–0.45.
        Returns 0.0 if any landmark is missing.
    """
    try:
        p = [np.array(landmarks[i], dtype=np.float64) for i in eye_idx]
    except KeyError:
        return 0.0

    A = np.linalg.norm(p[1] - p[5])   # ‖p2 - p6‖
    B = np.linalg.norm(p[2] - p[4])   # ‖p3 - p5‖
    C = np.linalg.norm(p[0] - p[3])   # ‖p1 - p4‖

    if C < 1e-6:          # degenerate / occluded landmark
        return 0.0

    return (A + B) / (2.0 * C)


def compute_mean_ear(landmarks: Dict[int, Tuple[int, int]]) -> float:
    """
    Average EAR across both eyes.

    Args:
        landmarks: same format as ``compute_ear``.

    Returns:
        Mean EAR float, or 0.0 if landmarks are missing.
    """
    right = compute_ear(landmarks, RIGHT_EYE_EAR_IDX)
    left  = compute_ear(landmarks, LEFT_EYE_EAR_IDX)
    return (right + left) / 2.0


# ─────────────────────────────────────────────────────────────────
# 3.  Temporal smoothing
#     Source: BlinkCounterandEARPlot._init_tracking_variables()
#             uses a rolling window over ear_values.
#     Extracted as a pure, reusable helper.
# ─────────────────────────────────────────────────────────────────

class EARSmoother:
    """
    Rolling-window mean smoother for EAR values.

    Reduces single-frame noise without introducing latency proportional
    to the full window (unlike a simple moving average over all history).

    Usage::

        smoother = EARSmoother(window=3)
        smooth_ear = smoother.update(raw_ear)
    """

    def __init__(self, window: int = 3) -> None:
        """
        Args:
            window: Number of frames to average.  3–5 is typically stable
                    for 25–30 fps streams without masking genuine blinks.
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, ear: float) -> float:
        """Push a new EAR sample and return the smoothed value."""
        self._buf.append(ear)
        return float(np.mean(self._buf))

    def reset(self) -> None:
        """Clear the internal buffer (e.g., on face loss)."""
        self._buf.clear()


# ─────────────────────────────────────────────────────────────────
# 4.  Blink state machine
#     Source: BlinkCounter.update_blink_count()
#     Decoupled from all video / display concerns.
# ─────────────────────────────────────────────────────────────────

@dataclass
class BlinkState:
    """
    Stateful blink detector.  Feed it one EAR value per frame.

    Attributes:
        ear_threshold:  EAR below this → eye considered closed.
                        Tune per user; 0.25–0.30 works for most subjects.
        consec_frames:  Minimum consecutive closed frames = 1 blink.
                        Typically 2–4 frames at 25-30 fps.
        blink_count:    Total confirmed blinks so far.
        frame_count:    Consecutive frames currently below threshold.
        is_closed:      True while eyes are below threshold.
    """

    ear_threshold:  float = 0.28
    consec_frames:  int   = 3
    blink_count:    int   = field(default=0, init=False)
    frame_count:    int   = field(default=0, init=False)
    is_closed:      bool  = field(default=False, init=False)

    def update(self, ear: float) -> bool:
        """
        Feed one EAR sample.

        Args:
            ear: Current (optionally smoothed) EAR value.

        Returns:
            True on the frame a new blink is *confirmed* (eye just reopened
            after enough consecutive closed frames), False otherwise.
        """
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
        """Reset counters (e.g., when face tracking is lost)."""
        self.blink_count = 0
        self.frame_count = 0
        self.is_closed   = False


# ─────────────────────────────────────────────────────────────────
# 5.  Convenience: single-call pipeline helper
#     Wraps EARSmoother + BlinkState into one object for drop-in use.
# ─────────────────────────────────────────────────────────────────

class BlinkDetector:
    """
    High-level blink detector.

    Plug-in replacement for the original ``BlinkCounter`` class,
    with no camera / UI / file dependencies.

    Example usage in your pipeline::

        from blink_core import BlinkDetector

        detector = BlinkDetector(ear_threshold=0.28, consec_frames=3, smooth_window=3)

        # Inside your per-frame loop:
        # ``landmarks`` = dict from MediaPipe:  {id: (x_px, y_px)}
        result = detector.process(landmarks)
        if result.blink_detected:
            print(f"Blink #{result.blink_count}  EAR={result.ear:.3f}")
    """

    def __init__(
        self,
        ear_threshold: float = 0.28,
        consec_frames: int   = 3,
        smooth_window:  int  = 3,
    ) -> None:
        """
        Args:
            ear_threshold:  EAR threshold for closed-eye detection.
            consec_frames:  Minimum closed frames to confirm a blink.
            smooth_window:  Rolling-average window applied to raw EAR.
                            Set to 1 to disable smoothing.
        """
        self._smoother = EARSmoother(window=smooth_window)
        self._state    = BlinkState(
            ear_threshold=ear_threshold,
            consec_frames=consec_frames,
        )

    # ── public API ───────────────────────────────────────────────

    def process(
        self,
        landmarks: Dict[int, Tuple[int, int]],
    ) -> "BlinkResult":
        """
        Process one frame's landmarks.

        Args:
            landmarks: MediaPipe landmark dict {id: (x_px, y_px)}.
                       Pass an empty dict if no face was detected —
                       the detector will reset its closed-frame counter.

        Returns:
            BlinkResult with ear, smooth_ear, blink_detected, blink_count.
        """
        if not landmarks:
            self._smoother.reset()
            self._state.reset()
            return BlinkResult(
                ear=0.0, smooth_ear=0.0,
                blink_detected=False,
                blink_count=self._state.blink_count,
                is_eye_closed=False,
            )

        raw_ear    = compute_mean_ear(landmarks)
        smooth_ear = self._smoother.update(raw_ear)
        blink      = self._state.update(smooth_ear)

        return BlinkResult(
            ear=raw_ear,
            smooth_ear=smooth_ear,
            blink_detected=blink,
            blink_count=self._state.blink_count,
            is_eye_closed=self._state.is_closed,
        )

    @property
    def blink_count(self) -> int:
        """Total confirmed blinks since instantiation / last reset."""
        return self._state.blink_count

    def reset(self) -> None:
        """Fully reset detector state (smoother buffer + blink counters)."""
        self._smoother.reset()
        self._state.reset()


@dataclass(frozen=True)
class BlinkResult:
    """
    Immutable result returned per frame by ``BlinkDetector.process()``.

    Attributes:
        ear:            Raw mean EAR (both eyes, this frame).
        smooth_ear:     Smoothed EAR used for blink decision.
        blink_detected: True on the first frame after a confirmed blink.
        blink_count:    Cumulative confirmed blinks.
        is_eye_closed:  True while EAR < threshold (closed phase).
    """
    ear:            float
    smooth_ear:     float
    blink_detected: bool
    blink_count:    int
    is_eye_closed:  bool


# ─────────────────────────────────────────────────────────────────
# 6.  MediaPipe landmark adapter
#     Converts the native mediapipe NormalizedLandmarkList to the
#     pixel-coordinate dict expected by the functions above.
# ─────────────────────────────────────────────────────────────────

def mediapipe_landmarks_to_dict(
    face_landmarks,          # mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
    frame_width:  int,
    frame_height: int,
) -> Dict[int, Tuple[int, int]]:
    """
    Convert a single MediaPipe face landmark object to a pixel-coordinate dict.

    Args:
        face_landmarks: One element from ``results.multi_face_landmarks``.
        frame_width:    Width of the source frame in pixels.
        frame_height:   Height of the source frame in pixels.

    Returns:
        ``{landmark_id: (x_px, y_px)}``

    Example::

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(...)
        results = mp_face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            lm_dict = mediapipe_landmarks_to_dict(
                results.multi_face_landmarks[0],
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
    """
    return {
        idx: (
            int(lm.x * frame_width),
            int(lm.y * frame_height),
        )
        for idx, lm in enumerate(face_landmarks.landmark)
    }
