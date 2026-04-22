"""
agent/liveness/head_pose.py

Head-pose estimation utilities plus two adapters:
- HeadPoseModule for the lightweight liveness pipeline
- HeadPoseEstimator for the event-bus based extractor expected by main.py
"""

from __future__ import annotations

import collections
from typing import Tuple

import cv2
import numpy as np

from agent.events import FrameEvent, LivenessSignal
from agent.liveness.base import SignalExtractor


HEAD_POSE_INDICES = [1, 33, 61, 199, 263, 291]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_OUTER_CORNER = 362

HISTORY_SEC = 5.0


class AngleBuffer:
    def __init__(self, size: int = 10) -> None:
        self._buffer: collections.deque = collections.deque(maxlen=size)

    def add(self, angles: list | np.ndarray) -> None:
        self._buffer.append(angles)

    def get_average(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(3)
        return np.mean(self._buffer, axis=0)

    def reset(self) -> None:
        self._buffer.clear()


def _build_camera_matrix(image_size: Tuple[int, int]) -> np.ndarray:
    img_h, img_w = image_size
    focal_length = float(img_w)
    return np.array(
        [
            [focal_length, 0, img_h / 2.0],
            [0, focal_length, img_w / 2.0],
            [0, 0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_head_pose(
    mesh_points_3D: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[float, float, float]:
    img_h, img_w = image_size

    pts_3D = np.multiply(mesh_points_3D[HEAD_POSE_INDICES], [img_w, img_h, 1]).astype(
        np.float64
    )
    pts_2D = pts_3D[:, :2].astype(np.float64)

    cam_matrix = _build_camera_matrix(image_size)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(pts_3D, pts_2D, cam_matrix, dist_matrix)
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = angles[0] * 360.0
    yaw = angles[1] * 360.0
    roll = angles[2] * 360.0
    return pitch, yaw, roll


def classify_head_direction(
    pitch: float,
    yaw: float,
    threshold: float = 10.0,
) -> str:
    if yaw < -threshold:
        return "Left"
    if yaw > threshold:
        return "Right"
    if pitch < -threshold:
        return "Down"
    if pitch > threshold:
        return "Up"
    return "Forward"


def compute_iris_gaze_vector(
    mesh_points: np.ndarray,
    eye: str = "left",
) -> Tuple[int, int, int, int, float]:
    if eye == "left":
        iris_idx = LEFT_IRIS
        corner_idx = LEFT_EYE_OUTER_CORNER
    else:
        iris_idx = RIGHT_IRIS
        corner_idx = RIGHT_EYE_OUTER_CORNER

    iris_pts = mesh_points[iris_idx]
    (cx, cy), radius = cv2.minEnclosingCircle(iris_pts)
    cx, cy = int(cx), int(cy)

    corner = mesh_points[corner_idx]
    dx = cx - int(corner[0])
    dy = cy - int(corner[1])

    return cx, cy, dx, dy, radius


class HeadPoseModule:
    def __init__(
        self,
        smooth_window: int = 10,
        direction_threshold: float = 10.0,
    ) -> None:
        self._buffer = AngleBuffer(size=smooth_window)
        self._threshold = direction_threshold

    def process(
        self,
        mesh_points_3D: np.ndarray,
        image_size: Tuple[int, int],
    ) -> dict:
        if mesh_points_3D is None or len(mesh_points_3D) == 0:
            self._buffer.reset()
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "direction": "Forward"}

        pitch, yaw, roll = estimate_head_pose(mesh_points_3D, image_size)
        self._buffer.add([pitch, yaw, roll])
        sp, sy, sr = self._buffer.get_average()

        return {
            "pitch": float(sp),
            "yaw": float(sy),
            "roll": float(sr),
            "direction": classify_head_direction(sp, sy, self._threshold),
        }

    def reset(self) -> None:
        self._buffer.reset()


class HeadPoseEstimator(SignalExtractor):
    name = "head_pose"

    def __init__(self) -> None:
        self._module = HeadPoseModule()
        self._yaw_history: collections.deque = collections.deque()
        self._pitch_history: collections.deque = collections.deque()
        self._ts_history: collections.deque = collections.deque()
        self._frame_count = 0

    def register(self) -> None:
        super().register()

    async def handle(self, event: FrameEvent) -> None:
        await super().handle(event)

    def extract(self, event: FrameEvent) -> LivenessSignal | None:
        if not event.face_landmarks:
            self._module.reset()
            return None

        mesh_points_3D = np.array(event.face_landmarks, dtype=np.float64)
        signal = self._module.process(mesh_points_3D, image_size=(event.frame_height, event.frame_width))

        pitch = float(signal["pitch"])
        yaw = float(signal["yaw"])
        roll = float(signal["roll"])

        now = event.timestamp
        self._frame_count += 1

        self._yaw_history.append(yaw)
        self._pitch_history.append(pitch)
        self._ts_history.append(now)

        cutoff = now - HISTORY_SEC
        while self._ts_history and self._ts_history[0] < cutoff:
            self._ts_history.popleft()
            self._yaw_history.popleft()
            self._pitch_history.popleft()

        if len(self._yaw_history) > 5:
            yaw_var = float(np.var(list(self._yaw_history)))
            pitch_var = float(np.var(list(self._pitch_history)))
            total_var = yaw_var + pitch_var
            if total_var < 0.1:
                score = 0.2
            elif total_var < 1.0:
                score = 0.5 + total_var * 0.5
            elif total_var < 50.0:
                score = 1.0
            else:
                score = max(0.3, 1.0 - (total_var - 50.0) / 200.0)
        else:
            score = 0.7
            total_var = 0.0

        confidence = (
            min(1.0, self._frame_count / 60.0) if len(self._yaw_history) > 5 else 0.05
        )
        print(f"[HEAD] yaw={round(yaw,2)} pitch={round(pitch,2)} score={round(score,2)}")

        return LivenessSignal(
            session_id=event.session_id,
            extractor_name=self.name,
            value=yaw,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            metadata={
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "roll": round(roll, 2),
                "direction": signal["direction"],
                "pose_variance": round(total_var, 4),
            },
        )
