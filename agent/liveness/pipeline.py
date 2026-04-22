from __future__ import annotations

import numpy as np

from agent.liveness.blink import BlinkModule
from agent.liveness.head_pose import HeadPoseModule
from agent.liveness.texture import TextureModule


class LivenessPipeline:
    def __init__(self):
        self.blink_module = BlinkModule()
        self.head_pose_module = HeadPoseModule()
        self.texture_module = TextureModule()

    def process(self, frame_bgr, landmarks, w, h):
        mesh_points_3D = self._to_mesh_points_3d(landmarks)
        mesh_points_2D = self._to_mesh_points_2d(landmarks, w, h)

        blink = self.blink_module.process(landmarks, w, h)
        head_pose = self.head_pose_module.process(mesh_points_3D, image_size=(h, w))
        texture = self.texture_module.process(frame_bgr, self._to_face_roi(mesh_points_2D))

        return {
            "blink": blink,
            "head_pose": head_pose,
            "texture": texture,
        }

    def _to_mesh_points_3d(self, landmarks):
        if not landmarks:
            return np.empty((0, 3), dtype=np.float64)

        return np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks.landmark],
            dtype=np.float64,
        )

    def _to_mesh_points_2d(self, landmarks, w, h):
        if not landmarks:
            return np.empty((0, 2), dtype=np.int32)

        return np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark],
            dtype=np.int32,
        )

    def _to_face_roi(self, mesh_points_2D):
        if mesh_points_2D.size == 0:
            return (0, 0, 0, 0)

        x_min = int(np.min(mesh_points_2D[:, 0]))
        x_max = int(np.max(mesh_points_2D[:, 0]))
        y_min = int(np.min(mesh_points_2D[:, 1]))
        y_max = int(np.max(mesh_points_2D[:, 1]))
        return (y_min, y_max, x_min, x_max)
