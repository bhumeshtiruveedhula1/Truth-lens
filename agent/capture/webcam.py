"""
capture/webcam.py — Capture Layer

Acquires raw webcam frames, runs MediaPipe FaceMesh,
and emits FrameEvents onto the bus at ~30fps.

This module has NO signal processing logic — it is pure acquisition.
To use a pre-recorded video for demo, pass video_path to CaptureSource.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from agent.events import FrameEvent
from agent.event_bus import bus

logger = logging.getLogger(__name__)


class CaptureSource:
    """
    Wraps a webcam or video file.
    Set video_path to replay a deepfake demo clip instead of live camera.
    """

    def __init__(
        self,
        session_id: str,
        device_index: int = 0,
        video_path: Optional[str] = None,
        target_fps: int = 30,
    ):
        self.session_id = session_id
        self.device_index = device_index
        self.video_path = video_path
        self.target_fps = target_fps
        self._frame_id = 0
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._face_mesh = None

    def _open(self) -> bool:
        source = self.video_path if self.video_path else self.device_index
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.error(f"Failed to open capture source: {source}")
            return False
        # Force resolution for consistent landmark coordinates
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Create FaceMesh per-instance during source lifecycle, not at module import.
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info(f"Capture source opened: {source}")
        return True

    def _close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def _process_frame(self, frame_bgr: np.ndarray) -> FrameEvent:
        """Run MediaPipe FaceMesh and build a FrameEvent."""
        print("Processing frame...")
        if self._face_mesh is None:
            raise RuntimeError("CaptureSource FaceMesh not initialized")

        h, w = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)
        print(
            "FaceMesh result:",
            results is not None and results.multi_face_landmarks is not None,
        )

        landmarks = []
        face_bbox = (0, 0, 0, 0)
        face_detected = False
        face_roi_bgr: Optional[bytes] = None
        face_roi_shape: Optional[tuple[int, int]] = None

        if results and results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0]
            landmarks = [(p.x, p.y, p.z) for p in lm.landmark]

            # Compute bounding box from landmarks
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            face_bbox = (x1, y1, x2 - x1, y2 - y1)

            # Pass only the face ROI bytes (~20KB) not the full frame (~920KB).
            # Only used by CompressionArtifactDetector every 5th frame.
            pad = 20
            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(w, x2 + pad)
            ry2 = min(h, y2 + pad)
            if rx2 > rx1 and ry2 > ry1:
                face_roi = frame_bgr[ry1:ry2, rx1:rx2]
                face_roi_shape = face_roi.shape[:2]
                face_roi_bgr = face_roi.tobytes()
        if not face_detected:
            print("NO FACE DETECTED")

        return FrameEvent(
            session_id=self.session_id,
            frame_id=self._frame_id,
            face_landmarks=landmarks,
            face_bbox=face_bbox,
            face_detected=face_detected,
            frame_width=w,
            frame_height=h,
            face_roi_bgr=face_roi_bgr,
            face_roi_shape=face_roi_shape,
        )

    async def run(self) -> None:
        """Main async loop — reads frames and publishes FrameEvents."""
        if not self._open():
            return

        self._running = True
        frame_interval = 1.0 / self.target_fps
        logger.info(f"Capture loop started (session={self.session_id})")

        loop = asyncio.get_running_loop()
        try:
            while self._running:
                loop_start = time.perf_counter()

                # Use run_in_executor so the blocking camera read does NOT
                # stall the asyncio event loop (risk engine, WS broadcast, etc.)
                ret, frame = await loop.run_in_executor(None, self._read_frame)
                print("READ FRAME:", ret)
                if not ret or frame is None:
                    if self.video_path:
                        # Loop video file for demo
                        if self._cap is not None:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("Camera read failed — retrying...")
                    await asyncio.sleep(0.1)
                    continue

                self._frame_id += 1
                self._last_raw_frame = frame  # retained for debug UI
                # FaceMesh is the expensive step; keep it off the asyncio thread.
                event = await loop.run_in_executor(None, self._process_frame, frame)
                await bus.publish(event)

                # Throttle to target fps
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

        finally:
            self._close()
            logger.info("Capture loop stopped")

    def stop(self) -> None:
        self._running = False
