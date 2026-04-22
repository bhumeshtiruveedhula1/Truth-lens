"""
demo/play_deepfake.py — Demo Mode: Pipe pre-recorded video through virtual camera

Usage:
  python demo/play_deepfake.py --video demo/deepfake_sample.mp4

What it does:
  1. Opens the video file with OpenCV
  2. Pipes frames to pyvirtualcam (OBS Virtual Camera / virtual webcam)
  3. The DeepShield agent should be pointed at device_index=1 (or the virtual cam index)
  4. Run Python agent in demo mode: python -m agent.main --demo demo/deepfake_sample.mp4

NOTE: For the simplest demo, just run the agent with --demo flag instead of
this script. This script is for the OBS-based demo where the deepfake plays
THROUGH a virtual camera device visible to Zoom/Meet.

Requirements:
  pip install pyvirtualcam opencv-python
  OBS Studio must be installed and OBS Virtual Camera started
"""

import argparse
import time
import sys
from pathlib import Path

try:
    import cv2
    import pyvirtualcam
except ImportError:
    print("Missing deps. Run: pip install pyvirtualcam opencv-python")
    sys.exit(1)


def play(video_path: str, fps: int = 30, loop: bool = True) -> None:
    path = Path(video_path)
    if not path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[demo] Video: {w}x{h} @ {fps}fps | loop={loop}")

    with pyvirtualcam.Camera(width=w, height=h, fps=fps) as cam:
        print(f"[demo] Virtual camera: {cam.device}")
        print("[demo] Start DeepShield agent with: python -m agent.main --device 1")
        print("[demo] Press Ctrl+C to stop.")

        frame_time = 1.0 / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # pyvirtualcam expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam.send(frame_rgb)
            cam.sleep_until_next_frame()

    cap.release()
    print("[demo] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Demo: Virtual Camera Playback")
    parser.add_argument("--video", required=True, help="Path to deepfake video file")
    parser.add_argument("--fps",   type=int, default=30)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()
    play(args.video, fps=args.fps, loop=not args.no_loop)
