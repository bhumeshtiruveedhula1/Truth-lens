"""
scripts/capture_real.py — Live Webcam Face Capture
====================================================
Captures 224x224 face crops from a live webcam for REAL class collection.

Usage:
    python scripts/capture_real.py --out data/deepfake_dataset/real/session_1
    python scripts/capture_real.py --out data/deepfake_dataset/real/session_2 --device 1 --every 5

Controls:
    Q / ESC → stop and save
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Optional MediaPipe ─────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP = mp.solutions.face_detection
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

TARGET_SIZE  = 224
FACE_PAD     = 0.25   # fractional padding around face bbox
MIN_FACE_PX  = 50     # relaxed — small face still valid (fix 6)


def build_detector():
    """Return (detector, backend_name). MediaPipe preferred over Haar."""
    if _MP_AVAILABLE:
        # Fix 6: relax confidence to 0.3 so faces at angle / distance are caught
        det = _MP.FaceDetection(model_selection=0, min_detection_confidence=0.30)
        return det, "mediapipe"
    # Haar fallback
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return haar, "haar"


def detect_face_mp(frame_rgb: np.ndarray, detector) -> tuple | None:
    """MediaPipe → (x1,y1,x2,y2) in pixel coords, or None."""
    results = detector.process(frame_rgb)
    if not results or not results.detections:
        return None
    det  = results.detections[0]
    bbox = det.location_data.relative_bounding_box
    h, w = frame_rgb.shape[:2]
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width)  * w)
    y2 = int((bbox.ymin + bbox.height) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def detect_face_haar(frame_gray: np.ndarray, detector) -> tuple | None:
    """Haar → (x1,y1,x2,y2) or None."""
    faces = detector.detectMultiScale(frame_gray, 1.1, 5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    return x, y, x + w, y + h


def pad_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Add padding to face bbox then crop + resize to TARGET_SIZE×TARGET_SIZE."""
    h, w = frame.shape[:2]
    fw, fh = x2 - x1, y2 - y1
    px = int(fw * FACE_PAD)
    py = int(fh * FACE_PAD)
    rx1 = max(0, x1 - px)
    ry1 = max(0, y1 - py)
    rx2 = min(w, x2 + px)
    ry2 = min(h, y2 + py)
    crop = frame[ry1:ry2, rx1:rx2]
    return cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture real face crops from webcam")
    p.add_argument("--out",    required=True, help="Output directory for saved crops")
    p.add_argument("--device", type=int, default=0,  help="Camera device index (default 0)")
    p.add_argument("--every",  type=int, default=3,  help="Save every N-th detected frame (default 3)")
    p.add_argument("--max",    type=int, default=0,  help="Stop after N crops saved (0 = unlimited)")
    p.add_argument("--jpg-quality", type=int, default=92, help="JPEG quality 1-100 (default 92)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    detector, backend = build_detector()
    print(f"[Capture] backend={backend}  device={args.device}  out={out}")
    print(f"[Capture] saving every {args.every} frame(s)  max={args.max or 'unlimited'}")

    # Fix 1: DirectShow backend (Windows) — avoids Media Foundation black frames
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.device} (DirectShow)")
        print("        Try --device 0 or run capture_obs.py --list-cameras")
        sys.exit(1)

    # Fix 2: Force resolution so driver negotiates a real mode
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)   # enable autofocus if supported

    # Fix 3: Warm-up — discard first 30 frames while sensor stabilises
    print("[Capture] Camera warming up (30 frames)...")
    for _ in range(30):
        cap.read()
    print("[Capture] Warm-up done.")

    frame_n = 0
    saved   = 0
    t_start = time.time()

    print("[Capture] Press  Q / ESC  to stop\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying...")
            time.sleep(0.05)
            continue

        # Fix 4: blank / black frame guard
        if frame is None or frame.mean() < 5:
            print("[WARN] Empty frame detected (mean<5) — skipping")
            time.sleep(0.02)
            continue

        # Fix 5: show raw feed BEFORE any processing so user can confirm camera
        cv2.imshow("RAW FEED (close = q)", frame)

        frame_n += 1
        display  = frame.copy()

        # ── Detect face ───────────────────────────────────────────────────────
        if backend == "mediapipe":
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox = detect_face_mp(rgb, detector)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox = detect_face_haar(gray, detector)

        face_found = bbox is not None
        crop       = None

        if face_found:
            x1, y1, x2, y2 = bbox
            fw, fh = x2 - x1, y2 - y1
            if fw >= MIN_FACE_PX and fh >= MIN_FACE_PX:
                crop = pad_crop(frame, x1, y1, x2, y2)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 2)
            else:
                face_found = False   # too small

        # Fix 7: fallback center box — confirms camera is live even when no face
        if not face_found:
            h_fr, w_fr = display.shape[:2]
            cx, cy = w_fr // 2, h_fr // 2
            half   = min(w_fr, h_fr) // 6
            cv2.rectangle(display, (cx-half, cy-half), (cx+half, cy+half), (0, 80, 200), 1)
            cv2.putText(display, "No face — aim camera at face",
                        (cx - half, cy - half - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 200), 1)

        # ── Save every N-th frame ─────────────────────────────────────────────
        if face_found and crop is not None and frame_n % args.every == 0:
            fname = out / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            saved += 1

        # ── HUD overlay ───────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        fps_est = frame_n / elapsed if elapsed > 0 else 0
        status  = f"Saved: {saved}  Frame: {frame_n}  FPS: {fps_est:.1f}"
        clr     = (0, 220, 0) if face_found else (0, 80, 200)
        cv2.putText(display, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
        cv2.putText(display, "Q/ESC = stop", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Capture REAL — DeepShield", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

        if args.max and saved >= args.max:
            print(f"[Capture] Reached max={args.max} — stopping.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if backend == "mediapipe":
        detector.close()

    elapsed = time.time() - t_start
    print(f"\n[Capture] Done.  saved={saved}  frames_read={frame_n}  time={elapsed:.1f}s")
    print(f"[Capture] Output: {out.resolve()}")


if __name__ == "__main__":
    main()
