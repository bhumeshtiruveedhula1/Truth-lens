"""
scripts/capture_obs.py — OBS Virtual Camera Face Capture (FAKE class)
Usage:
    python scripts/capture_obs.py --list-cameras
    python scripts/capture_obs.py --out data/deepfake_dataset/fake/obs_session_1 --device 1
    python scripts/capture_obs.py --out data/deepfake_dataset/fake/obs_session_2 --device 2 --every 1 --max 3000
Controls:  Q / ESC → stop
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP = mp.solutions.face_detection
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

TARGET_SIZE = 224
FACE_PAD    = 0.20
MIN_FACE_PX = 60


def list_cameras(max_idx: int = 6) -> None:
    print("Scanning camera devices (DirectShow backend)...")
    for i in range(max_idx):
        # Use DirectShow so OBS Virtual Camera shows up correctly
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Device {i}: {w}x{h}  <- available")
            cap.release()
        else:
            print(f"  Device {i}: not available")


def build_detector():
    if _MP_AVAILABLE:
        # Relaxed confidence — OBS feeds can have softer face edges
        det = _MP.FaceDetection(model_selection=0, min_detection_confidence=0.30)
        return det, "mediapipe"
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return haar, "haar"


def detect_bbox(frame, backend, detector):
    if backend == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        if not res or not res.detections:
            return None
        b = res.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        return (max(0, int(b.xmin*w)), max(0, int(b.ymin*h)),
                min(w, int((b.xmin+b.width)*w)), min(h, int((b.ymin+b.height)*h)))
    else:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
        if len(faces) == 0:
            return None
        x, y, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        return x, y, x+fw, y+fh


def pad_crop(frame, x1, y1, x2, y2):
    fh, fw = frame.shape[:2]
    px = int((x2-x1)*FACE_PAD);  py = int((y2-y1)*FACE_PAD)
    roi = frame[max(0,y1-py):min(fh,y2+py), max(0,x1-px):min(fw,x2+px)]
    return cv2.resize(roi, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


def parse_args():
    p = argparse.ArgumentParser(description="OBS virtual camera face capture (FAKE class)")
    p.add_argument("--out",          default=None, help="Output directory")
    p.add_argument("--device",       type=int, default=1,  help="Camera device index (default 1)")
    p.add_argument("--every",        type=int, default=2,  help="Save every N frames (default 2)")
    p.add_argument("--max",          type=int, default=0,  help="Max crops (0=unlimited)")
    p.add_argument("--jpg-quality",  type=int, default=92)
    p.add_argument("--list-cameras", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_cameras:
        list_cameras(); return
    if not args.out:
        print("[ERROR] --out required.  Example: --out data/deepfake_dataset/fake/obs_session_1")
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector, backend = build_detector()
    print(f"[OBS] backend={backend}  device={args.device}  out={out}")
    print("[OBS] Ensure OBS Virtual Camera is ON before starting.")

    # Fix 1: DirectShow backend — required for OBS Virtual Camera on Windows
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)

    # Fix 4: debug print immediately after open
    print(f"[DEBUG] device={args.device}  opened={cap.isOpened()}")

    if not cap.isOpened():
        print(f"[ERROR] Cannot open device {args.device} — run --list-cameras")
        sys.exit(1)

    # Fix 2: Force resolution so driver negotiates a valid mode
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Fix 3: Warm-up — discard first 30 frames while OBS stream stabilises
    print("[OBS] Warming up camera (30 frames)...")
    for _ in range(30):
        cap.read()
    print("[OBS] Warm-up done.")

    frame_n = saved = 0
    t0 = time.time()
    print("[OBS] Press Q/ESC to stop\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — is OBS Virtual Camera still running?")
            time.sleep(0.1); continue

        # Fix 5: skip black / blank frames
        if frame is None or frame.mean() < 5:
            continue

        frame_n += 1
        display  = frame.copy()
        bbox     = detect_bbox(frame, backend, detector)
        face_ok  = False

        if bbox:
            x1, y1, x2, y2 = bbox
            if (x2-x1) >= MIN_FACE_PX and (y2-y1) >= MIN_FACE_PX:
                face_ok = True
                if frame_n % args.every == 0:
                    crop = pad_crop(frame, x1, y1, x2, y2)
                    cv2.imwrite(str(out/f"frame_{saved:06d}.jpg"), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                    saved += 1
                cv2.rectangle(display, (x1,y1), (x2,y2), (0,60,255), 2)

        elapsed = time.time()-t0
        status  = f"[OBS/FAKE] Saved:{saved}  Frame:{frame_n}  FPS:{frame_n/elapsed:.1f}" if elapsed else ""
        clr = (0,60,255) if face_ok else (80,80,200)
        cv2.putText(display, status,       (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
        cv2.putText(display, "Q/ESC=stop", (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.imshow("Capture OBS/FAKE", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27): break
        if args.max and saved >= args.max:
            print(f"[OBS] Reached max={args.max}"); break

    cap.release()
    cv2.destroyAllWindows()
    if backend == "mediapipe": detector.close()
    print(f"\n[OBS] Done.  saved={saved}  frames={frame_n}  time={time.time()-t0:.1f}s")
    print(f"[OBS] Output: {out.resolve()}")


if __name__ == "__main__":
    main()
