"""
scripts/extract_faces.py — Batch Face Extractor from Video or Image Folder
===========================================================================
Extracts 224x224 face crops from:
  - a single video file (.mp4, .avi, .mov, .mkv)
  - a folder of video files
  - a folder of images (.jpg, .png)

Usage:
    # Extract from a video file
    python scripts/extract_faces.py --src data/raw_videos/deepfake.mp4 --out data/deepfake_dataset/fake/ff_session_1

    # Extract from a folder of videos
    python scripts/extract_faces.py --src data/raw_videos/ --out data/deepfake_dataset/fake/ff_bulk

    # Extract from an image folder
    python scripts/extract_faces.py --src data/celeb_df/images/ --out data/deepfake_dataset/fake/celeb_df

    # Skip every 10 frames (faster, less redundant):
    python scripts/extract_faces.py --src video.mp4 --out out/ --every 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP = mp.solutions.face_detection
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

TARGET_SIZE  = 224
FACE_PAD     = 0.20
MIN_FACE_PX  = 48
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_detector():
    if _MP_AVAILABLE:
        det = _MP.FaceDetection(model_selection=1, min_detection_confidence=0.50)
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
    if roi.size == 0:
        return None
    return cv2.resize(roi, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


def process_video(path: Path, out_dir: Path, every: int, max_crops: int,
                  detector, backend, quality: int, saved_ref: list) -> int:
    """Extract faces from one video file. Returns number of crops saved."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open video: {path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    local_saved  = 0
    frame_n      = 0

    print(f"  Processing video: {path.name}  ({total_frames} frames @ {fps:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % every != 0:
            continue

        bbox = detect_bbox(frame, backend, detector)
        if bbox:
            x1, y1, x2, y2 = bbox
            if (x2-x1) >= MIN_FACE_PX and (y2-y1) >= MIN_FACE_PX:
                crop = pad_crop(frame, x1, y1, x2, y2)
                if crop is not None:
                    idx   = saved_ref[0]
                    fname = out_dir / f"frame_{idx:07d}.jpg"
                    cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    saved_ref[0] += 1
                    local_saved  += 1
                    if max_crops and saved_ref[0] >= max_crops:
                        break

        if frame_n % 500 == 0:
            pct = 100 * frame_n / total_frames if total_frames else 0
            print(f"    {path.name}: frame {frame_n}/{total_frames} ({pct:.0f}%)  saved={local_saved}")

    cap.release()
    return local_saved


def process_image(path: Path, out_dir: Path, detector, backend,
                  quality: int, saved_ref: list) -> bool:
    frame = cv2.imread(str(path))
    if frame is None:
        return False
    bbox = detect_bbox(frame, backend, detector)
    if bbox:
        x1, y1, x2, y2 = bbox
        if (x2-x1) >= MIN_FACE_PX and (y2-y1) >= MIN_FACE_PX:
            crop = pad_crop(frame, x1, y1, x2, y2)
            if crop is not None:
                fname = out_dir / f"frame_{saved_ref[0]:07d}.jpg"
                cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved_ref[0] += 1
                return True
    return False


def parse_args():
    p = argparse.ArgumentParser(description="Batch face extractor from video or image folder")
    p.add_argument("--src",         required=True, help="Video file, video folder, or image folder")
    p.add_argument("--out",         required=True, help="Output directory for face crops")
    p.add_argument("--every",       type=int, default=5,  help="Process every N-th frame (default 5)")
    p.add_argument("--max",         type=int, default=0,  help="Max total crops (0=unlimited)")
    p.add_argument("--jpg-quality", type=int, default=92, help="JPEG quality (default 92)")
    return p.parse_args()


def main():
    args     = parse_args()
    src      = Path(args.src)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector, backend = build_detector()
    print(f"[Extract] backend={backend}  every={args.every}  max={args.max or 'unlimited'}")
    print(f"[Extract] src={src}  out={out_dir}\n")

    saved_ref = [0]   # mutable counter shared across helpers
    t0 = time.time()

    if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
        # Single video
        process_video(src, out_dir, args.every, args.max,
                      detector, backend, args.jpg_quality, saved_ref)

    elif src.is_dir():
        # Collect all videos and images
        videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
        images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

        if videos:
            print(f"[Extract] Found {len(videos)} video(s)")
            for vid in videos:
                if args.max and saved_ref[0] >= args.max:
                    break
                process_video(vid, out_dir, args.every, args.max,
                              detector, backend, args.jpg_quality, saved_ref)

        if images:
            print(f"[Extract] Found {len(images)} image(s) — extracting faces...")
            for i, img_path in enumerate(images):
                if args.max and saved_ref[0] >= args.max:
                    break
                process_image(img_path, out_dir, detector, backend,
                              args.jpg_quality, saved_ref)
                if (i+1) % 200 == 0:
                    print(f"  Images processed: {i+1}/{len(images)}  saved={saved_ref[0]}")

        if not videos and not images:
            print(f"[ERROR] No video or image files found in {src}")
            sys.exit(1)
    else:
        print(f"[ERROR] --src must be a video file or a directory: {src}")
        sys.exit(1)

    if backend == "mediapipe":
        detector.close()

    elapsed = time.time() - t0
    print(f"\n[Extract] Complete.  total_saved={saved_ref[0]}  time={elapsed:.1f}s")
    print(f"[Extract] Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
