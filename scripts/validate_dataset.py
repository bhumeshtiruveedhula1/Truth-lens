"""
scripts/validate_dataset.py — Dataset Quality Validator
========================================================
Audits a dataset directory (real/ and fake/ subdirs) before training.

Checks:
  - Image counts per class
  - Size consistency (all must be TARGET_SIZE x TARGET_SIZE)
  - Face presence (re-runs detector on a random sample)
  - Class balance
  - Corrupt / unreadable files

Usage:
    python scripts/validate_dataset.py --dir data/deepfake_dataset
    python scripts/validate_dataset.py --dir data/deepfake_dataset --sample 50
    python scripts/validate_dataset.py --dir data/deepfake_dataset --full-scan
"""
from __future__ import annotations

import argparse
import random
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

TARGET_SIZE = 224
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}
MIN_FACE_PX = 40


def collect_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def check_size(img_path: Path) -> tuple[bool, tuple[int,int] | None]:
    """Returns (ok, (h,w)) or (False, None) if unreadable."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False, None
    h, w = img.shape[:2]
    return (h == TARGET_SIZE and w == TARGET_SIZE), (h, w)


def has_face(img_path: Path, detector, backend: str) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    if backend == "mediapipe":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        return bool(res and res.detections)
    else:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
        return len(faces) > 0


def audit_class(
    class_dir:  Path,
    label:      str,
    sample_n:   int,
    full_scan:  bool,
    detector,
    backend:    str,
) -> dict:
    """Run all checks on one class directory. Returns audit dict."""
    result = {
        "label":        label,
        "dir":          str(class_dir),
        "exists":       class_dir.exists(),
        "total":        0,
        "corrupt":      0,
        "wrong_size":   0,
        "size_issues":  [],
        "no_face":      0,
        "no_face_examples": [],
        "sample_n":     sample_n,
        "face_ok_pct":  None,
    }

    if not class_dir.exists():
        print(f"  [{label}] Directory not found: {class_dir}")
        return result

    images = collect_images(class_dir)
    result["total"] = len(images)

    if len(images) == 0:
        print(f"  [{label}] EMPTY — no images found in {class_dir}")
        return result

    # ── Size + corrupt scan (full or sampled) ─────────────────────────────────
    scan_set = images if full_scan else random.sample(images, min(sample_n * 2, len(images)))
    print(f"  [{label}] Checking {len(scan_set)} images for size/corruption...")

    for p in scan_set:
        ok, dims = check_size(p)
        if dims is None:
            result["corrupt"] += 1
        elif not ok:
            result["wrong_size"] += 1
            result["size_issues"].append(f"{p.name}: {dims[1]}x{dims[0]}")

    # ── Face presence sample ──────────────────────────────────────────────────
    face_sample = random.sample(images, min(sample_n, len(images)))
    print(f"  [{label}] Checking faces in {len(face_sample)} sampled images...")

    no_face_count = 0
    for p in face_sample:
        if not has_face(p, detector, backend):
            no_face_count += 1
            result["no_face_examples"].append(p.name)

    result["no_face"]    = no_face_count
    result["face_ok_pct"] = round(100.0 * (len(face_sample) - no_face_count) / len(face_sample), 1)

    return result


def print_report(real_r: dict, fake_r: dict, target: int) -> None:
    total_real = real_r["total"]
    total_fake = fake_r["total"]
    total      = total_real + total_fake
    balance    = round(100.0 * total_real / total, 1) if total else 0

    sep = "=" * 60
    print(f"\n{sep}")
    print("DATASET VALIDATION REPORT")
    print(sep)

    for r in [real_r, fake_r]:
        lbl   = r["label"].upper()
        total_cls = r["total"]
        print(f"\n  [{lbl}]  dir={r['dir']}")
        if not r["exists"]:
            print(f"    STATUS: MISSING DIRECTORY")
            continue
        if total_cls == 0:
            print(f"    STATUS: EMPTY")
            continue

        print(f"    Total images  : {total_cls}")
        print(f"    Corrupt files : {r['corrupt']}")
        wrong = r["wrong_size"]
        print(f"    Wrong size    : {wrong}" + (f"  (expected {target}x{target})" if wrong else ""))
        if r["size_issues"]:
            for s in r["size_issues"][:5]:
                print(f"      - {s}")
            if len(r["size_issues"]) > 5:
                print(f"      ... and {len(r['size_issues'])-5} more")
        face_pct = r["face_ok_pct"]
        nf       = r["no_face"]
        face_ok  = "OK" if (face_pct is not None and face_pct >= 80) else "WARNING"
        print(f"    Face presence : {face_pct}% ({face_ok})  [no-face in {nf}/{r['sample_n']} samples]")
        if r["no_face_examples"]:
            for ex in r["no_face_examples"][:3]:
                print(f"      - {ex}")

    print(f"\n  CLASS BALANCE")
    print(f"    Real : {total_real:5d}  ({balance:.1f}%)")
    print(f"    Fake : {total_fake:5d}  ({100-balance:.1f}%)")
    print(f"    Total: {total:5d}")
    if total < 2000:
        print(f"    WARNING: dataset is small (<2000 images). Training may overfit.")
    if abs(balance - 50) > 20:
        print(f"    WARNING: class imbalance > 20% — use WeightedRandomSampler during training.")
    else:
        print(f"    Balance: OK")

    print(f"\n{sep}")
    # Overall verdict
    issues = 0
    if total_real == 0:     issues += 1; print("  FAIL: real/ directory is empty")
    if total_fake == 0:     issues += 1; print("  FAIL: fake/ directory is empty")
    real_fp = real_r.get("face_ok_pct") or 0
    fake_fp = fake_r.get("face_ok_pct") or 0
    if real_fp < 80:        issues += 1; print(f"  FAIL: real face detection too low ({real_fp}%)")
    if fake_fp < 80:        issues += 1; print(f"  FAIL: fake face detection too low ({fake_fp}%)")
    if real_r["corrupt"]:   print(f"  WARN: {real_r['corrupt']} corrupt file(s) in real/")
    if fake_r["corrupt"]:   print(f"  WARN: {fake_r['corrupt']} corrupt file(s) in fake/")

    if issues == 0:
        print("  VERDICT: PASS — dataset ready for training")
    else:
        print(f"  VERDICT: FAIL — {issues} critical issue(s) found")
    print(sep)


def parse_args():
    p = argparse.ArgumentParser(description="Validate deepfake detection dataset")
    p.add_argument("--dir",       required=True, help="Root dir containing real/ and fake/ subdirs")
    p.add_argument("--real",      default=None,  help="Override real class dir (default: <dir>/real)")
    p.add_argument("--fake",      default=None,  help="Override fake class dir (default: <dir>/fake)")
    p.add_argument("--sample",    type=int, default=30, help="N images to face-check per class (default 30)")
    p.add_argument("--full-scan", action="store_true",  help="Scan ALL images for size/corruption (slow)")
    return p.parse_args()


def main():
    args     = parse_args()
    root     = Path(args.dir)
    real_dir = Path(args.real) if args.real else root / "real"
    fake_dir = Path(args.fake) if args.fake else root / "fake"

    print(f"[Validate] root={root}")
    print(f"[Validate] real={real_dir}")
    print(f"[Validate] fake={fake_dir}")
    print(f"[Validate] sample={args.sample}  full_scan={args.full_scan}\n")

    # Build detector
    if _MP_AVAILABLE:
        detector = _MP.FaceDetection(model_selection=1, min_detection_confidence=0.45)
        backend  = "mediapipe"
    else:
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        backend  = "haar"

    print(f"[Validate] Face detector: {backend}\n")

    t0     = time.time()
    real_r = audit_class(real_dir, "real", args.sample, args.full_scan, detector, backend)
    fake_r = audit_class(fake_dir, "fake", args.sample, args.full_scan, detector, backend)

    if _MP_AVAILABLE:
        detector.close()

    print_report(real_r, fake_r, TARGET_SIZE)
    print(f"\n[Validate] Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
