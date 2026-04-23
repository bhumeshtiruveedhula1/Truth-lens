"""
scripts/build_cnn_dataset.py -- CNN Face-Crop Dataset Builder
=============================================================
Assembles CNN training dataset from session face crops produced
by the runtime pipeline with --save-crops enabled.

Alignment guarantee:
  - Same sessions, same session IDs, same labels as GRU dataset
  - No new data sources

Usage:
    python scripts/build_cnn_dataset.py

Collect crops for new sessions:
    python -m agent.main --record-ml-data --label real --save-crops
    python -m agent.main --record-ml-data --label fake --save-crops

NOTE: Sessions recorded WITHOUT --save-crops have frames.csv but no
frames/ directory. Re-record those sessions to generate crops.
"""

from __future__ import annotations
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
SESSIONS_DIR = Path("data/sessions")
OUT_ROOT     = Path("data/cnn_dataset")
BLUR_THRESH  = 50.0      # Laplacian variance — lowered from 80 (face_roi is small ROI)
CROP_SIZE    = 224

SEP  = "-" * 64
SEP2 = "=" * 64


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_metadata(session_dir: Path) -> dict | None:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  WARNING: bad metadata.json in {session_dir.name}: {exc}")
        return None


def _load_valid_frame_ids(session_dir: Path) -> set[int]:
    """
    Return frame_ids where face_present == 1 from frames.csv.
    Returns empty set (= skip no frames) if CSV missing.
    """
    csv_path = session_dir / "frames.csv"
    if not csv_path.exists():
        return set()
    valid: set[int] = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    if int(float(row.get("face_present", 0))) == 1:
                        valid.add(int(float(row.get("frame_id", -1))))
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        print(f"  WARNING: could not read frames.csv: {exc}")
    return valid


def _blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_readable(path: Path) -> np.ndarray | None:
    """Returns decoded image array or None if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


# ── Main ───────────────────────────────────────────────────────────────────────

def build() -> None:
    print(SEP2)
    print("  CNN DATASET BUILDER")
    print(SEP2)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not SESSIONS_DIR.exists():
        print(f"\nERROR: sessions directory not found: {SESSIONS_DIR}")
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "real").mkdir(exist_ok=True)
    (OUT_ROOT / "fake").mkdir(exist_ok=True)

    all_sessions = [s for s in sorted(SESSIONS_DIR.iterdir()) if s.is_dir()]
    print(f"\n  Sessions found : {len(all_sessions)}")
    print(f"  Output root    : {OUT_ROOT}\n")

    # ── Per-session counters ──────────────────────────────────────────────────
    n_real = 0; n_fake = 0

    skip_no_meta   : list[str] = []
    skip_bad_label : list[str] = []
    skip_no_crops  : list[str] = []  # has metadata but no frames/ dir
    skip_empty_crops: list[str] = [] # has frames/ but 0 crop files

    n_skip_unreadable = 0
    n_skip_blur       = 0
    n_skip_no_face    = 0

    print(f"  {'Session ID':<40} {'Label':<6} {'Crops':>6} {'Saved':>6}  Notes")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6}  -----")

    for sess_dir in all_sessions:
        sid   = sess_dir.name.replace("session_", "")
        sid_s = sid[:36]

        # ── 1. Metadata ───────────────────────────────────────────────────────
        meta = _load_metadata(sess_dir)
        if meta is None:
            skip_no_meta.append(sess_dir.name)
            print(f"  {sid_s:<40} {'?':<6} {'N/A':>6} {'SKIP':>6}  no metadata.json")
            continue

        label = meta.get("label", "").strip().lower()
        if label not in ("real", "fake"):
            skip_bad_label.append(f"{sess_dir.name} (label={label!r})")
            print(f"  {sid_s:<40} {label or '?':<6} {'N/A':>6} {'SKIP':>6}  invalid label")
            continue

        # ── 2. Crops directory ────────────────────────────────────────────────
        crops_dir = sess_dir / "frames"
        if not crops_dir.exists():
            skip_no_crops.append(sess_dir.name)
            print(f"  {sid_s:<40} {label:<6} {'N/A':>6} {'SKIP':>6}  no frames/ dir (re-record with --save-crops)")
            continue

        crop_files = sorted(crops_dir.glob("frame_*.jpg"))
        if not crop_files:
            skip_empty_crops.append(sess_dir.name)
            print(f"  {sid_s:<40} {label:<6} {0:>6} {'SKIP':>6}  frames/ is empty")
            continue

        # ── 3. Valid frame IDs from CSV ───────────────────────────────────────
        valid_ids = _load_valid_frame_ids(sess_dir)
        # If CSV is missing/empty valid_ids set, allow all frames through
        use_csv_filter = len(valid_ids) > 0

        # ── 4. Process each crop ──────────────────────────────────────────────
        sess_saved = 0
        sess_skip_face = 0; sess_skip_blur = 0; sess_skip_unread = 0

        for crop_path in crop_files:
            # Parse frame_id: frame_0000049.jpg → 49
            try:
                fid = int(crop_path.stem.replace("frame_", ""))
            except ValueError:
                sess_skip_unread += 1
                continue

            # CSV face_present filter
            if use_csv_filter and fid not in valid_ids:
                sess_skip_face += 1
                n_skip_no_face += 1
                continue

            # Load image
            img = _is_readable(crop_path)
            if img is None:
                sess_skip_unread += 1
                n_skip_unreadable += 1
                continue

            # Shape guard — warn but do NOT skip (resize instead)
            h, w = img.shape[:2]
            if (h, w) != (CROP_SIZE, CROP_SIZE):
                img = cv2.resize(img, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

            # Blur check
            blur = _blur_score(img)
            if blur < BLUR_THRESH:
                sess_skip_blur += 1
                n_skip_blur += 1
                continue

            # Copy/write to output
            out_name = f"session_{sid}_frame_{fid:07d}.jpg"
            out_path = OUT_ROOT / label / out_name
            try:
                # Write via OpenCV to ensure valid JPEG even after potential resize
                cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                sess_saved += 1
                if label == "real":
                    n_real += 1
                else:
                    n_fake += 1
            except Exception as exc:
                print(f"\n  WARNING: write failed {out_path}: {exc}")

        notes = []
        if sess_skip_face   > 0: notes.append(f"face_filter={sess_skip_face}")
        if sess_skip_blur   > 0: notes.append(f"blur={sess_skip_blur}")
        if sess_skip_unread > 0: notes.append(f"unreadable={sess_skip_unread}")
        note_str = "  " + ", ".join(notes) if notes else ""
        print(f"  {sid_s:<40} {label:<6} {len(crop_files):>6} {sess_saved:>6}{note_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = n_real + n_fake
    print(f"\n{SEP2}")
    print("  SUMMARY")
    print(SEP2)
    print(f"  Real images copied   : {n_real:,}")
    print(f"  Fake images copied   : {n_fake:,}")
    print(f"  Total                : {total:,}")

    if total > 0:
        print(f"  Ratio (real/fake)    : {n_real/total*100:.0f}% / {n_fake/total*100:.0f}%")

    print(f"\n  --- Skipped sessions ---")
    print(f"  No metadata.json     : {len(skip_no_meta)}")
    print(f"  Invalid label        : {len(skip_bad_label)}")
    if skip_bad_label:
        for s in skip_bad_label[:3]: print(f"    {s}")
    print(f"  No frames/ directory : {len(skip_no_crops)}  <- recorded without --save-crops")
    print(f"  Empty frames/        : {len(skip_empty_crops)}")

    print(f"\n  --- Skipped images ---")
    print(f"  face_present=0 (CSV) : {n_skip_no_face:,}")
    print(f"  Too blurry           : {n_skip_blur:,}  (Laplacian < {BLUR_THRESH})")
    print(f"  Unreadable           : {n_skip_unreadable:,}")

    # ── Actionable guidance ───────────────────────────────────────────────────
    print(f"\n{SEP2}")
    if total == 0:
        print("  RESULT: 0 images — no usable sessions found.")
        print()
        print("  ACTION REQUIRED:")
        print("  Record new sessions with --save-crops to generate face crops:")
        print()
        print("    # Real session (30-60 seconds, sit normally):")
        print("    python -m agent.main --record-ml-data --label real --save-crops")
        print()
        print("    # Fake session (show phone/image to webcam):")
        print("    python -m agent.main --record-ml-data --label fake --save-crops")
        print()
        print("  Then re-run:  python scripts/build_cnn_dataset.py")
    elif n_fake == 0:
        print(f"  RESULT: {total} real images — but 0 fake images.")
        print()
        print("  ACTION REQUIRED: Record at least 1 fake session:")
        print("    python -m agent.main --record-ml-data --label fake --save-crops")
        print()
        print("  Then re-run:  python scripts/build_cnn_dataset.py")
    elif n_real == 0:
        print(f"  RESULT: {total} fake images — but 0 real images.")
        print()
        print("  ACTION REQUIRED: Record at least 1 real session:")
        print("    python -m agent.main --record-ml-data --label real --save-crops")
        print()
        print("  Then re-run:  python scripts/build_cnn_dataset.py")
    else:
        real_pct = n_real / total * 100
        if real_pct > 70 or real_pct < 30:
            balance_warn = f"  WARNING: class imbalance ({n_real} real vs {n_fake} fake)\n"
        else:
            balance_warn = ""
        print(f"  RESULT: {total:,} images ready for CNN training.")
        if balance_warn:
            print(balance_warn)
        print(f"  Validate: python scripts/validate_cnn_dataset.py")
    print(SEP2)


if __name__ == "__main__":
    build()
