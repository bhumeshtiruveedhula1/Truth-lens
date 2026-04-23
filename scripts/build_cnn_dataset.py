"""
scripts/build_cnn_dataset.py -- CNN Face-Crop Dataset Builder
=============================================================
Reads EXISTING session data (frames/ + metadata.json) produced by
the runtime pipeline with --save-crops enabled, then assembles
a clean CNN training dataset.

Consistent with GRU dataset:
  - Same sessions as GRU sequences.npz
  - Same session IDs, same labels (from metadata.json)
  - No new data sources

Usage:
    python scripts/build_cnn_dataset.py

Output:
    data/cnn_dataset/
        real/   session_<id>_frame_<n>.jpg
        fake/   session_<id>_frame_<n>.jpg

How to collect crops for a new session:
    python -m agent.main --record-ml-data --label real --save-crops
    python -m agent.main --record-ml-data --label fake --save-crops
"""

from __future__ import annotations
import csv
import json
import shutil
import sys
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
SESSIONS_DIR = Path("data/sessions")
OUT_ROOT     = Path("data/cnn_dataset")
BLUR_THRESH  = 80.0       # Laplacian variance; crops already 224x224
MIN_FACE_PCT = 0.5        # skip sessions where < 50% frames have face_present=1

SEP = "-" * 64


def _blur_score(img_path: Path) -> float:
    """Laplacian variance as sharpness proxy.  Returns float."""
    import cv2, numpy as np
    img  = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _load_metadata(session_dir: Path) -> dict | None:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_valid_frame_ids(session_dir: Path) -> set[int]:
    """
    Read frames.csv and return set of frame_ids where face_present == 1.
    This ensures CNN crops align with GRU feature rows.
    """
    csv_path = session_dir / "frames.csv"
    if not csv_path.exists():
        return set()
    valid = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    if int(float(row.get("face_present", 0))) == 1:
                        valid.add(int(float(row.get("frame_id", -1))))
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return valid


def build() -> None:
    print(SEP)
    print("CNN DATASET BUILDER (from existing sessions)")
    print(SEP)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "real").mkdir(exist_ok=True)
    (OUT_ROOT / "fake").mkdir(exist_ok=True)

    sessions = sorted(SESSIONS_DIR.iterdir())
    print(f"  Found {len(sessions)} session(s) in {SESSIONS_DIR}")

    n_real = 0; n_fake = 0; n_skip_frame = 0; n_skip_blur = 0
    n_sess_no_crops = 0; n_sess_no_meta = 0; n_sess_bad_label = 0

    for sess_dir in sessions:
        if not sess_dir.is_dir():
            continue

        sid = sess_dir.name.replace("session_", "")

        # ── 1. Load metadata ──────────────────────────────────────────────────
        meta = _load_metadata(sess_dir)
        if meta is None:
            n_sess_no_meta += 1
            continue

        label = meta.get("label", "unknown").strip()
        if label not in ("real", "fake"):
            n_sess_bad_label += 1
            continue

        # ── 2. Find crops directory ───────────────────────────────────────────
        crops_dir = sess_dir / "frames"
        if not crops_dir.exists():
            n_sess_no_crops += 1
            continue

        crop_files = sorted(crops_dir.glob("frame_*.jpg"))
        if not crop_files:
            n_sess_no_crops += 1
            continue

        # ── 3. Load valid frame IDs from CSV (face_present == 1) ─────────────
        valid_ids = _load_valid_frame_ids(sess_dir)

        # ── 4. Process each crop ──────────────────────────────────────────────
        sess_saved = 0
        for crop_path in crop_files:
            # Parse frame_id from filename: frame_0000123.jpg
            try:
                fid = int(crop_path.stem.split("_")[1])
            except (IndexError, ValueError):
                n_skip_frame += 1
                continue

            # Skip if not face-present in CSV
            if valid_ids and fid not in valid_ids:
                n_skip_frame += 1
                continue

            # Blur check
            if _blur_score(crop_path) < BLUR_THRESH:
                n_skip_blur += 1
                continue

            # Copy to output (crop is already 224x224 letterboxed by logger)
            out_name = f"session_{sid}_frame_{fid:07d}.jpg"
            out_path = OUT_ROOT / label / out_name
            try:
                shutil.copy2(str(crop_path), str(out_path))
                sess_saved += 1
                if label == "real":
                    n_real += 1
                else:
                    n_fake += 1
            except Exception as exc:
                print(f"  WARNING: copy failed {crop_path}: {exc}", file=sys.stderr)

        print(f"  [{label:4s}] {sess_dir.name[:32]}  crops={len(crop_files):,}  saved={sess_saved:,}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("DATASET COMPLETE")
    print(SEP)
    print(f"  Real images     : {n_real:,}  ->  {OUT_ROOT / 'real'}")
    print(f"  Fake images     : {n_fake:,}  ->  {OUT_ROOT / 'fake'}")
    print(f"  Skipped (face)  : {n_skip_frame:,}  (face_present=0 in CSV)")
    print(f"  Skipped (blur)  : {n_skip_blur:,}  (Laplacian < {BLUR_THRESH})")
    print(f"  Sessions skipped: {n_sess_no_crops} (no crops)  "
          f"{n_sess_no_meta} (no metadata)  "
          f"{n_sess_bad_label} (unknown label)")

    total = n_real + n_fake
    if total == 0:
        print(f"\n  WARNING: 0 images written.")
        print(f"  Sessions must be recorded with: --record-ml-data --save-crops --label real/fake")
    else:
        print(f"  Total           : {total:,} images  "
              f"({n_real/total*100:.0f}% real / {n_fake/total*100:.0f}% fake)")
    print(SEP)


if __name__ == "__main__":
    build()
