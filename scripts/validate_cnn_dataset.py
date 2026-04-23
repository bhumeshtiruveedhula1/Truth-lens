"""
scripts/validate_cnn_dataset.py -- CNN Dataset Validation
==========================================================
Fast, streaming sanity check before CNN training.

Phase: real-human vs non-live-input detection
Input: data/cnn_dataset/real/*.jpg  +  data/cnn_dataset/fake/*.jpg

Usage:
    python scripts/validate_cnn_dataset.py
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_ROOT    = Path("data/cnn_dataset")
EXPECTED_SHAPE  = (224, 224)        # (H, W) — channels not checked here
IMBALANCE_LIMIT = 0.70              # warn if either class > 70% of total
BRIGHT_GAP_WARN = 20.0              # warn if mean brightness differs > 20 pts
DARK_THRESH     = 30.0              # mean grayscale < 30 → extremely dark
BRIGHT_THRESH   = 225.0             # mean grayscale > 225 → extremely bright
BLANK_STD_THRESH = 5.0              # std < 5 → near-empty / near-uniform image
SAMPLE_LIMIT    = 5                 # max filenames printed in error lists

SEP  = "-" * 56
SEP2 = "=" * 56


# ── Helpers ────────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_class(class_dir: Path, label: str) -> dict:
    """
    Stream through all JPEGs in class_dir.
    Returns a stats dict without holding images in memory.
    """
    files = sorted(class_dir.glob("*.jpg"))
    n = len(files)

    shape_errors:   list[str] = []
    dark_files:     list[str] = []
    bright_files:   list[str] = []
    blank_files:    list[str] = []
    unreadable:     list[str] = []

    brightness_vals: list[float] = []
    std_vals:        list[float] = []
    seen_hashes:     dict[str, str] = {}   # md5 -> first filename
    dup_count = 0

    for fpath in files:
        # ── Hash (duplicate check) ────────────────────────────────────────────
        digest = _md5(fpath)
        if digest in seen_hashes:
            dup_count += 1
        else:
            seen_hashes[digest] = fpath.name

        # ── Load image ────────────────────────────────────────────────────────
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            unreadable.append(fpath.name)
            continue

        h, w = img.shape[:2]

        # ── Shape check ───────────────────────────────────────────────────────
        if (h, w) != EXPECTED_SHAPE:
            shape_errors.append(f"{fpath.name} ({w}x{h})")

        # ── Pixel stats (grayscale) ───────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = float(gray.mean())
        std  = float(gray.std())
        brightness_vals.append(mean)
        std_vals.append(std)

        # ── Quality flags ─────────────────────────────────────────────────────
        if mean < DARK_THRESH:
            dark_files.append(fpath.name)
        if mean > BRIGHT_THRESH:
            bright_files.append(fpath.name)
        if std < BLANK_STD_THRESH:
            blank_files.append(fpath.name)

    # Aggregate
    b_arr = np.array(brightness_vals, dtype=np.float32) if brightness_vals else np.zeros(1)
    s_arr = np.array(std_vals,        dtype=np.float32) if std_vals        else np.zeros(1)

    return {
        "label":         label,
        "n":             n,
        "shape_errors":  shape_errors,
        "dark":          dark_files,
        "bright":        bright_files,
        "blank":         blank_files,
        "unreadable":    unreadable,
        "mean":          float(b_arr.mean()),
        "std":           float(s_arr.mean()),   # average per-image std
        "dup_count":     dup_count,
    }


def _truncated(items: list[str], limit: int = SAMPLE_LIMIT) -> str:
    if not items:
        return "none"
    shown = items[:limit]
    tail  = f" ... +{len(items)-limit} more" if len(items) > limit else ""
    return ", ".join(shown) + tail


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP2)
    print("  DATASET VALIDATION REPORT")
    print(f"  Phase: real-human vs non-live-input detection")
    print(SEP2)

    real_dir = DATASET_ROOT / "real"
    fake_dir = DATASET_ROOT / "fake"

    # Check dataset root exists
    if not DATASET_ROOT.exists():
        print(f"\nERROR: dataset root not found: {DATASET_ROOT}")
        print("  Run sessions with: --record-ml-data --save-crops --label real/fake")
        print("  Then run:  python scripts/build_cnn_dataset.py")
        sys.exit(1)

    missing = []
    if not real_dir.exists(): missing.append(str(real_dir))
    if not fake_dir.exists(): missing.append(str(fake_dir))
    if missing:
        print(f"\nERROR: missing directories: {missing}")
        sys.exit(1)

    # ── Scan both classes ─────────────────────────────────────────────────────
    print(f"\n  Scanning real/... ", end="", flush=True)
    real = _scan_class(real_dir, "real")
    print(f"{real['n']:,} images")

    print(f"  Scanning fake/... ", end="", flush=True)
    fake = _scan_class(fake_dir, "fake")
    print(f"{fake['n']:,} images")

    total = real["n"] + fake["n"]

    # ── 1. Class Distribution ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("1. CLASS DISTRIBUTION")
    print(SEP)
    if total == 0:
        print("  ERROR: No images found in either class.")
        print("  Record sessions with --save-crops then run build_cnn_dataset.py")
        sys.exit(1)

    real_pct = real["n"] / total * 100
    fake_pct = fake["n"] / total * 100
    print(f"  Real images : {real['n']:,}  ({real_pct:.1f}%)")
    print(f"  Fake images : {fake['n']:,}  ({fake_pct:.1f}%)")
    print(f"  Total       : {total:,}")
    print(f"  Ratio       : {real_pct:.0f} / {fake_pct:.0f}")

    imbalance_warn = real_pct > IMBALANCE_LIMIT * 100 or fake_pct > IMBALANCE_LIMIT * 100

    # ── 2. Shape Validation ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("2. IMAGE SHAPE VALIDATION  (expected 224x224)")
    print(SEP)
    total_shape_err = len(real["shape_errors"]) + len(fake["shape_errors"])
    print(f"  Shape issues (real) : {len(real['shape_errors'])}")
    if real["shape_errors"]:
        print(f"    Samples : {_truncated(real['shape_errors'])}")
    print(f"  Shape issues (fake) : {len(fake['shape_errors'])}")
    if fake["shape_errors"]:
        print(f"    Samples : {_truncated(fake['shape_errors'])}")
    print(f"  Total shape issues  : {total_shape_err}")

    unreadable_total = len(real["unreadable"]) + len(fake["unreadable"])
    if unreadable_total > 0:
        print(f"  Unreadable images   : {unreadable_total}")
        if real["unreadable"]:
            print(f"    Real: {_truncated(real['unreadable'])}")
        if fake["unreadable"]:
            print(f"    Fake: {_truncated(fake['unreadable'])}")

    # ── 3. Pixel Statistics ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("3. PIXEL STATISTICS (grayscale brightness per class)")
    print(SEP)
    print(f"  Real  ->  mean: {real['mean']:.1f}   std: {real['std']:.1f}")
    print(f"  Fake  ->  mean: {fake['mean']:.1f}   std: {fake['std']:.1f}")
    brightness_gap = abs(real["mean"] - fake["mean"])
    print(f"  Brightness gap  : {brightness_gap:.1f} pts  ", end="")
    brightness_warn = brightness_gap > BRIGHT_GAP_WARN
    print("(WARN: dataset bias risk)" if brightness_warn else "(OK)")

    # ── 4. Duplicate Detection ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("4. DUPLICATE DETECTION  (MD5 hash)")
    print(SEP)
    real_dup_pct = real["dup_count"] / real["n"] * 100 if real["n"] > 0 else 0.0
    fake_dup_pct = fake["dup_count"] / fake["n"] * 100 if fake["n"] > 0 else 0.0
    print(f"  Real  ->  {real['dup_count']} duplicates ({real_dup_pct:.1f}%)")
    print(f"  Fake  ->  {fake['dup_count']} duplicates ({fake_dup_pct:.1f}%)")

    # ── 5. Quality Flags ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("5. QUALITY FLAGS")
    print(SEP)

    def _qline(label: str, items: list, threshold_desc: str) -> None:
        n = len(items)
        flag = "WARN" if n > 0 else "OK"
        print(f"  [{flag}] {label:28s}: {n}  ({threshold_desc})")
        if n > 0:
            print(f"          Samples: {_truncated(items)}")

    print("  Real:")
    _qline("  Dark (mean < 30)",     real["dark"],   f"< {DARK_THRESH}")
    _qline("  Bright (mean > 225)",  real["bright"], f"> {BRIGHT_THRESH}")
    _qline("  Near-empty (std < 5)", real["blank"],  f"std < {BLANK_STD_THRESH}")

    print("  Fake:")
    _qline("  Dark (mean < 30)",     fake["dark"],   f"< {DARK_THRESH}")
    _qline("  Bright (mean > 225)",  fake["bright"], f"> {BRIGHT_THRESH}")
    _qline("  Near-empty (std < 5)", fake["blank"],  f"std < {BLANK_STD_THRESH}")

    # ── Warnings summary ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("WARNINGS")
    print(SEP)
    warnings = []

    if imbalance_warn:
        warnings.append(
            f"Class imbalance: real={real_pct:.0f}% fake={fake_pct:.0f}%  "
            f"(>{IMBALANCE_LIMIT*100:.0f}% threshold)"
        )
    if brightness_warn:
        warnings.append(
            f"Brightness bias: gap={brightness_gap:.1f} pts  "
            f"(real={real['mean']:.1f} vs fake={fake['mean']:.1f})"
        )
    if total_shape_err > 0:
        warnings.append(f"{total_shape_err} images are not 224x224 — resize before training")
    if unreadable_total > 0:
        warnings.append(f"{unreadable_total} unreadable images — remove or re-export")

    quality_issues = (
        len(real["dark"]) + len(fake["dark"]) +
        len(real["bright"]) + len(fake["bright"]) +
        len(real["blank"]) + len(fake["blank"])
    )
    if quality_issues > 0:
        warnings.append(
            f"{quality_issues} quality-flagged images "
            f"(dark/bright/blank — review before training)"
        )

    dup_total = real["dup_count"] + fake["dup_count"]
    if dup_total > 0:
        warnings.append(f"{dup_total} duplicate images detected")

    if warnings:
        for w in warnings:
            print(f"  * {w}")
    else:
        print("  None — dataset looks clean.")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    critical = total_shape_err > 0 or unreadable_total > 0 or total == 0
    if critical:
        verdict = "FAIL — fix critical issues before training"
    elif warnings:
        verdict = "PASS with warnings — review above before training"
    else:
        verdict = "PASS — dataset ready for CNN training"
    print(f"  VERDICT: {verdict}")
    print(SEP2)


if __name__ == "__main__":
    main()
