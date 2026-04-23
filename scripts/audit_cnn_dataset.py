"""
scripts/audit_cnn_dataset.py -- CNN Pipeline Audit
===================================================
Streaming, read-only audit of data/cnn_dataset/ and data/sessions/.
Covers: crop quality, frame filtering, label consistency, balance,
        bias detection, duplicate detection.

Usage:
    python scripts/audit_cnn_dataset.py
"""

from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Force UTF-8 output on Windows so Unicode symbols render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_ROOT = Path("data/cnn_dataset")
SESSIONS_DIR = Path("data/sessions")
REAL_DIR     = DATASET_ROOT / "real"
FAKE_DIR     = DATASET_ROOT / "fake"

IMBALANCE_WARN   = 0.70    # either class > 70% → warn
BRIGHT_GAP_WARN  = 20.0    # brightness mean gap > 20 pts
DARK_THRESH      = 30.0
BRIGHT_THRESH    = 225.0
BLANK_STD_THRESH = 5.0
BLUR_THRESH      = 50.0
DUPE_WARN_PCT    = 5.0     # > 5% dupes → warn
CLIPPING_MARGIN  = 8       # pixels — detect face cut off near edges

SEP  = "-" * 64
SEP2 = "=" * 64
SAMPLE = 4                 # filenames printed in lists


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _blur_score(gray: np.ndarray) -> float:
    g = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _face_clipping_score(gray: np.ndarray, margin: int = CLIPPING_MARGIN) -> float:
    """
    Check if significant brightness content is pressed against any edge.
    High value → face likely clipped.
    Score = max mean brightness in a border strip of `margin` pixels.
    """
    h, w = gray.shape
    strips = [
        gray[:margin, :],          # top
        gray[h - margin:, :],      # bottom
        gray[:, :margin],          # left
        gray[:, w - margin:],      # right
    ]
    return float(max(s.mean() for s in strips))


def _scan_images(class_dir: Path, label: str) -> dict:
    files = sorted(class_dir.glob("*.jpg"))
    n     = len(files)

    shape_errors: list[str] = []
    dark_files:   list[str] = []
    bright_files: list[str] = []
    blank_files:  list[str] = []
    blurry_files: list[str] = []
    clipped_files: list[str] = []
    unreadable:   list[str] = []

    brightness_vals: list[float] = []
    std_vals:        list[float] = []
    blur_vals:       list[float] = []
    hashes:          dict[str, str] = {}
    dup_count = 0

    for fpath in files:
        # ── Duplicate check ───────────────────────────────────────────────────
        digest = _md5(fpath)
        if digest in hashes:
            dup_count += 1
        else:
            hashes[digest] = fpath.name

        # ── Load ─────────────────────────────────────────────────────────────
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            unreadable.append(fpath.name)
            continue

        h, w = img.shape[:2]
        if (h, w) != (224, 224):
            shape_errors.append(f"{fpath.name}({w}x{h})")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = float(gray.mean())
        std  = float(gray.std())
        blur = _blur_score(gray)
        clip = _face_clipping_score(gray)

        brightness_vals.append(mean)
        std_vals.append(std)
        blur_vals.append(blur)

        if mean < DARK_THRESH:          dark_files.append(fpath.name)
        if mean > BRIGHT_THRESH:        bright_files.append(fpath.name)
        if std  < BLANK_STD_THRESH:     blank_files.append(fpath.name)
        if blur < BLUR_THRESH:          blurry_files.append(fpath.name)
        if clip > 200:                  clipped_files.append(fpath.name)  # bright strip at edge

    def _agg(vals):
        a = np.array(vals, dtype=np.float32) if vals else np.array([0.0])
        return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

    b_mean, b_std, b_min, b_max = _agg(brightness_vals)
    s_mean, *_                  = _agg(std_vals)
    bl_mean, bl_std, bl_min, bl_max = _agg(blur_vals)

    return {
        "label":         label,
        "n":             n,
        "shape_errors":  shape_errors,
        "dark":          dark_files,
        "bright":        bright_files,
        "blank":         blank_files,
        "blurry":        blurry_files,
        "clipped":       clipped_files,
        "unreadable":    unreadable,
        "b_mean": b_mean, "b_std": b_std, "b_min": b_min, "b_max": b_max,
        "s_mean": s_mean,
        "bl_mean": bl_mean, "bl_std": bl_std, "bl_min": bl_min, "bl_max": bl_max,
        "dup_count": dup_count,
    }


# ── Session-level checks ───────────────────────────────────────────────────────

def _audit_sessions() -> dict:
    """
    Read all sessions, check label consistency vs cnn_dataset output.
    Returns per-session stats dict.
    """
    if not SESSIONS_DIR.exists():
        return {}

    sessions = [s for s in sorted(SESSIONS_DIR.iterdir()) if s.is_dir()]
    with_crops = []
    without_crops = []
    label_map: dict[str, str] = {}   # session_id → label

    for sess in sessions:
        meta_path = sess / "metadata.json"
        crops_dir = sess / "frames"
        if not meta_path.exists():
            continue
        try:
            meta  = json.load(open(meta_path, encoding="utf-8"))
            label = meta.get("label", "unknown").strip().lower()
        except Exception:
            continue
        sid = sess.name.replace("session_", "")
        label_map[sid] = label

        if crops_dir.exists() and list(crops_dir.glob("frame_*.jpg")):
            with_crops.append((sid, label))
        else:
            without_crops.append((sid, label))

    return {
        "total":        len(sessions),
        "with_crops":   with_crops,
        "without_crops": without_crops,
        "label_map":    label_map,
    }


def _check_label_consistency(label_map: dict[str, str]) -> list[str]:
    """
    For every file in cnn_dataset/, verify the parent folder matches
    the session's label in metadata.json.
    Returns list of violation filenames.
    """
    violations: list[str] = []
    for folder_label in ("real", "fake"):
        class_dir = DATASET_ROOT / folder_label
        if not class_dir.exists():
            continue
        for fpath in class_dir.glob("*.jpg"):
            # Filename: session_<sid>_frame_<n>.jpg
            parts = fpath.stem.split("_frame_")
            if len(parts) < 2:
                continue
            sid = parts[0].replace("session_", "")
            meta_label = label_map.get(sid)
            if meta_label and meta_label != folder_label:
                violations.append(
                    f"{fpath.name}  [folder={folder_label}  meta={meta_label}]"
                )
    return violations


def _check_csv_filtering(sessions_info: dict) -> dict:
    """
    For sessions with crops, verify that face_present=0 frames
    are NOT present in the cnn_dataset output.
    """
    false_inclusions = 0
    checked_sessions = 0

    for sid, label in sessions_info.get("with_crops", []):
        sess_dir = SESSIONS_DIR / f"session_{sid}"
        csv_path = sess_dir / "frames.csv"
        if not csv_path.exists():
            continue

        # Collect frame_ids where face_present == 0
        no_face_ids: set[int] = set()
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        if int(float(row.get("face_present", 1))) == 0:
                            no_face_ids.add(int(float(row.get("frame_id", -1))))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            continue

        # Check if any no-face frame made it into cnn_dataset
        class_dir = DATASET_ROOT / label
        if not class_dir.exists():
            continue
        for fpath in class_dir.glob(f"session_{sid}_frame_*.jpg"):
            try:
                fid = int(fpath.stem.split("_frame_")[1])
                if fid in no_face_ids:
                    false_inclusions += 1
            except (IndexError, ValueError):
                pass

        checked_sessions += 1

    return {
        "checked":         checked_sessions,
        "false_inclusions": false_inclusions,
    }


def _truncated(items: list, limit: int = SAMPLE) -> str:
    if not items: return "none"
    tail = f" +{len(items)-limit} more" if len(items) > limit else ""
    return ", ".join(str(x) for x in items[:limit]) + tail


# ── Main audit ─────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP2)
    print("  CNN DATA PIPELINE AUDIT")
    print(f"  Dataset: {DATASET_ROOT}")
    print(SEP2)

    if not DATASET_ROOT.exists():
        print(f"\nERROR: {DATASET_ROOT} not found. Run build_cnn_dataset.py first.")
        sys.exit(1)

    # ── Scan both classes ─────────────────────────────────────────────────────
    print(f"\n  Scanning real/ ... ", end="", flush=True)
    real = _scan_images(REAL_DIR, "real")
    print(f"{real['n']:,} images")

    print(f"  Scanning fake/ ... ", end="", flush=True)
    fake = _scan_images(FAKE_DIR, "fake")
    print(f"{fake['n']:,} images")

    total = real["n"] + fake["n"]
    if total == 0:
        print("\nERROR: no images in dataset. Run build_cnn_dataset.py first.")
        sys.exit(1)

    # ── Session info ──────────────────────────────────────────────────────────
    print(f"  Scanning sessions ...", end="", flush=True)
    sess_info = _audit_sessions()
    print(f" {sess_info.get('total',0)} sessions")

    warnings:   list[str] = []
    failures:   list[str] = []

    # ═════════════════════════════════════════════════════════════════════════
    # CHECK 1 — Class Distribution
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("CHECK 1 — CLASS DISTRIBUTION")
    print(SEP)
    real_pct = real["n"] / total * 100
    fake_pct = fake["n"] / total * 100
    print(f"  Real images  : {real['n']:,}  ({real_pct:.1f}%)")
    print(f"  Fake images  : {fake['n']:,}  ({fake_pct:.1f}%)")
    print(f"  Total        : {total:,}")
    print(f"  Ratio        : {real_pct:.0f} / {fake_pct:.0f}")

    if real_pct > IMBALANCE_WARN*100 or fake_pct > IMBALANCE_WARN*100:
        msg = f"Class imbalance: real={real_pct:.0f}% fake={fake_pct:.0f}% (>{IMBALANCE_WARN*100:.0f}%)"
        warnings.append(msg)
        print(f"  ⚠ WARN: {msg}")
    else:
        print(f"  ✓ Balance: OK")

    # Session breakdown
    n_real_sess = sum(1 for _, l in sess_info.get("with_crops", []) if l == "real")
    n_fake_sess = sum(1 for _, l in sess_info.get("with_crops", []) if l == "fake")
    n_no_crops  = len(sess_info.get("without_crops", []))
    print(f"\n  Sessions with crops    : {len(sess_info.get('with_crops',[]))}  "
          f"(real={n_real_sess}  fake={n_fake_sess})")
    print(f"  Sessions without crops : {n_no_crops}  (recorded before --save-crops)")

    # ═════════════════════════════════════════════════════════════════════════
    # CHECK 2 — Crop Quality (shape, blur, clipping)
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("CHECK 2 — CROP QUALITY")
    print(SEP)

    total_shape = len(real["shape_errors"]) + len(fake["shape_errors"])
    print(f"  Shape (224×224):")
    print(f"    Real errors : {len(real['shape_errors'])}  {_truncated(real['shape_errors'])}")
    print(f"    Fake errors : {len(fake['shape_errors'])}  {_truncated(fake['shape_errors'])}")
    if total_shape > 0:
        failures.append(f"{total_shape} images not 224×224 — re-run build_cnn_dataset.py")

    total_blur = len(real["blurry"]) + len(fake["blurry"])
    print(f"\n  Blur (Laplacian < {BLUR_THRESH}):")
    print(f"    Real : {len(real['blurry'])}   Fake : {len(fake['blurry'])}")
    print(f"    Real blur stats → mean={real['bl_mean']:.0f}  min={real['bl_min']:.0f}  max={real['bl_max']:.0f}")
    print(f"    Fake blur stats → mean={fake['bl_mean']:.0f}  min={fake['bl_min']:.0f}  max={fake['bl_max']:.0f}")
    if total_blur > 0:
        warnings.append(f"{total_blur} blurry images survived filtering")

    total_clipped = len(real["clipped"]) + len(fake["clipped"])
    print(f"\n  Face clipping (bright edge strip):")
    print(f"    Real : {len(real['clipped'])}   Fake : {len(fake['clipped'])}")
    if total_clipped > 0:
        warnings.append(f"{total_clipped} images may have clipped face edges")

    # ═════════════════════════════════════════════════════════════════════════
    # CHECK 3 — Label Consistency
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("CHECK 3 — LABEL CONSISTENCY")
    print(SEP)
    violations = _check_label_consistency(sess_info.get("label_map", {}))
    if violations:
        failures.append(f"{len(violations)} label mismatches (file in wrong folder)")
        print(f"  ✗ FAIL: {len(violations)} mismatch(es):")
        for v in violations[:SAMPLE]: print(f"    {v}")
    else:
        print(f"  ✓ All {total:,} images are in the correct label folder")

    # ═════════════════════════════════════════════════════════════════════════
    # CHECK 4 — Frame Filtering (face_present=0 guard)
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("CHECK 4 — FRAME FILTERING (face_present guard)")
    print(SEP)
    filter_result = _check_csv_filtering(sess_info)
    if filter_result["checked"] == 0:
        print(f"  INFO: no sessions with both crops and CSV found to cross-check")
    elif filter_result["false_inclusions"] > 0:
        n = filter_result["false_inclusions"]
        failures.append(f"{n} face_present=0 frames leaked into dataset")
        print(f"  ✗ FAIL: {n} frames with face_present=0 found in cnn_dataset/")
    else:
        print(f"  ✓ Checked {filter_result['checked']} session(s) — "
              f"no face_present=0 frames leaked")

    # ═════════════════════════════════════════════════════════════════════════
    # CHECK 5 — Bias Detection (brightness, per-session variety)
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("CHECK 5 — BIAS DETECTION")
    print(SEP)

    b_gap = abs(real["b_mean"] - fake["b_mean"])
    print(f"  Brightness (grayscale mean ± std):")
    print(f"    Real : {real['b_mean']:.1f} ± {real['b_std']:.1f}  "
          f"[min={real['b_min']:.0f}  max={real['b_max']:.0f}]")
    print(f"    Fake : {fake['b_mean']:.1f} ± {fake['b_std']:.1f}  "
          f"[min={fake['b_min']:.0f}  max={fake['b_max']:.0f}]")
    print(f"    Gap  : {b_gap:.1f} pts", end="  ")
    if b_gap > BRIGHT_GAP_WARN:
        msg = f"Brightness bias: gap={b_gap:.1f}pts (real={real['b_mean']:.0f} vs fake={fake['b_mean']:.0f})"
        warnings.append(msg)
        print(f"⚠ WARN")
    else:
        print(f"✓ OK")

    # Texture diversity (std of brightness std across images = texture spread)
    print(f"\n  Texture diversity (per-image pixel std):")
    print(f"    Real mean std : {real['s_mean']:.1f}")
    print(f"    Fake mean std : {fake['s_mean']:.1f}")
    s_gap = abs(real["s_mean"] - fake["s_mean"])
    if s_gap > 20:
        warnings.append(f"Texture gap={s_gap:.1f}: real/fake have very different pixel variance")
        print(f"    Gap {s_gap:.1f} ⚠ WARN (model may overfit on texture alone)")
    else:
        print(f"    Gap {s_gap:.1f} ✓ OK")

    # Session diversity — detect if one session dominates a class
    print(f"\n  Session diversity (per-class):")
    for folder_label in ("real", "fake"):
        class_dir = DATASET_ROOT / folder_label
        if not class_dir.exists(): continue
        session_counts: dict[str, int] = {}
        for fpath in class_dir.glob("*.jpg"):
            parts = fpath.stem.split("_frame_")
            sid = parts[0].replace("session_", "")[:8] if len(parts) >= 2 else "?"
            session_counts[sid] = session_counts.get(sid, 0) + 1
        n_class = sum(session_counts.values())
        top_sid, top_n = max(session_counts.items(), key=lambda x: x[1]) if session_counts else ("?", 0)
        top_pct = top_n / n_class * 100 if n_class > 0 else 0
        n_sess  = len(session_counts)
        print(f"    {folder_label:4s}: {n_sess} session(s), "
              f"largest={top_pct:.0f}% (session {top_sid}...)")
        if n_sess == 1:
            warnings.append(f"Only 1 {folder_label} session — no within-class variation")
        elif top_pct > 70:
            warnings.append(f"{folder_label} dominated by one session ({top_pct:.0f}%) — add more variety")

    # ═════════════════════════════════════════════════════════════════════════
    # CHECK 6 — Duplicate Detection
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("CHECK 6 — DUPLICATE DETECTION (MD5)")
    print(SEP)

    real_dup_pct = real["dup_count"] / real["n"] * 100 if real["n"] > 0 else 0
    fake_dup_pct = fake["dup_count"] / fake["n"] * 100 if fake["n"] > 0 else 0
    print(f"  Real : {real['dup_count']} duplicates ({real_dup_pct:.1f}%)")
    print(f"  Fake : {fake['dup_count']} duplicates ({fake_dup_pct:.1f}%)")

    total_dups = real["dup_count"] + fake["dup_count"]
    dup_pct    = total_dups / total * 100 if total > 0 else 0
    if dup_pct > DUPE_WARN_PCT:
        warnings.append(f"High duplicate rate: {total_dups} ({dup_pct:.1f}%) images are exact copies")
    else:
        print(f"  ✓ Duplicate rate {dup_pct:.1f}% — OK")

    # ═════════════════════════════════════════════════════════════════════════
    # CHECK 7 — Quality Flags
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("CHECK 7 — QUALITY FLAGS")
    print(SEP)

    rows = [
        ("Extremely dark (mean<30)",  real["dark"],   fake["dark"]),
        ("Extremely bright (mean>225)",real["bright"], fake["bright"]),
        ("Near-blank (std<5)",        real["blank"],  fake["blank"]),
        ("Unreadable",                real["unreadable"], fake["unreadable"]),
    ]
    any_quality = False
    for name, r_list, f_list in rows:
        n_r, n_f = len(r_list), len(f_list)
        flag = "⚠" if (n_r + n_f) > 0 else "✓"
        print(f"  {flag} {name:<30} real={n_r}  fake={n_f}")
        if (n_r + n_f) > 0:
            any_quality = True
            if r_list: print(f"      real samples: {_truncated(r_list)}")
            if f_list: print(f"      fake samples: {_truncated(f_list)}")

    if any_quality:
        total_q = sum(len(r)+len(f) for _, r, f in rows)
        warnings.append(f"{total_q} quality-flagged images (dark/bright/blank/unreadable)")

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP2}")
    print("  AUDIT SUMMARY")
    print(SEP2)
    print(f"  Real images  : {real['n']:,}")
    print(f"  Fake images  : {fake['n']:,}")
    print(f"  Total        : {total:,}")
    print(f"  Sessions used: {len(sess_info.get('with_crops',[]))}")
    print(f"  Sessions skip: {n_no_crops}  (no crops — recorded before --save-crops)")

    print(f"\n  Warnings ({len(warnings)}):")
    if warnings:
        for w in warnings: print(f"    ⚠ {w}")
    else:
        print("    None")

    print(f"\n  Failures ({len(failures)}):")
    if failures:
        for f_ in failures: print(f"    ✗ {f_}")
    else:
        print("    None")

    print()
    if failures:
        verdict = "FAIL"
        detail  = "Critical issues must be fixed before training."
    elif warnings:
        verdict = "WARNING"
        detail  = "Review warnings above. Training can proceed with caution."
    else:
        verdict = "PASS"
        detail  = "Dataset is clean and training-ready."

    print(f"  VERDICT: {verdict} — {detail}")
    print(SEP2)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
