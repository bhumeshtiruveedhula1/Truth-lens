"""
scripts/prepare_dataset.py — Dataset Audit, Clean, Balance & Split
===================================================================
Tasks (run in order):
  1. Audit:    count images per source/class
  2. Clean:    remove black, corrupt, wrong-size, near-duplicate frames
  3. Balance:  enforce REAL 35-45% / FAKE 55-65%, OBS cap 20%
  4. Validate: brightness, face-presence sample, resolution check
  5. Split:    identity-aware train/test split (no identity leakage)

Usage:
  python scripts/prepare_dataset.py --src data/deepfake_dataset --out data/deepfake_final
  python scripts/prepare_dataset.py --src data/deepfake_dataset --out data/deepfake_final --audit-only
  python scripts/prepare_dataset.py --src data/deepfake_dataset --out data/deepfake_final --dry-run
"""
from __future__ import annotations
import argparse, hashlib, random, shutil, sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.35)
    _MP_AVAIL = True
except Exception:
    _MP_AVAIL = False

IMG_EXTS   = {".jpg", ".jpeg", ".png"}
TARGET_W   = TARGET_H = 224
REAL_MIN   = 0.35; REAL_MAX = 0.45
OBS_MAX_FRAC = 0.20        # OBS must not exceed 20% of total fake
HASH_BITS  = 8             # perceptual hash grid size
DUP_THRESH = 5             # hamming distance <= this -> duplicate
FACE_SAMPLE = 40           # images to face-check per class
TRAIN_RATIO = 0.80

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTS)


def phash(img_bgr: np.ndarray, bits: int = HASH_BITS) -> int:
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (bits + 1, bits))
    diff  = small[:, 1:] > small[:, :-1]
    return int(sum(1 << i for i, v in enumerate(diff.flatten()) if v))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def has_face(img_bgr: np.ndarray) -> bool:
    if not _MP_AVAIL:
        return True   # skip check if MP unavailable
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = _MP.process(rgb)
    return bool(res and res.detections)


def identity_of(path: Path) -> str:
    """
    Infer identity group from folder name.
    real/person_1_s1 -> 'person_1'
    real/celeb_real  -> 'celeb_real'
    fake/obs_p2_v3   -> 'obs_p2'
    fake/celeb_df    -> 'celeb_df'
    """
    name = path.parent.name
    parts = name.split("_")
    if name.startswith("person_"):
        return f"person_{parts[1]}"
    if name.startswith("obs_"):
        return f"obs_{parts[1]}"
    return name   # celeb_real, celeb_df


def source_of(path: Path) -> str:
    """Return source tag: 'webcam', 'celeb_real', 'celeb_df', 'obs'."""
    name = path.parent.name
    if name.startswith("person_"):  return "webcam"
    if name == "celeb_real":        return "celeb_real"
    if name == "celeb_df":          return "celeb_df"
    if name.startswith("obs_"):     return "obs"
    return "unknown"


# ── Task 1: Audit ─────────────────────────────────────────────────────────────

def audit(src: Path) -> dict:
    real_dir = src / "real"
    fake_dir = src / "fake"
    real_imgs = collect(real_dir)
    fake_imgs = collect(fake_dir)

    real_by_src: dict[str, int] = defaultdict(int)
    fake_by_src: dict[str, int] = defaultdict(int)
    for p in real_imgs: real_by_src[source_of(p)] += 1
    for p in fake_imgs: fake_by_src[source_of(p)] += 1

    total = len(real_imgs) + len(fake_imgs)
    real_pct = 100 * len(real_imgs) / total if total else 0
    fake_pct = 100 * len(fake_imgs) / total if total else 0
    obs_total = fake_by_src.get("obs", 0)
    obs_pct_of_fake = 100 * obs_total / len(fake_imgs) if fake_imgs else 0

    print("\n" + "="*60)
    print("TASK 1 — DATA MERGE AUDIT")
    print("="*60)
    print(f"  Total images   : {total}")
    print(f"  REAL total     : {len(real_imgs)}  ({real_pct:.1f}%)")
    for k, v in sorted(real_by_src.items()):
        print(f"    {k:20s}: {v}")
    print(f"  FAKE total     : {len(fake_imgs)}  ({fake_pct:.1f}%)")
    for k, v in sorted(fake_by_src.items()):
        pct = 100*v/len(fake_imgs) if fake_imgs else 0
        print(f"    {k:20s}: {v}  ({pct:.1f}% of fake)")

    flags = []
    if real_pct < 30:
        flags.append(f"REAL% = {real_pct:.1f}% < 30% — CAPTURE MORE REAL DATA")
    if obs_pct_of_fake > 20:
        flags.append(f"OBS = {obs_pct_of_fake:.1f}% of fake > 20% — will downsample")
    for f in flags:
        print(f"  !!  FLAG: {f}")

    return {
        "real": real_imgs, "fake": fake_imgs,
        "real_pct": real_pct, "obs_pct_of_fake": obs_pct_of_fake,
        "flags": flags,
    }


# ── Task 2: Distribution Correction ──────────────────────────────────────────

def correct_balance(audit_result: dict, dry_run: bool) -> tuple[list[Path], list[Path]]:
    real_imgs = list(audit_result["real"])
    fake_imgs = list(audit_result["fake"])
    obs_pct   = audit_result["obs_pct_of_fake"]

    print("\n" + "="*60)
    print("TASK 2 — DISTRIBUTION CORRECTION")
    print("="*60)

    # Hard gate: don't auto-fix if real < 30%
    if audit_result["real_pct"] < 30:
        print("  FAIL REAL < 30% — cannot auto-correct.")
        print("     ACTION REQUIRED: capture more real webcam data, then re-run.")
        return real_imgs, fake_imgs

    # Downsample OBS if > 20% of fake
    if obs_pct > 20:
        obs_paths  = [p for p in fake_imgs if source_of(p) == "obs"]
        non_obs    = [p for p in fake_imgs if source_of(p) != "obs"]
        target_obs = int(len(non_obs) * (OBS_MAX_FRAC / (1.0 - OBS_MAX_FRAC)))

        # Downsample evenly per session to preserve identity diversity
        by_session: dict[str, list[Path]] = defaultdict(list)
        for p in obs_paths:
            by_session[p.parent.name].append(p)
        per_session = max(1, target_obs // len(by_session))
        kept_obs: list[Path] = []
        for sess_paths in by_session.values():
            random.shuffle(sess_paths)
            kept_obs.extend(sess_paths[:per_session])
        random.shuffle(kept_obs)
        kept_obs = kept_obs[:target_obs]
        fake_imgs = non_obs + kept_obs
        print(f"  OBS downsampled: {len(obs_paths)} -> {len(kept_obs)}")
    else:
        print(f"  OBS {obs_pct:.1f}% — within limit, no change.")

    total = len(real_imgs) + len(fake_imgs)
    print(f"  After balance: real={len(real_imgs)} ({100*len(real_imgs)/total:.1f}%)  "
          f"fake={len(fake_imgs)} ({100*len(fake_imgs)/total:.1f}%)")
    return real_imgs, fake_imgs


# ── Task 3: Data Cleaning ─────────────────────────────────────────────────────

def clean(paths: list[Path], label: str, dry_run: bool) -> list[Path]:
    print(f"\n  Cleaning {label} ({len(paths)} images)...")
    kept:    list[Path] = []
    removed: dict[str, int] = defaultdict(int)
    seen_hashes: dict[int, Path] = {}

    for p in paths:
        img = cv2.imread(str(p))

        # Corrupt / unreadable
        if img is None:
            removed["corrupt"] += 1
            continue

        # Wrong size
        h, w = img.shape[:2]
        if h != TARGET_H or w != TARGET_W:
            removed["wrong_size"] += 1
            continue

        # Black / near-black (mean brightness < 8)
        if img.mean() < 8:
            removed["black"] += 1
            continue

        # Perceptual duplicate detection
        ph = phash(img)
        dup = False
        for seen_hash in seen_hashes:
            if hamming(ph, seen_hash) <= DUP_THRESH:
                removed["duplicate"] += 1
                dup = True
                break
        if dup:
            continue
        seen_hashes[ph] = p

        kept.append(p)

    total_removed = sum(removed.values())
    print(f"  [{label}] kept={len(kept)}  removed={total_removed} "
          f"(corrupt={removed['corrupt']} wrong_size={removed['wrong_size']} "
          f"black={removed['black']} dup={removed['duplicate']})")
    return kept


# ── Task 4: Validation ────────────────────────────────────────────────────────

def validate(real_imgs: list[Path], fake_imgs: list[Path]) -> bool:
    print("\n" + "="*60)
    print("TASK 4 — VALIDATION")
    print("="*60)
    total = len(real_imgs) + len(fake_imgs)
    real_pct = 100 * len(real_imgs) / total if total else 0
    fake_pct = 100 - real_pct
    print(f"  Balance   : real={len(real_imgs)} ({real_pct:.1f}%)  fake={len(fake_imgs)} ({fake_pct:.1f}%)")

    issues = []

    # Balance check
    if not (35 <= real_pct <= 45):
        issues.append(f"Class balance out of range: real={real_pct:.1f}% (target 35-45%)")

    # Brightness distribution
    for label, paths in [("real", real_imgs), ("fake", fake_imgs)]:
        sample = random.sample(paths, min(60, len(paths)))
        means  = []
        for p in sample:
            img = cv2.imread(str(p))
            if img is not None:
                means.append(float(img.mean()))
        avg_bright = sum(means) / len(means) if means else 0
        print(f"  Brightness[{label}]: avg={avg_bright:.1f}  "
              f"{'OK' if avg_bright > 20 else '!! LOW'}")
        if avg_bright < 15:
            issues.append(f"{label} brightness too low ({avg_bright:.1f}) — possible black frames remaining")

    # Resolution spot-check
    for label, paths in [("real", real_imgs), ("fake", fake_imgs)]:
        wrong = 0
        for p in random.sample(paths, min(50, len(paths))):
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                if h != TARGET_H or w != TARGET_W:
                    wrong += 1
        print(f"  Resolution[{label}]: wrong_size_in_sample={wrong}/50  {'OK' if wrong==0 else '!!'}")
        if wrong > 0:
            issues.append(f"{label} has wrong-size images remaining")

    # Face presence sample
    if _MP_AVAIL:
        for label, paths in [("real", real_imgs), ("fake", fake_imgs)]:
            sample = random.sample(paths, min(FACE_SAMPLE, len(paths)))
            no_face = 0
            for p in sample:
                img = cv2.imread(str(p))
                if img is not None and not has_face(img):
                    no_face += 1
            face_pct = 100 * (len(sample) - no_face) / len(sample)
            print(f"  Face presence[{label}]: {face_pct:.1f}%  {'OK' if face_pct >= 80 else '!! LOW'}")
            if face_pct < 80:
                issues.append(f"{label} face presence {face_pct:.1f}% < 80%")
    else:
        print("  Face presence: SKIPPED (MediaPipe not available)")

    if issues:
        print("\n  VALIDATION ISSUES:")
        for i in issues:
            print(f"    !!  {i}")
    else:
        print("\n  Validation: ALL CHECKS PASSED")

    return len(issues) == 0


# ── Task 5: Identity-Aware Split ──────────────────────────────────────────────

def identity_split(
    real_imgs: list[Path],
    fake_imgs: list[Path],
    out_dir:   Path,
    dry_run:   bool,
) -> None:
    print("\n" + "="*60)
    print("TASK 5 — IDENTITY-AWARE TRAIN/TEST SPLIT")
    print("="*60)

    def split_by_identity(paths: list[Path]):
        by_id: dict[str, list[Path]] = defaultdict(list)
        for p in paths:
            by_id[identity_of(p)].append(p)
        ids = sorted(by_id.keys())
        random.shuffle(ids)
        cut = max(1, int(len(ids) * TRAIN_RATIO))
        train_ids = set(ids[:cut])
        test_ids  = set(ids[cut:])
        assert not train_ids & test_ids, "Identity overlap detected!"
        train_paths = [p for p in paths if identity_of(p) in train_ids]
        test_paths  = [p for p in paths if identity_of(p) in test_ids]
        return train_paths, test_paths, train_ids, test_ids

    real_train, real_test, real_train_ids, real_test_ids = split_by_identity(real_imgs)
    fake_train, fake_test, fake_train_ids, fake_test_ids = split_by_identity(fake_imgs)

    print(f"  REAL identities : train={sorted(real_train_ids)}  test={sorted(real_test_ids)}")
    print(f"  FAKE identities : train={sorted(fake_train_ids)}  test={sorted(fake_test_ids)}")
    assert not (real_train_ids & real_test_ids), "REAL identity leak!"
    assert not (fake_train_ids & fake_test_ids), "FAKE identity leak!"
    print("  Identity overlap: NONE OK")

    print(f"\n  Train: real={len(real_train)}  fake={len(fake_train)}  "
          f"total={len(real_train)+len(fake_train)}")
    print(f"  Test : real={len(real_test)}   fake={len(fake_test)}   "
          f"total={len(real_test)+len(fake_test)}")

    if dry_run:
        print("  [DRY-RUN] No files written.")
        return

    # Copy files
    splits = [
        (real_train, out_dir / "train" / "real"),
        (real_test,  out_dir / "test"  / "real"),
        (fake_train, out_dir / "train" / "fake"),
        (fake_test,  out_dir / "test"  / "fake"),
    ]
    for paths, dest in splits:
        dest.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(paths):
            shutil.copy2(str(p), str(dest / f"frame_{i:07d}.jpg"))
        print(f"  Written: {dest.relative_to(out_dir.parent)}  ({len(paths)} files)")


# ── Task 6: Final Report ──────────────────────────────────────────────────────

def final_report(src: Path, out: Path, validation_ok: bool, flags: list[str]) -> None:
    print("\n" + "="*60)
    print("TASK 6 — FINAL REPORT")
    print("="*60)

    total = 0
    for split in ["train", "test"]:
        for cls in ["real", "fake"]:
            d = out / split / cls
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            total += n
            print(f"  {split}/{cls}: {n}")
    print(f"  Total in output : {total}")

    if flags:
        print("\n  FLAGS RAISED:")
        for f in flags:
            print(f"    !!  {f}")

    print("\n" + "="*60)
    if not validation_ok or any("CAPTURE MORE" in f for f in flags):
        print("  FINAL STATUS: NOT_READY")
        for f in flags:
            print(f"    Reason: {f}")
    else:
        print("  FINAL STATUS: READY_FOR_TRAINING")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dataset audit, clean, balance and split")
    p.add_argument("--src",        required=True, help="Source dataset root (contains real/ fake/)")
    p.add_argument("--out",        required=True, help="Output directory for train/test split")
    p.add_argument("--audit-only", action="store_true", help="Only run Task 1 (audit) and exit")
    p.add_argument("--dry-run",    action="store_true", help="Run all tasks but write no files")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    src = Path(args.src)
    out = Path(args.out)

    if not (src / "real").exists() or not (src / "fake").exists():
        print(f"[ERROR] {src} must contain real/ and fake/ subdirectories")
        sys.exit(1)

    # Task 1
    audit_result = audit(src)

    if args.audit_only:
        return

    # Stop immediately if real < 30%
    if audit_result["real_pct"] < 30:
        print("\n[STOP] Real data < 30%. Capture more real data before proceeding.")
        sys.exit(1)

    # Task 2
    real_imgs, fake_imgs = correct_balance(audit_result, args.dry_run)

    # Task 3
    print("\n" + "="*60)
    print("TASK 3 — DATA CLEANING")
    print("="*60)
    real_imgs = clean(real_imgs, "real", args.dry_run)
    fake_imgs = clean(fake_imgs, "fake", args.dry_run)

    # Task 4
    validation_ok = validate(real_imgs, fake_imgs)

    # Task 5
    identity_split(real_imgs, fake_imgs, out, args.dry_run)

    # Task 6
    final_report(src, out, validation_ok, audit_result["flags"])


if __name__ == "__main__":
    main()
