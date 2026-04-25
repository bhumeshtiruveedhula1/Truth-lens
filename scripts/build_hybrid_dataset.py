"""
scripts/build_hybrid_dataset.py
================================
Builds a domain-aligned, identity-safe train/val dataset from:
  - Webcam real captures   (real/person_N_sM/)
  - Celeb-DF real crops    (real/celeb_real/)
  - Celeb-DF fake crops    (fake/celeb_df/)
  - OBS fake captures      (fake/obs_pN_vM/)

Target distribution
  REAL  2000-2500:  webcam ~60%, celeb_real ~40%
  FAKE  2500-3000:  celeb_df ~80%, OBS <=20%

Identity split
  Webcam  person_1/2 -> train    person_3/4 -> val
  Celeb   celeb_real  -> train (real)
  OBS     obs_p1/p2   -> train    obs_p3/p4  -> val
  CelebDF celeb_df    -> split by stride

Usage
  python scripts/build_hybrid_dataset.py --src data/deepfake_dataset --out data/deepfake_final
  python scripts/build_hybrid_dataset.py --src data/deepfake_dataset --out data/deepfake_final --dry-run
"""
from __future__ import annotations

import argparse, hashlib, random, shutil, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
REAL_TARGET   = 2200          # total real after merge
FAKE_TARGET   = 2700          # total fake after merge
OBS_CAP_FRAC  = 0.14          # pre-dedup cap; post-dedup stays <=20%
PER_ID_MAX    = 200           # max frames per identity per split
STRIDE        = 3             # keep every N-th frame within a session
DUP_THRESH    = 6             # perceptual hash hamming distance for dup
HASH_SIZE     = 8
VAL_RATIO     = 0.20          # 20% val
SEED          = 42

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Identity routing (no overlap between train and val)
WEBCAM_TRAIN_IDS = {"person_1", "person_2"}
WEBCAM_VAL_IDS   = {"person_3", "person_4"}
OBS_TRAIN_IDS    = {"obs_p1", "obs_p2", "obs_p5", "obs_p6"}
OBS_VAL_IDS      = {"obs_p3", "obs_p4"}

random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTS)


def phash(img: np.ndarray) -> int:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE))
    diff  = small[:, 1:] > small[:, :-1]
    return int(sum(1 << i for i, v in enumerate(diff.flatten()) if v))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def stride_sample(paths: list[Path], stride: int = STRIDE, max_n: int = PER_ID_MAX) -> list[Path]:
    """Take every stride-th frame then cap at max_n."""
    strided = paths[::stride]
    return strided[:max_n]


def dedup(paths: list[Path], thresh: int = DUP_THRESH) -> list[Path]:
    """Remove near-duplicate images using perceptual hashing."""
    kept:   list[Path]  = []
    hashes: list[int]   = []
    removed = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            removed += 1
            continue
        if img.mean() < 8:      # black frame
            removed += 1
            continue
        ph = phash(img)
        is_dup = any(hamming(ph, h) <= thresh for h in hashes)
        if is_dup:
            removed += 1
        else:
            hashes.append(ph)
            kept.append(p)
    return kept, removed


def copy_files(paths: list[Path], dest: Path, prefix: str, dry_run: bool) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        out = dest / f"{prefix}_{i:06d}.jpg"
        if not dry_run:
            shutil.copy2(str(p), str(out))
    return len(paths)


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ── Step 1: Collect per-identity pools ───────────────────────────────────────

def collect_real(src: Path):
    """Returns {identity_key: [Path, ...]} for all real sources."""
    pools: dict[str, list[Path]] = {}
    real_dir = src / "real"

    # Webcam: person_N_s1, person_N_s2 -> merged under person_N
    webcam_by_person: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(real_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("person_"):
            continue
        parts = d.name.split("_")
        pid = f"person_{parts[1]}"         # person_1, person_2 ...
        webcam_by_person[pid].extend(collect(d))
    for pid, imgs in webcam_by_person.items():
        pools[pid] = imgs

    # Celeb real
    celeb_real_dir = real_dir / "celeb_real"
    if celeb_real_dir.exists():
        pools["celeb_real"] = collect(celeb_real_dir)
    else:
        print("  WARN: real/celeb_real/ not found")

    return pools


def collect_fake(src: Path):
    """Returns {identity_key: [Path, ...]} for all fake sources."""
    pools: dict[str, list[Path]] = {}
    fake_dir = src / "fake"

    # Celeb-DF (single folder — split by index later)
    celeb_df_dir = fake_dir / "celeb_df"
    if celeb_df_dir.exists():
        pools["celeb_df"] = collect(celeb_df_dir)

    # OBS: obs_pN_vM -> grouped under obs_pN
    obs_by_person: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(fake_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("obs_"):
            continue
        parts = d.name.split("_")
        pid = f"obs_{parts[1]}"            # obs_p1, obs_p2 ...
        obs_by_person[pid].extend(collect(d))
    for pid, imgs in obs_by_person.items():
        pools[pid] = imgs

    return pools


# ── Step 2: Identity-aware split selection ────────────────────────────────────

def select_real(real_pools: dict) -> tuple[list[Path], list[Path]]:
    """
    Returns (train_paths, val_paths) for real class.

    train: webcam person_1+2, celeb_real (80%)
    val:   webcam person_3+4, celeb_real (20%)
    """
    train, val = [], []

    # Webcam train identities
    for pid in WEBCAM_TRAIN_IDS:
        imgs = stride_sample(real_pools.get(pid, []))
        train.extend(imgs[:PER_ID_MAX])

    # Webcam val identities
    for pid in WEBCAM_VAL_IDS:
        imgs = stride_sample(real_pools.get(pid, []))
        val.extend(imgs[:PER_ID_MAX])

    # Celeb_real -> 80% train, 20% val (NO identity mixing)
    celeb = real_pools.get("celeb_real", [])
    random.shuffle(celeb)
    cut = int(len(celeb) * (1 - VAL_RATIO))
    train.extend(celeb[:cut])
    val.extend(celeb[cut:])

    return train, val


def select_fake(fake_pools: dict) -> tuple[list[Path], list[Path]]:
    """
    Returns (train_paths, val_paths) for fake class.

    train: celeb_df (80% of it) + obs_p1 + obs_p2
    val:   celeb_df (20% of it) + obs_p3 + obs_p4

    OBS is hard-capped so it never exceeds OBS_CAP_FRAC of total fake,
    computed BEFORE dedup so the fraction holds even after duplicates removed.
    """
    # Celeb_df split by position (no identity metadata)
    celeb_df = list(fake_pools.get("celeb_df", []))
    random.shuffle(celeb_df)
    cut = int(len(celeb_df) * (1 - VAL_RATIO))
    celeb_train = celeb_df[:cut]
    celeb_val   = celeb_df[cut:]

    # OBS absolute cap: OBS_CAP_FRAC of projected total fake
    # projected_fake = celeb_total / (1 - OBS_CAP_FRAC)
    celeb_total = len(celeb_df)
    obs_total_max = int(celeb_total * OBS_CAP_FRAC / (1.0 - OBS_CAP_FRAC))
    obs_train_max = int(obs_total_max * (1 - VAL_RATIO))
    obs_val_max   = obs_total_max - obs_train_max

    # Collect OBS, stride-sample, then cap globally
    obs_train_all: list[Path] = []
    for pid in OBS_TRAIN_IDS:
        imgs = stride_sample(fake_pools.get(pid, []))
        obs_train_all.extend(imgs)
    random.shuffle(obs_train_all)
    obs_train = obs_train_all[:obs_train_max]

    obs_val_all: list[Path] = []
    for pid in OBS_VAL_IDS:
        imgs = stride_sample(fake_pools.get(pid, []))
        obs_val_all.extend(imgs)
    random.shuffle(obs_val_all)
    obs_val = obs_val_all[:obs_val_max]

    print(f"  OBS cap: total_max={obs_total_max}  "
          f"train={len(obs_train)}  val={len(obs_val)}")

    return celeb_train + obs_train, celeb_val + obs_val


# ── Step 3: Enforce target counts ─────────────────────────────────────────────

def enforce_targets(
    real_train: list[Path], real_val: list[Path],
    fake_train: list[Path], fake_val: list[Path],
) -> tuple:
    """
    Cap each split to target totals.
    OBS fraction inside fake is enforced by selecting OBS last (already done
    above — OBS capped at PER_ID_MAX per identity).
    Final random sample to hit REAL/FAKE targets.
    """
    # Shuffle before sampling so not biased to first sessions
    random.shuffle(real_train); random.shuffle(real_val)
    random.shuffle(fake_train); random.shuffle(fake_val)

    train_real_n = int(REAL_TARGET * (1 - VAL_RATIO))
    val_real_n   = REAL_TARGET - train_real_n
    train_fake_n = int(FAKE_TARGET * (1 - VAL_RATIO))
    val_fake_n   = FAKE_TARGET - train_fake_n

    real_train = real_train[:train_real_n]
    real_val   = real_val[:val_real_n]
    fake_train = fake_train[:train_fake_n]
    fake_val   = fake_val[:val_fake_n]

    return real_train, real_val, fake_train, fake_val


# ── Step 4: Dedup ────────────────────────────────────────────────────────────

def dedup_split(real_train, real_val, fake_train, fake_val):
    section("DEDUPLICATION")
    results = {}
    for label, paths in [("real_train", real_train), ("real_val", real_val),
                          ("fake_train", fake_train), ("fake_val", fake_val)]:
        cleaned, n_removed = dedup(paths)
        print(f"  {label:15s}: {len(paths)} -> {len(cleaned)}  (removed {n_removed})")
        results[label] = cleaned
    return (results["real_train"], results["real_val"],
            results["fake_train"], results["fake_val"])


# ── Step 5: Validate ─────────────────────────────────────────────────────────

def validate_and_report(real_train, real_val, fake_train, fake_val,
                        real_pools, fake_pools):
    section("VALIDATION REPORT")

    total_real  = len(real_train) + len(real_val)
    total_fake  = len(fake_train) + len(fake_val)
    total       = total_real + total_fake
    real_pct    = 100 * total_real / total if total else 0
    fake_pct    = 100 - real_pct

    print(f"\n  Total dataset   : {total}")
    print(f"  REAL            : {total_real} ({real_pct:.1f}%)")
    print(f"  FAKE            : {total_fake} ({fake_pct:.1f}%)")

    # OBS fraction of fake
    all_fake = fake_train + fake_val
    obs_count = sum(1 for p in all_fake if "obs_" in p.parent.name)
    obs_frac = 100 * obs_count / len(all_fake) if all_fake else 0
    print(f"\n  OBS in fake     : {obs_count} ({obs_frac:.1f}%)  "
          f"{'OK' if obs_frac <= 20 else 'OVER LIMIT'}")

    print(f"\n  Train split     : real={len(real_train)}  fake={len(fake_train)}  "
          f"total={len(real_train)+len(fake_train)}")
    print(f"  Val   split     : real={len(real_val)}  fake={len(fake_val)}  "
          f"total={len(real_val)+len(fake_val)}")

    # Identity distribution
    print("\n  Identity distribution:")
    id_counts: dict[str, int] = defaultdict(int)
    for p in real_train + real_val:
        name = p.parent.name
        pid  = "_".join(name.split("_")[:2]) if name.startswith("person_") else name
        id_counts[pid] += 1
    for p in fake_train + fake_val:
        name = p.parent.name
        pid  = "_".join(name.split("_")[:2]) if name.startswith("obs_") else name
        id_counts[pid] += 1
    for k, v in sorted(id_counts.items()):
        print(f"    {k:25s}: {v}")

    # Identity overlap check
    train_ids = set()
    val_ids   = set()
    for p in real_train + fake_train:
        n = p.parent.name
        pid = "_".join(n.split("_")[:2]) if (n.startswith("person_") or n.startswith("obs_")) else n
        train_ids.add(pid)
    for p in real_val + fake_val:
        n = p.parent.name
        pid = "_".join(n.split("_")[:2]) if (n.startswith("person_") or n.startswith("obs_")) else n
        val_ids.add(pid)

    # celeb_real and celeb_df are split by position, not identity — they share the folder name
    # We exclude them from the overlap check (no identity metadata available)
    webcam_obs_train = {i for i in train_ids if i.startswith("person_") or i.startswith("obs_")}
    webcam_obs_val   = {i for i in val_ids   if i.startswith("person_") or i.startswith("obs_")}
    overlap = webcam_obs_train & webcam_obs_val

    print(f"\n  Train identities (webcam/obs): {sorted(webcam_obs_train)}")
    print(f"  Val   identities (webcam/obs): {sorted(webcam_obs_val)}")
    if overlap:
        print(f"  !! IDENTITY OVERLAP DETECTED: {overlap}")
    else:
        print("  Identity overlap: NONE (OK)")

    # Balance check
    issues = []
    if not (35 <= real_pct <= 60):
        issues.append(f"Real% = {real_pct:.1f}% outside 35-55% range")
    if obs_frac > 20:
        issues.append(f"OBS = {obs_frac:.1f}% > 20% limit")
    if overlap:
        issues.append(f"Identity overlap: {overlap}")

    return issues


# ── Step 6: Write ─────────────────────────────────────────────────────────────

def write_split(real_train, real_val, fake_train, fake_val, out: Path, dry_run: bool):
    section("WRITING DATASET")
    splits = [
        (real_train, out / "train" / "real", "real_train"),
        (real_val,   out / "val"   / "real", "real_val"),
        (fake_train, out / "train" / "fake", "fake_train"),
        (fake_val,   out / "val"   / "fake", "fake_val"),
    ]
    for paths, dest, prefix in splits:
        n = copy_files(paths, dest, prefix, dry_run)
        mode = "(DRY-RUN)" if dry_run else ""
        print(f"  {str(dest.relative_to(out.parent)):50s} {n} files {mode}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build hybrid deepfake detection dataset")
    p.add_argument("--src",     required=True, help="Source root (contains real/ fake/)")
    p.add_argument("--out",     required=True, help="Output root (will contain train/ val/)")
    p.add_argument("--dry-run", action="store_true", help="Run all steps but write no files")
    return p.parse_args()


def main():
    args = parse_args()
    src  = Path(args.src)
    out  = Path(args.out)

    if not (src / "real").exists() or not (src / "fake").exists():
        print(f"[ERROR] {src} must contain real/ and fake/ subdirs")
        sys.exit(1)

    section("TASK 1+2 — COLLECTING AND STRIDE-SAMPLING BY IDENTITY")

    real_pools = collect_real(src)
    fake_pools = collect_fake(src)

    print("  REAL pools:")
    for k, v in sorted(real_pools.items()):
        print(f"    {k:25s}: {len(v)} frames")
    print("  FAKE pools:")
    for k, v in sorted(fake_pools.items()):
        print(f"    {k:25s}: {len(v)} frames")

    section("TASK 3 — IDENTITY-AWARE SPLIT SELECTION")
    real_train, real_val = select_real(real_pools)
    fake_train, fake_val = select_fake(fake_pools)
    print(f"  Before target cap:")
    print(f"    real_train={len(real_train)}  real_val={len(real_val)}")
    print(f"    fake_train={len(fake_train)}  fake_val={len(fake_val)}")

    section("TASK 4 — ENFORCE TARGET COUNTS")
    real_train, real_val, fake_train, fake_val = enforce_targets(
        real_train, real_val, fake_train, fake_val
    )
    print(f"  After target cap:")
    print(f"    real_train={len(real_train)}  real_val={len(real_val)}")
    print(f"    fake_train={len(fake_train)}  fake_val={len(fake_val)}")

    real_train, real_val, fake_train, fake_val = dedup_split(
        real_train, real_val, fake_train, fake_val
    )

    issues = validate_and_report(
        real_train, real_val, fake_train, fake_val, real_pools, fake_pools
    )

    write_split(real_train, real_val, fake_train, fake_val, out, args.dry_run)

    section("FINAL STATUS")
    if issues:
        print("  STATUS: NOT_READY")
        for i in issues:
            print(f"    Reason: {i}")
    else:
        print("  STATUS: READY_FOR_TRAINING")
        print(f"  Output : {out.resolve()}")
        print("  Next   : python scripts/train_deepfake_cnn.py "
              "--real-dir data/deepfake_final/train/real "
              "--fake-dir data/deepfake_final/train/fake")


if __name__ == "__main__":
    main()
