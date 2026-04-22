"""
scripts/build_sequences.py — Sequence Dataset Generator
════════════════════════════════════════════════════════
Converts frame-level CSV logs into fixed-length temporal sequences
for GRU training.

Feature set (validated 2026-04-22, 15 sessions, 32 505 frames):
  yaw, pitch, roll, motion_score, temporal_score, texture_score, face_present

Dropped signals (data-quality analysis):
  ear          — delta=0.009, marginal separation
  blink_detected — 98.1 % zero across all sessions
  irregularity — globally flat, std=3.82e-05

Usage:
    # Single session
    python scripts/build_sequences.py --input data/sessions/session_<id>/frames.csv

    # All sessions at once
    python scripts/build_sequences.py \\
        --input data/sessions/session_*/frames.csv \\
        --out   data/sequences.npz \\
        --window 24 --stride 8

Output .npz:
    X            float32  (N, 24, 7)  — normalised feature sequences
    y            int8     (N,)        — 0=real, 1=fake
    groups       int32    (N,)        — session index per window (0..S-1)
    session_ids  str      (S,)        — ordered session UUID strings
    features     str      (7,)        — feature name list
    feat_mean    float32  (7,)        — per-feature mean (for inference)
    feat_std     float32  (7,)        — per-feature std  (for inference)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ── Validated feature set ─────────────────────────────────────────────────────
# Order matters — GRU weights are tied to this order at inference time.
# Do NOT add or remove columns without re-training.
FEATURE_COLS = [
    "yaw",
    "pitch",
    "roll",
    "motion_score",
    "temporal_score",
    "texture_score",
    "face_present",
]

LABEL_MAP = {"real": 0, "fake": 1}   # only these labels are accepted


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _f(val: str, default: float = 0.0) -> float:
    """Coerce CSV string to float; return default for empty / non-numeric."""
    try:
        v = val.strip()
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def load_csv(path: Path) -> list[dict]:
    """Read all rows from a frames.csv file."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── Core pipeline ─────────────────────────────────────────────────────────────

def filter_rows(rows: list[dict]) -> list[dict]:
    """Keep only rows whose label is 'real' or 'fake'."""
    return [r for r in rows if r.get("label", "unknown").strip() in LABEL_MAP]


def group_by_session(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by session_id, preserving insertion order.

    Windows never cross session boundaries — each session is windowed
    independently to avoid mixing temporal contexts.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        sid = r.get("session_id", "")
        groups.setdefault(sid, []).append(r)
    return groups


def extract_features(rows: list[dict]) -> list[list[float]]:
    """Sort rows by frame_id and extract the validated feature vector.

    face_present == 0 → all feature values are zeroed out for that frame.
    This prevents stale / carryover signal values from polluting sequences
    during face-absent intervals.

    Returns list of float lists — shape (N, F).
    """
    rows_sorted = sorted(rows, key=lambda r: int(r.get("frame_id", 0)))
    vectors: list[list[float]] = []
    for r in rows_sorted:
        face_ok = int(_f(r.get("face_present", "1")))
        if face_ok == 0:
            # Zero-fill: no reliable signal when face is absent
            vectors.append([0.0] * len(FEATURE_COLS))
        else:
            vectors.append([_f(r.get(col, "")) for col in FEATURE_COLS])
    return vectors


def session_label(rows: list[dict]) -> int:
    """Return the numeric label for a session (first valid label wins)."""
    for r in rows:
        lbl = r.get("label", "").strip()
        if lbl in LABEL_MAP:
            return LABEL_MAP[lbl]
    raise ValueError("Session has no valid label")


def sliding_windows(
    matrix: list[list[float]],
    label: int,
    window: int,
    stride: int,
) -> tuple[list[list[list[float]]], list[int]]:
    """Generate overlapping fixed-length windows within a single session.

    Windows never cross session boundaries (caller responsibility).
    Returns (windows, labels) — each window shape (window, F).
    """
    windows, labels = [], []
    n = len(matrix)
    start = 0
    while start + window <= n:
        windows.append(matrix[start : start + window])
        labels.append(label)
        start += stride
    return windows, labels


def standard_scale(
    X: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Z-score normalisation computed globally across (N * window, F).

    Returns (X_scaled, mean, std).
    Columns with std == 0 are left unchanged (avoids divide-by-zero for
    constant features such as face_present when always=1).
    """
    import numpy as np

    n, w, f = X.shape
    flat = X.reshape(-1, f)           # (N*window, F)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std[std == 0] = 1.0               # protect constant features
    X_scaled = ((X - mean) / std).astype(np.float32)
    return X_scaled, mean.astype(np.float32), std.astype(np.float32)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is required.  pip install numpy", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Build sliding-window GRU sequence dataset from frames.csv"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        type=Path,
        metavar="CSV",
        help="Path(s) to frames.csv files",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=Path("data/sequences.npz"),
        help="Output .npz path (default: data/sequences.npz)",
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=24,
        help="Sequence window size in frames (default: 24 ≈ 0.8 s at 30 fps)",
    )
    parser.add_argument(
        "--stride", "-s",
        type=int,
        default=8,
        help="Stride between windows in frames (default: 8)",
    )
    args = parser.parse_args()

    # ── Load & merge all CSVs ──────────────────────────────────────────────────
    all_rows: list[dict] = []
    for csv_path in args.input:
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — skipping", file=sys.stderr)
            continue
        rows = load_csv(csv_path)
        print(f"  Loaded {len(rows):,} rows from {csv_path}")
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: no rows loaded — nothing to process.", file=sys.stderr)
        sys.exit(1)

    # ── Filter to labeled rows only ────────────────────────────────────────────
    labeled = filter_rows(all_rows)
    skipped = len(all_rows) - len(labeled)
    print(f"\n  {len(labeled):,} labeled rows  ({skipped:,} 'unknown' / warmup skipped)")

    if not labeled:
        print("ERROR: no labeled rows found.  Re-run with --label real or --label fake.",
              file=sys.stderr)
        sys.exit(1)

    # ── Build windows per session (no cross-session contamination) ─────────────
    groups = group_by_session(labeled)
    print(f"  {len(groups)} session(s) found\n")

    all_windows:     list[list[list[float]]] = []
    all_labels:      list[int] = []
    all_group_ids:   list[int] = []   # session index per window
    session_id_list: list[str] = []   # ordered session UUIDs

    for sid, rows in groups.items():
        try:
            lbl = session_label(rows)
        except ValueError as exc:
            print(f"  WARNING: session {sid[:8]}... skipped — {exc}", file=sys.stderr)
            continue

        session_idx = len(session_id_list)
        session_id_list.append(sid)

        matrix = extract_features(rows)         # face_present=0 -> zero-filled
        wins, lbls = sliding_windows(matrix, lbl, args.window, args.stride)

        label_str = "real" if lbl == 0 else "fake"
        print(f"  session {sid[:8]}...  frames={len(matrix):,}  "
              f"windows={len(wins):,}  label={label_str}")

        all_windows.extend(wins)
        all_labels.extend(lbls)
        all_group_ids.extend([session_idx] * len(wins))

    if not all_windows:
        print(
            f"\nERROR: no windows generated.  Sessions may be too short "
            f"(need >= {args.window} labeled frames).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Convert to numpy ───────────────────────────────────────────────────────
    X_raw  = np.array(all_windows,   dtype=np.float32)   # (N, window, F)
    y      = np.array(all_labels,    dtype=np.int8)       # (N,)
    groups = np.array(all_group_ids, dtype=np.int32)      # (N,)

    # ── Standard scaling (global, per-feature) ─────────────────────────────────
    X, feat_mean, feat_std = standard_scale(X_raw)

    # ── Save ───────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        groups=groups,
        session_ids=np.array(session_id_list, dtype=str),
        features=np.array(FEATURE_COLS, dtype=str),
        feat_mean=feat_mean,
        feat_std=feat_std,
    )

    # ── Validation prints ──────────────────────────────────────────────────────
    n_real = int((y == 0).sum())
    n_fake = int((y == 1).sum())
    print(f"\n{'-'*56}")
    print(f"  Total samples  : {len(X):,}")
    print(f"  X shape        : {X.shape}  -> (N, window={args.window}, F={X.shape[2]})")
    print(f"  y shape        : {y.shape}")
    print(f"  Groups shape   : {groups.shape}  (session index per window)")
    print(f"  Sessions       : {len(session_id_list)} unique")
    print(f"  Feature list   : {FEATURE_COLS}")
    print(f"  Class balance  : real={n_real:,}  fake={n_fake:,}  "
          f"({n_real/len(y)*100:.1f}% / {n_fake/len(y)*100:.1f}%)")
    print(f"  feat_mean      : {np.round(feat_mean, 4).tolist()}")
    print(f"  feat_std       : {np.round(feat_std,  4).tolist()}")
    print(f"  Saved to       : {args.out}")
    print(f"{'-'*56}")


if __name__ == "__main__":
    main()
