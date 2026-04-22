"""
tools/build_windows.py — Offline Sliding Window Builder
════════════════════════════════════════════════════════
Standalone utility.  NOT imported by the runtime pipeline.

Reads frames.csv produced by agent/ml/logger.py and outputs
fixed-length overlapping sequences ready for CNN + temporal model training.

Usage:
    python tools/build_windows.py \\
        --session data/sessions/session_<id> \\
        --window  24 \\
        --stride  8  \\
        --out     data/windows/session_<id>_w24.npz

Output .npz contains:
    X     : float32 array of shape (N, window, n_features)
    labels: str array  of shape (N,)   — from the "label" column (may be empty)
    meta  : dict with session_id, window_size, stride, feature_names
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# ── Feature columns extracted from frames.csv ─────────────────────────────────
# Only numeric columns are included in the feature matrix.
# Columns with NULL values are zero-filled.
FEATURE_COLUMNS = [
    "face_present",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "ear",
    "blink_detected",
    "yaw", "pitch", "roll",
    "motion_raw", "motion_score",
    "irregularity", "temporal_score",
    "texture_score",
    "is_spoof",
    "trust_score",
]

LABEL_COLUMN = "label"


def _coerce(value: str, default: float = 0.0) -> float:
    """Coerce a CSV string value to float, returning default for empty/NULL."""
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def load_frames(csv_path: Path) -> tuple[list[dict], list[str]]:
    """Load all rows from frames.csv.  Returns (rows, fieldnames)."""
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    return rows, fieldnames


def build_feature_matrix(rows: list[dict]) -> tuple[list[list[float]], list[str]]:
    """Convert rows to a list of feature vectors (one per frame)."""
    matrix: list[list[float]] = []
    for row in rows:
        vec = [_coerce(row.get(col, "")) for col in FEATURE_COLUMNS]
        matrix.append(vec)
    labels = [row.get(LABEL_COLUMN, "") for row in rows]
    return matrix, labels


def sliding_windows(
    matrix: list[list[float]],
    labels: list[str],
    window_size: int,
    stride: int,
) -> tuple[list[list[list[float]]], list[str]]:
    """
    Build overlapping sliding windows.

    Returns:
        windows : list of (window_size, n_features) sequences
        win_labels : one label per window (label of the last frame in the window)
    """
    windows: list[list[list[float]]] = []
    win_labels: list[str] = []
    n = len(matrix)
    start = 0
    while start + window_size <= n:
        end = start + window_size
        windows.append(matrix[start:end])
        win_labels.append(labels[end - 1])   # label from last frame in window
        start += stride
    return windows, win_labels


def save_npz(
    out_path: Path,
    windows: list[list[list[float]]],
    win_labels: list[str],
    meta: dict[str, Any],
) -> None:
    """Save windows to .npz.  Requires numpy."""
    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is required.  pip install numpy", file=sys.stderr)
        sys.exit(1)

    X = np.array(windows, dtype=np.float32)
    labels_arr = np.array(win_labels, dtype=str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        labels=labels_arr,
        meta=json.dumps(meta),
    )
    print(f"Saved {X.shape[0]} windows of shape {X.shape[1:]} → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sliding windows from frames.csv")
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="Path to session directory (containing frames.csv)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Window size in frames (default: 24 ≈ 0.8 sec at 30 fps)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride between windows in frames (default: 8)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz path (default: <session>/windows_w<N>_s<S>.npz)",
    )
    args = parser.parse_args()

    session_dir = args.session
    csv_path = session_dir / "frames.csv"
    meta_path = session_dir / "metadata.json"

    if not csv_path.exists():
        print(f"ERROR: frames.csv not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load session metadata if available
    session_meta: dict = {}
    if meta_path.exists():
        session_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    out_path = args.out or (
        session_dir / f"windows_w{args.window}_s{args.stride}.npz"
    )

    print(f"Loading {csv_path} ...")
    rows, _ = load_frames(csv_path)
    print(f"  {len(rows)} frames loaded")

    matrix, labels = build_feature_matrix(rows)

    windows, win_labels = sliding_windows(
        matrix, labels, args.window, args.stride
    )
    print(f"  {len(windows)} windows (size={args.window}, stride={args.stride})")

    meta = {
        "session_id":    session_meta.get("session_id", "unknown"),
        "window_size":   args.window,
        "stride":        args.stride,
        "feature_names": FEATURE_COLUMNS,
        "n_features":    len(FEATURE_COLUMNS),
        "n_windows":     len(windows),
        "schema_version": 1,
    }

    save_npz(out_path, windows, win_labels, meta)


if __name__ == "__main__":
    main()
