"""
liveness/signals.py
-------------------
Standalone liveness signal functions extracted from Face-Recognition-main.

Input contract:
  - `landmarks`  : list of mediapipe NormalizedLandmark objects
                   (face_mesh.FaceMesh result → results.multi_face_landmarks[i].landmark)
  - `frame`      : numpy ndarray in BGR format (H, W, 3)
  - `frame_shape`: (height, width, channels) tuple  ← from frame.shape

No MediaPipe init, no camera, no display, no CLI.
Drop this file into any pipeline that already has MediaPipe running.

Dependencies (all standard):
  numpy, opencv-python, scikit-image
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ---------------------------------------------------------------------------
# MediaPipe landmark index constants (Face Mesh 468-point model)
# ---------------------------------------------------------------------------

LEFT_EYE_IDX  = [33, 133, 160, 158, 153, 144]   # P0..P5
RIGHT_EYE_IDX = [362, 263, 249, 338, 297, 334]  # P0..P5
MOUTH_IDX     = [61, 291, 81, 78, 13, 14, 87, 317]
NOSE_TIP_IDX  = 1
CHIN_IDX      = 152

# ---------------------------------------------------------------------------
# Default decision thresholds  (tune per deployment environment)
# ---------------------------------------------------------------------------

EAR_THRESHOLD              = 0.20   # below → possible blink
SMILE_THRESHOLD            = 0.55   # above → smile detected
DEPTH_RATIO_THRESHOLD      = 0.30   # head_depth / face_diagonal
LAPLACIAN_TEXTURE_THRESHOLD = 300   # scaled variance
LBP_THRESHOLD              = 110    # LBP variance upper bound for real face


# ---------------------------------------------------------------------------
# Helper: extract pixel-space (x, y) tuples for a list of landmark indices
# ---------------------------------------------------------------------------

def _get_pixel_coords(
    landmarks,
    indices: list[int],
    frame_h: int,
    frame_w: int,
) -> list[tuple[int, int]]:
    """Convert normalized MediaPipe landmarks to pixel coordinates."""
    return [
        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
        for i in indices
    ]


# ---------------------------------------------------------------------------
# 1. EAR — Eye Aspect Ratio  (blink indicator)
# ---------------------------------------------------------------------------

def calculate_ear(eye_points: list[tuple[int, int]]) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    Args:
        eye_points: 6 (x, y) pixel coords in order:
                    [corner_left, top_outer, top_inner,
                     corner_right, bot_inner, bot_outer]
                    → matches LEFT_EYE_IDX / RIGHT_EYE_IDX landmark order.

    Returns:
        EAR scalar.  Low value (< EAR_THRESHOLD) → eye is closing / blink.

    Formula:
        EAR = (‖P1-P5‖ + ‖P2-P4‖) / (2 · ‖P0-P3‖)
    """
    p = [np.array(pt, dtype=float) for pt in eye_points]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def ear_from_landmarks(
    landmarks,
    frame_h: int,
    frame_w: int,
) -> dict[str, float]:
    """
    Convenience wrapper: compute EAR for both eyes from raw MediaPipe landmarks.

    Returns:
        {"left": float, "right": float, "mean": float}
    """
    left_pts  = _get_pixel_coords(landmarks, LEFT_EYE_IDX,  frame_h, frame_w)
    right_pts = _get_pixel_coords(landmarks, RIGHT_EYE_IDX, frame_h, frame_w)
    ear_l = calculate_ear(left_pts)
    ear_r = calculate_ear(right_pts)
    return {"left": ear_l, "right": ear_r, "mean": (ear_l + ear_r) / 2.0}


# ---------------------------------------------------------------------------
# 2. Smile ratio  (expression signal)
# ---------------------------------------------------------------------------

def calculate_smile(mouth_points: list[tuple[int, int]]) -> float:
    """
    Compute mouth openness ratio as a smile indicator.

    Args:
        mouth_points: 8 (x, y) pixel coords matching MOUTH_IDX order.

    Returns:
        Scalar.  High value (> SMILE_THRESHOLD) → smile / open mouth.

    Formula:
        ratio = vertical_distance(P3, P7) / horizontal_distance(P0, P4)
    """
    p = [np.array(pt, dtype=float) for pt in mouth_points]
    vertical   = np.linalg.norm(p[3] - p[7])
    horizontal = np.linalg.norm(p[0] - p[4])
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def smile_from_landmarks(
    landmarks,
    frame_h: int,
    frame_w: int,
) -> float:
    """Convenience wrapper: compute smile ratio from raw MediaPipe landmarks."""
    mouth_pts = _get_pixel_coords(landmarks, MOUTH_IDX, frame_h, frame_w)
    return calculate_smile(mouth_pts)


# ---------------------------------------------------------------------------
# 3. Head depth (pseudo-pose — nose-to-chin distance normalized by face size)
# ---------------------------------------------------------------------------

def estimate_head_depth(
    landmarks,
    frame_h: int,
    frame_w: int,
) -> dict[str, float]:
    """
    Estimate a pseudo-depth signal using the nose-tip → chin Euclidean distance
    normalized by the face bounding-box diagonal.

    This is a lightweight proxy for solvePnP — it detects screen-printed / flat
    spoof images (very small normalized depth) without needing camera intrinsics.

    Args:
        landmarks: raw MediaPipe NormalizedLandmark list.
        frame_h, frame_w: frame dimensions.

    Returns:
        {
            "raw_depth"       : pixel distance nose → chin,
            "face_diagonal"   : pixel diagonal of face bounding box,
            "normalized_depth": raw_depth / face_diagonal,
        }
    """
    all_pts = np.array([
        (lm.x * frame_w, lm.y * frame_h) for lm in landmarks
    ])

    nose = all_pts[NOSE_TIP_IDX]
    chin = all_pts[CHIN_IDX]
    raw_depth = float(np.linalg.norm(nose - chin))

    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    face_diagonal = float(np.linalg.norm([x_max - x_min, y_max - y_min]))

    normalized = raw_depth / face_diagonal if face_diagonal > 1e-6 else 0.0

    return {
        "raw_depth"       : raw_depth,
        "face_diagonal"   : face_diagonal,
        "normalized_depth": normalized,
    }


# ---------------------------------------------------------------------------
# 4a. Texture — Laplacian variance
# ---------------------------------------------------------------------------

def detect_texture_laplacian(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],  # (y_min, y_max, x_min, x_max)
    padding: int = 10,
    downscale: bool = True,
) -> float:
    """
    Measure texture sharpness in a face ROI via Laplacian variance.

    Real faces have rich high-frequency texture; screen replays / printed photos
    tend to have lower or artificially uniform texture.

    Args:
        frame   : BGR numpy array (H, W, 3).
        roi     : (y_min, y_max, x_min, x_max) pixel bounding box.
        padding : extra pixels to add around the ROI.
        downscale: halve the ROI before processing (speeds up computation).

    Returns:
        Scaled Laplacian variance (float).
        High value (> LAPLACIAN_TEXTURE_THRESHOLD) can indicate screen glare /
        unnatural sharpness; low value → blurry / printed.
        Use in combination with other signals — not a standalone check.
    """
    y_min, y_max, x_min, x_max = roi
    h, w = frame.shape[:2]
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return 0.0

    if downscale:
        crop = cv2.resize(crop, (max(1, crop.shape[1] // 2),
                                 max(1, crop.shape[0] // 2)))

    gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(crop, (3, 3), 0)
    gray_blur = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    lap      = cv2.Laplacian(gray_blur, cv2.CV_64F, ksize=3)
    lap_abs  = cv2.convertScaleAbs(lap)
    variance = np.var(lap_abs) / max(1, gray.size)
    return float(variance * 1000)


# ---------------------------------------------------------------------------
# 4b. Texture — Local Binary Pattern (LBP) variance
# ---------------------------------------------------------------------------

def detect_texture_lbp(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],  # (y_min, y_max, x_min, x_max)
    P: int = 32,
    R: float = 4.0,
) -> float:
    """
    Measure micro-texture via Local Binary Pattern variance.

    Uniform LBP on real skin has a characteristic variance range.
    Printed / screen replays disrupt this pattern.

    Args:
        frame: BGR numpy array (H, W, 3).
        roi  : (y_min, y_max, x_min, x_max) pixel bounding box.
        P    : number of LBP neighbor points (default: 32).
        R    : LBP radius in pixels (default: 4.0).

    Returns:
        LBP variance (float).
        Lower values (< LBP_THRESHOLD) are more consistent with real faces
        (at default P=32, R=4 settings).

    Tip:
        Call multiple times with (P=8,R=1), (P=16,R=2), (P=32,R=4) for a
        multi-scale texture profile:
            scores = [detect_texture_lbp(frame, roi, P, R)
                      for P, R in [(8,1),(16,2),(32,4)]]
    """
    y_min, y_max, x_min, x_max = roi
    h, w = frame.shape[:2]
    y_min = max(0, y_min)
    y_max = min(h, y_max)
    x_min = max(0, x_min)
    x_max = min(w, x_max)

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, P=P, R=R, method="uniform")
    return float(np.var(lbp))


# ---------------------------------------------------------------------------
# 5. Composite liveness score  (ties signals into a single verdict)
# ---------------------------------------------------------------------------

def compute_liveness_score(
    landmarks,
    frame: np.ndarray,
    *,
    ear_threshold: float              = EAR_THRESHOLD,
    smile_threshold: float            = SMILE_THRESHOLD,
    depth_ratio_threshold: float      = DEPTH_RATIO_THRESHOLD,
    laplacian_threshold: float        = LAPLACIAN_TEXTURE_THRESHOLD,
    lbp_threshold: float              = LBP_THRESHOLD,
    min_raw_depth: float              = 70.0,
    passing_score: int                = 3,
) -> dict:
    """
    Combine all liveness signals into a score and a boolean verdict.

    Args:
        landmarks : MediaPipe NormalizedLandmark list for ONE face.
        frame     : BGR numpy array (H, W, 3).
        *         : keyword-only threshold overrides.

    Returns:
        {
            "score"         : int   (0–5),
            "is_real"       : bool,
            "ear"           : dict  (left, right, mean),
            "smile"         : float,
            "depth"         : dict  (raw_depth, face_diagonal, normalized_depth),
            "laplacian"     : float,
            "lbp"           : float,
            "signals"       : dict[str, bool]  — per-signal pass/fail,
        }
    """
    frame_h, frame_w = frame.shape[:2]

    # --- compute signals ---
    ear    = ear_from_landmarks(landmarks, frame_h, frame_w)
    smile  = smile_from_landmarks(landmarks, frame_h, frame_w)
    depth  = estimate_head_depth(landmarks, frame_h, frame_w)

    # face ROI from all landmark bounding box
    all_pts = [
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in landmarks
    ]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    roi = (min(ys), max(ys), min(xs), max(xs))

    laplacian = detect_texture_laplacian(frame, roi)
    lbp       = detect_texture_lbp(frame, roi)

    # --- score each signal ---
    sig_blink   = ear["left"] < ear_threshold or ear["right"] < ear_threshold
    sig_smile   = smile > smile_threshold
    sig_depth   = (
        depth["normalized_depth"] > depth_ratio_threshold
        and depth["raw_depth"] > min_raw_depth
    )
    sig_lap     = laplacian < laplacian_threshold   # real face → moderate texture
    sig_lbp     = lbp < lbp_threshold

    score = sum([sig_blink, sig_smile, sig_depth, sig_lap, sig_lbp])

    is_real = (
        score >= passing_score
        and depth["raw_depth"] > min_raw_depth
        and laplacian < laplacian_threshold
        and lbp < lbp_threshold
    )

    return {
        "score"   : score,
        "is_real" : is_real,
        "ear"     : ear,
        "smile"   : smile,
        "depth"   : depth,
        "laplacian": laplacian,
        "lbp"     : lbp,
        "signals" : {
            "blink"    : sig_blink,
            "smile"    : sig_smile,
            "depth"    : sig_depth,
            "laplacian": sig_lap,
            "lbp"      : sig_lbp,
        },
    }
