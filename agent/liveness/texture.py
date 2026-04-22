"""
agent/liveness/texture.py
─────────────────────────
Texture-based spoof-detection module for the liveness pipeline.

Input  : BGR frame (numpy ndarray H×W×3)  +  face ROI bounding box.
Output : dict
         {
             "laplacian_score": float,
             "lbp_score":       float,
             "is_spoof":        bool,
         }

Signal semantics
────────────────
laplacian_score
    Scaled Laplacian variance of the face crop.
    Real faces occupy a moderate band.  Very low → blurry / printed spoof.
    Very high → screen glare / unnatural sharpness.

lbp_score
    Local Binary Pattern variance of the face crop.
    Real skin has a characteristic micro-texture variance range.
    Spoofs (printed / replayed) disrupt this distribution.

is_spoof
    True when BOTH texture scores fall outside reasonable real-face bounds.
    Thresholds are constructor parameters — tune per deployment environment.

Optional:
    estimate_head_depth() — nose-to-chin depth ratio proxy (flat-spoof indicator).

No decision engine, no MediaPipe init, no camera, no UI.
Dependencies: numpy, opencv-python (cv2), scikit-image.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ---------------------------------------------------------------------------
# ROI type alias:  (y_min, y_max, x_min, x_max)  — pixel-space integers
# ---------------------------------------------------------------------------

FaceROI = Tuple[int, int, int, int]

# Landmark indices used by estimate_head_depth (MediaPipe 468-point model)
_NOSE_TIP_IDX = 1
_CHIN_IDX     = 152


# ---------------------------------------------------------------------------
# Core signal functions
# ---------------------------------------------------------------------------

def detect_texture_laplacian(
    frame: np.ndarray,
    roi: FaceROI,
    padding: int = 10,
    downscale: bool = True,
) -> float:
    """
    Measure texture sharpness in a face ROI via Laplacian variance.

    Real faces have rich high-frequency texture.  Screen replays / printed
    photos tend to have lower or artificially uniform texture.

    Args:
        frame:     BGR numpy array (H, W, 3).
        roi:       (y_min, y_max, x_min, x_max) pixel bounding box.
        padding:   Extra pixels to add around the ROI before cropping.
        downscale: Halve the ROI before processing (speeds up computation).

    Returns:
        Scaled Laplacian variance (float).
        Returns 0.0 if the crop is empty or invalid.
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
        crop = cv2.resize(
            crop,
            (max(1, crop.shape[1] // 2), max(1, crop.shape[0] // 2)),
        )

    gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(crop, (3, 3), 0)
    gray_blur = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    lap      = cv2.Laplacian(gray_blur, cv2.CV_64F, ksize=3)
    lap_abs  = cv2.convertScaleAbs(lap)
    variance = np.var(lap_abs) / max(1, gray.size)

    return float(variance * 1000)


def detect_texture_lbp(
    frame: np.ndarray,
    roi: FaceROI,
    P: int   = 32,
    R: float = 4.0,
) -> float:
    """
    Measure micro-texture via Local Binary Pattern (LBP) variance.

    Uniform LBP on real skin has a characteristic variance range.
    Printed / screen-replay spoofs disrupt this distribution.

    Args:
        frame: BGR numpy array (H, W, 3).
        roi:   (y_min, y_max, x_min, x_max) pixel bounding box.
        P:     Number of LBP neighbour points (default 32).
        R:     LBP radius in pixels (default 4.0).

    Returns:
        LBP variance (float).
        Returns 0.0 if the crop is empty or invalid.

    Tip:
        For a multi-scale texture profile call at several (P, R) pairs::

            scores = [detect_texture_lbp(frame, roi, P, R)
                      for P, R in [(8, 1), (16, 2), (32, 4)]]
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
# Optional: pseudo-depth signal (flat-spoof indicator)
# ---------------------------------------------------------------------------

def estimate_head_depth(
    landmarks,
    frame_h: int,
    frame_w: int,
) -> dict:
    """
    Estimate a pseudo-depth signal from nose-tip → chin distance normalised
    by the face bounding-box diagonal.

    Lightweight proxy for solvePnP — detects flat (printed / screen) spoofs
    that show abnormally small normalised depth without needing camera intrinsics.

    Args:
        landmarks: MediaPipe NormalizedLandmark list for one face
                   (results.multi_face_landmarks[i].landmark).
        frame_h:   Frame height in pixels.
        frame_w:   Frame width in pixels.

    Returns:
        {
            "raw_depth":        float,  # pixel distance nose → chin
            "face_diagonal":    float,  # pixel diagonal of face bounding box
            "normalized_depth": float,  # raw_depth / face_diagonal
        }
    """
    all_pts = np.array(
        [(lm.x * frame_w, lm.y * frame_h) for lm in landmarks],
        dtype=np.float64,
    )

    nose = all_pts[_NOSE_TIP_IDX]
    chin = all_pts[_CHIN_IDX]
    raw_depth = float(np.linalg.norm(nose - chin))

    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    face_diagonal = float(np.linalg.norm([x_max - x_min, y_max - y_min]))

    normalized = raw_depth / face_diagonal if face_diagonal > 1e-6 else 0.0

    return {
        "raw_depth":        raw_depth,
        "face_diagonal":    face_diagonal,
        "normalized_depth": normalized,
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

class TextureModule:
    """
    Texture-based spoof-detection adapter for the liveness pipeline.

    One instance per session.  Thresholds are injected at construction time
    so they can be tuned per deployment without touching signal logic.

    Usage::

        module = TextureModule(
            laplacian_spoof_below=50.0,   # flag as spoof if score < this
            lbp_spoof_below=30.0,         # flag as spoof if score < this
        )

        # Per-frame:
        signal = module.process(frame_bgr, face_roi)
        # → {"laplacian_score": float, "lbp_score": float, "is_spoof": bool}

    Args:
        laplacian_spoof_below:
            Laplacian score threshold.  ROI scores *below* this value are
            consistent with blur / flat-spoof material.

        lbp_spoof_below:
            LBP variance threshold.  Scores *below* this value indicate
            unnaturally uniform micro-texture (printed / screen spoof).

        laplacian_padding:
            Pixel padding added around the ROI for the Laplacian crop.

        laplacian_downscale:
            Halve the crop before Laplacian processing for speed.

        lbp_P, lbp_R:
            LBP neighbour count and radius.
    """

    def __init__(
        self,
        laplacian_spoof_below: float = 50.0,
        lbp_spoof_below:       float = 30.0,
        laplacian_padding:     int   = 10,
        laplacian_downscale:   bool  = True,
        lbp_P:                 int   = 32,
        lbp_R:                 float = 4.0,
    ) -> None:
        self._lap_threshold = laplacian_spoof_below
        self._lbp_threshold = lbp_spoof_below
        self._lap_padding   = laplacian_padding
        self._lap_downscale = laplacian_downscale
        self._lbp_P         = lbp_P
        self._lbp_R         = lbp_R

    def process(
        self,
        frame_bgr: np.ndarray,
        face_roi: FaceROI,
    ) -> dict:
        """
        Run texture analysis on one frame.

        Args:
            frame_bgr: BGR numpy array (H, W, 3) — the full camera frame.
            face_roi:  (y_min, y_max, x_min, x_max) pixel bounding box of
                       the detected face region.

        Returns:
            {
                "laplacian_score": float,
                "lbp_score":       float,
                "is_spoof":        bool,
            }

            ``is_spoof`` is True when BOTH scores fall below their respective
            thresholds (both signals must agree to avoid false positives).
        """
        lap = detect_texture_laplacian(
            frame_bgr,
            face_roi,
            padding=self._lap_padding,
            downscale=self._lap_downscale,
        )
        lbp = detect_texture_lbp(
            frame_bgr,
            face_roi,
            P=self._lbp_P,
            R=self._lbp_R,
        )

        is_spoof = (lap < self._lap_threshold) and (lbp < self._lbp_threshold)
        score = lap
        print(f"[TEXTURE] score={round(score,2)}")

        return {
            "laplacian_score": lap,
            "lbp_score":       lbp,
            "is_spoof":        is_spoof,
        }
