"""
gaze_core.py
------------
Minimal, standalone extraction of the core logic from Python-Gaze-Face-Tracker.

Extracted:
  - EAR-based blinking ratio (euclidean_distance_3D / blinking_ratio)
  - Head pose estimation (solvePnP + RQDecomp3x3 path)
  - Pitch normalization
  - Iris gaze vector calculation
  - Temporal smoothing (AngleBuffer)

Removed:
  - Camera/VideoCapture code
  - cv2.imshow / UI drawing code
  - UDP socket transmission
  - CSV logging
  - CLI / argparse
  - Top-level script globals and main loop

Input contract:
  All functions accept MediaPipe FaceMesh landmark arrays directly.
  Two forms are used:
    - mesh_points    : np.ndarray shape (478, 2) — pixel-space (x, y)
    - mesh_points_3D : np.ndarray shape (478, 3) — normalized (x, y, z)

Dependencies: numpy, opencv-python (cv2)
"""

import collections
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe FaceMesh with refine_landmarks=True)
# ---------------------------------------------------------------------------

# Iris rings (only present when refine_landmarks=True)
LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]

# Eye-corner indices
LEFT_EYE_OUTER_CORNER  = 33
LEFT_EYE_INNER_CORNER  = 133
RIGHT_EYE_OUTER_CORNER = 362
RIGHT_EYE_INNER_CORNER = 263

# EAR landmark rings  (P0, P3, P4, P5, P8, P11, P12, P13)
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS  = [362, 385, 386, 387, 263, 373, 374, 380]

# Head-pose anchor indices
HEAD_POSE_INDICES = [1, 33, 61, 199, 263, 291]  # nose, eye corners, mouth corners


# ---------------------------------------------------------------------------
# 1. EAR / Blinking
# ---------------------------------------------------------------------------

def _euclidean_distance_3D(points: np.ndarray) -> float:
    """
    Computes the Eye Aspect Ratio (EAR) variant from 8 3-D facial landmarks.

    The metric is:
        EAR = (||P3-P13||³ + ||P4-P12||³ + ||P5-P11||³) / (3 * ||P0-P8||³)

    Args:
        points: np.ndarray of shape (8, 3), ordered as
                [P0, P3, P4, P5, P8, P11, P12, P13]
                in MediaPipe normalized coordinates.

    Returns:
        float: EAR value (lower → more closed).
    """
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator   = (np.linalg.norm(P3 - P13) ** 3
                   + np.linalg.norm(P4 - P12) ** 3
                   + np.linalg.norm(P5 - P11) ** 3)
    denominator = 3.0 * np.linalg.norm(P0 - P8) ** 3
    return numerator / denominator


def compute_blinking_ratio(mesh_points_3D: np.ndarray) -> float:
    """
    Calculates the combined (left + right) eye blinking ratio from
    MediaPipe 3-D normalized landmarks.

    Args:
        mesh_points_3D: np.ndarray, shape (478, 3), MediaPipe normalized coords.

    Returns:
        float: Average blinking ratio. Values <= BLINK_THRESHOLD (~0.51)
               indicate a closed eye.
    """
    right_ear = _euclidean_distance_3D(mesh_points_3D[RIGHT_EYE_POINTS])
    left_ear  = _euclidean_distance_3D(mesh_points_3D[LEFT_EYE_POINTS])
    return (right_ear + left_ear + 1.0) / 2.0


def update_blink_state(
    ear: float,
    frame_counter: int,
    total_blinks: int,
    threshold: float = 0.51,
    consec_frames: int = 2,
) -> tuple[int, int]:
    """
    Finite-state blink counter — call once per frame.

    Args:
        ear:           Current blinking ratio from compute_blinking_ratio().
        frame_counter: Running count of consecutive sub-threshold frames.
        total_blinks:  Cumulative blink count so far.
        threshold:     EAR threshold below which eye is considered closing.
        consec_frames: Minimum consecutive frames required to register a blink.

    Returns:
        (frame_counter, total_blinks): Updated state to pass into the next call.
    """
    if ear <= threshold:
        frame_counter += 1
    else:
        if frame_counter > consec_frames:
            total_blinks += 1
        frame_counter = 0
    return frame_counter, total_blinks


# ---------------------------------------------------------------------------
# 2. Head Pose (solvePnP path — uses raw 3-D MediaPipe points as object pts)
# ---------------------------------------------------------------------------

def _build_camera_matrix(image_size: tuple[int, int]) -> np.ndarray:
    """
    Constructs a simple pinhole camera matrix from image dimensions.

    Args:
        image_size: (height, width) of the frame.

    Returns:
        Camera matrix K, shape (3, 3).
    """
    img_h, img_w = image_size
    focal_length = float(img_w)
    return np.array(
        [[focal_length, 0,            img_h / 2.0],
         [0,            focal_length, img_w / 2.0],
         [0,            0,            1.0         ]],
        dtype=np.float64,
    )


def estimate_head_pose(
    mesh_points_3D: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float, float]:
    """
    Estimates head pitch, yaw, roll from MediaPipe 3-D landmarks using solvePnP.

    Strategy (mirrors original repo):
      - Uses 6 anchor points (nose, eye corners, mouth corners).
      - Treats MediaPipe's (x*w, y*h, z) as 3-D object points.
      - Drops z to form 2-D image points.
      - Solves PnP → rotation matrix → RQDecomp3x3 → Euler angles.

    Args:
        mesh_points_3D: np.ndarray shape (478, 3), MediaPipe normalized coords.
        image_size:     (height, width) of the source frame.

    Returns:
        (pitch, yaw, roll) in degrees.
        pitch: positive = looking up, negative = looking down.
        yaw:   positive = looking right, negative = looking left.
        roll:  in-plane rotation.
    """
    img_h, img_w = image_size

    # Scale normalized landmarks to pixel space (keep z as-is)
    pts_3D = np.multiply(
        mesh_points_3D[HEAD_POSE_INDICES], [img_w, img_h, 1]
    ).astype(np.float64)

    # Drop z dimension for 2-D correspondences
    pts_2D = pts_3D[:, :2].astype(np.float64)

    cam_matrix  = _build_camera_matrix(image_size)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        pts_3D, pts_2D, cam_matrix, dist_matrix
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    # Scale to degrees (RQDecomp3x3 returns values in [-1, 1] range × 360)
    pitch = angles[0] * 360.0
    yaw   = angles[1] * 360.0
    roll  = angles[2] * 360.0

    return pitch, yaw, roll


def classify_head_direction(
    pitch: float,
    yaw: float,
    threshold: float = 10.0,
) -> str:
    """
    Maps (pitch, yaw) angles to a human-readable direction label.

    Args:
        pitch:     Pitch angle in degrees (from estimate_head_pose).
        yaw:       Yaw angle in degrees.
        threshold: Dead-zone in degrees around centre.

    Returns:
        One of: "Left", "Right", "Up", "Down", "Forward".
    """
    if yaw < -threshold:
        return "Left"
    elif yaw > threshold:
        return "Right"
    elif pitch < -threshold:
        return "Down"
    elif pitch > threshold:
        return "Up"
    return "Forward"


def normalize_pitch(pitch: float) -> float:
    """
    Normalizes raw pitch from decomposeProjectionMatrix to [-90, 90].

    Note: Only needed if you use the *decomposeProjectionMatrix* code path
    (not required for the RQDecomp3x3 path above).

    Args:
        pitch: Raw pitch angle in degrees.

    Returns:
        Normalized pitch, positive = looking up.
    """
    if pitch > 180:
        pitch -= 360
    pitch = -pitch
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
    pitch = -pitch
    return pitch


# ---------------------------------------------------------------------------
# 3. Gaze / Iris tracking
# ---------------------------------------------------------------------------

def compute_iris_gaze_vector(
    mesh_points: np.ndarray,
    eye: str = "left",
) -> tuple[int, int, int, int, float]:
    """
    Computes iris centre and its displacement from the eye outer corner.

    Args:
        mesh_points: np.ndarray shape (478, 2), pixel-space landmarks.
        eye:         "left" or "right".

    Returns:
        (cx, cy, dx, dy, radius)
        cx, cy  : iris centre in pixels.
        dx, dy  : displacement of iris from the outer eye corner.
        radius  : radius of the minimum enclosing circle around iris ring.
    """
    if eye == "left":
        iris_idx   = LEFT_IRIS
        corner_idx = LEFT_EYE_OUTER_CORNER
    else:
        iris_idx   = RIGHT_IRIS
        corner_idx = RIGHT_EYE_OUTER_CORNER

    iris_pts = mesh_points[iris_idx]
    (cx, cy), radius = cv2.minEnclosingCircle(iris_pts)
    cx, cy = int(cx), int(cy)

    corner = mesh_points[corner_idx]          # shape (2,)
    dx = cx - int(corner[0])
    dy = cy - int(corner[1])

    return cx, cy, dx, dy, radius


# ---------------------------------------------------------------------------
# 4. Temporal Smoothing (AngleBuffer)
# ---------------------------------------------------------------------------

class AngleBuffer:
    """
    A fixed-size circular buffer that computes a moving-average over the
    last `size` angle triplets (pitch, yaw, roll).

    Usage:
        buf = AngleBuffer(size=10)
        buf.add([pitch, yaw, roll])
        smoothed_pitch, smoothed_yaw, smoothed_roll = buf.get_average()
    """

    def __init__(self, size: int = 10) -> None:
        self.buffer: collections.deque = collections.deque(maxlen=size)

    def add(self, angles: list[float] | np.ndarray) -> None:
        """Append a new [pitch, yaw, roll] sample."""
        self.buffer.append(angles)

    def get_average(self) -> np.ndarray:
        """
        Returns element-wise mean of buffered samples.

        Returns:
            np.ndarray of shape (3,) → [pitch, yaw, roll].
            Returns zeros if buffer is empty.
        """
        if not self.buffer:
            return np.zeros(3)
        return np.mean(self.buffer, axis=0)

    def reset(self) -> None:
        """Clear the buffer (e.g., after a calibration event)."""
        self.buffer.clear()
