"""
scripts/capture_identity.py — Identity Enrollment Frame Capture
===============================================================
Standalone utility. Zero dependency on the main pipeline.

Usage:
    python scripts/capture_identity.py
    python scripts/capture_identity.py --camera 1 --max-frames 20 --out data/identity

Keys:
    C      → capture current frame (if quality checks pass)
    SPACE  → same as C (convenience)
    Q / ESC → quit

Quality gates (all must pass before a frame is saved):
    • Face detected by MediaPipe FaceMesh
    • Face bbox area  ≥ MIN_FACE_AREA  (default 60×60 px)
    • Frame brightness ≥ MIN_BRIGHTNESS (default mean pixel > 40)
    • Sharpness (Laplacian variance) ≥ MIN_SHARPNESS (default 80)

Diversity enforcement:
    • Minimum 0.5 s between successive captures
    • On-screen prompts cycle through: look left/right/up/down/blink

Output:
    data/identity/<session_id>/frame_001.jpg …
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

# ── Optional MediaPipe (preferred) ────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[CAPTURE] mediapipe not found — using OpenCV Haar cascade for face detection")

# ── Quality thresholds ─────────────────────────────────────────────────────────
MIN_FACE_AREA   = 60 * 60    # px² — reject tiny/distant faces
MIN_BRIGHTNESS  = 40         # mean pixel value (0–255)
MIN_SHARPNESS   = 80         # Laplacian variance — low = blurry
MIN_GAP_SEC     = 0.5        # seconds between successive captures

# ── Capture target ─────────────────────────────────────────────────────────────
DEFAULT_MAX_FRAMES = 20
DEFAULT_OUT_DIR    = "data/identity"

# ── UI colours (BGR) ──────────────────────────────────────────────────────────
C_GREEN  = (50,  220,  80)
C_RED    = (60,   60, 220)
C_AMBER  = (40,  180, 240)
C_WHITE  = (230, 230, 230)
C_DARK   = ( 14,  14,  20)
C_CYAN   = (200, 210,  60)

# ── Diversity prompts (cycled after each capture) ──────────────────────────────
_PROMPTS = [
    "Look straight at camera",
    "Tilt head slightly LEFT",
    "Tilt head slightly RIGHT",
    "Look slightly UP",
    "Look slightly DOWN",
    "Blink slowly",
    "Neutral expression",
    "Slight smile",
    "Turn slightly LEFT",
    "Turn slightly RIGHT",
]


# ─────────────────────────────────────────────────────────────────────────────
# Face detection backends
# ─────────────────────────────────────────────────────────────────────────────

class _MediaPipeDetector:
    """FaceMesh-based face detector — returns (x1, y1, x2, y2) or None."""

    def __init__(self) -> None:
        self._mp_face = mp.solutions.face_mesh
        self._mesh    = self._mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = frame_bgr.shape[:2]
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res   = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        xs = [p.x * w for p in lm.landmark]
        ys = [p.y * h for p in lm.landmark]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        return x1, y1, x2, y2

    def close(self) -> None:
        self._mesh.close()


class _HaarDetector:
    """OpenCV Haar cascade fallback."""

    def __init__(self) -> None:
        cv_data = cv2.data.haarcascades
        xml     = Path(cv_data) / "haarcascade_frontalface_default.xml"
        self._clf = cv2.CascadeClassifier(str(xml))
        if self._clf.empty():
            raise RuntimeError("Haar cascade XML not found in cv2.data")

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if not len(faces):
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return x, y, x + w, y + h

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Quality checks
# ─────────────────────────────────────────────────────────────────────────────

def _face_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


def _brightness(roi: np.ndarray) -> float:
    return float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())


def _sharpness(roi: np.ndarray) -> float:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _quality_check(
    roi: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> tuple[bool, str]:
    """Return (passed, rejection_reason)."""
    area = _face_area(x1, y1, x2, y2)
    if area < MIN_FACE_AREA:
        return False, f"face too small ({area} px² < {MIN_FACE_AREA})"

    bright = _brightness(roi)
    if bright < MIN_BRIGHTNESS:
        return False, f"too dark (mean={bright:.0f} < {MIN_BRIGHTNESS})"

    sharp = _sharpness(roi)
    if sharp < MIN_SHARPNESS:
        return False, f"blurry (lap_var={sharp:.1f} < {MIN_SHARPNESS})"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text(canvas, text, pos, scale=0.55, color=C_WHITE, bold=False):
    thickness = 2 if bold else 1
    cv2.putText(canvas, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _overlay(canvas, captured: int, max_frames: int,
             face_ok: bool, reject_reason: str,
             prompt: str, last_cap_quality: str) -> None:
    h, w = canvas.shape[:2]

    # ── Top bar ───────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (w, 38), C_DARK, -1)
    title = f"Identity Enrollment  |  Frames: {captured}/{max_frames}"
    _text(canvas, title, (12, 26), scale=0.60, color=C_CYAN, bold=True)

    # ── Face status badge ─────────────────────────────────────────────────────
    if face_ok:
        badge_col, badge_txt = C_GREEN, "FACE OK"
    else:
        badge_col, badge_txt = C_RED, "NO FACE"
    cv2.rectangle(canvas, (w - 110, 6), (w - 4, 32), badge_col, -1)
    _text(canvas, badge_txt, (w - 104, 26), scale=0.50, color=C_DARK, bold=True)

    # ── Progress bar ──────────────────────────────────────────────────────────
    bar_w  = w - 24
    bar_h  = 6
    bar_y  = 38
    cv2.rectangle(canvas, (12, bar_y), (12 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill   = int(bar_w * min(captured / max(max_frames, 1), 1.0))
    fill_c = C_GREEN if captured >= max_frames else C_AMBER
    cv2.rectangle(canvas, (12, bar_y), (12 + fill, bar_y + bar_h), fill_c, -1)

    # ── Bottom instruction bar ────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, h - 72), (w, h), C_DARK, -1)
    cv2.line(canvas, (0, h - 72), (w, h - 72), (50, 50, 60), 1)

    _text(canvas, f"Prompt: {prompt}", (12, h - 52), scale=0.50, color=C_AMBER)
    _text(canvas, "Press C / SPACE to capture   |   Q / ESC to quit",
          (12, h - 30), scale=0.45, color=C_WHITE)

    # Rejection reason or last capture quality
    if reject_reason:
        _text(canvas, f"Rejected: {reject_reason}", (12, h - 10),
              scale=0.42, color=C_RED)
    elif last_cap_quality:
        _text(canvas, f"Saved: {last_cap_quality}", (12, h - 10),
              scale=0.42, color=C_GREEN)


def _draw_face_box(canvas, x1, y1, x2, y2, ok: bool) -> None:
    col  = C_GREEN if ok else C_RED
    cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
    # Corner ticks
    tl = 16
    for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(canvas, (px, py), (px + dx*tl, py), col, 3, cv2.LINE_AA)
        cv2.line(canvas, (px, py), (px, py + dy*tl), col, 3, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main capture loop
# ─────────────────────────────────────────────────────────────────────────────

def run_capture(
    camera_idx: int  = 0,
    max_frames: int  = DEFAULT_MAX_FRAMES,
    out_dir:    str  = DEFAULT_OUT_DIR,
) -> list[str]:
    """
    Run the interactive capture loop.

    Returns:
        list of absolute paths to saved frame files.
    """
    # ── Session output directory ───────────────────────────────────────────────
    session_id  = uuid.uuid4().hex[:8]
    save_dir    = Path(out_dir) / session_id
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CAPTURE] session={session_id}  saving to: {save_dir}")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[CAPTURE] ERROR: cannot open camera {camera_idx}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    print(f"[CAPTURE] camera {camera_idx} opened  "
          f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    # ── Face detector — MediaPipe preferred, Haar cascade fallback ────────────
    detector: _MediaPipeDetector | _HaarDetector
    if _MP_AVAILABLE:
        try:
            detector = _MediaPipeDetector()
            print("[CAPTURE] detector: MediaPipe FaceMesh")
        except Exception as mp_err:
            print(f"[WARNING] MediaPipe failed ({mp_err}), falling back to OpenCV Haar Cascade")
            detector = _HaarDetector()
            print("[CAPTURE] detector: OpenCV Haar Cascade (fallback)")
    else:
        detector = _HaarDetector()
        print("[CAPTURE] detector: OpenCV Haar Cascade (mediapipe not installed)")

    # ── State ──────────────────────────────────────────────────────────────────
    saved_paths:   list[str] = []
    last_cap_time: float     = 0.0
    last_quality:  str       = ""
    reject_reason: str       = ""
    prompt_idx:    int       = 0
    face_bbox:     tuple | None = None

    win_name = "DeepShield — Identity Enrollment"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 600)

    print("[CAPTURE] window open — press C to capture, Q to quit")

    # ── Capture loop ───────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CAPTURE] frame read failed — camera disconnected?")
            break

        frame   = cv2.flip(frame, 1)   # mirror for natural UX
        canvas  = frame.copy()
        h, w    = frame.shape[:2]

        # Face detection
        bbox     = detector.detect(frame)
        face_ok  = False
        roi_bgr  = None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                face_bbox = (x1, y1, x2, y2)
                roi_bgr   = frame[y1:y2, x1:x2].copy()
                ok, _     = _quality_check(roi_bgr, x1, y1, x2, y2)
                face_ok   = ok
                _draw_face_box(canvas, x1, y1, x2, y2, face_ok)

        # ── Overlay ───────────────────────────────────────────────────────────
        prompt = _PROMPTS[prompt_idx % len(_PROMPTS)]
        _overlay(
            canvas,
            captured      = len(saved_paths),
            max_frames    = max_frames,
            face_ok       = face_ok,
            reject_reason = reject_reason,
            prompt        = prompt,
            last_cap_quality = last_quality,
        )

        cv2.imshow(win_name, canvas)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):   # Q or ESC
            print("[CAPTURE] user quit")
            break

        if key in (ord('c'), ord('C'), 32):   # C / SPACE
            now     = time.monotonic()
            elapsed = now - last_cap_time

            # Gate: cooldown
            if elapsed < MIN_GAP_SEC:
                reject_reason = f"too soon ({elapsed:.2f}s < {MIN_GAP_SEC}s cooldown)"
                last_quality  = ""

            # Gate: face required
            elif roi_bgr is None or face_bbox is None:
                reject_reason = "no face detected"
                last_quality  = ""

            else:
                x1, y1, x2, y2 = face_bbox
                passed, reason  = _quality_check(roi_bgr, x1, y1, x2, y2)

                if not passed:
                    reject_reason = reason
                    last_quality  = ""
                    print(f"[CAPTURE] rejected: {reason}")
                else:
                    # ── Save ─────────────────────────────────────────────────
                    idx       = len(saved_paths) + 1
                    fname     = save_dir / f"frame_{idx:03d}.jpg"
                    cv2.imwrite(str(fname), roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_paths.append(str(fname.resolve()))
                    last_cap_time = now
                    prompt_idx   += 1
                    reject_reason = ""
                    area   = _face_area(x1, y1, x2, y2)
                    sharp  = _sharpness(roi_bgr)
                    bright = _brightness(roi_bgr)
                    last_quality = (
                        f"frame_{idx:03d}.jpg  "
                        f"area={area}  sharp={sharp:.0f}  bright={bright:.0f}"
                    )
                    print(f"[CAPTURE] saved {fname.name}  {last_quality}")

        # ── Auto-stop ─────────────────────────────────────────────────────────
        if len(saved_paths) >= max_frames:
            print(f"[CAPTURE] target reached: {len(saved_paths)} frames captured")
            # Brief "done" screen
            done = canvas.copy()
            cv2.rectangle(done, (0, 0), (w, h), (14, 14, 20), -1)
            _text(done, "Enrollment complete!", (w//2 - 130, h//2 - 20),
                  scale=1.0, color=C_GREEN, bold=True)
            _text(done, f"{len(saved_paths)} frames saved to {save_dir}",
                  (w//2 - 200, h//2 + 20), scale=0.55, color=C_WHITE)
            _text(done, "Press any key to exit.", (w//2 - 90, h//2 + 55),
                  scale=0.50, color=C_AMBER)
            cv2.imshow(win_name, done)
            cv2.waitKey(0)
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    return saved_paths


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment into IdentityVerifier (optional — called if --enroll flag set)
# ─────────────────────────────────────────────────────────────────────────────

def enroll_from_files(paths: list[str]) -> None:
    """
    Convenience: load saved frames and enrol into IdentityVerifier immediately.
    Only available when running inside the deepfake project environment.
    """
    try:
        import sys, os
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from agent.ml.identity_verifier import IdentityVerifier
        print("\n[ENROLL] loading IdentityVerifier (ArcFace)...")
        verifier = IdentityVerifier()

        if not verifier._ready:
            print("[ENROLL] model not ready — skipping enrolment")
            return

        frames = [cv2.imread(p) for p in paths if p]
        frames = [f for f in frames if f is not None]

        ok = verifier.enroll_identity(frames)
        if ok:
            print(f"[ENROLL] SUCCESS — reference embedding set from {len(frames)} frames")
            # Persist embedding to the same session directory the frames were captured to.
            if paths:
                session_dir = Path(paths[0]).parent
                saved = verifier.save_identity(session_dir)
                if saved:
                    print(f"[ENROLL] embedding saved → {session_dir / 'embedding.npy'}")
                    print(f"[ENROLL] Runtime will auto-load from: {session_dir}")
                else:
                    print("[ENROLL] WARNING: save_identity failed — embedding not persisted")
        else:
            print("[ENROLL] FAILED — not enough valid face embeddings")
    except ImportError as exc:
        print(f"[ENROLL] skipped (not in project env): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture face frames for identity enrollment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/capture_identity.py
  python scripts/capture_identity.py --camera 1 --max-frames 15
  python scripts/capture_identity.py --out data/identity --enroll
        """,
    )
    p.add_argument("--camera",     type=int, default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                   help=f"Stop after N good frames (default: {DEFAULT_MAX_FRAMES})")
    p.add_argument("--out",        type=str, default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--enroll",     action="store_true",
                   help="Auto-enroll into IdentityVerifier after capture")
    p.add_argument("--sharpness",  type=float, default=MIN_SHARPNESS,
                   help=f"Min Laplacian variance for sharpness (default: {MIN_SHARPNESS})")
    p.add_argument("--brightness", type=float, default=MIN_BRIGHTNESS,
                   help=f"Min mean pixel brightness (default: {MIN_BRIGHTNESS})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Allow CLI overrides of quality thresholds
    global MIN_SHARPNESS, MIN_BRIGHTNESS
    MIN_SHARPNESS  = args.sharpness
    MIN_BRIGHTNESS = args.brightness

    print("=" * 60)
    print("  DeepShield — Identity Enrollment Capture")
    print("=" * 60)
    print(f"  Camera:     {args.camera}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Output:     {args.out}")
    print(f"  Sharpness ≥ {MIN_SHARPNESS}   Brightness ≥ {MIN_BRIGHTNESS}")
    print("=" * 60)
    print("  C / SPACE → capture   Q / ESC → quit")
    print()

    paths = run_capture(
        camera_idx = args.camera,
        max_frames = args.max_frames,
        out_dir    = args.out,
    )

    print()
    print("=" * 60)
    print(f"  Capture complete: {len(paths)} frames saved")
    for p in paths:
        print(f"    {p}")
    print("=" * 60)

    if args.enroll and paths:
        enroll_from_files(paths)

    # Return paths (useful when imported as a module)
    return paths


if __name__ == "__main__":
    main()
