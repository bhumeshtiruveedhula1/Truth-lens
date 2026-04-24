"""
agent/ml/identity_verifier.py — ArcFace Identity Verification Module
=====================================================================
Model:    InsightFace ArcFace (buffalo_sc — lightweight, 224×224)
Input:    face_roi_bgr  (OpenCV BGR numpy array, any resolution)
Output:   identity_score (0–1 cosine similarity)
          identity_match (True if similarity >= MATCH_THRESHOLD)

Pipeline:
  face_roi_bgr
    → insightface.app.FaceAnalysis (det + recog)
    → 512-d L2-normalized ArcFace embedding
    → cosine_similarity(embedding, reference_embedding)

Enrolment:
  enroll_identity(frames_list)
    → extract embedding from each frame
    → average + L2-normalize → self.reference_embedding

Runtime contract:
  - Non-blocking: all inference runs in ThreadPoolExecutor
  - Runs every _INFER_EVERY_N frames (default 4)
  - Subscribes to FrameEvent (same as CNN/Deepfake)
  - Publishes IdentityEvent after each inference
  - Graceful fallback: if model fails → status=NO_MODEL, no crash

Model download:
  InsightFace downloads buffalo_sc on first use (~50 MB) to:
  ~/.insightface/models/buffalo_sc/
  Subsequent runs use the cached model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
MATCH_THRESHOLD   = 0.60    # cosine similarity >= this → identity_match=True
_INFER_EVERY_N    = 4       # run every Nth frame (≈7–8 Hz at 30 fps camera)
_MIN_ENROLL_FRAMES = 3      # minimum frames required for a valid enrolment
_MODEL_NAME       = "buffalo_sc"   # lightweight det+recog pack (~50 MB)
_DET_SIZE         = (160, 160)     # face detector input size
_CTX_ID           = 0              # 0=CPU, set >0 for GPU
_IDENTITY_EMA_ALPHA = 0.6          # EMA weight for per-frame identity score smoothing


class IdentityVerifier:
    """
    ArcFace-based identity verification engine.

    Usage:
      verifier = IdentityVerifier()
      verifier.register()               # hook into FrameEvent bus

      # Enrol the target identity once (e.g. from first N frames):
      verifier.enroll_identity(frames)  # list of BGR numpy arrays

      # After enrolment, inference runs automatically on every Nth frame.
      verifier.latest_result            # polled by FusionEngine / DebugUI
    """

    def __init__(
        self,
        match_threshold: float = MATCH_THRESHOLD,
        infer_every_n:   int   = _INFER_EVERY_N,
    ) -> None:
        self._threshold     = match_threshold
        self._every_n       = infer_every_n

        self._app           = None      # insightface FaceAnalysis
        self._ready         = False     # True once model loaded
        self._enrolled      = False     # True once reference_embedding set

        self.reference_embedding: Optional[np.ndarray] = None

        # Frame counter for throttle
        self._frame_count   = 0

        # EMA state for identity score smoothing (alpha=0.6)
        # Initialized to None; set on first inference result.
        self._smooth_identity: Optional[float] = None

        # Thread pool for non-blocking inference
        self._executor      = ThreadPoolExecutor(max_workers=1, thread_name_prefix="identity")

        # Public result — polled by DebugUI and FusionEngine.
        # SAFE DEFAULTS: identity_score=1.0 and identity_match=True until enrolled.
        # This prevents fusion Rule 2 (i < 0.45) and Rule 5 (i < 0.60) from
        # firing false alerts before the reference embedding is established.
        self.latest_result: dict = {
            "status":         "NOT_ENROLLED",
            "identity_score": 1.0,    # safe default — no mismatch assumed
            "identity_match": True,   # safe default — assume match until proven otherwise
            "enrolled":       False,
        }

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """
        Load InsightFace ArcFace model (buffalo_sc).
        Downloads ~50 MB on first run, cached in ~/.insightface/models/.
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis

            print(
                f"[IDENTITY] loading InsightFace {insightface.__version__}  "
                f"model={_MODEL_NAME}  det_size={_DET_SIZE}"
            )
            app = FaceAnalysis(
                name=_MODEL_NAME,
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=_CTX_ID, det_size=_DET_SIZE)

            self._app   = app
            self._ready = True
            self.latest_result["status"] = "NO_REFERENCE"
            print(f"[IDENTITY] model ready  threshold={self._threshold}")

        except Exception as exc:
            print(f"[IDENTITY] LOAD FAILED: {exc}")
            logger.error(f"IdentityVerifier load failed: {exc}", exc_info=True)
            self.latest_result["status"] = "NO_MODEL"

    # ─────────────────────────────────────────────────────────────────────────
    # Event bus wiring
    # ─────────────────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to FrameEvent for automatic per-frame inference."""
        try:
            from agent.event_bus import bus
            from agent.events import FrameEvent

            bus.subscribe(FrameEvent, self._on_frame_event)
            print(
                f"[IDENTITY] registered on FrameEvent  "
                f"ready={self._ready}  enrolled={self._enrolled}  "
                f"every_n={self._every_n}"
            )
        except Exception as exc:
            print(f"[IDENTITY] register FAILED: {exc}")

    async def _on_frame_event(self, event) -> None:
        """
        FrameEvent handler — mirrors CNN/Deepfake pattern exactly.
        Throttles to every _INFER_EVERY_N frames.
        """
        if not self._ready or not self._enrolled:
            return

        if not event.face_detected or event.face_roi_bgr is None:
            return

        self._frame_count += 1
        if self._frame_count % self._every_n != 0:
            return

        roi_bgr   = event.face_roi_bgr
        roi_shape = event.face_roi_shape

        print(f"[IDENTITY] dispatching  fid={event.frame_id}  shape={roi_shape}")

        try:
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._verify_sync,
                roi_bgr,
                roi_shape,
            )
        except Exception as exc:
            print(f"[IDENTITY] executor error: {exc}")
            return

        if result is None:
            return

        score, match = result

        # ── EMA smoothing ───────────────────────────────────────────────────
        # Smooths per-frame flicker in cosine similarity.
        # alpha=0.6: recent frame has 60% weight, history has 40%.
        if self._smooth_identity is None:
            self._smooth_identity = score
        else:
            self._smooth_identity = (
                _IDENTITY_EMA_ALPHA * score +
                (1.0 - _IDENTITY_EMA_ALPHA) * self._smooth_identity
            )
        smoothed_score = round(self._smooth_identity, 4)
        match          = smoothed_score >= self._threshold

        self.latest_result = {
            "status":         "READY",
            "identity_score": smoothed_score,
            "identity_match": match,
            "enrolled":       True,
        }

        print(
            f"[IDENTITY] raw={score:.3f}  smooth={smoothed_score:.3f}  "
            f"match={match}  fid={event.frame_id}  thr={self._threshold}"
        )

        # Publish IdentityEvent
        try:
            from agent.event_bus import bus
            from agent.events import IdentityEvent

            await bus.publish(IdentityEvent(
                session_id     = event.session_id,
                frame_id       = event.frame_id,
                identity_score = smoothed_score,
                identity_match = match,
            ))
        except Exception as exc:
            logger.debug(f"[IDENTITY] publish error: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — enrolment
    # ─────────────────────────────────────────────────────────────────────────

    def enroll_identity(self, frames: list[bytes | np.ndarray], shapes: list[tuple] = None) -> bool:
        """
        Enrol a reference identity from a list of face crops.

        Args:
          frames: list of face_roi_bgr (bytes or numpy arrays)
          shapes: list of (h, w) tuples — required if frames are bytes

        Returns:
          True if enrolment succeeded, False if not enough valid embeddings.
        """
        if not self._ready:
            print("[IDENTITY] enroll skipped — model not loaded")
            return False

        embeddings = []
        for i, frame in enumerate(frames):
            if isinstance(frame, (bytes, bytearray)):
                if shapes is None or i >= len(shapes):
                    continue
                h, w = shapes[i]
                try:
                    bgr = np.frombuffer(frame, dtype=np.uint8).reshape(h, w, 3)
                except Exception:
                    continue
            else:
                bgr = frame

            emb = self._extract_embedding(bgr)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < _MIN_ENROLL_FRAMES:
            print(
                f"[IDENTITY] enroll FAILED: only {len(embeddings)} valid embeddings "
                f"(need >= {_MIN_ENROLL_FRAMES})"
            )
            return False

        # Mean embedding → L2 normalize
        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm     = np.linalg.norm(mean_emb)
        if norm < 1e-8:
            print("[IDENTITY] enroll FAILED: zero-norm mean embedding")
            return False

        self.reference_embedding             = mean_emb / norm
        self._enrolled                       = True
        self.latest_result["status"]         = "READY"
        self.latest_result["enrolled"]       = True
        print(
            f"[IDENTITY] enrolled  frames={len(frames)}  "
            f"valid={len(embeddings)}  emb_dim={self.reference_embedding.shape[0]}"
        )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence — save / load reference embedding
    # ─────────────────────────────────────────────────────────────────────────

    def save_identity(self, session_dir: str | Path) -> bool:
        """
        Persist the current reference_embedding to disk.

        Writes:
          <session_dir>/embedding.npy   — raw float32 array (512,)
          <session_dir>/meta.json       — {frames, dim, threshold}

        Returns:
          True on success, False if not enrolled or write fails.
        """
        if not self._enrolled or self.reference_embedding is None:
            print("[IDENTITY] save_identity: nothing to save — not enrolled")
            return False

        try:
            session_dir = Path(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)

            emb_path  = session_dir / "embedding.npy"
            meta_path = session_dir / "meta.json"

            np.save(str(emb_path), self.reference_embedding)

            meta = {
                "frames":    int(self.reference_embedding.shape[0]),   # = 512, dim
                "dim":       int(self.reference_embedding.shape[0]),
                "threshold": self._threshold,
            }
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, indent=2)

            print(
                f"[IDENTITY] saved embedding → {emb_path}  "
                f"dim={self.reference_embedding.shape[0]}"
            )
            return True

        except Exception as exc:
            print(f"[IDENTITY] save_identity FAILED: {exc}")
            logger.error(f"save_identity error: {exc}", exc_info=True)
            return False

    def load_identity(self, path: str | Path) -> bool:
        """
        Load a persisted reference embedding from disk and activate enrollment.

        Expects:
          <path>/embedding.npy   — float32 array of shape (512,)

        Returns:
          True on success (self.reference_embedding set, self._enrolled=True),
          False if file not found or load fails.
        """
        emb_path = Path(path) / "embedding.npy"

        if not emb_path.exists():
            print(f"[IDENTITY] no embedding found at {emb_path} → NOT ENROLLED")
            return False

        try:
            emb  = np.load(str(emb_path)).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                print("[IDENTITY] loaded embedding has zero norm — rejected")
                return False

            self.reference_embedding          = emb / norm   # ensure unit-norm
            self._enrolled                    = True
            self.latest_result["status"]      = "READY"
            self.latest_result["enrolled"]    = True

            print(
                f"[IDENTITY] loaded embedding  path={emb_path}  "
                f"dim={self.reference_embedding.shape[0]}  "
                f"threshold={self._threshold}"
            )
            return True

        except Exception as exc:
            print(f"[IDENTITY] load_identity FAILED: {exc}")
            logger.error(f"load_identity error: {exc}", exc_info=True)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Core inference — synchronous, runs in thread pool
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_embedding(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run ArcFace embedding extraction on a BGR face crop.

        Steps:
          1. Convert BGR → RGB  (InsightFace expects RGB)
          2. Run FaceAnalysis.get() — detects + embeds
          3. Take the largest detected face (by bbox area)
          4. L2-normalize the 512-d embedding

        Returns:
          np.ndarray shape (512,), normalized — or None if no face detected.
        """
        try:
            rgb = bgr[..., ::-1].copy().astype(np.uint8)
            faces = self._app.get(rgb)

            if not faces:
                return None

            # Pick largest face by bounding box area
            def _area(f):
                x1, y1, x2, y2 = f.bbox
                return max(0.0, (x2 - x1) * (y2 - y1))

            face = max(faces, key=_area)
            emb  = face.embedding                      # shape (512,)

            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return None
            return (emb / norm).astype(np.float32)

        except Exception as exc:
            logger.debug(f"[IDENTITY] _extract_embedding error: {exc}")
            return None

    def _verify_sync(
        self,
        roi_bytes: bytes,
        roi_shape: tuple,
    ) -> Optional[tuple[float, bool]]:
        """
        Synchronous verification — called in thread pool.

        Returns:
          (identity_score, identity_match)  or  None on failure.
        """
        if self.reference_embedding is None:
            return None

        try:
            h, w = roi_shape
            bgr  = np.frombuffer(roi_bytes, dtype=np.uint8).reshape(h, w, 3)
            emb  = self._extract_embedding(bgr)

            if emb is None:
                print("[IDENTITY] no face detected in frame — skip")
                return None

            # Cosine similarity (both vectors are L2-normalized → dot product = cosine)
            score = float(np.dot(emb, self.reference_embedding))
            # Clamp to [0, 1] — cosine can be negative for dissimilar faces
            score = max(0.0, min(1.0, score))
            match = score >= self._threshold

            return score, match

        except Exception as exc:
            print(f"[IDENTITY] _verify_sync error: {exc}")
            logger.debug(f"_verify_sync error: {exc}", exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience — extract_embedding public API
    # ─────────────────────────────────────────────────────────────────────────

    def extract_embedding(self, face_roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Public API: extract 512-d ArcFace embedding from a BGR face crop.
        Converts BGR → RGB internally.

        Returns:
          np.ndarray shape (512,), L2-normalized — or None.
        """
        if not self._ready:
            return None
        return self._extract_embedding(face_roi_bgr)

    def verify_identity(self, face_roi_bgr: np.ndarray) -> tuple[float, bool]:
        """
        Public API: run one-shot synchronous verification.

        Returns:
          (identity_score, identity_match)
          identity_score: float [0, 1]
          identity_match: bool
        """
        if not self._ready or not self._enrolled:
            return 0.0, False

        emb = self._extract_embedding(face_roi_bgr)
        if emb is None:
            return 0.0, False

        score = float(np.dot(emb, self.reference_embedding))
        score = max(0.0, min(1.0, score))
        return score, score >= self._threshold
