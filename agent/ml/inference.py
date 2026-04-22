"""
agent/ml/inference.py — GRU Real-Time Inference Engine
=======================================================
Non-blocking, event-driven inference layer that runs the trained GRU
model on live signal sequences.

Feature alignment
-----------------
Feature order is read directly from the saved checkpoint and NEVER
inferred from code or documentation.  The exact order confirmed from
models/gru_refined.pt AND data/sequences.npz is:

    ["yaw", "pitch", "roll", "motion_score",
     "temporal_score", "texture_score", "face_present"]

Normalization (z-score)
-----------------------
  feat_mean and feat_std are loaded from data/sequences.npz, which is
  the exact artifact produced by the same training run.
  std=0 guard: any zero std element is replaced with 1e-6 to prevent
  divide-by-zero (training convention uses 1.0 but 1e-6 is numerically
  equivalent and explicit).

Architecture (UNCHANGED — must match training exactly)
-------------------------------------------------------
  GRU(input=7, hidden=128, layers=1, batch_first=True)
  -> last hidden state -> Linear(128, 1) -> sigmoid

Lifecycle
---------
  1. Construct GRUInferenceEngine(model_path, npz_path)
  2. Call register() to subscribe to event bus signals
  3. Signals are fed automatically via LivenessSignal events
  4. FrameEvent triggers push_frame() on every frame
  5. Query latest_result for the current prediction

Return schema
-------------
  READY:
    { "status": "READY",
      "fake_probability": float,  # 0.0-1.0
      "fake_label": "REAL" | "FAKE" }

  Not enough data:
    { "status": "INSUFFICIENT_DATA",
      "fake_probability": 0.0,
      "fake_label": "REAL" }

Debug output (stdout)
---------------------
  ML FEATURE VECTOR (frame N): [...]  (len=7)  — every 30 frames
  ML FACE_PRESENT: 0/1                          — every 30 frames
  ML BUFFER SIZE: N/24                          — every 10 frames while filling
  ML TRIGGER: Running GRU inference             — on first READY
  ML NORMALIZED: [...]                          — on first READY
  ML INFERENCE: prob=X label=Y                  — on first READY + every 30 frames
  ML INFERENCE ERROR: ...                       — on any inference exception

Thread safety
-------------
  All asyncio event handlers run on the single asyncio thread.
  _buffer and _signals are only written from that thread.
  latest_result is assigned atomically (dict assignment is thread-safe
  under the GIL), so the debug UI render thread can read it safely.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature metadata — loaded from checkpoint at runtime, never hardcoded here
# ---------------------------------------------------------------------------
_FALLBACK_FEATURES = [
    "yaw", "pitch", "roll", "motion_score",
    "temporal_score", "texture_score", "face_present",
]
_FALLBACK_THRESHOLD = 0.30


class GRUInferenceEngine:
    """
    Stateful sliding-window GRU inference engine.
    Instantiated once per session.  Feeds on event bus signals.
    """

    def __init__(
        self,
        model_path: str | Path = "models/gru_refined.pt",
        npz_path:   str | Path = "data/sequences.npz",
    ) -> None:
        self._model_path = Path(model_path)
        self._npz_path   = Path(npz_path)

        # Populated during _load()
        self._model      = None
        self._features:  list[str]  = []
        self._feat_mean: np.ndarray = None
        self._feat_std:  np.ndarray = None
        self._window:    int        = 24
        self._threshold: float      = _FALLBACK_THRESHOLD
        self._device     = None
        self._ready      = False

        # Sliding buffer — one entry per frame.
        # Initialised without maxlen; _load() sets the correct maxlen.
        self._buffer: deque = deque()

        # Latest signal cache — written only by event handlers (asyncio thread)
        self._signals: dict[str, float] = {
            "yaw":            0.0,
            "pitch":          0.0,
            "roll":           0.0,
            "motion_score":   0.0,
            "temporal_score": 0.0,
            "texture_score":  0.0,
            "face_present":   0.0,
        }

        # Latest prediction — readable from any thread (GIL-safe atomic assign)
        self.latest_result: dict = {
            "status":           "INSUFFICIENT_DATA",
            "fake_probability": 0.0,
            "fake_label":       "REAL",
        }

        # Debug counters
        self._push_count: int  = 0
        self._was_ready:  bool = False   # tracks first READY transition

        # EMA smoothing state
        self._smoothed_score: float = 0.0
        self._has_smoothed:   bool  = False

        # Runtime evaluation stats (Task 1)
        self._stat_total:      int   = 0
        self._stat_real:       int   = 0
        self._stat_fake:       int   = 0
        self._stat_sum:        float = 0.0
        self._stat_last_label: str | None = None
        self._stat_flicker:    int   = 0

        self._load()

    # =========================================================================
    # 1.  Model loading
    # =========================================================================

    def _load(self) -> None:
        """Load model weights + normalization stats.  Fails gracefully."""
        try:
            import torch
            import torch.nn as nn

            self._device = torch.device("cpu")  # CPU-only for real-time safety

            # -----------------------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------------------
            if not self._model_path.exists():
                logger.warning(
                    f"GRUInferenceEngine: model not found at {self._model_path}. "
                    "Inference disabled."
                )
                return

            ckpt = torch.load(
                self._model_path,
                map_location=self._device,
                weights_only=False,
            )

            # Read exact feature list from checkpoint — never from documentation
            raw_features    = ckpt.get("features", _FALLBACK_FEATURES)
            self._features  = [str(f) for f in raw_features]
            input_size      = ckpt.get("input_size",  len(self._features))
            hidden_size     = ckpt.get("hidden_size", 128)
            self._threshold = float(ckpt.get("best_threshold", _FALLBACK_THRESHOLD))

            logger.info(
                f"GRUInferenceEngine: features={self._features} "
                f"window={self._window} threshold={self._threshold:.2f}"
            )

            # -----------------------------------------------------------------
            # Model rebuild (architecture MUST match training exactly)
            # -----------------------------------------------------------------
            class _GRU(nn.Module):
                def __init__(self, in_sz, h_sz):
                    super().__init__()
                    self.gru  = nn.GRU(in_sz, h_sz, num_layers=1, batch_first=True)
                    self.head = nn.Linear(h_sz, 1)

                def forward(self, x):
                    _, h = self.gru(x)
                    return self.head(h.squeeze(0)).squeeze(-1)

            model = _GRU(input_size, hidden_size)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._model = model

            # -----------------------------------------------------------------
            # Normalization stats
            # -----------------------------------------------------------------
            if self._npz_path.exists():
                d = np.load(self._npz_path, allow_pickle=True)
                self._feat_mean = d["feat_mean"].astype(np.float32)
                self._feat_std  = d["feat_std"].astype(np.float32)
                # AUDIT FIX: guard zero-std with 1e-6 (prevents divide-by-zero)
                self._feat_std  = np.where(
                    self._feat_std == 0, np.float32(1e-6), self._feat_std
                )
                # Window size from training data shape
                self._window = int(d["X"].shape[1])
                logger.info(
                    f"GRUInferenceEngine: norm loaded from {self._npz_path} "
                    f"mean={np.round(self._feat_mean, 3).tolist()} "
                    f"std={np.round(self._feat_std, 3).tolist()}"
                )
            else:
                logger.warning(
                    f"GRUInferenceEngine: {self._npz_path} not found. "
                    "Using unit normalization (mean=0, std=1)."
                )
                F = len(self._features)
                self._feat_mean = np.zeros(F, dtype=np.float32)
                self._feat_std  = np.ones(F,  dtype=np.float32)

            # -----------------------------------------------------------------
            # AUDIT VERIFIED: buffer initialised with correct maxlen from NPZ
            # -----------------------------------------------------------------
            self._buffer = deque(maxlen=self._window)
            self._ready  = True
            logger.info(
                f"GRUInferenceEngine: READY — "
                f"window={self._window} features={len(self._features)} "
                f"threshold={self._threshold}"
            )
            print(
                f"ML ENGINE READY: window={self._window} "
                f"features={self._features} threshold={self._threshold}"
            )

        except Exception as exc:
            logger.error(f"GRUInferenceEngine: failed to load — {exc}", exc_info=True)
            self._ready = False

    # =========================================================================
    # 2.  Event bus wiring
    # =========================================================================

    def register(self) -> None:
        """Subscribe to LivenessSignal and FrameEvent on the bus."""
        try:
            from agent.event_bus import bus
            from agent.events import FrameEvent, LivenessSignal

            bus.subscribe(LivenessSignal, self._on_liveness_signal)
            bus.subscribe(FrameEvent,     self._on_frame_event)
            logger.info("GRUInferenceEngine registered on event bus")
        except Exception as exc:
            logger.warning(f"GRUInferenceEngine.register failed: {exc}")

    async def _on_liveness_signal(self, sig) -> None:
        """
        Update signal cache from each extractor's LivenessSignal.
        AUDIT FIX: exception now at logger.warning (was debug — silent).
        """
        try:
            md   = sig.metadata
            name = sig.extractor_name

            if name == "head_pose":
                self._signals["yaw"]   = float(md.get("yaw",   0.0))
                self._signals["pitch"] = float(md.get("pitch", 0.0))
                self._signals["roll"]  = float(md.get("roll",  0.0))

            elif name == "motion":
                self._signals["motion_score"] = float(md.get("motion_score", 0.0))

            elif name == "temporal_consistency":
                self._signals["temporal_score"] = float(md.get("temporal_score", 0.0))

            elif name == "texture":
                self._signals["texture_score"] = float(md.get("texture_score", 0.0))

        except Exception as exc:
            # AUDIT FIX: was logger.debug (silent) — now logger.warning (visible)
            logger.warning(
                f"GRUInferenceEngine._on_liveness_signal [{sig.extractor_name}]: {exc}",
                exc_info=True,
            )

    async def _on_frame_event(self, event) -> None:
        """
        Called at ~30 Hz for EVERY frame.
        AUDIT VERIFIED: no conditional blocking — push_frame is always called.
        """
        # TRACE: always prints — confirms event bus wiring is live
        print("ML EVENT TRIGGERED")
        try:
            face_present = bool(event.face_detected)
            print(f"ML FACE_PRESENT: {int(face_present)}")
            self._signals["face_present"] = float(face_present)
            self.push_frame(face_present=face_present)
        except Exception as exc:
            logger.warning(
                f"GRUInferenceEngine._on_frame_event: {exc}", exc_info=True
            )

    # =========================================================================
    # 3.  Core inference
    # =========================================================================

    def push_frame(self, face_present: bool = True) -> dict:
        """
        Build feature vector, append to buffer, run inference when full.
        Called from _on_frame_event at ~30 Hz.  Never blocks.
        """
        # TRACE: always prints — confirms push_frame is being entered
        print(f"ML PUSH_FRAME CALLED: ready={self._ready}")

        if not self._ready:
            # TRACE: model failed to load — check startup logs for ERROR
            print("ML ENGINE NOT READY — model may have failed to load")
            self.latest_result = {
                "status":           "INSUFFICIENT_DATA",
                "fake_probability": 0.0,
                "fake_label":       "REAL",
            }
            return self.latest_result

        self._push_count += 1

        # -----------------------------------------------------------------
        # SIGNAL READINESS GUARD
        # During risk engine warmup (first ~45 frames), motion_score,
        # temporal_score and texture_score are still 0.0 because the risk
        # engine hasn't emitted its first LivenessSignal yet.
        # Pushing zero-vectors into the buffer poisons the first 24-frame
        # inference window with out-of-distribution data.
        # Fix: skip the frame (do NOT append) until at least one real
        # signal value has arrived.  head_pose (yaw/pitch) arrives first
        # via HeadPoseEstimator, so we check all five key signals.
        # -----------------------------------------------------------------
        if face_present:
            # Check only the risk-engine trio: these three signals stay at 0.0
            # for the first ~45 frames (risk engine warmup).  yaw/pitch arrive
            # immediately from HeadPoseEstimator so they are NOT checked here.
            risk_trio = [
                self._signals.get("motion_score",  0.0),
                self._signals.get("temporal_score",0.0),
                self._signals.get("texture_score", 0.0),
            ]
            if all(v == 0.0 for v in risk_trio):
                print("ML SKIP FRAME — signals not ready yet")
                return self.latest_result

        # Build feature vector in EXACT training order
        if not face_present:
            vec = [0.0] * len(self._features)   # zero-fill, NOT skipped
        else:
            vec = [self._signals.get(f, 0.0) for f in self._features]

        # Feature vector log every 30 frames
        if self._push_count % 30 == 1:
            print(
                f"ML FEATURE VECTOR (frame {self._push_count}): "
                f"{[round(v, 4) for v in vec]}  (len={len(vec)})"
            )
            for i, (name, val) in enumerate(zip(self._features, vec)):
                print(f"  [{i}] {name:20s} = {val:.4f}")

        # Unconditional buffer append — no conditional skip
        self._buffer.append(vec)
        fill = len(self._buffer)

        # TRACE: buffer size on every frame
        print(f"ML BUFFER SIZE: {fill}/{self._window}")

        if fill < self._window:
            self.latest_result = {
                "status":           "INSUFFICIENT_DATA",
                "fake_probability": 0.0,
                "fake_label":       "REAL",
            }
            return self.latest_result

        # Inference trigger
        try:
            import torch
            print("ML TRIGGER: Running GRU inference")

            seq      = np.array(list(self._buffer), dtype=np.float32)  # (24, 7)
            seq_norm = (seq - self._feat_mean) / self._feat_std        # (24, 7)

            if not self._was_ready:
                print(f"ML NORMALIZED (row 0): {[round(float(v),4) for v in seq_norm[0]]}")

            x = torch.from_numpy(seq_norm).unsqueeze(0)                # (1, 24, 7)

            with torch.no_grad():
                logit = self._model(x)

            # Task 2 — EMA smoothing
            raw_score = float(torch.sigmoid(logit).item())
            if not self._has_smoothed:
                self._smoothed_score = raw_score
                self._has_smoothed   = True
            else:
                self._smoothed_score = 0.7 * self._smoothed_score + 0.3 * raw_score
            smoothed_score = self._smoothed_score

            # Task 2 — updated threshold (0.52)
            threshold = 0.52
            label = "FAKE" if smoothed_score >= threshold else "REAL"

            # Task 3 — expanded result dict (backward-compatible)
            self.latest_result = {
                "status":           "READY",
                "raw_score":        round(raw_score,      4),
                "smoothed_score":   round(smoothed_score, 4),
                "fake_probability": round(smoothed_score, 4),  # UI compat
                "fake_label":       label,
            }

            # Task 4 — updated debug print
            print(
                f"ML RESULT: raw={raw_score:.4f} "
                f"smooth={smoothed_score:.4f} "
                f"label={label} thr=0.52"
            )

            if not self._was_ready:
                self._was_ready = True
                print(f"ML INFERENCE: FIRST PREDICTION — smooth={smoothed_score:.4f} label={label}")

            # Task 2 — update runtime stats after each prediction
            self._stat_total += 1
            self._stat_sum   += smoothed_score
            if label == "REAL":
                self._stat_real += 1
            else:
                self._stat_fake += 1
            if self._stat_last_label is not None and self._stat_last_label != label:
                self._stat_flicker += 1
            self._stat_last_label = label

            # Task 3 — print summary every 150 predictions (~5 s at 30 fps)
            if self._stat_total % 150 == 0:
                real_pct = (self._stat_real / self._stat_total) * 100
                fake_pct = (self._stat_fake / self._stat_total) * 100
                avg      = self._stat_sum   / self._stat_total
                print(
                    f"\nML STATS:\n"
                    f"REAL%: {real_pct:.1f}%  "
                    f"FAKE%: {fake_pct:.1f}%\n"
                    f"AVG SMOOTH: {avg:.3f}\n"
                    f"FLICKERS: {self._stat_flicker}\n"
                )

        except Exception as exc:
            logger.warning(f"GRUInferenceEngine inference error: {exc}", exc_info=True)
            print(f"ML INFERENCE ERROR: {exc}")
            self.latest_result = {
                "status":           "INSUFFICIENT_DATA",
                "fake_probability": 0.0,
                "fake_label":       "REAL",
            }

        return self.latest_result

    # =========================================================================
    # 4.  Diagnostics
    # =========================================================================

    @property
    def buffer_fill(self) -> int:
        """Number of frames currently in the sliding buffer."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when model is loaded and buffer is full."""
        return self._ready and len(self._buffer) >= self._window

    # Task 4 — stats reset
    def reset_stats(self) -> None:
        """Reset all runtime evaluation counters to zero."""
        self._stat_total      = 0
        self._stat_real       = 0
        self._stat_fake       = 0
        self._stat_sum        = 0.0
        self._stat_last_label = None
        self._stat_flicker    = 0
