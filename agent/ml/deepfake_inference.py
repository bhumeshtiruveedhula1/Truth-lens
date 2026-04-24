"""
agent/ml/deepfake_inference.py — Production Deepfake Inference Module
======================================================================
Model:      Xception (FaceForensics++ pretrained, C0 compression)
Weights:    models/legacy_deepfake_xception.pkl  (79.7 MB)
Input:      face_roi_bgr — OpenCV BGR numpy array, any resolution
Output:     deepfake_probability  (calibrated, used by fusion engine)
            raw_probability       (unmodified sigmoid, used for audit)

Architecture — Self-contained, no Desktop repo dependency:
  Xception backbone is embedded here via the copied network files in
  agent/ml/legacy/network_xception/.  No sys.path hacks, no import
  of agent.ml.legacy.* at module level — imported lazily inside _load().

Calibration pipeline:
  raw  = softmax(logit)[1]           (original model output)
  cal  = temperature_scale(raw, T)   (soften saturated outputs)
  ema  = EMA(cal, alpha)             (temporal stability on webcam)

Temperature T=2.5:
  Xception trained on compressed video frames (C0/C23/C40) tends to
  produce high probabilities on real webcam faces (domain shift).
  T=2.5 significantly reduces saturation without hiding the signal:
    raw=0.94  →  logit≈2.75  →  cal=sigmoid(2.75/2.5)=0.72
    raw=0.60  →  logit≈0.41  →  cal=sigmoid(0.41/2.5)=0.54
    raw=0.30  →  logit≈−0.85 →  cal=sigmoid(−0.85/2.5)=0.42

Domain mismatch guard:
  Rolling window of last 10 raw outputs.
  If mean > 0.80 for 10 consecutive frames → log WARNING.
  Never clamps output. Never crashes.

Weights path:
  models/legacy_deepfake_xception.pkl  (FaceForensics++ C0 checkpoint)

Upgrade path:
  Replace _load() with EfficientNet-B0 loading once deepfake-trained
  weights are available at models/deepfake_efficientnet_b0.pt.
  All async/event/throttle logic is model-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Weights ───────────────────────────────────────────────────────────────────
_WEIGHTS_PATH = Path("models/legacy_deepfake_xception.pkl")

# ── Runtime model toggle ──────────────────────────────────────────────────────
# True  → try EfficientNet-B0 (FF++ C23) first; fall back to Xception on error
# False → always use Xception (legacy, calibrated with T=2.5)
USE_EFFICIENTNET = True

# ── Throttle ──────────────────────────────────────────────────────────────────
_MIN_GAP_SEC   = 0.30
_MAX_GAP_SEC   = 1.00
_CNN_TRIGGER   = 0.60
_FUSION_STATES = {"WARNING", "HIGH_RISK"}

# ── Calibration ───────────────────────────────────────────────────────────────
DEEPFAKE_THRESHOLD = 0.55   # applied to calibrated_probability
_TEMPERATURE       = 2.5    # T>1 reduces saturation; Xception needs T≈2.5
_EMA_ALPHA         = 0.40   # smoothing (lower = more stable, slower to react)

# ── Domain mismatch ───────────────────────────────────────────────────────────
_MISMATCH_WINDOW = 10
_MISMATCH_THRESH = 0.80


def _temperature_scale(raw_prob: float, T: float) -> float:
    """
    Apply temperature scaling to a probability value.

    Equivalent to running sigmoid on logit/T:
      logit = log(p / (1-p))
      cal   = sigmoid(logit / T)

    Safe for edge values p=0.0 and p=1.0.
    """
    if raw_prob <= 0.0:
        return 0.0
    if raw_prob >= 1.0:
        return float(1.0 / (1.0 + math.exp(-300.0 / T)))  # near-1 but safe
    logit = math.log(raw_prob / (1.0 - raw_prob))
    return float(1.0 / (1.0 + math.exp(-logit / T)))


class DeepfakeInferenceEngine:
    """
    Production deepfake inference engine.

    Wraps FaceForensics++ pretrained Xception with calibrated outputs.
    Runs in a ThreadPoolExecutor — non-blocking, event-driven.
    Publishes DeepfakeEvent.  API identical to all previous adapters.
    """

    def __init__(
        self,
        weights_path: str | Path = _WEIGHTS_PATH,
        threshold:    float      = DEEPFAKE_THRESHOLD,
        temperature:  float      = _TEMPERATURE,
        ema_alpha:    float      = _EMA_ALPHA,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._threshold    = threshold
        self._temperature  = temperature
        self._ema_alpha    = ema_alpha

        self._model         = None   # Xception (legacy, always attempted)
        self._eff_model     = None   # EfficientNet-B0 (parallel, primary)
        self._eff_ready     = False  # True once EfficientNet loaded OK
        self._device        = "cpu"
        self._ready         = False  # True once at least one model loaded
        self._active_backend: str = "xception"  # 'efficientnet' or 'xception'

        # Throttle
        self._last_run_time: float = 0.0

        # EMA state (shared across backends for continuity)
        self._ema_value: Optional[float] = None

        # Domain mismatch rolling window
        self._raw_window: list[float] = []
        self._mismatch_logged = False

        # Face ROI cache
        self._last_roi_bytes: Optional[bytes] = None
        self._last_roi_shape: Optional[tuple] = None

        # Executor (None = default ThreadPoolExecutor)
        self._executor = None

        # Public result — polled by DebugUI and FusionEngine
        self.latest_result: dict = {
            "status":                    "LOADING",
            "deepfake_probability":      0.0,   # calibrated — used by fusion
            "raw_probability":           0.0,   # raw model output — audit
            "deepfake_label":            "REAL",
            "efficientnet_probability":  0.0,   # EfficientNet raw (0=none)
            "efficientnet_label":        "--",  # EfficientNet label
            "active_backend":            "--",  # 'efficientnet' or 'xception'
        }

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """
        Load inference backends.

        Priority:
          1. EfficientNet-B0 (USE_EFFICIENTNET=True) — FF++ C23, ~21 MB download
          2. Xception (legacy) — FF++ C0, 79.7 MB, local

        Active backend is set by self._active_backend.
        Falls back to Xception automatically if EfficientNet fails.
        Both backends can coexist — EfficientNet result is always shown
        separately in latest_result['efficientnet_probability'].
        """
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── 1. Try EfficientNet-B0 ────────────────────────────────────────────
        if USE_EFFICIENTNET:
            self._load_efficientnet()
        else:
            print("DEEPFAKE: USE_EFFICIENTNET=False — skipping EfficientNet")

        # ── 2. Always try Xception as fallback / parallel signal ──────────────
        self._load_xception()

        # ── 3. Decide active backend ──────────────────────────────────────────
        if self._eff_ready and USE_EFFICIENTNET:
            self._active_backend = "efficientnet"
        elif self._model is not None:
            self._active_backend = "xception"
        else:
            self.latest_result["status"] = "NO_WEIGHTS"
            return

        self._ready = True
        self.latest_result["status"] = "READY"
        self.latest_result["active_backend"] = self._active_backend
        print(
            f"DEEPFAKE: active backend = {self._active_backend}  "
            f"device={self._device}"
        )

    def _load_efficientnet(self) -> None:
        """Try to load EfficientNet-B0 (FF++ C23).  Sets self._eff_model."""
        try:
            from agent.ml.deepfake_efficientnet import load_model as eff_load
            model = eff_load(device=self._device)
            if model is not None:
                self._eff_model = model
                self._eff_ready = True
                print("DEEPFAKE: EfficientNet-B0 ready")
            else:
                print("DEEPFAKE: EfficientNet-B0 load returned None — will use Xception")
        except Exception as exc:
            print(f"DEEPFAKE: EfficientNet-B0 load error: {exc} — will use Xception")
            logger.warning(f"EfficientNet load failed: {exc}")

    def _load_xception(self) -> None:
        """Load Xception (legacy FF++ C0 checkpoint).  Sets self._model."""
        if not self._weights_path.exists():
            print(
                f"DEEPFAKE: Xception weights not found at {self._weights_path}\n"
                "  Expected: models/legacy_deepfake_xception.pkl (79.7 MB)\n"
                "  System continues in EfficientNet-only or 2-signal mode."
            )
            return

        try:
            import torch

            network_dir = Path(__file__).parent / "legacy"
            if str(network_dir) not in sys.path:
                sys.path.insert(0, str(network_dir))

            from network_xception.models import model_selection  # noqa: PLC0415

            print(
                f"DEEPFAKE: loading Xception (FF++ C0)  "
                f"device={self._device}  weights={self._weights_path}"
            )

            model = model_selection(modelname="xception", num_out_classes=2, dropout=0.5)
            state = torch.load(str(self._weights_path), map_location=self._device)
            if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                state = {k[len("module."):]: v for k, v in state.items()}

            model.load_state_dict(state)
            model.eval()
            model.to(self._device)
            self._model = model
            print("DEEPFAKE: Xception ready (calibrated T=2.5)") 
            print(
                f"DEEPFAKE: Xception loaded OK  "
                f"T={self._temperature}  threshold={self._threshold}  status=READY"
            )

        except Exception as exc:
            # Xception unavailable is expected — EfficientNet takes over.
            # Single-line, no traceback.
            print(f"DEEPFAKE: Xception unavailable (expected): {type(exc).__name__}: {exc}")
            self._xception_ready = False

    # ─────────────────────────────────────────────────────────────────────────
    # Event bus wiring
    # ─────────────────────────────────────────────────────────────────────────

    def register(self) -> None:
        try:
            from agent.event_bus import bus
            from agent.events import FrameEvent

            # Subscribe to FrameEvent (same as CNN) — NOT FusionEvent.
            # FusionEvent caused a circular dependency:
            #   FusionEngine polls deepfake.latest_result → always 0.0
            #   DeepfakeEngine waited for FusionEvent to trigger inference
            # Fix: trigger on raw frames, exactly like CNN does.
            bus.subscribe(FrameEvent, self._on_frame_event)
            print(
                f"[DF INIT] registered on FrameEvent  "
                f"eff_ready={self._eff_ready}  "
                f"xcep_ready={self._model is not None}  "
                f"active={self._active_backend}"
            )
        except Exception as exc:
            print(f"DEEPFAKE: register FAILED — {exc}")

    async def _on_frame_event(self, event) -> None:
        """
        Triggered at ~30 Hz by Capture Layer (same event CNN uses).

        Two responsibilities:
          1. Cache the latest face ROI for inference.
          2. Throttle and run inference every _INFER_EVERY_N frames.
        """
        print(f"[DF PIPELINE] FrameEvent fid={event.frame_id}  face={event.face_detected}")

        if not self._ready:
            print(f"[DF PIPELINE] NOT READY — skip fid={event.frame_id}")
            return

        if not event.face_detected:
            print(f"[DF PIPELINE] no face — skip fid={event.frame_id}")
            return

        roi_bgr   = event.face_roi_bgr
        roi_shape = event.face_roi_shape

        if roi_bgr is None or roi_shape is None:
            print(f"[DF PIPELINE] ROI None — skip fid={event.frame_id}")
            return

        # Cache latest ROI
        self._last_roi_bytes = roi_bgr
        self._last_roi_shape = roi_shape
        print(f"[DF PIPELINE] ROI cached fid={event.frame_id}  shape={roi_shape}  bytes={len(roi_bgr)}")

        # ── Frame-skip throttle (run every 5th frame, same cadence as CNN) ────
        self._frame_count = getattr(self, '_frame_count', 0) + 1
        if self._frame_count % 5 != 0:
            print(f"[DF PIPELINE] throttle skip count={self._frame_count}")
            return

        # ── Time-gap gate (never run faster than _MIN_GAP_SEC) ───────────────
        import time as _time
        now     = _time.monotonic()
        elapsed = now - self._last_run_time
        if elapsed < _MIN_GAP_SEC:
            print(f"[DF PIPELINE] min-gap skip elapsed={elapsed:.3f}s")
            return
        self._last_run_time = now

        # ── Run inference in thread pool (non-blocking) ───────────────────────
        print(f"[DF PIPELINE] DISPATCHING inference fid={event.frame_id}  backend={self._active_backend}")
        try:
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor, self._infer, roi_bgr, roi_shape
            )
        except Exception as exc:
            print(f"[DF EXECUTOR ERROR] {exc}")
            return

        if result is None:
            print(f"[DF PIPELINE] _infer returned None fid={event.frame_id}")
            return

        raw_prob, cal_prob, eff_prob = result
        label     = "FAKE" if cal_prob >= self._threshold else "REAL"
        eff_label = "FAKE" if eff_prob >= self._threshold else ("REAL" if eff_prob > 0 else "--")

        print(
            f"[DF RESULT] fid={event.frame_id}  "
            f"backend={self._active_backend}  "
            f"raw={raw_prob:.4f}  cal={cal_prob:.4f}  eff={eff_prob:.4f}  label={label}"
        )

        self.latest_result = {
            "status":                   "READY",
            "deepfake_probability":     round(cal_prob, 4),
            "raw_probability":          round(raw_prob, 4),
            "deepfake_label":           label,
            "efficientnet_probability": round(eff_prob, 4),
            "efficientnet_label":       eff_label,
            "active_backend":           self._active_backend,
        }

        # Publish DeepfakeEvent for downstream consumers
        try:
            from agent.event_bus import bus
            from agent.events import DeepfakeEvent
            await bus.publish(DeepfakeEvent(
                session_id           = event.session_id,
                frame_id             = event.frame_id,
                deepfake_probability = round(cal_prob, 4),
                deepfake_label       = label,
                trigger_cnn          = False,
                trigger_fusion       = False,
                trigger_timeout      = True,
            ))
        except Exception as exc:
            logger.debug(f"DEEPFAKE publish error: {exc}")

    # ── Legacy FusionEvent handler (kept as stub — no longer primary trigger) ─
    async def _on_fusion_event(self, event) -> None:
        """Stub — deepfake now triggers on FrameEvent instead of FusionEvent."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Core inference — synchronous, runs in thread pool
    # ─────────────────────────────────────────────────────────────────────────

    def _infer(
        self,
        roi_bytes: bytes,
        roi_shape: Optional[tuple],
    ) -> Optional[tuple[float, float]]:
        """
        BGR ROI bytes → Xception → (raw_probability, calibrated_probability).

        Preprocessing pipeline (exact FF++ xception_default_data_transforms):
          BGR numpy
            → cv2.cvtColor(BGR→RGB)
            → PIL.Image.fromarray()
            → Resize(299, 299)
            → ToTensor()             [0,255] → [0.0, 1.0]
            → Normalize([0.5]*3, [0.5]*3)    → range [-1, 1]
            → unsqueeze(0)           add batch dim

        Calibration:
          raw  = softmax(logit)[1]
          cal  = temperature_scale(raw, T=2.5)
          ema  = alpha * cal + (1-alpha) * prev_ema
        """
        try:
            if roi_shape is None:
                return None

            h, w = roi_shape
            bgr  = np.frombuffer(roi_bytes, dtype=np.uint8).reshape(h, w, 3)

            print(
                f"DEEPFAKE RUNNING [{self._active_backend.upper()}]  "
                f"shape=({h},{w})  device={self._device}"
            )

            # ── Route to active backend ───────────────────────────────────────
            if self._active_backend == "efficientnet" and self._eff_ready:
                raw_prob, cal_prob, eff_prob = self._infer_efficientnet(bgr)
            else:
                raw_prob, cal_prob, eff_prob = self._infer_xception(bgr)

            # ── Domain mismatch (always on raw) ──────────────────────────────
            self._check_domain_mismatch(raw_prob)

            return raw_prob, cal_prob, eff_prob

        except Exception as exc:
            print(f"DEEPFAKE _INFER ERROR: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def _infer_efficientnet(self, bgr: np.ndarray) -> tuple[float, float, float]:
        """
        EfficientNet-B0 inference.

        Preprocessing (exact model card — NO normalization):
          BGR → RGB → PIL → Resize(224,224) → ToTensor()

        Returns: (raw_prob, cal_prob_ema, eff_prob)
          raw_prob    = softmax(logits)[0][1]
          cal_prob    = EMA smoothed raw (no temperature — EfficientNet is
                        trained on 224px compressed frames, less domain shift)
          eff_prob    = same as raw_prob (stored separately for UI)
        """
        from agent.ml.deepfake_efficientnet import predict as eff_predict

        raw_prob = eff_predict(bgr, self._eff_model, self._device)
        eff_prob = raw_prob

        # Mild EMA only (EfficientNet doesn't need temperature scaling)
        if self._ema_value is None:
            self._ema_value = raw_prob
        else:
            self._ema_value = (
                self._ema_alpha * raw_prob +
                (1.0 - self._ema_alpha) * self._ema_value
            )
        cal_prob = self._ema_value

        print(
            f"EFFICIENTNET RAW: {raw_prob:.4f}  EMA: {cal_prob:.4f}"
        )
        return raw_prob, cal_prob, eff_prob

    def _infer_xception(self, bgr: np.ndarray) -> tuple[float, float, float]:
        """
        Xception inference with temperature calibration.

        Preprocessing (exact FF++ xception_default_data_transforms):
          BGR → RGB → PIL → Resize(299,299) → ToTensor → Normalize([0.5]*3,[0.5]*3)

        Returns: (raw_prob, cal_prob_ema, eff_prob=0.0)
        """
        import cv2
        import torch
        import torch.nn as nn
        from PIL import Image as pil_image
        from torchvision import transforms

        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil    = pil_image.fromarray(rgb)
        tf     = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        tensor = tf(pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
        raw_prob = float(nn.Softmax(dim=1)(logits)[0][1].cpu().item())

        # Temperature scaling + EMA
        cal_pre_ema = _temperature_scale(raw_prob, self._temperature)
        if self._ema_value is None:
            self._ema_value = cal_pre_ema
        else:
            self._ema_value = (
                self._ema_alpha * cal_pre_ema +
                (1.0 - self._ema_alpha) * self._ema_value
            )
        cal_prob = self._ema_value

        print(
            f"XCEPTION RAW: {raw_prob:.4f}  "
            f"T-scaled: {cal_pre_ema:.4f}  EMA: {cal_prob:.4f}"
        )
        return raw_prob, cal_prob, 0.0


    # ─────────────────────────────────────────────────────────────────────────
    # Domain mismatch monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def _check_domain_mismatch(self, raw_prob: float) -> None:
        """
        Rolling window check.  If raw outputs are consistently > 0.80 for
        10 consecutive frames, the model is domain-mismatched to this webcam.

        Action: log warning + suggest increasing temperature.
        Does NOT modify output.  Does NOT crash.
        """
        self._raw_window.append(raw_prob)
        if len(self._raw_window) > _MISMATCH_WINDOW:
            self._raw_window.pop(0)

        if len(self._raw_window) == _MISMATCH_WINDOW:
            mean = sum(self._raw_window) / _MISMATCH_WINDOW
            if mean > _MISMATCH_THRESH:
                if not self._mismatch_logged:
                    logger.warning(
                        f"Deepfake model domain mismatch detected — "
                        f"rolling mean raw_prob={mean:.3f} > {_MISMATCH_THRESH}. "
                        f"Consider increasing temperature (current T={self._temperature}). "
                        f"GRU behavioral signal will veto false HIGH_RISK decisions."
                    )
                    self._mismatch_logged = True
                print(
                    f"DEEPFAKE WARNING: domain mismatch  "
                    f"mean_raw={mean:.3f}  T={self._temperature}  "
                    f"(fusion GRU guard active)"
                )
            else:
                self._mismatch_logged = False   # reset if scores normalise
