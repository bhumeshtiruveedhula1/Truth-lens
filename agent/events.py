"""
events.py — Canonical event schema for DeepShield.

This file is the single shared contract between ALL modules.
It has zero dependencies on any other DeepShield module.
This is intentional: it can be imported by the SDK, browser extension,
or enterprise backend without pulling in the rest of the agent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import time
import uuid


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def new_session_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Raw frame data — emitted by Capture Layer at ~30fps
# ---------------------------------------------------------------------------

@dataclass
class FrameEvent:
    """Emitted by Capture Layer every processed frame."""
    session_id: str
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    # Face ROI BGR bytes only (not the full frame) to keep per-event allocations small
    face_roi_bgr: Optional[bytes] = None
    # (height, width) for face_roi_bgr reconstruction
    face_roi_shape: Optional[tuple[int, int]] = None
    # 468 MediaPipe FaceMesh landmarks as list of (x, y, z) normalized tuples
    face_landmarks: list = field(default_factory=list)
    # (x, y, w, h) pixel bbox of detected face in frame
    face_bbox: tuple = (0, 0, 0, 0)
    # True if a face was detected in this frame
    face_detected: bool = False
    # Frame dimensions
    frame_width: int = 640
    frame_height: int = 480


@dataclass
class CNNEvent:
    """
    Emitted by CNNInferenceEngine after each visual inference call.
    Published at ~10 Hz (every 3rd frame).  Consumed by DebugUI and
    any future fusion layer.
    """
    session_id:          str
    frame_id:            int
    cnn_fake_probability: float   # 0.0 = real, 1.0 = fake
    cnn_label:           str      # "REAL" or "FAKE"
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusionEvent:
    """
    Final decision produced by FusionEngine combining GRU + CNN + Deepfake scores.
    Published at ~10 Hz (whenever CNNEvent fires).
    Consumed by DebugUI and any downstream API/overlay.
    """
    session_id:     str
    frame_id:       int
    gru_score:      float   # GRU behavioral fake probability (0=real, 1=fake)
    cnn_score:      float   # CNN liveness fake probability  (0=real, 1=fake)
    deepfake_score: float   # Xception deepfake probability  (0=real, 1=fake)
    fusion_score:   float   # weighted combination
    fusion_smooth:  float   # EMA-smoothed fusion_score
    final_status:   str     # "SAFE" | "WARNING" | "HIGH_RISK" | "LOW_CONFIDENCE"
    reason:         str     # human-readable explanation
    timestamp: float = field(default_factory=time.time)


@dataclass
class DeepfakeEvent:
    """
    Emitted by DeepfakeInferenceEngine after each Xception forward pass.
    Fired at most every 300ms, only when CNN or fusion signals are elevated.
    """
    session_id:           str
    frame_id:             int
    deepfake_probability: float   # 0.0=real, 1.0=deepfake
    deepfake_label:       str     # "REAL" | "FAKE"
    trigger_cnn:          bool    # fired because cnn_score > 0.6
    trigger_fusion:       bool    # fired because fusion WARNING/HIGH_RISK
    trigger_timeout:      bool    # fired because >1s since last run
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Biometric signal — emitted by each SignalExtractor in Liveness Engine
# ---------------------------------------------------------------------------

@dataclass
class LivenessSignal:
    """
    Emitted by each SignalExtractor.
    score: 0.0 = very likely fake/anomalous, 1.0 = very likely real/human
    confidence: how reliable this reading is (low confidence → less weight)
    """
    session_id: str
    extractor_name: str   # "blink" | "head_pose" | "gaze" | "micro_expr"
    value: float          # raw signal value (e.g. EAR ratio, yaw angle)
    score: float          # 0.0 (fake) → 1.0 (human)
    confidence: float     # 0.0 → 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provenance signal — emitted by Provenance Heuristics module
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceSignal:
    """
    Emitted by each provenance checker.
    anomaly_score: 0.0 = clean, 1.0 = very suspicious
    score property inverts this so it's compatible with weighted fusion.
    """
    session_id: str
    check_name: str       # "compression_artifact" | "temporal_jitter"
    anomaly_score: float  # 0.0 (clean) → 1.0 (suspicious)
    evidence: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def extractor_name(self) -> str:
        return self.check_name

    @property
    def score(self) -> float:
        """Convert anomaly_score to trust-compatible score."""
        return 1.0 - self.anomaly_score

    @property
    def confidence(self) -> float:
        return self.evidence.get("confidence", 0.7)


# ---------------------------------------------------------------------------
# Trust event — primary output of Risk Engine, streamed to Overlay UI
# ---------------------------------------------------------------------------

@dataclass
class TrustEvent:
    """
    The primary event consumed by the Overlay UI and Local API.
    Emitted by Risk Engine at ~2Hz.
    """
    session_id: str
    trust_score: float                            # 0.0 → 1.0
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    contributing_signals: dict = field(default_factory=dict)  # {name: score}
    alert: bool = False
    alert_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "trust_score": round(self.trust_score, 4),
            "risk_level": self.risk_level,
            "contributing_signals": {
                k: round(v, 4) for k, v in self.contributing_signals.items()
            },
            "alert": self.alert,
            "alert_reason": self.alert_reason,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Alert event — emitted on HIGH risk, triggers overlay red modal + audit write
# ---------------------------------------------------------------------------

@dataclass
class AlertEvent:
    """Emitted when Risk Engine classifies a frame window as HIGH risk."""
    session_id: str
    trust_score: float
    primary_trigger: str        # which signal name contributed most
    recommended_action: str     # "warn_user" | "log_only"
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Session record — written to audit DB on alert or session end
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    """Immutable audit record written to local SQLite."""
    session_id: str
    start_time: float
    end_time: float
    peak_risk: Literal["LOW", "MEDIUM", "HIGH"]
    alert_count: int
    trust_score_trace: list   # [(timestamp, score), ...]
    face_embedding_hash: str  # SHA-256 of face embedding, NOT raw pixels
    policy_snapshot: dict
