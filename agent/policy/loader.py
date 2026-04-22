"""
policy/loader.py — Policy configuration loader.

Loads and validates policy.yaml using Pydantic.
The policy controls all tunable thresholds and behaviors of the risk engine.

In MVP: loaded from local YAML.
In Phase 2: fetched from enterprise policy server with JWT auth.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SignalWeights(BaseModel):
    blink: float                = Field(0.25, ge=0.0, le=1.0)
    head_pose: float            = Field(0.20, ge=0.0, le=1.0)
    gaze: float                 = Field(0.15, ge=0.0, le=1.0)
    micro_expr: float           = Field(0.10, ge=0.0, le=1.0)
    compression_artifact: float = Field(0.15, ge=0.0, le=1.0)
    temporal_jitter: float      = Field(0.10, ge=0.0, le=1.0)

    def as_dict(self) -> dict:
        return self.dict()


class AlertPolicy(BaseModel):
    action: str         = "warn_user"   # "warn_user" | "log_only"
    cooldown_sec: float = 10.0
    modal_timeout_sec: float = 8.0


class ThresholdPolicy(BaseModel):
    low_threshold: float    = Field(0.80, ge=0.0, le=1.0)
    medium_threshold: float = Field(0.55, ge=0.0, le=1.0)
    rolling_window_sec: float = 30.0
    blink_override_threshold: float  = 0.10
    compression_override_threshold: float = 0.15


class DeepShieldPolicy(BaseModel):
    version: str                = "1.0"
    organization: str           = "hackathon-demo"
    weights: SignalWeights      = Field(default_factory=SignalWeights)
    thresholds: ThresholdPolicy = Field(default_factory=ThresholdPolicy)
    alert: AlertPolicy          = Field(default_factory=AlertPolicy)

    # Paranoia mode: lowers thresholds for stricter enterprise enforcement
    paranoia_mode: bool = False

    def apply_paranoia(self) -> "DeepShieldPolicy":
        if self.paranoia_mode:
            self.thresholds.low_threshold    = 0.88
            self.thresholds.medium_threshold = 0.65
        return self


_DEFAULT_POLICY_PATH = Path(__file__).parent / "policy.yaml"


def load_policy(path: Optional[Path] = None) -> DeepShieldPolicy:
    """Load and validate policy from YAML file. Falls back to defaults."""
    policy_path = path or _DEFAULT_POLICY_PATH

    if not policy_path.exists():
        logger.warning(f"Policy file not found at {policy_path} — using defaults")
        return DeepShieldPolicy()

    try:
        with open(policy_path, "r") as f:
            data = yaml.safe_load(f) or {}
        policy = DeepShieldPolicy(**data)
        policy.apply_paranoia()
        logger.info(f"Policy loaded from {policy_path} (org={policy.organization})")
        return policy
    except Exception as exc:
        logger.error(f"Failed to load policy: {exc} — using defaults")
        return DeepShieldPolicy()
