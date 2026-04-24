"""
agent/ml/fusion_engine.py -- GRU + CNN + Deepfake Fusion Layer
===============================================================
Combines three independent intelligence signals into a single,
stable, interpretable final decision.

Signals:
  g = gru_fake_probability   (behavioral — primary ground truth)
  c = cnn_fake_probability   (presentation attack detection)
  d = deepfake_probability   (identity manipulation detection)

Decision rules (evaluated in strict priority order):

  Rule 1 — Strong REAL (behavior dominates):
    IF g < 0.30
    → SAFE  "behavior stable"

  Rule 2 — Strong deepfake detection (highest risk priority):
    ELIF d > 0.80
    → HIGH_RISK  "deepfake detected"

  Rule 3 — Combined behavioral + visual anomaly:
    ELIF g > 0.60 AND c > 0.60
    → HIGH_RISK  "behavior + visual anomaly"

  Rule 4 — Presentation attack (replay / screen):
    ELIF c > 0.85
    → WARNING  "possible replay / screen"

  Rule 5 — Suspicious disagreement (visual elevated, behavior OK):
    ELIF (c > 0.60 OR d > 0.60) AND g < 0.40
    → WARNING  "visual anomaly but behavior stable"

  Rule 6 — Default:
    → SAFE  "no strong anomaly"

Fusion score (3-signal weighted):
  fusion_score  = 0.50*g + 0.25*c + 0.25*d
  fusion_smooth = EMA(alpha=0.35, fusion_score)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_GRU_W      = 0.50
_CNN_W      = 0.25
_DF_W       = 0.25
_EMA_ALPHA  = 0.35


class FusionEngine:
    """
    Three-signal fusion layer.
    Subscribes to CNNEvent; polls GRU + Deepfake engines by reference.
    Pure arithmetic — no model calls, non-blocking.
    """

    def __init__(self, gru_engine, cnn_engine, deepfake_engine=None) -> None:
        self._gru_engine      = gru_engine
        self._cnn_engine      = cnn_engine
        self._deepfake_engine = deepfake_engine

        # EMA state
        self._fusion_smooth: float = 0.0
        self._has_smooth:    bool  = False

        # Latest result — polled by DebugUI
        self.latest_result: dict = {
            "status":         "LOADING",
            "gru_score":      0.0,
            "cnn_score":      0.0,
            "deepfake_score": 0.0,
            "fusion_score":   0.0,
            "fusion_smooth":  0.0,
            "final_status":   "LOW_CONFIDENCE",
            "reason":         "waiting for signals",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Event bus wiring
    # ─────────────────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to CNNEvent — triggers fusion on every CNN inference (~10 Hz)."""
        try:
            from agent.event_bus import bus
            from agent.events import CNNEvent
            bus.subscribe(CNNEvent, self._on_cnn_event)
            print("FUSION: registered (GRU + CNN + Deepfake)")
        except Exception as exc:
            print(f"FUSION: register FAILED — {exc}")

    async def _on_cnn_event(self, event) -> None:
        """
        Triggered at ~10 Hz by CNNInferenceEngine.
        Polls all three engine results, computes 3-signal fusion, publishes FusionEvent.
        """
        try:
            # ── Pull latest scores from all three engines ─────────────────────
            gru_r = self._gru_engine.latest_result
            cnn_r = self._cnn_engine.latest_result
            df_r  = self._deepfake_engine.latest_result if self._deepfake_engine else {}

            gru_ok = gru_r.get("status") == "READY"
            cnn_ok = cnn_r.get("status") == "READY"
            df_ok  = df_r.get("status")  == "READY"

            g = float(gru_r.get("smoothed_score",
                      gru_r.get("fake_probability", 0.0))) if gru_ok else None
            c = float(cnn_r.get("cnn_fake_probability", 0.0)) if cnn_ok else None
            d = float(df_r.get("deepfake_probability",  0.0)) if df_ok  else None

            # ── Fallback: no signals at all ───────────────────────────────────
            if g is None and c is None:
                self.latest_result = {
                    "status":         "READY",
                    "gru_score":      0.0,
                    "cnn_score":      0.0,
                    "deepfake_score": d if d is not None else 0.0,
                    "fusion_score":   0.0,
                    "fusion_smooth":  0.0,
                    "final_status":   "LOW_CONFIDENCE",
                    "reason":         "waiting for GRU + CNN",
                }
                return

            # Substitute 0.0 for unavailable signals (conservative)
            g = g if g is not None else 0.0
            c = c if c is not None else 0.0
            d = d if d is not None else 0.0

            # ── 6-rule decision tree ──────────────────────────────────────────
            #
            # Rule 1: behavioral ground truth overrides all
            if g < 0.30:
                final_status = "SAFE"
                reason       = "behavior stable"

            # Rule 2: strong deepfake signal — requires behavioral corroboration
            elif d > 0.80 and g > 0.30:
                final_status = "HIGH_RISK"
                reason       = "deepfake detected"

            # Rule 3: combined behavioral + visual anomaly
            elif g > 0.60 and c > 0.60:
                final_status = "HIGH_RISK"
                reason       = "behavior + visual anomaly"

            # Rule 4: presentation attack (replay / screen)
            elif c > 0.85:
                final_status = "WARNING"
                reason       = "possible replay / screen"

            # Rule 5: suspicious visual signal despite stable behavior
            elif (c > 0.60 or d > 0.60) and g < 0.40:
                final_status = "WARNING"
                reason       = "visual anomaly but behavior stable"

            # Rule 6: default — nothing alarming
            else:
                final_status = "SAFE"
                reason       = "no strong anomaly"

            # ── 3-signal weighted fusion score ────────────────────────────────
            fusion_score = round(_GRU_W * g + _CNN_W * c + _DF_W * d, 4)

            # ── EMA smoothing ─────────────────────────────────────────────────
            if not self._has_smooth:
                self._fusion_smooth = fusion_score
                self._has_smooth    = True
            else:
                self._fusion_smooth = (
                    _EMA_ALPHA * fusion_score +
                    (1.0 - _EMA_ALPHA) * self._fusion_smooth
                )
            fusion_smooth = round(self._fusion_smooth, 4)

            print(
                f"FUSION: gru={g:.3f}  cnn={c:.3f}  df={d:.3f}  "
                f"score={fusion_score:.3f}  smooth={fusion_smooth:.3f}  "
                f"status={final_status}  reason={reason!r}"
            )

            # ── Cache result ──────────────────────────────────────────────────
            self.latest_result = {
                "status":         "READY",
                "gru_score":      round(g, 4),
                "cnn_score":      round(c, 4),
                "deepfake_score": round(d, 4),
                "fusion_score":   fusion_score,
                "fusion_smooth":  fusion_smooth,
                "final_status":   final_status,
                "reason":         reason,
            }

            # ── Publish FusionEvent ───────────────────────────────────────────
            from agent.event_bus import bus
            from agent.events import FusionEvent

            await bus.publish(FusionEvent(
                session_id     = event.session_id,
                frame_id       = event.frame_id,
                gru_score      = round(g, 4),
                cnn_score      = round(c, 4),
                deepfake_score = round(d, 4),
                fusion_score   = fusion_score,
                fusion_smooth  = fusion_smooth,
                final_status   = final_status,
                reason         = reason,
            ))

        except Exception as exc:
            logger.warning(f"FusionEngine._on_cnn_event error: {exc}", exc_info=True)
