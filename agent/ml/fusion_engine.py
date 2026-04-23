"""
agent/ml/fusion_engine.py -- GRU + CNN Fusion Layer
=====================================================
Combines GRU behavioral score and CNN visual score into a single,
stable, interpretable final decision.

Design:
  - Subscribes to CNNEvent and polls GRUInferenceEngine.latest_result
  - Zero coupling to model internals — reads published scores only
  - All logic is pure arithmetic (no model calls, non-blocking)
  - Publishes FusionEvent on every CNN update (~10 Hz)
  - Caches latest_result dict for DebugUI polling

Fusion rules (in priority order):
  CASE 1: gru < 0.30                          → SAFE      "behavior stable"
  CASE 2: gru > 0.60 AND cnn > 0.60           → HIGH_RISK "both signals indicate risk"
  CASE 3: cnn > 0.85                           → WARNING   "visual anomaly detected"
  CASE 4: gru < 0.30 AND cnn > 0.60           → WARNING   "visual anomaly but behavior normal"
  DEFAULT:                                      → SAFE      "normal conditions"

Fusion score:
  fusion_score  = 0.6 * gru + 0.4 * cnn
  fusion_smooth = EMA(alpha=0.4, fusion_score)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GRU_WEIGHT = 0.6
_CNN_WEIGHT = 0.4
_EMA_ALPHA  = 0.4          # smoothing factor (higher = more reactive)


class FusionEngine:
    """
    Stateless fusion layer — holds only the latest scores and EMA state.
    Subscribes to CNNEvent; reads GRU score from injected engine reference.
    """

    def __init__(self, gru_engine, cnn_engine) -> None:
        self._gru_engine = gru_engine   # GRUInferenceEngine — read .latest_result
        self._cnn_engine = cnn_engine   # CNNInferenceEngine — read .latest_result

        # Score cache
        self._gru_score: float = 0.0
        self._cnn_score: float = 0.0
        self._gru_ready: bool  = False
        self._cnn_ready: bool  = False

        # EMA state
        self._fusion_smooth: float = 0.0
        self._has_smooth:    bool  = False

        # Latest result — polled by DebugUI
        self.latest_result: dict = {
            "status":        "LOADING",
            "gru_score":     0.0,
            "cnn_score":     0.0,
            "fusion_score":  0.0,
            "fusion_smooth": 0.0,
            "final_status":  "LOW_CONFIDENCE",
            "reason":        "waiting for signals",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Event bus wiring
    # ─────────────────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to CNNEvent.  GRU score is polled on each CNN update."""
        try:
            from agent.event_bus import bus
            from agent.events import CNNEvent

            bus.subscribe(CNNEvent, self._on_cnn_event)
            logger.info("FusionEngine registered on event bus")
            print("FUSION: registered on event bus")
        except Exception as exc:
            print(f"FUSION: register FAILED — {exc}")

    async def _on_cnn_event(self, event) -> None:
        """
        Triggered at ~10 Hz by CNNInferenceEngine.
        Reads latest GRU score, computes fusion, publishes FusionEvent.
        Pure arithmetic — never blocks.
        """
        try:
            # ── Pull latest scores ────────────────────────────────────────────
            cnn_result = self._cnn_engine.latest_result
            gru_result = self._gru_engine.latest_result

            cnn_status = cnn_result.get("status", "LOADING")
            gru_status = gru_result.get("status", "INSUFFICIENT_DATA")

            cnn_ok = cnn_status == "READY"
            gru_ok = gru_status == "READY"

            cnn_prob = float(cnn_result.get("cnn_fake_probability", 0.0)) if cnn_ok else None
            gru_prob = float(
                gru_result.get("smoothed_score",
                gru_result.get("fake_probability", 0.0))
            ) if gru_ok else None

            # ── Fallback handling ─────────────────────────────────────────────
            if cnn_prob is None and gru_prob is None:
                self.latest_result = {
                    "status":        "READY",
                    "gru_score":     0.0,
                    "cnn_score":     0.0,
                    "fusion_score":  0.0,
                    "fusion_smooth": 0.0,
                    "final_status":  "LOW_CONFIDENCE",
                    "reason":        "no signals available",
                }
                return

            if gru_prob is None:
                # Only CNN available — conservative: treat as WARNING only if very high
                g = 0.0
                c = cnn_prob
                final_status = "WARNING" if c > 0.85 else "SAFE"
                reason = "CNN only (GRU buffering)"
            elif cnn_prob is None:
                # Only GRU available — trust it fully
                g = gru_prob
                c = 0.0
                final_status = "HIGH_RISK" if g > 0.60 else ("WARNING" if g > 0.40 else "SAFE")
                reason = "GRU only (CNN loading)"
            else:
                g = gru_prob
                c = cnn_prob
                # ── Fusion decision rules (priority order) ────────────────────
                if g < 0.30:
                    final_status = "SAFE"
                    reason       = "behavior stable"
                elif g > 0.60 and c > 0.60:
                    final_status = "HIGH_RISK"
                    reason       = "both signals indicate risk"
                elif c > 0.85:
                    final_status = "WARNING"
                    reason       = "visual anomaly detected"
                elif g < 0.30 and c > 0.60:
                    final_status = "WARNING"
                    reason       = "visual anomaly but behavior normal"
                else:
                    final_status = "SAFE"
                    reason       = "normal conditions"

            # ── Fusion score ──────────────────────────────────────────────────
            fusion_score = round(_GRU_WEIGHT * g + _CNN_WEIGHT * c, 4)

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
                f"FUSION: gru={g:.3f}  cnn={c:.3f}  "
                f"score={fusion_score:.3f}  smooth={fusion_smooth:.3f}  "
                f"status={final_status}  reason={reason!r}"
            )

            # ── Update cached result ──────────────────────────────────────────
            self.latest_result = {
                "status":        "READY",
                "gru_score":     round(g, 4),
                "cnn_score":     round(c, 4),
                "fusion_score":  fusion_score,
                "fusion_smooth": fusion_smooth,
                "final_status":  final_status,
                "reason":        reason,
            }

            # ── Publish FusionEvent ───────────────────────────────────────────
            from agent.event_bus import bus
            from agent.events import FusionEvent

            await bus.publish(FusionEvent(
                session_id    = event.session_id,
                frame_id      = event.frame_id,
                gru_score     = round(g, 4),
                cnn_score     = round(c, 4),
                fusion_score  = fusion_score,
                fusion_smooth = fusion_smooth,
                final_status  = final_status,
                reason        = reason,
            ))

        except Exception as exc:
            logger.warning(f"FusionEngine._on_cnn_event error: {exc}", exc_info=True)
