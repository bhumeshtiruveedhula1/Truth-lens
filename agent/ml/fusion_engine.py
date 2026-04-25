"""
agent/ml/fusion_engine.py -- GRU + CNN + Deepfake Fusion Layer
===============================================================
Combines three independent intelligence signals into a single,
stable, interpretable final decision.

Signals:
  g = gru_fake_probability   (behavioral — primary ground truth)
  c = cnn_fake_probability   (presentation attack detection)
  d = deepfake_probability   (artifact / identity manipulation)

Decision rules (evaluated in strict RISK-FIRST priority order):

  Rule 0 — Definitive presentation attack (CNN alone is conclusive):
    IF c > 0.85 AND d > 0.30
    → HIGH_RISK  "presentation attack confirmed"

  Rule 1 — Strong REAL (behavioral ground truth — LAST, not first):
    ELIF g < 0.30 AND c < 0.30 AND d < 0.40
    → SAFE  "behavior stable"
    NOTE: GRU only gates SAFE when CNN + DF are also low.
          GRU can NEVER suppress a strong CNN or DF signal.

  Rule 2 — Strong deepfake regardless of GRU buffer state:
    ELIF d > 0.70 AND g > 0.20
    → HIGH_RISK  "deepfake detected"

  Rule 2b — Strong deepfake + strong CNN (GRU buffer filling):
    ELIF d > 0.70 AND c > 0.60
    → HIGH_RISK  "deepfake + visual anomaly"

  Rule 3 — Combined behavioral + visual anomaly:
    ELIF g > 0.60 AND c > 0.60
    → HIGH_RISK  "behavior + visual anomaly"

  Rule 4 — Presentation attack (replay / screen — softer signal):
    ELIF c > 0.85
    → WARNING  "possible replay / screen"

  Rule 5 — Synthetic consistency (temporal uniformity anomaly):
    ELIF var_gru < 0.002 AND var_df < 0.01
         AND 0.30 < d < 0.70 AND c < 0.30
    → WARNING  "synthetic consistency detected"

  Rule 6 — Visual elevated but behavior stable:
    ELIF (c > 0.60 OR d > 0.60) AND g < 0.40
    → WARNING  "visual anomaly but behavior stable"

  Rule 7 — Default:
    → SAFE  "no strong anomaly"

Fusion score (3-signal weighted):
  fusion_score  = 0.50*g + 0.25*c + 0.25*d
  fusion_smooth = EMA(alpha=0.35, fusion_score)

Consistency layer:
  Rolling window of 25 GRU + DF values.
  Low variance in both = temporally uniform = synthetic signal flag.
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)

_GRU_W     = 0.50
_CNN_W     = 0.25
_DF_W      = 0.25
_EMA_ALPHA = 0.35

# Consistency layer config
_CONSISTENCY_WINDOW   = 25      # rolling buffer length (frames)
_VAR_GRU_THRESH       = 0.002   # variance threshold for GRU
_VAR_DF_THRESH        = 0.010   # variance threshold for DF
_DF_MID_LOW           = 0.30    # mid-range DF lower bound
_DF_MID_HIGH          = 0.70    # mid-range DF upper bound
_CNN_LOW_THRESH        = 0.30   # CNN must be low for consistency rule


def _variance(buf: deque) -> float:
    """Population variance of a deque of floats."""
    if len(buf) < 2:
        return 999.0   # not enough data → do NOT trigger rule
    m = sum(buf) / len(buf)
    return sum((x - m) ** 2 for x in buf) / len(buf)


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

        # Consistency layer rolling buffers
        self._gru_buf: deque[float] = deque(maxlen=_CONSISTENCY_WINDOW)
        self._df_buf:  deque[float] = deque(maxlen=_CONSISTENCY_WINDOW)

        # Latest result — polled by DebugUI and /result endpoint
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
            print("FUSION: registered (GRU + CNN + Deepfake)  [risk-first + consistency layer]")
        except Exception as exc:
            print(f"FUSION: register FAILED — {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Core inference
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_cnn_event(self, event) -> None:
        """
        Triggered at ~10 Hz by CNNInferenceEngine.
        Polls all three engine results, computes 3-signal fusion, publishes FusionEvent.
        """
        try:
            # ── Pull latest scores ────────────────────────────────────────────
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

            # ── Update consistency buffers ────────────────────────────────────
            self._gru_buf.append(g)
            self._df_buf.append(d)
            var_gru = _variance(self._gru_buf)
            var_df  = _variance(self._df_buf)

            # ── RISK-FIRST decision tree (8 rules) ────────────────────────────

            # Rule 0 — Definitive presentation attack (CNN conclusive, DF corroborates)
            # GRU cannot suppress this — video replay always shows static behavior.
            if c > 0.85 and d > 0.30:
                final_status = "HIGH_RISK"
                reason       = "presentation attack confirmed"

            # Rule 1 — Strong REAL: GRU gates SAFE only when CNN + DF are also quiet.
            # CRITICAL: GRU < 0.30 alone is NOT enough for SAFE.
            elif g < 0.30 and c < 0.30 and d < 0.40:
                final_status = "SAFE"
                reason       = "behavior stable"

            # Rule 2 — Strong deepfake + any behavioral signal above floor.
            # Handles GRU-buffer-filling window (g > 0.20 means GRU is loading,
            # not that the person is real).
            elif d > 0.70 and g > 0.20:
                final_status = "HIGH_RISK"
                reason       = "deepfake detected"

            # Rule 2b — Strong deepfake + strong CNN (covers GRU=0.0 startup).
            elif d > 0.70 and c > 0.60:
                final_status = "HIGH_RISK"
                reason       = "deepfake + visual anomaly"

            # Rule 3 — Combined behavioral + visual anomaly
            elif g > 0.60 and c > 0.60:
                final_status = "HIGH_RISK"
                reason       = "behavior + visual anomaly"

            # Rule 4 — Presentation attack soft signal
            elif c > 0.85:
                final_status = "WARNING"
                reason       = "possible replay / screen"

            # Rule 5 — Synthetic consistency: low variance in both GRU and DF
            # over the last 25 frames signals unnatural temporal uniformity.
            # OBS-injected deepfakes appear consistent; real faces have micro-variation.
            elif (
                var_gru < _VAR_GRU_THRESH
                and var_df  < _VAR_DF_THRESH
                and _DF_MID_LOW < d < _DF_MID_HIGH
                and c < _CNN_LOW_THRESH
                and len(self._gru_buf) >= _CONSISTENCY_WINDOW
            ):
                final_status = "WARNING"
                reason       = "synthetic consistency detected"

            # Rule 6 — Visual elevated but behavior stable
            elif (c > 0.60 or d > 0.60) and g < 0.40:
                final_status = "WARNING"
                reason       = "visual anomaly but behavior stable"

            # Rule 7 — Default
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
                f"vg={var_gru:.4f}  vd={var_df:.4f}  "
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
                "var_gru":        round(var_gru, 6),
                "var_df":         round(var_df,  6),
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
