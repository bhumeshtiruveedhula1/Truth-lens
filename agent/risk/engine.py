"""
risk/engine.py — Risk Engine

Processes FrameEvents through the LivenessPipeline and emits TrustEvents
with simple deterministic risk scoring.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from types import SimpleNamespace

import numpy as np

from agent.event_bus import bus
from agent.events import AlertEvent, FrameEvent, LivenessSignal, TrustEvent
from agent.liveness.pipeline import LivenessPipeline

logger = logging.getLogger(__name__)

EMIT_INTERVAL_SEC = 0.5
ALERT_COOLDOWN_SEC = 10.0
ROI_PAD = 20


class RiskEngine:
    """
    Temporal-confidence risk engine with FSM state machine.

    Pipeline: normalise signals → weighted fusion → EMA → temporal buffer → FSM.
    Hysteresis prevents oscillation at state boundaries.
    """

    WARMUP_FRAMES = 45              # ~1.5 sec at 30 fps
    OCCLUSION_GRACE_SEC = 2.0       # face missing < this → hold score
    BUFFER_SIZE = 15                # temporal buffer (~0.5 s at 30 fps)

    # Signal weights for weighted fusion (must sum to 1.0)
    # motion + temporal_consistency are the primary anti-spoof anchors.
    WEIGHTS = {
        "face":                0.22,
        "temporal":            0.15,
        "pose":                0.08,
        "blink":               0.05,
        "texture":             0.18,
        "motion":              0.18,
        "temporal_consistency": 0.14,
    }

    # EMA smoothing:  L_t = α · current + (1 − α) · previous
    EMA_ALPHA = 0.3

    # FSM thresholds & hysteresis
    SAFE_THRESHOLD = 0.7
    HIGH_RISK_ENTER = 0.3           # must drop below to enter HIGH_RISK
    HIGH_RISK_EXIT = 0.5            # must rise above to exit HIGH_RISK
    HIGH_RISK_SUSTAIN_SEC = 1.0     # seconds below entry before transition

    # Non-rigid motion consistency (rigid vs non-rigid decomposition)
    MOTION_WINDOW = 20              # landmark frames kept (~0.67 s at 30 fps)
    NR_VAR_SCALE  = 0.005           # variance that maps to score = 1.0
    NR_STILL_THRESH = 0.0005        # variance below this is "still"
    NR_STILL_SEC  = 3.0             # sustain before penalising
    # Key landmark indices: nose tip, inner eye corners, outer eye corners, mouth corners
    _MOTION_LM = [1, 33, 133, 263, 362, 61, 291]

    # Blink gating — anti-spoof validation
    BLINK_NR_MIN     = 0.0015       # NR_var must be >= this for blink to be accepted
    BLINK_RIGID_MAX  = 0.70         # rigid_ratio must be <= this
    BLINK_EAR_LOW    = 0.20         # EAR threshold for "eye closing"
    BLINK_EAR_HIGH   = 0.25         # EAR threshold for "eye re-opening"
    BLINK_CLOSE_MIN  = 2            # min consecutive frames EAR must be below thr_low
    BLINK_OPEN_MAX   = 10           # max frames allowed for eye to re-open
    BLINK_IBI_MIN    = 2.0          # min inter-blink interval (seconds)
    BLINK_IBI_MAX    = 10.0         # max inter-blink interval (seconds, beyond = no credit)

    # Temporal consistency (irregularity of motion history)
    TEMPORAL_HIST_SIZE   = 30       # ~1 sec at 30 fps
    TEMPORAL_IRR_SCALE   = 0.0005   # irregularity that maps to score = 1.0 (was 0.002)
    TEMPORAL_LOW_THRESH  = 0.08     # score below this triggers WARNING cap (was 0.15)
    TEMPORAL_LOW_SEC     = 2.0      # sustained duration before cap fires
    TEMPORAL_SAFE_SCORE  = 0.20     # min temporal_score to enter SAFE (was 0.30)
    TEMPORAL_MOTION_MIN  = 0.15     # min motion_score to enter SAFE (was 0.20)
    TEMPORAL_SAFE_SEC    = 1.0      # both must hold for this long

    # Rigid-motion veto
    RIGID_VETO_THRESH = 0.80        # rigid_ratio above this is suspicious
    RIGID_VETO_SEC    = 1.0         # must be sustained this long before veto fires

    def __init__(self, session_id: str, weights: dict = None, policy=None):
        self.session_id = session_id
        self.pipeline = LivenessPipeline()
        self._policy = policy
        self._last_alert_time = 0.0
        self.last_blink_time = 0.0
        self.last_movement_time = 0.0
        self._prev_score = 0.6
        self._frame_count = 0
        self._last_face_time = 0.0
        self._fsm_state = "WARMING_UP"
        self._high_risk_since = 0.0
        self._score_buffer: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._trust_history: collections.deque = collections.deque(maxlen=3600)
        # Motion tracking
        self._landmark_history: collections.deque = collections.deque(
            maxlen=self.MOTION_WINDOW
        )
        self._still_since: float = 0.0   # when sustained stillness started

        # Blink temporal validation state machine
        # States: "IDLE" | "CLOSING" | "OPEN"
        self._blink_state: str = "IDLE"
        self._blink_close_frames: int = 0   # frames EAR has been below thr_low
        self._blink_open_frames: int = 0    # frames since eye started re-opening
        self._blink_validated: bool = False # was the last detected blink validated?
        self._last_validated_blink: float = 0.0  # time of last validated blink

        # Rigid-motion veto
        self._rigid_high_since: float = 0.0  # when rigid_ratio first crossed threshold

        # Temporal consistency tracking
        self._motion_history: collections.deque = collections.deque(
            maxlen=self.TEMPORAL_HIST_SIZE
        )
        self._temporal_low_since: float = 0.0  # when temporal_score went below threshold
        self._safe_ready_since:   float = 0.0  # when conditions for SAFE first met

        self._running = False
        self._latest_result: dict | None = None

    def register(self) -> None:
        bus.subscribe(FrameEvent, self._on_frame)
        logger.info("RiskEngine registered")

    # ── Signal normalizers (each → 0.0–1.0) ────────────────────

    def _norm_face(self, face_detected: bool, ear: float,
                   yaw: float, pitch: float, now: float) -> float:
        """Face presence + landmark stability → [0, 1]."""
        if not face_detected:
            absent = now - self._last_face_time if self._last_face_time else 0.0
            if absent < self.OCCLUSION_GRACE_SEC:
                return 0.8  # grace period
            return max(0.0, 0.8 - 0.1 * (absent - self.OCCLUSION_GRACE_SEC))
        # Corrupted landmarks (hand, extreme occlusion)
        if ear < 0.05 or yaw > 40.0 or pitch > 40.0:
            return 0.3
        return 1.0

    def _is_low_confidence(self, face_detected: bool, face_score: float) -> bool:
        """True when tracking data is too unreliable for a risk verdict."""
        return not face_detected or face_score < 0.4

    def _norm_temporal(self, now: float) -> float:
        """Blink recency / temporal liveness → [0, 1]."""
        if self.last_blink_time == 0.0:
            return 0.4  # never blinked yet — neutral-low
        age = now - self.last_blink_time
        if age < 3.0:
            return 1.0
        if age < 6.0:
            return 1.0 - (age - 3.0) / 6.0   # 1.0 → 0.5
        if age < 12.0:
            return 0.5 - (age - 6.0) / 12.0  # 0.5 → 0.0
        return 0.0

    def _norm_pose(self, yaw: float, pitch: float,
                   face_detected: bool) -> float:
        """Head-pose naturalness → [0, 1]."""
        if not face_detected:
            return 0.5  # unknown — neutral
        if yaw < 20.0 and pitch < 20.0:
            return 1.0
        if yaw < 30.0 and pitch < 30.0:
            return max(0.0, 1.0 - (max(yaw, pitch) - 20.0) / 10.0)
        return 0.0

    def _norm_blink(self, validated_blink: bool, ear: float,
                    face_detected: bool) -> float:
        """
        Blink signal quality → [0, 1].
        Only a VALIDATED blink (gated by motion + temporal rules) gives full credit.
        """
        if not face_detected:
            return 0.5  # unknown — neutral
        if validated_blink:
            return 1.0
        if ear > 0.20:
            return 0.7  # eyes open — partial credit
        if ear > 0.10:
            return 0.5
        return 0.3

    def _norm_texture(self, texture_data: dict) -> float:
        """Texture anti-spoof → [0, 1]."""
        return 0.2 if texture_data["is_spoof"] else 1.0

    def _update_landmark_history(self, event: "FrameEvent") -> None:
        """Append a snapshot of key landmark positions to the rolling history."""
        lm = event.face_landmarks
        if not lm:
            return
        pts = []
        for idx in self._MOTION_LM:
            if idx < len(lm):
                pts.append((lm[idx][0], lm[idx][1]))
        if pts:
            self._landmark_history.append(np.array(pts, dtype=np.float32))

    def _compute_motion_score(self, now: float, face_detected: bool) -> tuple[float, float]:
        """
        Rigid vs non-rigid motion decomposition.

        Returns (nr_variance, still_duration).

        Algorithm (per consecutive frame pair):
          1. displacement_i = current_lm_i - previous_lm_i   (per landmark)
          2. global_d       = mean(displacement_i)            (rigid translation)
          3. residual_i     = displacement_i - global_d       (non-rigid component)
          4. frame_var      = mean(norm(residual_i))

        Aggregate: nr_variance = mean(frame_var) across the window.

        Why this works:
          * Moving phone  : all landmarks shift by ~same vector → residuals ≈ 0 → low score
          * Real human    : eyes/mouth move differently from nose → residuals > 0 → higher score
          * Static image  : no displacement at all → triggers still timer
        """
        if not face_detected or len(self._landmark_history) < 4:
            still_dur = now - self._still_since if self._still_since else 0.0
            return 0.0, still_dur

        history = list(self._landmark_history)
        frame_vars: list[float] = []

        for prev, curr in zip(history, history[1:]):
            if prev.shape != curr.shape:
                continue
            # (N, 2) displacement vectors
            disp = curr - prev                          # shape (N, 2)
            global_d = disp.mean(axis=0)                # shape (2,)  — rigid component
            residual = disp - global_d                  # shape (N, 2) — non-rigid component
            # Mean norm of residual vectors across landmarks
            frame_var = float(np.mean(np.linalg.norm(residual, axis=1)))
            frame_vars.append(frame_var)

        if not frame_vars:
            return 0.0, 0.0

        nr_var = float(np.mean(frame_vars))

        # Still timer: only accumulates when variance is truly negligible
        if nr_var < self.NR_STILL_THRESH:
            if self._still_since == 0.0:
                self._still_since = now
            still_dur = now - self._still_since
        else:
            self._still_since = 0.0
            still_dur = 0.0

        return nr_var, still_dur

    def _compute_rigid_metrics(self) -> tuple[float, float, float]:
        """
        Returns (nr_variance, mean_d_mag, rigid_ratio) from the landmark history.

        nr_variance : mean non-rigid residual norm (same as _compute_motion_score)
        mean_d_mag  : magnitude of the average global (rigid) displacement vector
        rigid_ratio : mean_d_mag / (mean_d_mag + nr_variance + 1e-9)
                      → 1.0 = purely rigid (phone waving)
                      → 0.0 = purely non-rigid (real expressions)
        """
        history = list(self._landmark_history)
        if len(history) < 4:
            return 0.0, 0.0, 0.0

        nr_vars, mean_d_mags = [], []
        for prev, curr in zip(history, history[1:]):
            if prev.shape != curr.shape:
                continue
            disp     = curr - prev
            global_d = disp.mean(axis=0)
            residual = disp - global_d
            nr_vars.append(float(np.mean(np.linalg.norm(residual, axis=1))))
            mean_d_mags.append(float(np.linalg.norm(global_d)))

        if not nr_vars:
            return 0.0, 0.0, 0.0

        nr_var    = float(np.mean(nr_vars))
        mean_d    = float(np.mean(mean_d_mags))
        rigid_ratio = mean_d / (mean_d + nr_var + 1e-9)
        return nr_var, mean_d, rigid_ratio

    def _validate_blink(self, raw_blink: bool, ear: float,
                        nr_var: float, rigid_ratio: float, now: float) -> bool:
        """
        Temporal blink state machine with motion gating.

        States: IDLE → CLOSING → OPEN → IDLE
        Accepts the blink only if:
          - EAR falls below BLINK_EAR_LOW for ≥ BLINK_CLOSE_MIN frames
          - EAR rises above BLINK_EAR_HIGH within BLINK_OPEN_MAX frames
          - NR_var ≥ BLINK_NR_MIN  (motion present — not a static image)
          - rigid_ratio ≤ BLINK_RIGID_MAX  (non-rigid motion — not a moving phone)
          - inter-blink interval ≥ BLINK_IBI_MIN
        """
        motion_ok = (nr_var >= self.BLINK_NR_MIN and
                     rigid_ratio <= self.BLINK_RIGID_MAX)
        validated = False

        if self._blink_state == "IDLE":
            if ear < self.BLINK_EAR_LOW:
                self._blink_state = "CLOSING"
                self._blink_close_frames = 1
                self._blink_open_frames = 0

        elif self._blink_state == "CLOSING":
            if ear < self.BLINK_EAR_LOW:
                self._blink_close_frames += 1
            else:
                # Eye started opening
                if self._blink_close_frames >= self.BLINK_CLOSE_MIN:
                    self._blink_state = "OPEN"
                    self._blink_open_frames = 0
                else:
                    self._blink_state = "IDLE"  # too brief — noise

        elif self._blink_state == "OPEN":
            self._blink_open_frames += 1
            if ear >= self.BLINK_EAR_HIGH:
                # Full blink cycle complete — apply gates
                ibi = now - self._last_validated_blink
                ibi_ok = (self._last_validated_blink == 0.0 or
                           self.BLINK_IBI_MIN <= ibi <= self.BLINK_IBI_MAX)
                if motion_ok and ibi_ok:
                    validated = True
                    self._last_validated_blink = now
                self._blink_state = "IDLE"
            elif self._blink_open_frames > self.BLINK_OPEN_MAX:
                self._blink_state = "IDLE"  # re-open window expired

        self._blink_validated = validated
        return validated

    def _apply_rigid_veto(self, rigid_ratio: float, now: float, status: str) -> str:
        """
        If rigid_ratio > RIGID_VETO_THRESH for > RIGID_VETO_SEC, cap state at WARNING.
        A moving phone cannot reach SAFE.
        """
        if rigid_ratio > self.RIGID_VETO_THRESH:
            if self._rigid_high_since == 0.0:
                self._rigid_high_since = now
            elif now - self._rigid_high_since > self.RIGID_VETO_SEC:
                if status == "SAFE":
                    return "WARNING"   # veto SAFE — too much rigid motion
        else:
            self._rigid_high_since = 0.0
        return status

    def _norm_motion(self, nr_var: float, still_dur: float) -> float:
        """
        Normalise non-rigid variance to [0, 1].

        Calibration targets (MediaPipe normalised coords):
          nr_var = 0.0            → score = 0.0  (static image)
          nr_var = 0.001          → score ≈ 0.20 (very subtle micro-motion)
          nr_var = 0.003          → score ≈ 0.60 (typical live human)
          nr_var >= NR_VAR_SCALE  → score = 1.0  (active movement)

        Penalty: if variance stays near-zero for > NR_STILL_SEC, score decays.
        """
        score = min(1.0, nr_var / self.NR_VAR_SCALE)

        if still_dur > self.NR_STILL_SEC:
            excess = still_dur - self.NR_STILL_SEC
            decay = min(0.8, excess / 10.0)   # -0.08/sec, max -0.8
            score = max(0.0, score - decay)

        return score

    def _norm_temporal_consistency(self, motion_raw: float,
                                   now: float) -> tuple[float, float]:
        """
        Append motion_raw to history; return (irregularity, temporal_score).

        Metric: mean absolute diff of successive values (irregularity).

        Why this is better than variance:
          Variance is near-zero for slow STEADY motion (real human moving slowly
          but uniformly) — this caused false negatives.
          Mean-abs-diff captures frame-to-frame CHANGE regardless of speed:
            Static image  : motion_raw ≈ 0 always     → diffs ≈ 0 → score = 0
            Moving phone  : motion_raw high but steady  → diffs ≈ 0 → score = 0
            Real human    : motion_raw fluctuates        → diffs > 0 → score > 0
        """
        self._motion_history.append(motion_raw)
        if len(self._motion_history) < 3:
            return 0.0, 0.5   # not enough history — neutral
        hist = list(self._motion_history)
        diffs = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        irregularity = float(np.mean(diffs))
        t_score = min(1.0, irregularity / self.TEMPORAL_IRR_SCALE)
        return irregularity, t_score

    def _apply_temporal_caps(self, status: str, t_score: float,
                             s_motion: float, face_detected: bool,
                             now: float) -> str:
        """
        Two temporal-consistency rules applied after FSM:

        Rule 1 — WARNING cap:
          If temporal_score < TEMPORAL_LOW_THRESH for > TEMPORAL_LOW_SEC:
          cap state at WARNING (never SAFE).

        Rule 2 — SAFE gate:
          SAFE is only allowed when:
            temporal_score > TEMPORAL_SAFE_SCORE
            AND motion_score > TEMPORAL_MOTION_MIN
            AND face_detected
          ... sustained for >= TEMPORAL_SAFE_SEC.
          If conditions are not met, demote SAFE → WARNING.
        """
        # ── Rule 1: Warning cap for persistent low temporal consistency ──
        if t_score < self.TEMPORAL_LOW_THRESH:
            if self._temporal_low_since == 0.0:
                self._temporal_low_since = now
            if now - self._temporal_low_since > self.TEMPORAL_LOW_SEC:
                if status == "SAFE":
                    status = "WARNING"
        else:
            self._temporal_low_since = 0.0

        # ── Rule 2: SAFE requires sustained liveness evidence ──
        if status == "SAFE":
            conditions_met = (
                t_score > self.TEMPORAL_SAFE_SCORE
                and s_motion > self.TEMPORAL_MOTION_MIN
                and face_detected
            )
            if conditions_met:
                if self._safe_ready_since == 0.0:
                    self._safe_ready_since = now
                if now - self._safe_ready_since < self.TEMPORAL_SAFE_SEC:
                    status = "WARNING"   # conditions met but not yet sustained
            else:
                self._safe_ready_since = 0.0
                status = "WARNING"       # conditions not met — deny SAFE
        else:
            # Reset safe-ready timer if we leave SAFE
            if status != "SAFE":
                self._safe_ready_since = 0.0

        return status

    # ── FSM with hysteresis ────────────────────────────────────

    def _update_fsm(self, score: float, now: float) -> str:
        """
        Update finite state machine.  Returns the new status string.

        Hysteresis:
          enter HIGH_RISK at score < 0.3 sustained > 1 s
          exit  HIGH_RISK at score > 0.5
        """
        if self._fsm_state == "HIGH_RISK":
            # Exit requires score above exit threshold
            if score > self.HIGH_RISK_EXIT:
                self._fsm_state = (
                    "SAFE" if score > self.SAFE_THRESHOLD else "WARNING"
                )
                self._high_risk_since = 0.0
            return self._fsm_state

        # Not in HIGH_RISK — check for entry
        if score < self.HIGH_RISK_ENTER:
            if self._high_risk_since == 0.0:
                self._high_risk_since = now
            elif now - self._high_risk_since > self.HIGH_RISK_SUSTAIN_SEC:
                self._fsm_state = "HIGH_RISK"
                return self._fsm_state
        else:
            self._high_risk_since = 0.0

        # Normal state transitions
        self._fsm_state = (
            "SAFE" if score > self.SAFE_THRESHOLD else "WARNING"
        )
        return self._fsm_state

    # ── Main frame handler ─────────────────────────────────────

    async def _on_frame(self, event: FrameEvent) -> None:
        now = event.timestamp
        self._frame_count += 1

        frame_bgr, landmarks, w, h = self._build_pipeline_inputs(event)
        face_detected = landmarks is not None

        if face_detected:
            self._last_face_time = now

        signals = self.pipeline.process(frame_bgr, landmarks, w, h)

        blink = signals["blink"]
        head_pose = signals["head_pose"]
        texture = signals["texture"]
        yaw = abs(float(head_pose["yaw"]))
        pitch = abs(float(head_pose["pitch"]))
        ear = float(blink.get("ear", 0.0))

        if blink["blink_detected"]:
            self.last_blink_time = now

        # Update landmark history for motion tracking
        if face_detected:
            self._update_landmark_history(event)

        # Compute rigid metrics from current history
        nr_var_now, mean_d_now, rigid_ratio = self._compute_rigid_metrics()

        # Validated blink: temporal state machine + motion gate
        validated_blink = self._validate_blink(
            blink["blink_detected"], ear, nr_var_now, rigid_ratio, now
        )
        if validated_blink:
            self.last_blink_time = now  # update temporal signal on validated blink only

        movement_mag = max(yaw, pitch)
        if 3.0 <= movement_mag <= 20.0:
            self.last_movement_time = now

        # ── Warmup phase ──
        if self._frame_count <= self.WARMUP_FRAMES:
            score = 0.6
            status = "WARMING_UP"
            self._prev_score = score
            self._score_buffer.append(score)
            # Seed blink timer at end of warmup so user gets grace window
            if self._frame_count == self.WARMUP_FRAMES and self.last_blink_time == 0.0:
                self.last_blink_time = now

            blink_age = 0.0
            movement_age = 0.0
            fused = score
        else:
            # ── Step 1: Normalize all signals to [0, 1] ──
            s_face     = self._norm_face(face_detected, ear, yaw, pitch, now)
            s_temporal = self._norm_temporal(now)
            s_pose     = self._norm_pose(yaw, pitch, face_detected)
            s_blink    = self._norm_blink(validated_blink, ear, face_detected)
            s_texture  = self._norm_texture(texture)

            # Motion consistency: primary anti-static-image signal
            raw_motion, still_dur = self._compute_motion_score(now, face_detected)
            s_motion = self._norm_motion(raw_motion, still_dur)

            # Temporal consistency: variance of motion history
            t_var, t_score = self._norm_temporal_consistency(raw_motion, now)

            # ── Step 2: Weighted fusion ──
            W = self.WEIGHTS
            fused = (
                W["face"]                * s_face
                + W["temporal"]          * s_temporal
                + W["pose"]              * s_pose
                + W["blink"]             * s_blink
                + W["texture"]           * s_texture
                + W["motion"]            * s_motion
                + W["temporal_consistency"] * t_score
            )

            # ── Step 3: EMA smoothing ──
            score = self.EMA_ALPHA * fused + (1.0 - self.EMA_ALPHA) * self._prev_score
            score = max(0.0, min(1.0, score))

            # ── Human floor: face + temporal + motion + temporal_consistency ──
            # All three liveness anchors must be present — spoof cannot satisfy all.
            if (s_face >= 0.8 and s_temporal >= 0.7
                    and s_motion >= 0.25 and t_score >= self.TEMPORAL_SAFE_SCORE):
                score = max(score, 0.72)

            self._prev_score = score

            # ── Step 4: Temporal buffer ──
            self._score_buffer.append(score)

            # ── Step 5 & 6: FSM with hysteresis + veto layers ──
            if self._is_low_confidence(face_detected, s_face):
                status = "LOW_CONFIDENCE"
                self._high_risk_since = 0.0
            else:
                status = self._update_fsm(score, now)
                status = self._apply_rigid_veto(rigid_ratio, now, status)
                status = self._apply_temporal_caps(status, t_score, s_motion,
                                                   face_detected, now)

            blink_age = (
                now - self.last_blink_time if self.last_blink_time else float("inf")
            )
            movement_age = (
                now - self.last_movement_time if self.last_movement_time else float("inf")
            )

        self._latest_result = {
            "score": score,
            "status": status,
            "signals": {
                "blink": blink,
                "head_pose": head_pose,
                "texture": texture,
            },
            "memory": {
                "blink_age": blink_age,
                "movement_age": movement_age,
                "raw_score": fused,
                "motion_raw": raw_motion if self._frame_count > self.WARMUP_FRAMES else 0.0,
                "motion_score": s_motion if self._frame_count > self.WARMUP_FRAMES else 0.5,
                "still_duration": still_dur if self._frame_count > self.WARMUP_FRAMES else 0.0,
                "rigid_ratio": rigid_ratio if self._frame_count > self.WARMUP_FRAMES else 0.0,
                "blink_validated": validated_blink if self._frame_count > self.WARMUP_FRAMES else False,
                "blink_state": self._blink_state,
                "temporal_var": t_var if self._frame_count > self.WARMUP_FRAMES else 0.0,
                "temporal_score": t_score if self._frame_count > self.WARMUP_FRAMES else 0.5,
            },
        }

        # ── ML signal bus emissions (pure side-effect, no logic impact) ──────
        # Emitted AFTER all scoring is complete so they never affect the engine.
        # BlinkDetector and HeadPoseEstimator already emit their own signals via
        # the base extractor path.  Motion, temporal, and texture are only
        # available here, so we publish them for the ML logger.
        if self._frame_count > self.WARMUP_FRAMES:
            _sid = event.session_id
            asyncio.ensure_future(bus.publish(LivenessSignal(
                session_id=_sid,
                extractor_name="motion",
                value=raw_motion,
                score=s_motion,
                confidence=1.0,
                metadata={
                    "motion_raw":   round(raw_motion, 6),
                    "motion_score": round(s_motion, 4),
                    "rigid_ratio":  round(rigid_ratio, 4),
                },
            )))
            asyncio.ensure_future(bus.publish(LivenessSignal(
                session_id=_sid,
                extractor_name="temporal_consistency",
                value=t_var,
                score=t_score,
                confidence=1.0,
                metadata={
                    "irregularity":   round(t_var, 6),
                    "temporal_score": round(t_score, 4),
                },
            )))
            asyncio.ensure_future(bus.publish(LivenessSignal(
                session_id=_sid,
                extractor_name="texture",
                value=float(texture.get("laplacian_score", 0.0)),
                score=s_texture,
                confidence=1.0,
                metadata={
                    "texture_score": round(float(texture.get("laplacian_score", 0.0)), 4),
                    "lbp_score":     round(float(texture.get("lbp_score", 0.0)), 4),
                    "is_spoof":      int(bool(texture.get("is_spoof", False))),
                },
            )))

    def _build_pipeline_inputs(
        self,
        event: FrameEvent,
    ) -> tuple[np.ndarray, object | None, int, int]:
        if not event.face_roi_bgr or not event.face_roi_shape:
            frame_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
            return frame_bgr, None, 1, 1

        roi_h, roi_w = event.face_roi_shape
        frame_bgr = np.frombuffer(event.face_roi_bgr, dtype=np.uint8).reshape(roi_h, roi_w, 3)

        if not event.face_detected or not event.face_landmarks:
            return frame_bgr, None, roi_w, roi_h

        x, y, w, h = event.face_bbox
        rx1 = max(0, x - ROI_PAD)
        ry1 = max(0, y - ROI_PAD)

        adapted = []
        for px, py, pz in event.face_landmarks:
            full_x = px * event.frame_width
            full_y = py * event.frame_height
            roi_x = (full_x - rx1) / max(roi_w, 1)
            roi_y = (full_y - ry1) / max(roi_h, 1)
            adapted.append(
                SimpleNamespace(
                    x=float(min(max(roi_x, 0.0), 1.0)),
                    y=float(min(max(roi_y, 0.0), 1.0)),
                    z=float(pz),
                )
            )

        return frame_bgr, SimpleNamespace(landmark=adapted), roi_w, roi_h

    async def run(self) -> None:
        self._running = True
        logger.info("RiskEngine emission loop started")
        while self._running:
            await asyncio.sleep(EMIT_INTERVAL_SEC)
            try:
                await self._emit()
            except Exception as exc:
                logger.error(f"RiskEngine emit error: {exc}", exc_info=True)

    async def _emit(self) -> None:
        if self._latest_result is None:
            return

        now = time.time()
        score = self._latest_result["score"]
        status = self._latest_result["status"]
        signals = self._latest_result["signals"]
        print(f"[RISK] score={round(score,2)} status={status}")

        alert = False
        alert_reason = None
        if status == "HIGH_RISK":
            alert_reason = "Pipeline signaled high spoof risk"
            if now - self._last_alert_time > ALERT_COOLDOWN_SEC:
                self._last_alert_time = now
                alert = True
                await bus.publish(
                    AlertEvent(
                        session_id=self.session_id,
                        trust_score=score,
                        primary_trigger="pipeline",
                        recommended_action="warn_user",
                    )
                )

        trust_event = TrustEvent(
            session_id=self.session_id,
            trust_score=score,
            risk_level=status,
            contributing_signals={
                "blink": float(not signals["blink"]["blink_detected"]),
                "head_pose": max(
                    abs(float(signals["head_pose"]["yaw"])),
                    abs(float(signals["head_pose"]["pitch"])),
                ),
                "texture": float(bool(signals["texture"]["is_spoof"])),
            },
            alert=alert,
            alert_reason=alert_reason,
        )

        self._trust_history.append((now, score))
        await bus.publish(trust_event)

    def stop(self) -> None:
        self._running = False

    @property
    def trust_history(self) -> list[tuple[float, float]]:
        return list(self._trust_history)
