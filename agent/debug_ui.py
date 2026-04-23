"""
agent/debug_ui.py — Lightweight OpenCV debug overlay
─────────────────────────────────────────────────────
Subscribes to FrameEvents, reads the latest risk-engine result,
and renders a live camera preview with debug telemetry overlay.

Activated via --debug-ui flag.  Zero impact on the pipeline when off.
Only dependency: cv2 (already in the project).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np

from agent.event_bus import bus
from agent.events import FrameEvent

logger = logging.getLogger(__name__)

# ── MediaPipe FaceMesh landmark subsets for drawing ────────────
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]
LEFT_EYE  = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]
MOUTH     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61]

# ── Colour palette ─────────────────────────────────────────────
STATUS_COLOURS = {
    "SAFE":           (0, 200, 80),    # green
    "WARNING":        (0, 180, 255),   # orange
    "HIGH_RISK":      (0, 0, 230),     # red
    "LOW_CONFIDENCE": (180, 180, 0),   # teal
    "WARMING_UP":     (200, 160, 0),   # blue-ish
}
PANEL_BG   = (30, 30, 30)
TEXT_WHITE  = (240, 240, 240)
TEXT_DIM    = (160, 160, 160)
LANDMARK_C = (0, 255, 200)
BBOX_C     = (255, 200, 0)


class DebugUI:
    """
    OpenCV-based debug overlay.  Call ``register()`` to subscribe to the
    event bus, then ``run()`` as an asyncio task.
    """

    WINDOW_NAME = "DeepShield Debug"

    def __init__(self, risk_engine, capture_source, gru_engine=None, cnn_engine=None, fusion_engine=None) -> None:
        self._risk_engine   = risk_engine
        self._capture       = capture_source
        self._gru_engine    = gru_engine
        self._cnn_engine    = cnn_engine
        self._fusion_engine = fusion_engine
        self._running = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_event: Optional[FrameEvent] = None

    def register(self) -> None:
        bus.subscribe(FrameEvent, self._on_frame)
        logger.info("DebugUI registered")

    async def _on_frame(self, event: FrameEvent) -> None:
        """Stash last event metadata (landmarks, bbox, face_detected)."""
        self._last_event = event

    async def run(self) -> None:
        """Main render loop — runs at ~30 fps alongside the pipeline."""
        self._running = True
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 800, 600)
        logger.info("DebugUI render loop started")

        while self._running:
            frame = self._grab_frame()
            if frame is not None:
                overlay = self._render(frame)
                cv2.imshow(self.WINDOW_NAME, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._running = False
                break

            await asyncio.sleep(1 / 30)

        cv2.destroyWindow(self.WINDOW_NAME)

    def stop(self) -> None:
        self._running = False

    # ── Frame acquisition ──────────────────────────────────────

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Read the latest raw frame from the capture source."""
        cap = getattr(self._capture, "_cap", None)
        if cap is None or not cap.isOpened():
            return self._last_frame  # return stale frame if cam busy

        # The capture source already reads frames in its own loop.
        # We reconstruct the full frame from the stored reference.
        raw = getattr(self._capture, "_last_raw_frame", None)
        if raw is not None:
            self._last_frame = raw.copy()
        return self._last_frame

    # ── Rendering ──────────────────────────────────────────────

    def _render(self, frame: np.ndarray) -> np.ndarray:
        """Compose the final overlay frame."""
        out = frame.copy()
        event = self._last_event
        result = self._risk_engine._latest_result

        # Draw landmarks + bbox if available
        if event and event.face_detected and event.face_landmarks:
            self._draw_landmarks(out, event)
            self._draw_bbox(out, event)

        # Draw telemetry panel
        self._draw_panel(out, event, result)

        return out

    def _draw_landmarks(self, img: np.ndarray, event: FrameEvent) -> None:
        h, w = img.shape[:2]
        lm = event.face_landmarks

        def _draw_strip(indices, colour, thickness=1):
            pts = []
            for i in indices:
                if i < len(lm):
                    x = int(lm[i][0] * w)
                    y = int(lm[i][1] * h)
                    pts.append((x, y))
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)

        _draw_strip(FACE_OVAL, (100, 200, 100), 1)
        _draw_strip(LEFT_EYE,  (0, 255, 255), 1)
        _draw_strip(RIGHT_EYE, (0, 255, 255), 1)
        _draw_strip(MOUTH,     (200, 100, 255), 1)

        # Draw sparse landmark dots
        for i in range(0, len(lm), 6):
            x = int(lm[i][0] * w)
            y = int(lm[i][1] * h)
            cv2.circle(img, (x, y), 1, LANDMARK_C, -1, cv2.LINE_AA)

    def _draw_bbox(self, img: np.ndarray, event: FrameEvent) -> None:
        x, y, bw, bh = event.face_bbox
        cv2.rectangle(img, (x, y), (x + bw, y + bh), BBOX_C, 2, cv2.LINE_AA)

    def _draw_panel(self, img: np.ndarray, event: Optional[FrameEvent],
                    result: Optional[dict]) -> None:
        """Draw the semi-transparent telemetry panel on the left side."""
        h, w = img.shape[:2]
        panel_w = 280
        panel_h = 420  # extended for ML inference rows

        # Semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay, (4, 4), (panel_w, panel_h), PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        y_pos = 26
        line_h = 22

        def _put(label: str, value: str, colour=TEXT_WHITE):
            nonlocal y_pos
            cv2.putText(img, f"{label}:", (12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(img, str(value), (130, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            y_pos += line_h

        # ── Status + Score ──
        if result:
            status = result.get("status", "—")
            score  = result.get("score", 0.0)
            sc = STATUS_COLOURS.get(status, TEXT_WHITE)

            # Big status badge
            cv2.putText(img, status, (12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2, cv2.LINE_AA)
            cv2.putText(img, f"{score:.2f}", (200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2, cv2.LINE_AA)
            y_pos += line_h + 6

            # ── Score bar ──
            bar_x, bar_y, bar_w, bar_h = 12, y_pos - 4, panel_w - 24, 10
            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            fill = int(bar_w * max(0, min(1, score)))
            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + fill, bar_y + bar_h), sc, -1)
            y_pos += line_h

            signals = result.get("signals", {})
            blink = signals.get("blink", {})
            head  = signals.get("head_pose", {})
            tex   = signals.get("texture", {})

            # ── Signal lines ──
            ear = blink.get("ear", 0.0)
            blink_det = blink.get("blink_detected", False)
            _put("Face", "DETECTED" if (event and event.face_detected) else "MISSING",
                 (0, 220, 0) if (event and event.face_detected) else (0, 0, 220))

            blink_str = f"{'YES' if blink_det else 'no'}  EAR={ear:.3f}"
            _put("Blink", blink_str,
                 (0, 255, 100) if blink_det else TEXT_WHITE)

            yaw   = head.get("yaw", 0.0)
            pitch = head.get("pitch", 0.0)
            _put("Yaw / Pitch", f"{yaw:+.1f}  /  {pitch:+.1f}")

            direction = head.get("direction", "—")
            _put("Direction", direction)

            spoof = tex.get("is_spoof", False)
            lap   = tex.get("laplacian_score", 0.0)
            _put("Texture", f"{'SPOOF' if spoof else 'OK'}  lap={lap:.0f}",
                 (0, 0, 220) if spoof else (0, 220, 0))

            memory = result.get("memory", {})

            # ── Non-rigid motion (rigid vs non-rigid decomposition) ──
            nr_var  = memory.get("motion_raw", 0.0)   # raw residual variance
            m_score = memory.get("motion_score", 0.5)  # normalised 0–1
            still_d = memory.get("still_duration", 0.0)
            if m_score >= 0.5:
                m_colour = (0, 200, 80)     # green — non-rigid motion detected
            elif m_score >= 0.2:
                m_colour = (0, 180, 255)    # orange — low non-rigid motion
            else:
                m_colour = (0, 0, 230)      # red — rigid/static suspected
            still_str = f" still:{still_d:.1f}s" if still_d > 0.5 else ""
            _put("NR-var", f"{nr_var:.5f}  sc={m_score:.2f}{still_str}", m_colour)

            # ── Temporal consistency (irregularity = mean-abs-diff of motion history) ──
            irr     = memory.get("temporal_var", 0.0)   # key kept for compat
            t_score = memory.get("temporal_score", 0.5)
            if t_score >= 0.20:
                tc_colour = (0, 200, 80)    # green — natural irregular motion
            elif t_score >= 0.08:
                tc_colour = (0, 180, 255)   # orange — borderline
            else:
                tc_colour = (0, 0, 230)     # red — flat/static/rigid signal
            _put("Irregularity", f"{irr:.6f}  sc={t_score:.2f}", tc_colour)

            # ── Rigid ratio (1.0 = purely rigid/phone, 0.0 = purely non-rigid/live) ──
            rigid_r = memory.get("rigid_ratio", 0.0)
            if rigid_r <= 0.5:
                rr_colour = (0, 200, 80)    # green — mostly non-rigid
            elif rigid_r <= 0.75:
                rr_colour = (0, 180, 255)   # orange — borderline
            else:
                rr_colour = (0, 0, 230)     # red — rigid/phone suspected
            _put("Rigid ratio", f"{rigid_r:.3f}", rr_colour)

            # ── Blink validation status ──
            blink_valid = memory.get("blink_validated", False)
            blink_st    = memory.get("blink_state", "IDLE")
            if blink_valid:
                bv_str    = "VALIDATED"
                bv_colour = (0, 220, 80)
            else:
                bv_str    = f"ignored [{blink_st}]"
                bv_colour = TEXT_DIM
            _put("Blink gate", bv_str, bv_colour)

            # ── Last detected blink (informational only) ──
            blink_age = memory.get("blink_age", 0.0)
            if blink_age == float("inf"):
                _put("Last blink", "never detected", TEXT_DIM)
            else:
                _put("Last blink", f"{blink_age:.1f} sec ago", TEXT_DIM)

            # ── GRU ML inference block ───────────────────────────────────
            # Separator line
            cv2.line(img, (12, y_pos - 6), (panel_w - 12, y_pos - 6),
                     (70, 70, 70), 1, cv2.LINE_AA)

            ml = self._gru_engine.latest_result if self._gru_engine else None

            if ml is None:
                _put("ML", "disabled", TEXT_DIM)
            else:
                ml_status    = ml.get("status", "INSUFFICIENT_DATA")
                ml_raw       = ml.get("raw_score",      ml.get("fake_probability", 0.0))
                ml_smooth    = ml.get("smoothed_score", ml.get("fake_probability", 0.0))
                ml_label     = ml.get("fake_label", "REAL")

                # Colour coding — driven by smoothed score and label
                if ml_status == "INSUFFICIENT_DATA":
                    ml_col = (140, 140, 140)   # gray
                elif ml_label == "FAKE":
                    ml_col = (0, 0, 220)        # red  (BGR)
                else:
                    ml_col = (0, 200, 80)       # green

                status_str = "READY" if ml_status == "READY" else "BUFFERING"
                _put("ML Status", status_str,           ml_col)
                _put("ML Raw",    f"{ml_raw:.4f}",      TEXT_DIM)
                _put("ML Smooth", f"{ml_smooth:.4f}",   ml_col)
                _put("ML Label",  ml_label,             ml_col)

            # ── CNN visual inference block ─────────────────────────────────
            cv2.line(img, (12, y_pos - 6), (panel_w - 12, y_pos - 6),
                     (70, 70, 70), 1, cv2.LINE_AA)

            cnn = self._cnn_engine.latest_result if self._cnn_engine else None

            if cnn is None:
                _put("CNN", "disabled", TEXT_DIM)
            else:
                cnn_status = cnn.get("status", "LOADING")
                cnn_prob   = cnn.get("cnn_fake_probability", 0.0)
                cnn_label  = cnn.get("cnn_label", "REAL")

                if cnn_status in ("LOADING", "NO_MODEL", "LOAD_ERROR"):
                    cnn_col = (140, 140, 140)   # gray
                elif cnn_label == "FAKE":
                    cnn_col = (0, 0, 220)        # red (BGR)
                else:
                    cnn_col = (0, 200, 80)       # green

                _put("CNN Status",  cnn_status,            cnn_col if cnn_status == "READY" else TEXT_DIM)
                _put("CNN Score",   f"{cnn_prob:.4f}",     cnn_col)
                _put("CNN Label",   cnn_label,             cnn_col)

            # ── FUSION decision block ───────────────────────────────────────
            cv2.line(img, (12, y_pos - 6), (panel_w - 12, y_pos - 6),
                     (55, 120, 180), 1, cv2.LINE_AA)   # blue separator for fusion

            fusion = self._fusion_engine.latest_result if self._fusion_engine else None

            if fusion is None:
                _put("FUSION", "disabled", TEXT_DIM)
            else:
                f_status  = fusion.get("status", "LOADING")
                f_final   = fusion.get("final_status",  "LOW_CONFIDENCE")
                f_reason  = fusion.get("reason",        "")
                f_gru     = fusion.get("gru_score",     0.0)
                f_cnn     = fusion.get("cnn_score",     0.0)
                f_score   = fusion.get("fusion_score",  0.0)
                f_smooth  = fusion.get("fusion_smooth", 0.0)

                # Colour by final_status
                if f_final == "HIGH_RISK":
                    f_col = (0, 0, 220)         # red (BGR)
                elif f_final == "WARNING":
                    f_col = (0, 140, 255)       # orange
                elif f_final == "SAFE":
                    f_col = (0, 200, 80)        # green
                else:
                    f_col = (140, 140, 140)     # gray (LOW_CONFIDENCE)

                _put("-- FUSION --",   "",                        f_col)
                _put("GRU Score",      f"{f_gru:.4f}",            TEXT_DIM)
                _put("CNN Score",      f"{f_cnn:.4f}",            TEXT_DIM)
                _put("Fusion Score",   f"{f_score:.4f}",          f_col)
                _put("Fus Smooth",     f"{f_smooth:.4f}",         f_col)
                _put("Status",         f_final,                   f_col)
                # Truncate reason to fit panel width
                reason_short = f_reason[:28] if len(f_reason) > 28 else f_reason
                _put("Reason",         reason_short,              TEXT_DIM)

        else:
            cv2.putText(img, "INITIALIZING", (12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_DIM, 2, cv2.LINE_AA)

        # ── Bottom-right: frame counter + fps hint ──
        if event:
            cv2.putText(img, f"#{event.frame_id}",
                        (w - 80, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
