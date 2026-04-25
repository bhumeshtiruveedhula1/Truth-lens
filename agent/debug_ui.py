"""
agent/debug_ui.py — Analysis-First Debug Overlay
─────────────────────────────────────────────────
Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  TOP BAR : STATUS | FUSION SCORE | Frame# | MODE:DEBUG   │
  ├────────────────────────────┬─────────────────────────────┤
  │                            │  GRU   CNN   DEEPFAKE FUSION│
  │   CAMERA  (landmarks+box)  │  analysis panel (right)     │
  │                            │                             │
  ├────────────────────────────┴─────────────────────────────┤
  │  ADVANCED signals (smaller font, secondary)               │
  └──────────────────────────────────────────────────────────┘
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

# ── Landmark subsets ────────────────────────────────────────────────────────
FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109,10,
]
LEFT_EYE  = [33,160,158,133,153,144,33]
RIGHT_EYE = [362,385,387,263,373,380,362]
MOUTH     = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,61]

# ── Colour palette (BGR) ────────────────────────────────────────────────────
C_SAFE    = (0, 210, 90)
C_WARN    = (0, 165, 255)
C_RISK    = (0, 0, 230)
C_DIM     = (120, 125, 135)
C_WHITE   = (235, 235, 240)
C_HEAD    = (175, 180, 205)
C_PANEL   = (18, 18, 24)
C_TOPBAR  = (12, 12, 18)
C_SEP     = (48, 52, 62)

STATUS_COL = {"SAFE": C_SAFE, "WARNING": C_WARN, "HIGH_RISK": C_RISK,
              "LOW_CONFIDENCE": (160,160,0), "WARMING_UP": (190,150,0)}

# ── Canvas dimensions ───────────────────────────────────────────────────────
_W       = 1160
_H       = 710
_TOP_H   = 50
_CAM_W   = 700
_ADV_H   = 80
_PANEL_W = _W - _CAM_W          # 460
_CAM_H   = _H - _TOP_H - _ADV_H # 580


class DebugUI:
    WINDOW_NAME = "DeepShield — Analysis Monitor"

    def __init__(self, risk_engine, capture_source,
                 gru_engine=None, cnn_engine=None,
                 fusion_engine=None, deepfake_engine=None) -> None:
        self._risk_engine     = risk_engine
        self._capture         = capture_source
        self._gru_engine      = gru_engine
        self._cnn_engine      = cnn_engine
        self._fusion_engine   = fusion_engine
        self._deepfake_engine = deepfake_engine
        self._running         = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_event: Optional[FrameEvent]  = None

    def register(self) -> None:
        bus.subscribe(FrameEvent, self._on_frame)
        logger.info("DebugUI registered")

    async def _on_frame(self, event: FrameEvent) -> None:
        self._last_event = event

    async def run(self) -> None:
        self._running = True
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, _W, _H)
        logger.info("DebugUI render loop started")

        while self._running:
            raw = getattr(self._capture, "_last_raw_frame", None)
            if raw is not None:
                self._last_frame = raw.copy()

            canvas = self._compose(self._last_frame, self._last_event)
            cv2.imshow(self.WINDOW_NAME, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False
                break
            await asyncio.sleep(1 / 30)

        cv2.destroyWindow(self.WINDOW_NAME)

    def stop(self) -> None:
        self._running = False

    # ── Composition entry point ─────────────────────────────────────────────

    def _compose(self, frame: Optional[np.ndarray],
                 event: Optional[FrameEvent]) -> np.ndarray:
        canvas = np.full((_H, _W, 3), C_PANEL, dtype=np.uint8)
        self._draw_top_bar(canvas, event)
        self._draw_camera(canvas, frame, event)
        self._draw_right_panel(canvas)
        self._draw_advanced_bar(canvas, event)
        return canvas

    # ── TOP BAR ─────────────────────────────────────────────────────────────

    def _draw_top_bar(self, canvas: np.ndarray, event: Optional[FrameEvent]) -> None:
        cv2.rectangle(canvas, (0, 0), (_W, _TOP_H), C_TOPBAR, -1)
        cv2.line(canvas, (0, _TOP_H), (_W, _TOP_H), C_SEP, 1)

        # Fusion status (left)
        fusion   = self._fusion_engine.latest_result if self._fusion_engine else {}
        f_status = fusion.get("final_status", "LOADING")
        f_score  = fusion.get("fusion_smooth", 0.0)
        f_reason = fusion.get("reason", "")
        col      = STATUS_COL.get(f_status, C_WHITE)

        # Status badge
        badge_text = f_status
        cv2.putText(canvas, badge_text, (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)

        bw, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0], 0
        x2 = 14 + bw[0] + 16

        # Score
        cv2.putText(canvas, f"score {f_score:.3f}", (x2, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

        # Reason (center)
        reason_short = f_reason[:40]
        r_sz = cv2.getTextSize(reason_short, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        rx = (_CAM_W - r_sz[0]) // 2
        cv2.putText(canvas, reason_short, (rx, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)

        # Frame + mode (right)
        fid = event.frame_id if event else 0
        right_text = f"#{fid}   MODE:DEBUG"
        rt_sz = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        cv2.putText(canvas, right_text, (_W - rt_sz[0] - 12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)

    # ── CAMERA ZONE ─────────────────────────────────────────────────────────

    def _draw_camera(self, canvas: np.ndarray, frame: Optional[np.ndarray],
                     event: Optional[FrameEvent]) -> None:
        cam_y0, cam_y1 = _TOP_H, _TOP_H + _CAM_H

        # Dark camera background
        cv2.rectangle(canvas, (0, cam_y0), (_CAM_W, cam_y1), (8, 8, 12), -1)
        cv2.line(canvas, (_CAM_W, cam_y0), (_CAM_W, cam_y1), C_SEP, 1)

        if frame is None:
            cv2.putText(canvas, "NO SIGNAL", (80, cam_y0 + _CAM_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_DIM, 2, cv2.LINE_AA)
            return

        # Scale frame to fit camera zone with letterbox
        fh, fw = frame.shape[:2]
        scale  = min(_CAM_W / fw, _CAM_H / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = (_CAM_W - nw) // 2
        y0 = cam_y0 + (_CAM_H - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized

        # Draw landmarks + bbox on the camera portion
        if event and event.face_detected:
            self._draw_landmarks(canvas, event, x0, y0, nw, nh)
            self._draw_bbox(canvas, event, x0, y0, nw, nh)

        # Inline camera labels (top-left of camera zone)
        face_col  = C_SAFE if (event and event.face_detected) else C_RISK
        face_text = "FACE DETECTED" if (event and event.face_detected) else "NO FACE"
        cv2.putText(canvas, face_text, (x0 + 8, cam_y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_col, 1, cv2.LINE_AA)

    def _draw_landmarks(self, img, event, x0, y0, nw, nh) -> None:
        lm = event.face_landmarks
        fw, fh = event.frame_width, event.frame_height

        def _strip(indices, colour):
            pts = []
            for i in indices:
                if i < len(lm):
                    px = x0 + int(lm[i][0] * nw)
                    py = y0 + int(lm[i][1] * nh)
                    pts.append((px, py))
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, colour, 1, cv2.LINE_AA)

        _strip(FACE_OVAL, (80, 185, 80))
        _strip(LEFT_EYE,  (0, 230, 230))
        _strip(RIGHT_EYE, (0, 230, 230))
        _strip(MOUTH,     (180, 90, 230))
        for i in range(0, len(lm), 8):
            px = x0 + int(lm[i][0] * nw)
            py = y0 + int(lm[i][1] * nh)
            cv2.circle(img, (px, py), 1, (0, 220, 180), -1, cv2.LINE_AA)

    def _draw_bbox(self, img, event, x0, y0, nw, nh) -> None:
        bx, by, bw, bh = event.face_bbox
        fw, fh = event.frame_width, event.frame_height
        sx = nw / fw;  sy = nh / fh
        rx1 = x0 + int(bx * sx)
        ry1 = y0 + int(by * sy)
        rx2 = x0 + int((bx + bw) * sx)
        ry2 = y0 + int((by + bh) * sy)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (210, 185, 0), 2, cv2.LINE_AA)

    # ── RIGHT ANALYSIS PANEL ─────────────────────────────────────────────────

    def _draw_right_panel(self, canvas: np.ndarray) -> None:
        px = _CAM_W
        py0, py1 = _TOP_H, _TOP_H + _CAM_H
        lh = 24      # line height
        sh = 30      # section header height
        pad = 14     # left padding inside panel

        y = py0 + 10

        def _section(title: str, border_col) -> None:
            nonlocal y
            cv2.line(canvas, (px + 8, y + 4), (px + _PANEL_W - 8, y + 4),
                     border_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, title, (px + pad, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, border_col, 1, cv2.LINE_AA)
            y += sh

        def _row(label: str, value: str, col=C_WHITE) -> None:
            nonlocal y
            if y > py1 - 12:
                return
            cv2.putText(canvas, label, (px + pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, value, (px + pad + 118, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
            y += lh

        def _spacer(n=6) -> None:
            nonlocal y; y += n

        # ── GRU ──────────────────────────────────────────────────────────────
        _section("GRU  /  BEHAVIORAL", (90, 160, 255))
        gru = self._gru_engine.latest_result if self._gru_engine else {}
        g_status = gru.get("status", "BUFFERING")
        g_raw    = gru.get("raw_score",    gru.get("fake_probability", 0.0))
        g_smooth = gru.get("smoothed_score", g_raw)
        g_label  = gru.get("fake_label", "REAL")
        g_col    = C_SAFE if g_label == "REAL" else C_RISK
        if g_status != "READY":
            g_col = C_DIM
        _row("Status",  "READY" if g_status == "READY" else "BUFFERING", g_col)
        _row("Raw",     f"{g_raw:.4f}", C_DIM)
        _row("Smooth",  f"{g_smooth:.4f}", g_col)
        _row("Label",   g_label, g_col)
        _spacer()

        # ── CNN ───────────────────────────────────────────────────────────────
        _section("CNN  /  LIVENESS", (0, 195, 120))
        cnn = self._cnn_engine.latest_result if self._cnn_engine else {}
        c_status = cnn.get("status", "LOADING")
        c_prob   = cnn.get("cnn_fake_probability", 0.0)
        c_label  = cnn.get("cnn_label", "REAL")
        c_col    = C_SAFE if c_label == "REAL" else C_RISK
        if c_status != "READY":
            c_col = C_DIM
        _row("Status",  c_status, c_col if c_status == "READY" else C_DIM)
        _row("Score",   f"{c_prob:.4f}", c_col)
        _row("Label",   c_label, c_col)
        _spacer()

        # ── DEEPFAKE ──────────────────────────────────────────────────────────
        _section("DEEPFAKE  /  EFFICIENTNET", (160, 70, 220))
        df       = self._deepfake_engine.latest_result if self._deepfake_engine else {}
        df_st    = df.get("status", "INIT")
        df_prob  = df.get("deepfake_probability", 0.0)
        df_label = df.get("deepfake_label", "REAL")
        df_back  = df.get("active_backend", "--")
        df_col   = C_SAFE if df_label == "REAL" else C_RISK
        if df_st != "READY":
            _row("Status",  df_st,   C_DIM)
            _row("Score",   "--",    C_DIM)
            _row("Label",   "--",    C_DIM)
            _row("Backend", df_back, C_DIM)
        else:
            _row("Status",  "READY",           C_DIM)
            _row("Score",   f"{df_prob:.4f}",  df_col)
            _row("Label",   df_label,           df_col)
            _row("Backend", df_back,            C_DIM)
        _spacer()

        # ── FUSION ────────────────────────────────────────────────────────────
        _section("FUSION  /  DECISION", C_WARN)
        fusion   = self._fusion_engine.latest_result if self._fusion_engine else {}
        f_status = fusion.get("final_status",   "LOW_CONFIDENCE")
        f_gscore = fusion.get("gru_score",      0.0)
        f_cscore = fusion.get("cnn_score",       0.0)
        f_dscore = fusion.get("deepfake_score",  0.0)
        f_score  = fusion.get("fusion_score",    0.0)
        f_smooth = fusion.get("fusion_smooth",   0.0)
        f_reason = fusion.get("reason",          "")
        f_col    = STATUS_COL.get(f_status, C_DIM)
        _row("Status",    f_status,          f_col)
        _row("GRU  in",   f"{f_gscore:.4f}", C_DIM)
        _row("CNN  in",   f"{f_cscore:.4f}", C_DIM)
        _row("DF   in",   f"{f_dscore:.4f}", C_DIM)
        _row("Score",     f"{f_score:.4f}",  f_col)
        _row("Smooth",    f"{f_smooth:.4f}", f_col)
        _row("Reason",    f_reason[:24],     C_DIM)

        # ── Structured signal log line ───────────────────────────────────────
        import logging as _log
        _log.getLogger("deepshield.signals").debug(
            f"g={f_gscore:.3f} c={f_cscore:.3f} "
            f"id=-- d={f_dscore:.3f} -> {f_status} ({f_reason})"
        )

    # ── ADVANCED BAR ─────────────────────────────────────────────────────────

    def _draw_advanced_bar(self, canvas: np.ndarray,
                           event: Optional[FrameEvent]) -> None:
        y0 = _TOP_H + _CAM_H
        cv2.rectangle(canvas, (0, y0), (_W, _H), (14, 14, 20), -1)
        cv2.line(canvas, (0, y0), (_W, y0), C_SEP, 1)

        result   = getattr(self._risk_engine, "_latest_result", None) or {}
        signals  = result.get("signals", {})
        blink    = signals.get("blink", {})
        head     = signals.get("head_pose", {})
        tex      = signals.get("texture", {})
        memory   = result.get("memory", {})

        ear      = blink.get("ear", 0.0)
        bdet     = blink.get("blink_detected", False)
        yaw      = head.get("yaw", 0.0)
        pitch    = head.get("pitch", 0.0)
        lap      = tex.get("laplacian_score", 0.0)
        m_score  = memory.get("motion_score", 0.0)
        t_score  = memory.get("temporal_score", 0.0)
        rigid_r  = memory.get("rigid_ratio", 0.0)
        bv       = memory.get("blink_validated", False)

        items = [
            ("EAR",    f"{ear:.3f}"),
            ("Blink",  "YES" if bdet else "no"),
            ("Yaw",    f"{yaw:+.1f}"),
            ("Pitch",  f"{pitch:+.1f}"),
            ("Texture",f"{lap:.0f}"),
            ("Motion", f"{m_score:.2f}"),
            ("Temp",   f"{t_score:.2f}"),
            ("Rigid",  f"{rigid_r:.3f}"),
            ("BlinkV", "OK" if bv else "wait"),
        ]

        x = 14
        for label, val in items:
            col_v = C_WHITE
            if label == "Blink" and bdet:  col_v = C_SAFE
            if label == "BlinkV" and bv:   col_v = C_SAFE
            cv2.putText(canvas, f"{label}:", (x, y0 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (x, y0 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_v, 1, cv2.LINE_AA)
            x += 108
            if x > _W - 80:
                break
