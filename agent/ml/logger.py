"""
agent/ml/logger.py — ML Training Data Logger
═════════════════════════════════════════════
Pure side-effect data collection layer.  Zero impact on runtime logic.

Signal sources
──────────────
  FrameEvent              → face presence, bbox, frame metadata  (30 Hz)
  LivenessSignal "blink"  → ear, blink_detected                  (30 Hz)
  LivenessSignal "head_pose" → yaw, pitch, roll                  (30 Hz)
  LivenessSignal "motion"    → motion_raw, motion_score          (30 Hz)
  LivenessSignal "temporal_consistency" → irregularity, temporal_score (30 Hz)
  LivenessSignal "texture"   → texture_score, is_spoof           (30 Hz)
  TrustEvent              → trust_score, trust_status            (2 Hz, cached)

All subscriptions are read-only observers on the public event bus.
No private state from any module is accessed.

File layout:
  data/sessions/session_<id>/frames.csv
  data/sessions/session_<id>/metadata.json

CSV schema (fixed, 22 columns — never changes at runtime):
  timestamp, session_id, frame_id,
  face_present, bbox_x, bbox_y, bbox_w, bbox_h,
  ear, blink_detected,
  yaw, pitch, roll,
  motion_raw, motion_score,
  irregularity, temporal_score,
  texture_score, is_spoof,
  trust_score, trust_status,
  label
"""

from __future__ import annotations

import csv
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from agent.event_bus import bus
from agent.events import FrameEvent, LivenessSignal, TrustEvent

logger = logging.getLogger(__name__)

# ── Fixed CSV schema ───────────────────────────────────────────────────────────
# NEVER change the order or names of these columns.
CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "session_id",
    "frame_id",
    "face_present",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "ear",
    "blink_detected",
    "yaw",
    "pitch",
    "roll",
    "motion_raw",
    "motion_score",
    "irregularity",
    "temporal_score",
    "texture_score",
    "is_spoof",
    "trust_score",
    "trust_status",
    "label",
)

# ── Signal default values (used when a signal hasn't arrived yet) ──────────────
_DEFAULTS: dict = {
    "ear":            0.0,
    "blink_detected": 0,
    "yaw":            0.0,
    "pitch":          0.0,
    "roll":           0.0,
    "motion_raw":     0.0,
    "motion_score":   0.0,
    "irregularity":   0.0,
    "temporal_score": 0.0,
    "texture_score":  0.0,
    "is_spoof":       0,
    "trust_score":    0.0,
    "trust_status":   "WARMING_UP",
}


class MLDataLogger:
    """
    Non-blocking ML data logger.

    Lifecycle:
      1. Construct with session_id and output_dir
      2. Call register() to subscribe to the event bus
      3. Call start() to launch the background writer thread
      4. Call stop() during shutdown to flush the queue and close files
    """

    # Max items in the write queue before we start dropping frames.
    # At 30 fps this is ~100 seconds of buffer.
    _QUEUE_MAXSIZE = 3000

    def __init__(self, session_id: str, base_dir: Path,
                 label: str = "unknown") -> None:
        self._session_id = session_id
        self._label = label                              # session-level ML label
        self._session_dir = base_dir / f"session_{session_id}"
        self._csv_path = self._session_dir / "frames.csv"
        self._meta_path = self._session_dir / "metadata.json"

        # Queue between async handlers (producers) and writer thread (consumer)
        self._queue: queue.Queue[Optional[dict]] = queue.Queue(
            maxsize=self._QUEUE_MAXSIZE
        )

        # ── Per-extractor signal cache (updated by LivenessSignal events) ──
        # These are replaced atomically per field — no locking needed because
        # Python GIL protects dict key assignment.
        self._signals: dict = dict(_DEFAULTS)

        self._writer_thread: Optional[threading.Thread] = None
        self._frame_count: int = 0
        self._start_time: float = time.time()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to public events on the bus.  No-op if bus unavailable."""
        try:
            bus.subscribe(FrameEvent,      self._on_frame_event)
            bus.subscribe(LivenessSignal,  self._on_liveness_signal)
            bus.subscribe(TrustEvent,      self._on_trust_event)
            logger.info("MLDataLogger registered on event bus")
        except Exception as exc:
            logger.warning(f"MLDataLogger.register failed: {exc}")

    def start(self) -> None:
        """Create output directory and launch background writer thread."""
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._write_metadata()
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="ml-data-writer",
                daemon=True,
            )
            self._writer_thread.start()
            logger.info(f"MLDataLogger writing to {self._session_dir}")
        except Exception as exc:
            logger.warning(f"MLDataLogger.start failed — logging disabled: {exc}")

    # ── Event handlers (called on the asyncio event loop) ──────────────────────

    async def _on_liveness_signal(self, sig: LivenessSignal) -> None:
        """
        Update the per-extractor signal cache from a LivenessSignal event.
        Each extractor publishes specific metadata keys that map to CSV columns.
        """
        try:
            md = sig.metadata
            name = sig.extractor_name

            if name == "blink":
                self._signals["ear"]            = float(md.get("ear", self._signals["ear"]))
                self._signals["blink_detected"] = int(bool(md.get("blink_detected", False)))

            elif name == "head_pose":
                self._signals["yaw"]   = float(md.get("yaw",   self._signals["yaw"]))
                self._signals["pitch"] = float(md.get("pitch", self._signals["pitch"]))
                self._signals["roll"]  = float(md.get("roll",  self._signals["roll"]))

            elif name == "motion":
                self._signals["motion_raw"]   = float(md.get("motion_raw",   self._signals["motion_raw"]))
                self._signals["motion_score"] = float(md.get("motion_score", self._signals["motion_score"]))

            elif name == "temporal_consistency":
                self._signals["irregularity"]   = float(md.get("irregularity",   self._signals["irregularity"]))
                self._signals["temporal_score"] = float(md.get("temporal_score", self._signals["temporal_score"]))

            elif name == "texture":
                self._signals["texture_score"] = float(md.get("texture_score", self._signals["texture_score"]))
                self._signals["is_spoof"]       = int(bool(md.get("is_spoof", False)))

        except Exception as exc:
            logger.debug(f"MLDataLogger._on_liveness_signal error: {exc}")

    async def _on_trust_event(self, event: TrustEvent) -> None:
        """Cache the latest trust values.  These annotate subsequent frame rows."""
        try:
            self._signals["trust_score"]  = round(float(event.trust_score), 6)
            self._signals["trust_status"] = str(event.risk_level)
        except Exception:
            pass

    async def _on_frame_event(self, event: FrameEvent) -> None:
        """
        Snapshot all cached signals and enqueue one CSV row.
        Called at ~30 Hz.  Must never block.

        Warmup frames (trust_status == "WARMING_UP") are silently skipped —
        their signals are all zero/default and would pollute ML training data.
        """
        try:
            # Skip warmup frames — no valid signals yet
            if self._signals["trust_status"] == "WARMING_UP":
                return
            bx, by, bw, bh = event.face_bbox if event.face_bbox else (0, 0, 0, 0)
            s = self._signals   # local alias — snapshot is implicit since dicts are mutable
            row = {
                "timestamp":      round(event.timestamp, 6),
                "session_id":     event.session_id,
                "frame_id":       event.frame_id,
                "face_present":   int(event.face_detected),
                "bbox_x":         bx,
                "bbox_y":         by,
                "bbox_w":         bw,
                "bbox_h":         bh,
                "ear":            round(s["ear"], 4),
                "blink_detected": s["blink_detected"],
                "yaw":            round(s["yaw"], 3),
                "pitch":          round(s["pitch"], 3),
                "roll":           round(s["roll"], 3),
                "motion_raw":     round(s["motion_raw"], 6),
                "motion_score":   round(s["motion_score"], 4),
                "irregularity":   round(s["irregularity"], 6),
                "temporal_score": round(s["temporal_score"], 4),
                "texture_score":  round(s["texture_score"], 4),
                "is_spoof":       s["is_spoof"],
                "trust_score":    s["trust_score"],
                "trust_status":   s["trust_status"],
                "label":          self._label,
            }
            # Non-blocking — drop if queue is full rather than stalling the loop
            self._queue.put_nowait(row)
        except queue.Full:
            pass
        except Exception as exc:
            logger.debug(f"MLDataLogger._on_frame_event error: {exc}")

    # ── Background writer thread ────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        """
        Daemon thread — blocks on the queue and writes rows to CSV.
        Receives None sentinel to stop.
        """
        try:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                fh.flush()

                while True:
                    try:
                        row = self._queue.get(timeout=1.0)
                    except queue.Empty:
                        fh.flush()
                        continue

                    if row is None:   # shutdown sentinel
                        fh.flush()
                        break

                    try:
                        writer.writerow(row)
                        self._frame_count += 1
                        # Flush every 30 rows (~1 sec at 30 fps)
                        if self._frame_count % 30 == 0:
                            fh.flush()
                    except Exception as exc:
                        logger.debug(f"MLDataLogger write error: {exc}")

        except Exception as exc:
            logger.warning(f"MLDataLogger writer thread failed: {exc}")

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Flush queue, join writer thread, update metadata.  Blocks up to 5s."""
        try:
            self._queue.put(None)   # sentinel
            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=5.0)
            self._update_metadata()
            logger.info(
                f"MLDataLogger stopped — {self._frame_count} frames written "
                f"to {self._csv_path}"
            )
        except Exception as exc:
            logger.warning(f"MLDataLogger.stop error: {exc}")

    # ── Metadata helpers ───────────────────────────────────────────────────────

    def _write_metadata(self) -> None:
        meta = {
            "session_id":    self._session_id,
            "label":         self._label,
            "start_time":    self._start_time,
            "csv_columns":   list(CSV_COLUMNS),
            "schema_version": 1,
            "frame_count":   0,
            "end_time":      None,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _update_metadata(self) -> None:
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            meta["frame_count"] = self._frame_count
            meta["end_time"] = time.time()
            self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
