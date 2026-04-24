"""
main.py — DeepShield Agent Entry Point

Boots all modules in order:
  1. Load policy
  2. Initialize audit store
  3. Register all signal extractors on event bus
  4. Start capture loop
  5. Start risk engine emission loop
  6. Start FastAPI server
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from pathlib import Path

import uvicorn
from PyQt6.QtWidgets import QApplication

from agent.event_bus import bus
from agent.events import TrustEvent
from agent.ui.overlay import OverlayHUD


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deepshield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepShield Agent")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index (0=default, 1=OBS virtual cam)",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Path to a video file to replay as the capture source",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to a custom policy.yaml",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--debug-ui",
        action="store_true",
        default=False,
        help="Open an OpenCV debug preview with live signal telemetry",
    )
    parser.add_argument(
        "--record-ml-data",
        action="store_true",
        default=False,
        help="Record ML training data to data/sessions/session_<id>/",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="unknown",
        choices=["real", "fake", "unknown"],
        help="Session label for ML training (real / fake / unknown)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        default=False,
        help="Save face crop JPEGs to data/sessions/<id>/frames/ for CNN training",
    )
    return parser.parse_args()


async def _run_ui_pump(app: QApplication) -> None:
    while True:
        app.processEvents()
        await asyncio.sleep(1 / 60)


async def main_async(args: argparse.Namespace) -> None:
    session_id = str(uuid.uuid4())
    logger.info(f"DeepShield starting — session={session_id}")

    app = QApplication.instance() or QApplication([])
    hud = OverlayHUD()
    hud.show()

    async def _on_trust_event(event: TrustEvent) -> None:
        status = event.risk_level
        if status == "SAFE":
            hud.update_status("✅ HUMAN VERIFIED")
        elif status == "WARNING":
            hud.update_status("⚠️ SUSPICIOUS BEHAVIOR")
        else:
            hud.update_status("🚨 POSSIBLE DEEPFAKE")

    bus.subscribe(TrustEvent, _on_trust_event)

    from agent.policy.loader import load_policy

    policy = load_policy(Path(args.policy) if args.policy else None)
    logger.info(
        f"Policy loaded: org={policy.organization}, paranoia={policy.paranoia_mode}"
    )

    from agent.audit.store import AuditStore

    audit = AuditStore()
    await audit.initialize(session_id)

    from agent.liveness.blink import BlinkDetector
    from agent.liveness.gaze import GazeTracker
    from agent.liveness.head_pose import HeadPoseEstimator
    from agent.liveness.micro_expr import MicroExpressionDetector
    from agent.provenance.compression import CompressionArtifactDetector
    from agent.provenance.temporal import TemporalJitterDetector

    extractors = [
        BlinkDetector(),
        HeadPoseEstimator(),
        GazeTracker(),
        MicroExpressionDetector(),
    ]
    for ext in extractors:
        ext.register()

    provenance_checkers = [
        CompressionArtifactDetector(),
        TemporalJitterDetector(),
    ]
    for chk in provenance_checkers:
        chk.register()

    from agent.risk.engine import RiskEngine

    risk_engine = RiskEngine(
        session_id=session_id,
        weights=policy.weights.as_dict(),
        policy=policy,
    )
    risk_engine.register()

    from agent.capture.webcam import CaptureSource

    capture = CaptureSource(
        session_id=session_id,
        device_index=args.device,
        video_path=args.demo,
        target_fps=30,
    )

    # ── Optional ML data logger ──
    ml_logger = None
    if getattr(args, "record_ml_data", False):
        from agent.ml.logger import MLDataLogger

        ml_logger = MLDataLogger(
            session_id=session_id,
            base_dir=Path("data/sessions"),
            label=getattr(args, "label", "unknown"),
            save_crops=getattr(args, "save_crops", False),
        )
        ml_logger.register()
        ml_logger.start()
        logger.info("ML data recording enabled (--record-ml-data)")

    # ── GRU real-time inference engine (always active, fails gracefully) ──
    from agent.ml.inference import GRUInferenceEngine

    gru_engine = GRUInferenceEngine(
        model_path=Path("models/gru_refined_v3.pt"),   # v3: balanced split, thr=0.45
        npz_path=Path("data/sequences.npz"),
    )
    gru_engine.register()
    logger.info("GRU inference engine registered")

    # ── CNN visual inference engine (parallel to GRU, non-blocking) ──
    from agent.ml.cnn_inference import CNNInferenceEngine

    cnn_engine = CNNInferenceEngine(
        model_path=Path("models/cnn_baseline.pt"),
    )
    cnn_engine.register()
    logger.info("CNN inference engine registered")

    # ── Deepfake engine (Xception — triggered on elevated suspicion) ──
    from agent.ml.deepfake_inference import DeepfakeInferenceEngine

    deepfake_engine = DeepfakeInferenceEngine(
        weights_path=Path("models/deepfake_xception.pkl"),
    )
    deepfake_engine.register()
    logger.info("Deepfake engine registered")

    # ── Fusion engine (GRU + CNN + Deepfake 3-signal decision layer) ──
    from agent.ml.fusion_engine import FusionEngine

    fusion_engine = FusionEngine(
        gru_engine=gru_engine,
        cnn_engine=cnn_engine,
        deepfake_engine=deepfake_engine,
    )
    fusion_engine.register()
    logger.info("Fusion engine registered (3-signal: GRU + CNN + Deepfake)")

    # ── Optional debug overlay ──
    debug_ui = None
    if getattr(args, "debug_ui", False):
        from agent.debug_ui import DebugUI

        debug_ui = DebugUI(
            risk_engine=risk_engine,
            capture_source=capture,
            gru_engine=gru_engine,
            cnn_engine=cnn_engine,
            fusion_engine=fusion_engine,
            deepfake_engine=deepfake_engine,
        )
        debug_ui.register()
        logger.info("Debug UI enabled (--debug-ui)")


    from agent.api.server import app as api_app
    from agent.api.server import register_handlers, set_audit_store

    set_audit_store(audit)
    register_handlers()

    logger.info(f"Starting API server on {args.host}:{args.port}")

    uvicorn_config = uvicorn.Config(
        api_app,
        host=args.host,
        port=args.port,
        log_level="warning",
        loop="asyncio",
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    tasks = [
        asyncio.create_task(capture.run(), name="capture"),
        asyncio.create_task(risk_engine.run(), name="risk-engine"),
        asyncio.create_task(uvicorn_server.serve(), name="uvicorn"),
        asyncio.create_task(_run_ui_pump(app), name="overlay-ui"),
    ]
    if debug_ui is not None:
        tasks.append(asyncio.create_task(debug_ui.run(), name="debug-ui"))

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        capture.stop()
        risk_engine.stop()
        uvicorn_server.should_exit = True
        if debug_ui is not None:
            debug_ui.stop()
        if ml_logger is not None:
            ml_logger.stop()
        hud.close()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        session_id_done = await audit.close_session()
        logger.info(f"Session {session_id_done} closed. Audit saved.")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("DeepShield stopped.")


if __name__ == "__main__":
    main()
