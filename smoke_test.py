from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass

from agent.capture.webcam import CaptureSource
from agent.event_bus import bus
from agent.events import FrameEvent, LivenessSignal, TrustEvent
from agent.liveness.blink import BlinkDetector
from agent.liveness.head_pose import HeadPoseEstimator
from agent.risk.engine import RiskEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class SmokeStats:
    frames_seen: int = 0
    face_frames: int = 0
    blink_signals: int = 0
    head_pose_signals: int = 0
    trust_events: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepShield smoke test")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--demo", type=str, default=None, help="Optional demo video path")
    parser.add_argument("--duration", type=float, default=8.0, help="Seconds to run")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    session_id = str(uuid.uuid4())
    stats = SmokeStats()

    async def on_frame(event: FrameEvent) -> None:
        stats.frames_seen += 1
        if event.face_detected:
            stats.face_frames += 1

    async def on_liveness(signal: LivenessSignal) -> None:
        if signal.extractor_name == "blink":
            stats.blink_signals += 1
        elif signal.extractor_name == "head_pose":
            stats.head_pose_signals += 1

    async def on_trust(event: TrustEvent) -> None:
        stats.trust_events += 1

    bus.subscribe(FrameEvent, on_frame)
    bus.subscribe(LivenessSignal, on_liveness)
    bus.subscribe(TrustEvent, on_trust)

    BlinkDetector().register()
    HeadPoseEstimator().register()
    risk_engine = RiskEngine(session_id=session_id)
    risk_engine.register()
    capture = CaptureSource(
        session_id=session_id,
        device_index=args.device,
        video_path=args.demo,
        target_fps=30,
    )

    capture_task = asyncio.create_task(capture.run(), name="smoke-capture")
    risk_task = asyncio.create_task(risk_engine.run(), name="smoke-risk")
    shutdown_clean = True

    try:
        await asyncio.sleep(args.duration)
    finally:
        capture.stop()
        risk_engine.stop()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(capture_task, risk_task, return_exceptions=True),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            shutdown_clean = False
            for task in (capture_task, risk_task):
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(capture_task, risk_task, return_exceptions=True)

    failures: list[str] = []
    if stats.frames_seen == 0:
        failures.append("webcam start")
    if stats.face_frames == 0:
        failures.append("face detection")
    if stats.blink_signals == 0:
        failures.append("blink signal")
    if stats.head_pose_signals == 0:
        failures.append("head pose signal")
    if stats.trust_events == 0:
        failures.append("risk engine output")
    if not shutdown_clean:
        failures.append("clean shutdown")
    if any(isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError) for result in results):
        failures.append("clean shutdown")

    if failures:
        raise SystemExit(f"SMOKE TEST FAILED: missing {', '.join(failures)}")

    print("SMOKE TEST PASSED")
    print(
        "frames={0} face_frames={1} blink_signals={2} head_pose_signals={3} trust_events={4}".format(
            stats.frames_seen,
            stats.face_frames,
            stats.blink_signals,
            stats.head_pose_signals,
            stats.trust_events,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
