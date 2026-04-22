from agent.events import TrustEvent, FrameEvent, LivenessSignal
from agent.event_bus import bus
from agent.liveness.base import SignalExtractor
from agent.liveness.blink import BlinkDetector, BLINK_TIMEOUT_SEC
from agent.liveness.head_pose import HeadPoseEstimator
from agent.liveness.gaze import GazeTracker
from agent.liveness.micro_expr import MicroExpressionDetector
from agent.provenance.compression import CompressionArtifactDetector
from agent.provenance.temporal import TemporalJitterDetector, STRUCTURAL_INDICES
from agent.risk.engine import RiskEngine, DEFAULT_WEIGHTS
from agent.policy.loader import load_policy

print("ALL IMPORTS OK")
print(f"BLINK_TIMEOUT_SEC = {BLINK_TIMEOUT_SEC}  (target: 4.0)")
print(f"texture_freq in weights: {'texture_freq' in DEFAULT_WEIGHTS}  (target: False)")
print(f"STRUCTURAL_INDICES ordered: {STRUCTURAL_INDICES == sorted(STRUCTURAL_INDICES)}  (target: True)")
p = load_policy()
print(f"Policy: {p.organization}")
