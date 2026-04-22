"""
liveness/base.py — SignalExtractor abstract base class.

Every liveness detector inherits from this.
The interface contract: receive a FrameEvent, emit a LivenessSignal.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod

from agent.events import FrameEvent, LivenessSignal
from agent.event_bus import bus

logger = logging.getLogger(__name__)


class SignalExtractor(ABC):
    """
    Base class for all liveness signal extractors.

    Subclasses implement `extract(frame_event) -> LivenessSignal`.
    The `handle` method is subscribed to FrameEvent on the bus.
    """

    name: str = "base"

    def register(self) -> None:
        """Subscribe this extractor to FrameEvents on the global bus."""
        bus.subscribe(FrameEvent, self.handle)
        logger.debug(f"Registered extractor: {self.name}")

    async def handle(self, event: FrameEvent) -> None:
        """Called by event bus for every FrameEvent. Publishes a LivenessSignal."""
        if not event.face_detected or not event.face_landmarks:
            return  # Skip frames with no face

        try:
            signal = self.extract(event)
            if signal is not None:
                await bus.publish(signal)
        except Exception as exc:
            logger.error(f"{self.name} extractor error: {exc}", exc_info=True)

    @abstractmethod
    def extract(self, event: FrameEvent) -> LivenessSignal | None:
        """Extract a liveness signal from a FrameEvent. Return None to skip."""
        ...
