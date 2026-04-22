"""
event_bus.py — Lightweight asyncio pub/sub internal event bus.

All modules communicate exclusively through this bus.
No module imports another module directly (only events.py).
This is what makes the architecture swappable.
"""

from __future__ import annotations
import asyncio
import logging
from collections import defaultdict
from typing import Callable, Type, Any

logger = logging.getLogger(__name__)


class EventBus:
    """
    Simple asyncio pub/sub bus.
    Subscribers register for a specific event type (class).
    Publishers call await bus.publish(event_instance).
    """

    def __init__(self):
        self._subscribers: dict[Type, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: Type, handler: Callable) -> None:
        """Register an async handler for a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__qualname__} to {event_type.__name__}")

    def unsubscribe(self, event_type: Type, handler: Callable) -> None:
        subscribers = self._subscribers.get(event_type, [])
        if handler in subscribers:
            subscribers.remove(handler)

    async def publish(self, event: Any) -> None:
        """Publish an event to all registered subscribers. Fire-and-forget."""
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return
        # Run all handlers concurrently
        await asyncio.gather(
            *[self._safe_call(h, event) for h in handlers],
            return_exceptions=True,
        )

    async def _safe_call(self, handler: Callable, event: Any) -> None:
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.error(
                f"Event handler {handler.__qualname__} raised {type(exc).__name__}: {exc}",
                exc_info=True,
            )


# Global singleton bus — imported by all modules
bus = EventBus()
