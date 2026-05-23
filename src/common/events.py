"""
Event system for pub/sub communication between components.

Enables loose coupling between different parts of the system through
an event-driven architecture.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events in the system."""

    # Market Data Events
    TICKER_UPDATE = "ticker_update"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_UPDATE = "trade_update"
    OHLCV_UPDATE = "ohlcv_update"

    # Order Events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Position Events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"

    # Portfolio Events
    BALANCE_UPDATE = "balance_update"
    PNL_UPDATE = "pnl_update"

    # Model Events
    MODEL_PREDICTION = "model_prediction"
    MODEL_UPDATED = "model_updated"

    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR_OCCURRED = "error_occurred"
    RISK_LIMIT_BREACH = "risk_limit_breach"


@dataclass
class Event:
    """Base event class."""

    event_type: EventType
    timestamp: datetime
    data: dict[str, Any]
    source: str | None = None

    def __post_init__(self):
        """Validate event after initialization."""
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")


EventHandler = Callable[[Event], None]


class EventBus:
    """
    Central event bus for pub/sub messaging.

    Allows components to publish events and subscribe to specific event types
    without tight coupling.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._event_history: list[Event] = []
        self._max_history_size = 1000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle the event
        """
        self._subscribers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Handler unsubscribed from {event_type.value}")

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

        # Notify subscribers
        handlers = self._subscribers.get(event.event_type, [])

        logger.debug(f"Publishing {event.event_type.value} to {len(handlers)} subscribers")

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler for {event.event_type.value}: {e}", exc_info=True
                )

    def get_history(self, event_type: EventType | None = None, limit: int = 100) -> list[Event]:
        """
        Get recent event history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")


# Global event bus instance
_global_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.

    Returns:
        Global EventBus instance
    """
    global _global_event_bus

    if _global_event_bus is None:
        _global_event_bus = EventBus()
        logger.info("Global event bus initialized")

    return _global_event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (useful for testing)."""
    global _global_event_bus
    _global_event_bus = None
    logger.info("Global event bus reset")


# Convenience functions


def publish_event(event_type: EventType, data: dict[str, Any], source: str | None = None) -> None:
    """
    Publish an event to the global event bus.

    Args:
        event_type: Type of event
        data: Event data
        source: Source component name
    """
    event = Event(
        event_type=event_type,
        timestamp=datetime.now(),
        data=data,
        source=source,
    )
    get_event_bus().publish(event)


def subscribe_to_event(event_type: EventType, handler: EventHandler) -> None:
    """
    Subscribe to an event on the global event bus.

    Args:
        event_type: Type of event to subscribe to
        handler: Callback function
    """
    get_event_bus().subscribe(event_type, handler)
