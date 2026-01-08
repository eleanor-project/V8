"""
ELEANOR V8 — Event Bus
----------------------

Event-driven architecture for decoupled component communication.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Type, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types."""
    CRITIC_EVALUATED = "critic_evaluated"
    ROUTER_SELECTED = "router_selected"
    PRECEDENT_RETRIEVED = "precedent_retrieved"
    UNCERTAINTY_COMPUTED = "uncertainty_computed"
    AGGREGATION_COMPLETE = "aggregation_complete"
    ESCALATION_REQUIRED = "escalation_required"
    DECISION_MADE = "decision_made"
    EVIDENCE_RECORDED = "evidence_recorded"


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    timestamp: datetime
    trace_id: str
    data: Dict[str, Any]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


@dataclass
class CriticEvaluatedEvent(Event):
    """Event fired when critic evaluation completes."""
    critic_name: str
    severity: float
    violations: List[str]


@dataclass
class RouterSelectedEvent(Event):
    """Event fired when router selects a model."""
    model_name: str
    selection_reason: str
    cost_estimate: Optional[float]


@dataclass
class EscalationRequiredEvent(Event):
    """Event fired when escalation is required."""
    tier: int
    critic: str
    clause: str
    rationale: str


@dataclass
class DecisionMadeEvent(Event):
    """Event fired when final decision is made."""
    decision: str
    confidence: float
    escalated: bool


class EventBus:
    """
    Event bus for decoupled component communication.
    
    Supports:
    - Async event publishing
    - Multiple subscribers per event type
    - Event filtering
    - Error handling
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._enabled: bool = True
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """
        Subscribe to event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async or sync handler function
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe handler from event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.
        
        Args:
            event: Event to publish
        """
        if not self._enabled:
            return
        
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Get subscribers for this event type
        handlers = self._subscribers.get(event.event_type, [])
        
        if not handlers:
            return
        
        # Publish to all subscribers
        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    # Run sync handler in executor
                    tasks.append(asyncio.to_thread(handler, event))
            except Exception as exc:
                logger.error(
                    "event_handler_error",
                    extra={
                        "event_type": event.event_type.value,
                        "error": str(exc),
                    },
                    exc_info=True,
                )
        
        # Wait for all handlers (fire and forget by default)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history."""
        history = self._event_history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    def enable(self) -> None:
        """Enable event bus."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable event bus."""
        self._enabled = False


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


__all__ = [
    "Event",
    "EventType",
    "CriticEvaluatedEvent",
    "RouterSelectedEvent",
    "EscalationRequiredEvent",
    "DecisionMadeEvent",
    "EventBus",
    "get_event_bus",
]
