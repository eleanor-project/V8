import asyncio

import pytest

from engine.events.event_bus import (
    CriticEvaluatedEvent,
    DecisionMadeEvent,
    Event,
    EventType,
    EscalationRequiredEvent,
    RouterSelectedEvent,
    get_event_bus,
)


@pytest.fixture(autouse=True)
def clear_event_bus_history():
    bus = get_event_bus()
    bus.clear_history()
    yield bus
    bus.clear_history()


def _build_event_for_type(event_type: EventType) -> Event:
    if event_type == EventType.CRITIC_EVALUATED:
        return CriticEvaluatedEvent(
            event_type=event_type,
            trace_id="trace-critic",
            data={"duration_ms": 1.0},
            critic_name="test-critic",
            severity=1.0,
            violations=[],
        )
    if event_type == EventType.ROUTER_SELECTED:
        return RouterSelectedEvent(
            event_type=event_type,
            trace_id="trace-router",
            data={"context": "test"},
            model_name="dummy",
            selection_reason="test",
            cost_estimate=0.0,
        )
    if event_type == EventType.ESCALATION_REQUIRED:
        return EscalationRequiredEvent(
            event_type=event_type,
            trace_id="trace-escalation",
            data={"metadata": "escalate"},
            tier=2,
            critic="rights",
            clause="001",
            rationale="test rationale",
        )
    if event_type == EventType.DECISION_MADE:
        return DecisionMadeEvent(
            event_type=event_type,
            trace_id="trace-decision",
            data={"decision_meta": "ok"},
            decision="requires_human_review",
            confidence=0.5,
            escalated=True,
        )
    # Generic event for remaining types (precedent, uncertainty, aggregation, evidence)
    return Event(event_type=event_type, trace_id=f"trace-{event_type.value}", data={"note": "audit"})


@pytest.mark.asyncio
async def test_all_event_types_are_publishable():
    bus = get_event_bus()
    published = []

    async def recorder(event):
        published.append(event.event_type)

    # Subscribe a recorder for each EventType for coverage
    for event_type in EventType:
        bus.subscribe(event_type, recorder)

    # Publish each event type once
    for event_type in EventType:
        event = _build_event_for_type(event_type)
        await bus.publish(event)

    # Allow async handlers to flush
    await asyncio.sleep(0)

    assert set(published) >= set(EventType)
    history_types = {event.event_type for event in bus.get_history()}
    assert set(history_types) >= set(EventType)
