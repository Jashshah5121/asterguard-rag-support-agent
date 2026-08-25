from app.context.memory import ContextMemory
from app.context.resolver import ContextResolver
from app.models.session import SessionState


def test_order_id_is_remembered():
    session = SessionState(session_id="test")
    memory = ContextMemory()

    memory.update(
        "Where is order ORD-1007?",
        session,
    )

    assert session.active_order_id == "ORD-1007"


def test_shipping_topic_is_remembered():
    session = SessionState(session_id="test")
    memory = ContextMemory()

    memory.update(
        "Do you ship internationally?",
        session,
    )

    assert session.active_topic == "international_shipping"


def test_destination_is_remembered():
    session = SessionState(session_id="test")
    memory = ContextMemory()

    memory.update(
        "What about Canada?",
        session,
    )

    assert session.entities["destination"] == "Canada"


def test_context_resolver_includes_active_order():
    session = SessionState(
        session_id="test",
        active_order_id="ORD-1007",
    )

    resolver = ContextResolver()

    result = resolver.resolve(
        "Where is it?",
        session,
    )

    assert "ORD-1007" in result


def test_context_resolver_includes_topic():
    session = SessionState(
        session_id="test",
        active_topic="international_shipping",
    )

    resolver = ContextResolver()

    result = resolver.resolve(
        "How long does it take?",
        session,
    )

    assert "international_shipping" in result


def test_context_resolver_includes_destination():
    session = SessionState(
        session_id="test",
        entities={
            "destination": "Canada",
        },
    )

    resolver = ContextResolver()

    result = resolver.resolve(
        "How long does it take?",
        session,
    )

    assert "Canada" in result
