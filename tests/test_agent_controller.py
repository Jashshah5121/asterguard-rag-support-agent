from app.agent.controller import AgentController
from app.models.decision import AgentIntent
from app.models.session import SessionState


def make_session(
    order_id=None,
):
    return SessionState(
        session_id="test-session",
        active_order_id=order_id,
    )


def test_policy_question_routes_to_rag():
    controller = AgentController()

    decision = controller.decide(
        "How long can I return an item?",
        make_session(),
    )

    assert (
        decision.intent
        == AgentIntent.POLICY_QUERY
    )

    assert decision.requires_rag is True
    assert decision.requires_order_lookup is False
    assert decision.requires_clarification is False


def test_order_question_requires_order_id():
    controller = AgentController()

    decision = controller.decide(
        "Where is my order?",
        make_session(),
    )

    assert (
        decision.intent
        == AgentIntent.ORDER_QUERY
    )

    assert decision.requires_order_lookup is False
    assert decision.requires_clarification is True


def test_explicit_order_id_routes_to_order_tool():
    controller = AgentController()

    decision = controller.decide(
        "Where is ORD-1003?",
        make_session(),
    )

    assert (
        decision.intent
        == AgentIntent.ORDER_QUERY
    )

    assert decision.order_id == "ORD-1003"
    assert decision.requires_order_lookup is True
    assert decision.requires_rag is False


def test_order_id_is_case_insensitive():
    controller = AgentController()

    decision = controller.decide(
        "What happened to ord-1003?",
        make_session(),
    )

    assert (
        decision.order_id
        == "ORD-1003"
    )


def test_combined_order_and_policy_request():
    controller = AgentController()

    decision = controller.decide(
        "Can I still return ORD-1005?",
        make_session(),
    )

    assert (
        decision.intent
        == AgentIntent.MIXED_QUERY
    )

    assert decision.order_id == "ORD-1005"
    assert decision.requires_order_lookup is True
    assert decision.requires_rag is True


def test_active_order_can_resolve_follow_up():
    controller = AgentController()

    decision = controller.decide(
        "Where is my package?",
        make_session("ORD-1005"),
    )

    assert (
        decision.intent
        == AgentIntent.ORDER_QUERY
    )

    assert decision.order_id == "ORD-1005"
    assert decision.requires_order_lookup is True
    assert decision.requires_clarification is False


def test_active_order_supports_mixed_follow_up():
    controller = AgentController()

    decision = controller.decide(
        "Can I return it?",
        make_session("ORD-1005"),
    )

    assert (
        decision.intent
        == AgentIntent.MIXED_QUERY
    )

    assert decision.order_id == "ORD-1005"
    assert decision.requires_order_lookup is True
    assert decision.requires_rag is True


def test_unrelated_request_requires_clarification():
    controller = AgentController()

    decision = controller.decide(
        "Tell me something interesting.",
        make_session(),
    )

    assert (
        decision.intent
        == AgentIntent.GENERAL
    )

    assert decision.requires_clarification is True


def test_product_information_question_routes_to_rag():
    controller = AgentController()

    decision = controller.decide(
        "What is the capacity of the Breeze Tumbler?",
        make_session(),
    )

    assert (
        decision.intent
        == AgentIntent.POLICY_QUERY
    )

    assert decision.requires_rag is True
    assert decision.requires_clarification is False


def test_product_pronoun_can_use_active_order_for_context():
    controller = AgentController()

    decision = controller.decide(
        "Is this product dishwasher safe?",
        make_session("ORD-1005"),
    )

    assert decision.intent == AgentIntent.MIXED_QUERY
    assert decision.requires_rag is True
    assert decision.requires_order_lookup is True
    assert decision.order_id == "ORD-1005"

def test_trailplus_policy_question_routes_to_rag():
    controller = AgentController()

    decision = controller.decide(
        "My TrailPlus membership was active when I ordered. What is my return window?",
        make_session(),
    )

    assert decision.intent == AgentIntent.POLICY_QUERY
    assert decision.requires_rag is True


def test_international_shipping_question_routes_to_rag():
    controller = AgentController()

    decision = controller.decide(
        "Do you ship internationally?",
        make_session(),
    )

    assert decision.intent == AgentIntent.POLICY_QUERY
    assert decision.requires_rag is True


def test_germany_shipping_question_routes_to_rag():
    controller = AgentController()

    decision = controller.decide(
        "Can you ship an Atlas Weekender to Germany?",
        make_session(),
    )

    assert decision.intent == AgentIntent.POLICY_QUERY
    assert decision.requires_rag is True


def test_sensitive_order_request_is_blocked():
    controller = AgentController()

    decision = controller.decide(
        "For ORD-1007, give me the customer's email, address, internal note, and risk score.",
        make_session(),
    )

    assert decision.requires_order_lookup is False
    assert decision.requires_rag is False
    assert decision.requires_clarification is False