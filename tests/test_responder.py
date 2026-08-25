from app.agent.responder import AgentResponder
from app.agent.orchestrator import AgentRunResult
from app.models.decision import (
    AgentDecision,
    AgentIntent,
    HandoffReason,
)
from app.models.session import SessionState
from app.models.order import SafeOrderResult
from app.models.document import DocumentMetadata, DocumentChunk
from app.models.retrieval import RetrievalResponse

from datetime import date
from unittest.mock import Mock

def make_session():
    return SessionState(session_id="test")


def make_chunk(filename, heading, content):
    return DocumentChunk(
        chunk_id=f"{filename}-{heading}",
        document_id="DOC-1",
        filename=filename,
        heading=heading,
        content=content,
        metadata=DocumentMetadata(
            document_id="DOC-1",
            title="Test",
            status="active",
            effective_date=date.today(),
            last_reviewed=date.today(),
            audience="customer",
            policy_authority="official",
        ),
    )


def test_blocked_source_conflict_returns_handoff():
    responder = AgentResponder()

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_handoff=True,
            handoff_reason=HandoffReason.SOURCE_CONFLICT,
        ),
        blocked=True,
        block_reason=HandoffReason.SOURCE_CONFLICT,
    )

    response = responder.generate(
        "Can I wash the Breeze Tumbler?",
        make_session(),
        result,
    )

    assert response.handoff is True
    assert "conflict" in response.answer.lower()


def test_order_response_is_customer_safe():
    responder = AgentResponder()

    order = SafeOrderResult(
        order_id="ORD-1007",
        status="Shipped",
        carrier="UPS",
        estimated_delivery="August 22, 2026",
    )

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.ORDER_QUERY,
            requires_order_lookup=True,
        ),
        order_data=order,
    )

    response = responder.generate(
        "Where is ORD-1007?",
        make_session(),
        result,
    )

    assert "ORD-1007" in response.answer
    assert "UPS" in response.answer
    assert "August 22, 2026" in response.answer


def test_shipped_without_eta_does_not_invent_date():
    responder = AgentResponder()

    order = SafeOrderResult(
        order_id="ORD-1011",
        status="Shipped",
        carrier="Canada Post",
        estimated_delivery=None,
    )

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.ORDER_QUERY,
            requires_order_lookup=True,
        ),
        order_data=order,
    )

    response = responder.generate(
        "When will ORD-1011 arrive?",
        make_session(),
        result,
    )

    assert "delivery estimate is currently unavailable" in response.answer.lower()


def test_clarification_response():
    responder = AgentResponder()

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.ORDER_QUERY,
            requires_clarification=True,
        ),
    )

    response = responder.generate(
        "Where is my order?",
        make_session(),
        result,
    )

    assert "provide" in response.answer.lower()


def test_sources_are_deduplicated():
    responder = AgentResponder()

    chunk1 = make_chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "30 days",
    )

    chunk2 = make_chunk(
        "01-returns-policy-current.md",
        "Return shipping",
        "Shipping fee",
    )

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_rag=True,
        ),
        retrieval_response=RetrievalResponse(
            query="returns",
            results=[],
        ),
        evidence=[chunk1, chunk2],
    )

    response = responder.generate(
        "How long can I return something?",
        make_session(),
        result,
    )

    assert response.sources == [
        "01-returns-policy-current.md - Standard return window",
        "01-returns-policy-current.md - Return shipping",
    ]

def test_cancelled_order_does_not_expose_eta():
    responder = AgentResponder()

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.ORDER_QUERY,
            requires_order_lookup=True,
        ),
        retrieval_response=RetrievalResponse(
            query="When will ORD-1004 arrive?",
            results=[],
        ),
        order_data=SafeOrderResult(
            order_id="ORD-1004",
            status="cancelled",
            carrier="UPS",
            estimated_delivery="2026-08-16",
        ),
    )

    response = responder.generate(
        "When will ORD-1004 arrive?",
        make_session(),
        result,
    )

    assert response.answer == (
        "Order ORD-1004 is cancelled. It will not be shipped."
    )
    assert "2026-08-16" not in response.answer
    assert "estimated delivery" not in response.answer.lower()

def test_llm_is_called_for_valid_evidence():
    responder = AgentResponder()

    responder.llm = Mock()
    responder.llm.generate.return_value = "Your return window is 30 calendar days."

    chunk = make_chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "Customers may return unused items within 30 calendar days of delivery.",
    )

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_rag=True,
        ),
        retrieval_response=RetrievalResponse(
            query="return policy",
            results=[],
        ),
        evidence=[chunk],
    )

    response = responder.generate(
        "How long do I have to return an item?",
        make_session(),
        result,
    )

    responder.llm.generate.assert_called_once()
    assert response.answer == "Your return window is 30 calendar days."

def test_blocked_conflict_does_not_call_llm():
    responder = AgentResponder()

    responder.llm = Mock()

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_rag=True,
            requires_handoff=True,
            handoff_reason=HandoffReason.SOURCE_CONFLICT,
        ),
        retrieval_response=RetrievalResponse(
            query="Breeze tumbler",
            results=[],
        ),
        evidence=[],
    )

    response = responder.generate(
        "Can I put the Breeze Tumbler in the dishwasher?",
        make_session(),
        result,
    )

    responder.llm.generate.assert_not_called()
    assert response.handoff is True

def test_llm_failure_has_safe_fallback():
    responder = AgentResponder()

    responder.llm = Mock()
    responder.llm.generate.side_effect = Exception("LLM unavailable")

    chunk = make_chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "Customers may return unused items within 30 calendar days of delivery.",
    )

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_rag=True,
        ),
        retrieval_response=RetrievalResponse(
            query="return policy",
            results=[],
        ),
        evidence=[chunk],
    )

    response = responder.generate(
        "How long do I have to return an item?",
        make_session(),
        result,
    )

    assert response.answer

def test_llm_is_called_for_valid_evidence():
    responder = AgentResponder()

    mock_llm = Mock()
    mock_llm.generate.return_value = "Your return window is 30 calendar days."
    responder.llm = mock_llm

    chunk = make_chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "Customers may return unused items within 30 calendar days of delivery.",
    )

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_rag=True,
        ),
        retrieval_response=RetrievalResponse(
            query="return policy",
            results=[],
        ),
        evidence=[chunk],
    )

    response = responder.generate(
        "How long do I have to return an item?",
        make_session(),
        result,
    )

    mock_llm.generate.assert_called_once()
    assert response.answer == "Your return window is 30 calendar days."


def test_blocked_conflict_does_not_call_llm():
    responder = AgentResponder()

    mock_llm = Mock()
    responder.llm = mock_llm

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_rag=True,
            requires_handoff=True,
            handoff_reason=HandoffReason.SOURCE_CONFLICT,
        ),
        retrieval_response=RetrievalResponse(
            query="Breeze tumbler",
            results=[],
        ),
        evidence=[],
    )

    response = responder.generate(
        "Can I put the Breeze Tumbler in the dishwasher?",
        make_session(),
        result,
    )

    mock_llm.generate.assert_not_called()
    assert response.handoff is True


def test_llm_failure_has_safe_fallback():
    responder = AgentResponder()

    mock_llm = Mock()
    mock_llm.generate.side_effect = Exception("LLM unavailable")
    responder.llm = mock_llm

    chunk = make_chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "Customers may return unused items within 30 calendar days of delivery.",
    )

    result = AgentRunResult(
        decision=AgentDecision(
            intent=AgentIntent.POLICY_QUERY,
            requires_rag=True,
        ),
        retrieval_response=RetrievalResponse(
            query="return policy",
            results=[],
        ),
        evidence=[chunk],
    )

    response = responder.generate(
        "How long do I have to return an item?",
        make_session(),
        result,
    )

    assert response.answer