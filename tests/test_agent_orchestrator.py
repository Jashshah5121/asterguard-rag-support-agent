from app.agent.controller import AgentController
from app.agent.orchestrator import AgentOrchestrator
from app.models.decision import (
    AgentIntent,
    HandoffReason,
)
from app.models.session import SessionState
from app.rag.conflicts import ConflictDetector
from app.rag.evidence import EvidenceSelector
from app.rag.retriever import HybridRetriever
from app.rag.pipeline import RAGPipeline
from app.rag.index import VectorIndex
from app.rag.parser import KnowledgeBaseParser
from app.config import settings
from app.orders.repository import OrderRepository
from app.orders.service import OrderService
from app.tools.order_lookup import OrderLookupTool
from app.rag.lexical import LexicalRetriever

def build_orchestrator():
    parser = KnowledgeBaseParser()

    chunks = parser.parse_directory(
        settings.knowledge_base_path
    )

    vector_index = VectorIndex()
    vector_index.build(chunks)

    lexical_retriever = LexicalRetriever()
    lexical_retriever.build(chunks)

    retriever = HybridRetriever(
        vector_index=vector_index,
        lexical_retriever=lexical_retriever,
    )

    evidence_selector = EvidenceSelector()
    conflict_detector = ConflictDetector()

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        evidence_selector=evidence_selector,
        conflict_detector=conflict_detector,
    )

    repository = OrderRepository(
        settings.orders_path
    )

    service = OrderService(repository)

    order_tool = OrderLookupTool(service)

    controller = AgentController()

    orchestrator = AgentOrchestrator(
        controller=controller,
        rag_pipeline=rag_pipeline,
        order_tool=order_tool,
        chunks=chunks,
    )

    return orchestrator


def make_session(
    order_id=None,
):
    return SessionState(
        session_id="test-session",
        active_order_id=order_id,
    )


def test_policy_query_executes_rag():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "How long can I return an item?",
        make_session(),
    )

    assert (
        result.decision.intent
        == AgentIntent.POLICY_QUERY
    )

    assert result.retrieval_response is not None
    assert len(result.evidence) > 0
    assert result.blocked is False


def test_order_query_executes_order_tool():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "Where is ORD-1003?",
        make_session(),
    )

    assert (
        result.decision.intent
        == AgentIntent.ORDER_QUERY
    )

    assert result.order_data is not None
    assert (
        result.order_data.order_id
        == "ORD-1003"
    )

    assert result.retrieval_response is None
    assert result.blocked is False


def test_mixed_query_executes_both_capabilities():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "Can I still return ORD-1005?",
        make_session(),
    )

    assert (
        result.decision.intent
        == AgentIntent.MIXED_QUERY
    )

    assert result.retrieval_response is not None
    assert len(result.evidence) > 0

    assert result.order_data is not None
    assert (
        result.order_data.order_id
        == "ORD-1005"
    )

    assert result.blocked is False


def test_missing_order_id_does_not_call_tools():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "Where is my order?",
        make_session(),
    )

    assert (
        result.decision.requires_clarification
        is True
    )

    assert result.retrieval_response is None
    assert result.order_data is None


def test_active_order_supports_follow_up():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "Can I return it?",
        make_session("ORD-1005"),
    )

    assert (
        result.decision.intent
        == AgentIntent.MIXED_QUERY
    )

    assert result.order_data is not None
    assert (
        result.order_data.order_id
        == "ORD-1005"
    )

    assert result.retrieval_response is not None
    assert len(result.evidence) > 0


def test_unrelated_breeze_conflict_does_not_block_capacity_answer():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "What is the capacity of the Breeze Tumbler?",
        make_session(),
    )

    assert result.blocked is False
    assert result.evidence


def test_breeze_cleaning_conflict_blocks_answer():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        make_session(),
    )

    assert result.blocked is True
    assert result.block_reason == HandoffReason.SOURCE_CONFLICT
    assert result.decision.requires_handoff is True


def test_unknown_order_is_handled_safely():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "Where is ORD-9999?",
        make_session(),
    )

    assert result.blocked is True

    assert (
        result.block_reason
        == HandoffReason.ORDER_NOT_FOUND
    )

    assert result.order_data is None


def test_internal_order_data_never_reaches_result():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        "What happened to ORD-1005?",
        make_session(),
    )

    assert result.order_data is not None

    serialized = str(
        result.order_data.model_dump()
    ).lower()

    assert "risk_score" not in serialized
    assert "warehouse_note" not in serialized
    assert "support_tags" not in serialized