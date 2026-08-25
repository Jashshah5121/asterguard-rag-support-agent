from typing import List, Optional

from app.agent.controller import AgentController
from app.models.decision import (
    AgentDecision,
    HandoffReason,
)
from app.models.document import DocumentChunk
from app.models.order import SafeOrderResult
from app.models.retrieval import RetrievalResponse
from app.models.conflict import ConflictAnalysis
from app.models.session import SessionState
from app.rag.pipeline import RAGPipeline
from app.tools.order_lookup import OrderLookupTool


class AgentRunResult:
    """
    Result of one deterministic agent execution.

    This object contains the information required by the
    response-generation layer. It does not generate the
    customer-facing response itself.
    """

    def __init__(
        self,
        decision: AgentDecision,
        retrieval_response: Optional[RetrievalResponse] = None,
        evidence: Optional[List] = None,
        conflict_analysis: Optional[ConflictAnalysis] = None,
        order_data: Optional[SafeOrderResult] = None,
        blocked: bool = False,
        block_reason: Optional[HandoffReason] = None,
    ):
        self.decision = decision
        self.retrieval_response = retrieval_response
        self.evidence = evidence or []
        self.conflict_analysis = conflict_analysis
        self.order_data = order_data
        self.blocked = blocked
        self.block_reason = block_reason


class AgentOrchestrator:
    """
    Coordinates the agent controller, RAG pipeline,
    and order lookup tool.

    Responsibilities:
    - obtain the routing decision
    - execute required capabilities
    - enforce safety boundaries
    - return structured evidence for response generation

    This layer does not generate natural-language responses.
    """

    def __init__(
        self,
        controller: AgentController,
        rag_pipeline: RAGPipeline,
        order_tool: OrderLookupTool,
        chunks: List[DocumentChunk],
    ) -> None:
        self.controller = controller
        self.rag_pipeline = rag_pipeline
        self.order_tool = order_tool
        self.chunks = chunks

    def run(
        self,
        query: str,
        session: SessionState,
        retrieval_query: str | None = None,
    ) -> AgentRunResult:

        decision = self.controller.decide(
            query=query,
            session=session,
        )

        if decision.requires_clarification:
            return AgentRunResult(decision=decision)

        if (
            decision.requires_handoff
            and not decision.requires_rag
            and not decision.requires_order_lookup
        ):
            return AgentRunResult(
                decision=decision,
                blocked=True,
                block_reason=decision.handoff_reason,
            )

        retrieval_response = None
        evidence = []
        conflict_analysis = None
        order_data = None

        # For mixed questions, look up the order first. The safe order facts
        # (membership/item flags) materially improve which policy is retrieved.
        if decision.requires_order_lookup and decision.requires_rag:
            lookup = self._lookup_order(decision)
            if isinstance(lookup, AgentRunResult):
                return lookup
            order_data = lookup

        if decision.requires_rag:
            rag_query = retrieval_query or query
            if order_data is not None:
                rag_query += self._retrieval_order_context(order_data)

            (
                retrieval_response,
                evidence,
                conflict_analysis,
            ) = self.rag_pipeline.retrieve(
                query=rag_query,
                chunks=self.chunks,
            )

            if conflict_analysis.has_conflict:
                decision.requires_handoff = True
                decision.handoff_reason = HandoffReason.SOURCE_CONFLICT
                return AgentRunResult(
                    decision=decision,
                    retrieval_response=retrieval_response,
                    evidence=evidence,
                    conflict_analysis=conflict_analysis,
                    order_data=order_data,
                    blocked=True,
                    block_reason=HandoffReason.SOURCE_CONFLICT,
                )

            if not evidence:
                decision.requires_handoff = True
                decision.handoff_reason = HandoffReason.INSUFFICIENT_EVIDENCE
                return AgentRunResult(
                    decision=decision,
                    retrieval_response=retrieval_response,
                    evidence=evidence,
                    conflict_analysis=conflict_analysis,
                    order_data=order_data,
                    blocked=True,
                    block_reason=HandoffReason.INSUFFICIENT_EVIDENCE,
                )

        # Pure order-status requests perform the lookup here. Mixed requests
        # already performed it before RAG above.
        if decision.requires_order_lookup and order_data is None:
            lookup = self._lookup_order(decision)
            if isinstance(lookup, AgentRunResult):
                lookup.retrieval_response = retrieval_response
                lookup.evidence = evidence
                lookup.conflict_analysis = conflict_analysis
                return lookup
            order_data = lookup

        return AgentRunResult(
            decision=decision,
            retrieval_response=retrieval_response,
            evidence=evidence,
            conflict_analysis=conflict_analysis,
            order_data=order_data,
        )

    def _lookup_order(self, decision: AgentDecision):
        if not decision.order_id:
            decision.requires_clarification = True
            return AgentRunResult(decision=decision)

        order_result = self.order_tool.execute(decision.order_id)
        if not order_result["success"]:
            decision.requires_handoff = True
            decision.handoff_reason = HandoffReason.ORDER_NOT_FOUND
            return AgentRunResult(
                decision=decision,
                blocked=True,
                block_reason=HandoffReason.ORDER_NOT_FOUND,
            )

        return SafeOrderResult.model_validate(order_result["data"])

    @staticmethod
    def _retrieval_order_context(order: SafeOrderResult) -> str:
        item_terms = []
        for item in order.items or []:
            name = str(item.get("name", ""))
            normalized_name = name.lower()
            category = ""
            if any(term in normalized_name for term in ("daypack", "backpack", "bag", "weekender", "tote")):
                category = " category bags backpacks"
            elif "cube" in normalized_name:
                category = " category packing cubes"
            elif "tumbler" in normalized_name:
                category = " category drinkware breeze tumbler"

            item_terms.append(
                f"item {name}{category} final_sale {item.get('final_sale', False)}"
            )

        return (
            "\nSanitized order context for policy retrieval: "
            f"membership {order.membership_tier or 'unknown'}; "
            + "; ".join(item_terms)
        )
