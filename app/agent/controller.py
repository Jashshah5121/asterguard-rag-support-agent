from typing import Optional

from app.agent.intent import (
    extract_order_id,
    has_action_intent,
    has_knowledge_intent,
    has_order_intent,
    has_policy_intent,
    has_security_intent,
    has_sensitive_data_intent,
    is_contextual_followup,
    references_active_order,
)
from app.models.decision import (
    AgentDecision,
    AgentIntent,
    HandoffReason,
)
from app.models.session import SessionState


class AgentController:
    """Deterministic routing layer.

    Routing is based on the current user message plus compact durable session
    state. Historical assistant text is intentionally excluded so old order
    IDs or policy words cannot hijack a new request.
    """

    POLICY_TOPICS = {
        "international_shipping",
        "returns",
        "warranty",
        "product_care",
        "damaged_items",
    }

    def decide(
        self,
        query: str,
        session: SessionState,
    ) -> AgentDecision:
        order_id = extract_order_id(query)
        order_intent = has_order_intent(query)
        policy_intent = has_policy_intent(query)
        knowledge_intent = has_knowledge_intent(query)

        # Security and private-data requests must never reach retrieval or the
        # order tool just because they happen to contain an order ID.
        if has_security_intent(query):
            return AgentDecision(
                intent=AgentIntent.SECURITY_REQUEST,
                requires_handoff=True,
                handoff_reason=HandoffReason.SECURITY_RESTRICTION,
            )

        if has_sensitive_data_intent(query):
            return AgentDecision(
                intent=AgentIntent.PRIVACY_REQUEST,
                requires_handoff=True,
                handoff_reason=HandoffReason.PRIVACY_RESTRICTION,
            )

        # The assignment only supports read-only lookup. Requests to actually
        # cancel/refund/replace/change data are handed to a human rather than
        # being misrepresented as completed actions.
        if has_action_intent(query):
            return AgentDecision(
                intent=AgentIntent.ACTION_REQUEST,
                requires_handoff=True,
                handoff_reason=HandoffReason.UNSUPPORTED_ACTION,
            )

        # Explicit order IDs are strong references. Policy/product questions
        # about that order need both policy evidence and sanitized order facts.
        if order_id:
            if policy_intent or knowledge_intent:
                return AgentDecision(
                    intent=AgentIntent.MIXED_QUERY,
                    requires_rag=True,
                    requires_order_lookup=True,
                    order_id=order_id,
                )

            return AgentDecision(
                intent=AgentIntent.ORDER_QUERY,
                requires_order_lookup=True,
                order_id=order_id,
            )

        # Genuine tracking/status intent without an explicit ID can use the
        # session's active order, otherwise the user needs a concise prompt.
        if order_intent:
            active_order_id: Optional[str] = session.active_order_id

            if active_order_id:
                return AgentDecision(
                    intent=AgentIntent.ORDER_QUERY,
                    requires_order_lookup=True,
                    order_id=active_order_id,
                )

            return AgentDecision(
                intent=AgentIntent.ORDER_QUERY,
                requires_clarification=True,
            )

        # A vague product pronoun must not be resolved by whichever product
        # happens to rank highest in retrieval. If an active order exists, the
        # sanitized order items can provide the reference; otherwise clarify.
        if knowledge_intent and references_active_order(query) and not session.active_order_id:
            return AgentDecision(
                intent=AgentIntent.GENERAL,
                requires_clarification=True,
            )

        # A prior order should not force every later policy question through an
        # order lookup. Only pronoun/order-specific follow-ups are mixed.
        if policy_intent or knowledge_intent:
            if session.active_order_id and references_active_order(query):
                return AgentDecision(
                    intent=AgentIntent.MIXED_QUERY,
                    requires_rag=True,
                    requires_order_lookup=True,
                    order_id=session.active_order_id,
                )

            needs_human_review = any(
                term in query.lower()
                for term in ("damaged", "broken", "defective", "wrong item")
            )

            return AgentDecision(
                intent=AgentIntent.POLICY_QUERY,
                requires_rag=True,
                requires_handoff=needs_human_review,
            )

        # Short follow-ups such as "How long does it take?" can inherit a
        # compact topic, but unrelated messages are not forced into old context.
        if is_contextual_followup(query) and session.active_topic:
            if session.active_topic == "order_delivery":
                if session.active_order_id:
                    return AgentDecision(
                        intent=AgentIntent.ORDER_QUERY,
                        requires_order_lookup=True,
                        order_id=session.active_order_id,
                    )

                return AgentDecision(
                    intent=AgentIntent.ORDER_QUERY,
                    requires_clarification=True,
                )

            if session.active_topic in self.POLICY_TOPICS:
                return AgentDecision(
                    intent=AgentIntent.POLICY_QUERY,
                    requires_rag=True,
                )

        return AgentDecision(
            intent=AgentIntent.GENERAL,
            requires_clarification=True,
        )
