from app.agent.intent import is_contextual_followup, references_active_order
from app.models.session import SessionState


class ContextResolver:
    """Add only compact context that is relevant to a follow-up query.

    Full conversation transcripts are deliberately not appended to retrieval
    queries. Doing so caused old order IDs, old policy terms, and assistant
    wording to contaminate routing/retrieval on later turns.
    """

    def resolve(
        self,
        query: str,
        session: SessionState,
    ) -> str:
        query = query.strip()
        if not query:
            return query

        needs_context = (
            is_contextual_followup(query)
            or references_active_order(query)
        )

        if not needs_context:
            return query

        context = self._build_context(query, session)
        if not context:
            return query

        return (
            f"{query}\n\n"
            "Relevant prior conversation context:\n"
            f"{context}"
        )

    def _build_context(
        self,
        query: str,
        session: SessionState,
    ) -> str:
        parts: list[str] = []

        if session.active_topic:
            parts.append(f"Previous topic: {session.active_topic}")

        if session.active_order_id and references_active_order(query):
            parts.append(f"Active order: {session.active_order_id}")

        if (
            session.entities.get("destination")
            and (
                session.active_topic == "international_shipping"
                or session.active_topic is None
            )
        ):
            parts.append(
                f"Previous destination: {session.entities['destination']}"
            )

        return "\n".join(parts)
