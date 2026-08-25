import re

from app.agent.intent import extract_order_id
from app.models.session import SessionState


class ContextMemory:
    """
    Compact conversational memory for the support agent.

    The goal is NOT to blindly inject the entire conversation into every
    request. Instead, we preserve stable entities and use recent turns only
    when the new message is clearly a follow-up.

    This prevents old order IDs or old topics from contaminating unrelated
    new questions while still allowing natural follow-ups such as:

        "Why?"
        "What about cleaning it normally?"
        "How long does that take?"
        "What about Canada?"
        "When will it arrive?"
    """

    FOLLOWUP_EXACT = {
        "why",
        "why?",
        "how",
        "how?",
        "when",
        "when?",
        "where",
        "where?",
        "what else",
        "what else?",
        "how so",
        "how so?",
        "why not",
        "why not?",
        "what about that",
        "what about that?",
        "what about it",
        "what about it?",
        "what about this",
        "what about this?",
        "and then",
        "and then?",
    }

    FOLLOWUP_PREFIXES = (
        "what about ",
        "how about ",
        "why is that",
        "why does",
        "why do",
        "does it",
        "can it",
        "is it",
        "can they",
        "do they",
        "what does that",
        "what do you mean",
        "what happens if",
        "and what about",
        "and how about",
        "how long",
        "how much time",
        "when will",
        "when does",
        "where is it",
    )

    PRODUCT_ALIASES = {
        "breeze tumbler": "Breeze Tumbler",
        "atlas weekender": "Atlas Weekender",
    }

    # ============================================================
    # PUBLIC API
    # ============================================================

    def update(
        self,
        query: str,
        session: SessionState,
        answer: str | None = None,
    ) -> None:
        """
        Update durable conversational state after a response is generated.

        Important:
        This should be called AFTER routing and retrieval so the current
        message does not overwrite context needed to resolve itself.
        """

        query = query.strip()

        if not query:
            return

        # --------------------------------------------------------
        # Order ID
        # --------------------------------------------------------

        order_id = extract_order_id(query)

        if order_id:
            session.active_order_id = order_id

        # --------------------------------------------------------
        # Topic
        # --------------------------------------------------------

        topic = self._detect_topic(query)

        if topic:
            session.active_topic = topic

        # --------------------------------------------------------
        # Destination
        # --------------------------------------------------------

        destination = self._detect_destination(query)

        if destination:
            session.entities["destination"] = destination

        # --------------------------------------------------------
        # Product
        # --------------------------------------------------------

        product = self._detect_product(query)

        if product:
            session.entities["product"] = product

        # --------------------------------------------------------
        # Previous question/answer
        # --------------------------------------------------------

        session.entities["last_user_query"] = query

        if answer:
            # Limit stored answer size so memory remains compact.
            session.entities["last_assistant_answer"] = answer[:1500]

        # --------------------------------------------------------
        # Turn counter
        # --------------------------------------------------------

        session.turn_count += 1

    def resolve_followup(
        self,
        query: str,
        session: SessionState,
    ) -> str:
        """
        Convert short contextual follow-ups into explicit standalone queries.

        Examples:

            Why?
            ->
            Explain why the previous answer said there is conflicting
            dishwasher guidance for the Breeze Tumbler.

            What about cleaning it normally?
            ->
            What are the normal cleaning and care instructions for
            the Breeze Tumbler?

        The returned text is intended for retrieval/RAG only.
        The original query should still be used for intent routing.
        """

        query = query.strip()

        if not query:
            return query

        normalized = self._normalize(query)

        if not self.is_followup(query):
            return query

        product = session.entities.get("product")

        previous_query = session.entities.get(
            "last_user_query"
        )

        previous_answer = session.entities.get(
            "last_assistant_answer"
        )

        destination = session.entities.get(
            "destination"
        )

        active_topic = getattr(
            session,
            "active_topic",
            None,
        )

        active_order_id = getattr(
            session,
            "active_order_id",
            None,
        )

        # --------------------------------------------------------
        # WHY?
        # --------------------------------------------------------

        if normalized in {
            "why",
            "why?",
            "why is that",
            "why is that?",
            "how so",
            "how so?",
        }:

            # Product-care conflict is an important special case.
            if (
                active_topic == "product_care"
                and product
                and previous_answer
                and self._looks_like_conflict(previous_answer)
            ):
                return (
                    f"Explain why the official product-care sources "
                    f"conflict for {product}. "
                    f"The previous user question was: "
                    f"{previous_query or 'unknown'}. "
                    f"The previous answer said: "
                    f"{previous_answer[:800]}"
                )

            if previous_query:
                return (
                    f"Explain the reason behind the previous answer "
                    f"to this question: {previous_query}. "
                    f"Previous answer: "
                    f"{(previous_answer or '')[:800]}"
                )

        # --------------------------------------------------------
        # PRODUCT CARE FOLLOW-UPS
        # --------------------------------------------------------

        if (
            product
            and any(
                phrase in normalized
                for phrase in (
                    "cleaning",
                    "clean it",
                    "cleaning it",
                    "wash it",
                    "wash normally",
                    "clean normally",
                    "clean it normally",
                    "what about cleaning",
                    "how about cleaning",
                )
            )
        ):
            return (
                f"What are the normal cleaning and care instructions "
                f"for {product}?"
            )

        if (
            product
            and normalized
            in {
                "can it go in the dishwasher",
                "can it go in the dishwasher?",
                "is it dishwasher safe",
                "is it dishwasher safe?",
                "can i wash it",
                "can i wash it?",
            }
        ):
            return (
                f"Is {product} dishwasher safe and what are "
                f"the official cleaning instructions?"
            )

        # --------------------------------------------------------
        # ORDER FOLLOW-UPS
        # --------------------------------------------------------

        if active_order_id:

            if any(
                phrase in normalized
                for phrase in (
                    "when will",
                    "when does",
                    "how long",
                    "how much time",
                    "arrive",
                    "delivery",
                    "get here",
                )
            ):
                return (
                    f"What is the current delivery status and estimated "
                    f"arrival time for order {active_order_id}?"
                )

            if normalized in {
                "where is it",
                "where is it?",
                "where?",
            }:
                return (
                    f"What is the current shipping status of "
                    f"order {active_order_id}?"
                )

            if any(
                phrase in normalized
                for phrase in (
                    "can i return it",
                    "can i return this",
                    "what about returning it",
                    "return it",
                    "return this",
                )
            ):
                return (
                    f"Can order {active_order_id} be returned under "
                    f"the current return policy?"
                )

        # --------------------------------------------------------
        # DESTINATION FOLLOW-UPS
        # --------------------------------------------------------

        if (
            destination
            and active_topic == "international_shipping"
        ):
            if any(
                phrase in normalized
                for phrase in (
                    "how long",
                    "how much time",
                    "shipping time",
                    "delivery time",
                    "take to ship",
                    "takes to ship",
                    "arrive",
                )
            ):
                return (
                    f"What is the estimated processing and delivery "
                    f"time for shipping to {destination}?"
                )

        # --------------------------------------------------------
        # GENERIC CONTEXTUAL QUERY
        # --------------------------------------------------------

        context_parts: list[str] = []

        if product:
            context_parts.append(
                f"Current product: {product}"
            )

        if active_order_id:
            context_parts.append(
                f"Current order: {active_order_id}"
            )

        if destination:
            context_parts.append(
                f"Current destination: {destination}"
            )

        if active_topic:
            context_parts.append(
                f"Previous topic: {active_topic}"
            )

        if previous_query:
            context_parts.append(
                f"Previous user question: {previous_query}"
            )

        if previous_answer:
            context_parts.append(
                f"Previous assistant answer: "
                f"{previous_answer[:700]}"
            )

        if not context_parts:
            return query

        return (
            "\n".join(context_parts)
            + "\n\n"
            + f"Current follow-up question: {query}"
        )

    def is_followup(
        self,
        query: str,
    ) -> bool:
        """
        Return True when the current message appears to depend on previous
        conversational context.
        """

        normalized = self._normalize(query)

        if normalized in self.FOLLOWUP_EXACT:
            return True

        if any(
            normalized.startswith(prefix)
            for prefix in self.FOLLOWUP_PREFIXES
        ):
            return True

        # Pronouns often indicate context dependence.
        pronoun_pattern = re.compile(
            r"\b("
            r"it|that|this|there|they|them|those|these"
            r")\b",
            re.IGNORECASE,
        )

        # Keep this conservative: only short messages should automatically
        # become contextual based on pronouns.
        if (
            len(normalized.split()) <= 10
            and pronoun_pattern.search(normalized)
        ):
            return True

        return False

    # ============================================================
    # TOPIC DETECTION
    # ============================================================

    def _detect_topic(
        self,
        query: str,
    ) -> str | None:

        normalized = query.lower()

        if any(
            term in normalized
            for term in (
                "international",
                "internationally",
                "canada",
                "canadian",
                "germany",
                "german",
                "destination",
            )
        ):
            return "international_shipping"

        if any(
            term in normalized
            for term in (
                "damaged",
                "damage",
                "broken",
                "wrong item",
                "wrong product",
            )
        ):
            return "damaged_items"

        if any(
            term in normalized
            for term in (
                "return",
                "returns",
                "refund",
                "exchange",
                "final sale",
            )
        ):
            return "returns"

        if "warranty" in normalized:
            return "warranty"

        if "trailplus" in normalized:
            return "trailplus"

        if any(
            term in normalized
            for term in (
                "dishwasher",
                "cleaning",
                "clean",
                "care",
                "wash",
                "breeze tumbler",
            )
        ):
            return "product_care"

        if (
            extract_order_id(query)
            or re.search(
                r"\b(?:"
                r"track|tracking|order status|"
                r"where is my order|package status"
                r")\b",
                normalized,
            )
            or re.search(
                r"\bwhen\b.*\b(?:"
                r"arrive|get here|delivered"
                r")\b",
                normalized,
            )
        ):
            return "order_delivery"

        return None

    # ============================================================
    # ENTITY DETECTION
    # ============================================================

    def _detect_destination(
        self,
        query: str,
    ) -> str | None:

        normalized = query.lower()

        if "canada" in normalized:
            return "Canada"

        if (
            "germany" in normalized
            or "german" in normalized
        ):
            return "Germany"

        return None

    def _detect_product(
        self,
        query: str,
    ) -> str | None:

        normalized = query.lower()

        for alias, canonical_name in self.PRODUCT_ALIASES.items():

            if alias in normalized:
                return canonical_name

        return None

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            text.strip().lower(),
        )

    @staticmethod
    def _looks_like_conflict(
        answer: str,
    ) -> bool:

        normalized = answer.lower()

        indicators = (
            "conflict",
            "conflicting",
            "one says",
            "another says",
            "sources disagree",
            "official sources",
            "cannot safely",
            "human support",
            "human representative",
        )

        return any(
            indicator in normalized
            for indicator in indicators
        )