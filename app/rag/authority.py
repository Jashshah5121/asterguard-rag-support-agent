from dataclasses import dataclass
from typing import Iterable, List, Optional

from app.models.document import DocumentChunk


@dataclass(frozen=True)
class AuthorityDecision:
    """
    Describes whether a document is suitable as customer-facing evidence.
    """

    usable: bool
    priority: int
    reason: str


class AuthorityResolver:
    """
    Applies deterministic source-authority rules.

    The LLM must not decide whether a document is authoritative.
    """

    def evaluate(self, chunk: DocumentChunk) -> AuthorityDecision:
        metadata = chunk.metadata

        # Explicitly non-customer-answering content is never usable
        # as customer-facing policy evidence.
        if metadata.customer_answering is False:
            return AuthorityDecision(
                usable=False,
                priority=0,
                reason="document explicitly disallows customer answering",
            )

        # Draft content is not authoritative for customer answers.
        if metadata.status == "draft":
            return AuthorityDecision(
                usable=False,
                priority=0,
                reason="document is a draft",
            )

        # Superseded documents should not be used when a current
        # authoritative source exists.
        if metadata.status == "superseded":
            return AuthorityDecision(
                usable=False,
                priority=10,
                reason="document has been superseded",
            )

        # Internal content should not normally be used as
        # customer-facing policy.
        if metadata.audience == "internal":
            return AuthorityDecision(
                usable=False,
                priority=20,
                reason="document is internal",
            )

        # Official active customer-facing content is the strongest
        # normal source.
        if (
            metadata.status == "active"
            and metadata.policy_authority == "official"
            and metadata.audience == "customer"
        ):
            return AuthorityDecision(
                usable=True,
                priority=100,
                reason="active official customer-facing source",
            )

        # Active official content without an explicit audience.
        if (
            metadata.status == "active"
            and metadata.policy_authority == "official"
        ):
            return AuthorityDecision(
                usable=True,
                priority=90,
                reason="active official source",
            )

        # Active customer-facing content with unspecified authority.
        if (
            metadata.status == "active"
            and metadata.audience == "customer"
        ):
            return AuthorityDecision(
                usable=True,
                priority=70,
                reason="active customer-facing source",
            )

        # Conservative fallback.
        return AuthorityDecision(
            usable=False,
            priority=0,
            reason="source does not meet customer-answering authority requirements",
        )

    def filter_usable(
        self,
        chunks: Iterable[DocumentChunk],
    ) -> List[DocumentChunk]:
        return [
            chunk
            for chunk in chunks
            if self.evaluate(chunk).usable
        ]

    def rank(
        self,
        chunks: Iterable[DocumentChunk],
    ) -> List[DocumentChunk]:
        return sorted(
            chunks,
            key=lambda chunk: self.evaluate(chunk).priority,
            reverse=True,
        )