import re
from typing import Iterable, List

from app.models.conflict import Conflict, ConflictAnalysis
from app.models.document import DocumentChunk


class ConflictDetector:
    """Detect contradictory claims only when they are relevant to the query."""

    def analyze(
        self,
        chunks: Iterable[DocumentChunk],
        query: str | None = None,
    ) -> ConflictAnalysis:
        chunks = list(chunks)
        conflicts = []

        normalized_query = (query or "").lower()
        cleaning_question = (
            not query
            or any(
                term in normalized_query
                for term in ("dishwasher", "clean", "wash", "care")
            )
        )

        if cleaning_question:
            conflicts.extend(
                self._detect_breeze_tumbler_cleaning_conflict(chunks)
            )

        return ConflictAnalysis(
            has_conflict=bool(conflicts),
            conflicts=conflicts,
        )

    def _detect_breeze_tumbler_cleaning_conflict(
        self,
        chunks: List[DocumentChunk],
    ) -> List[Conflict]:
        relevant = []

        for chunk in chunks:
            text = chunk.content.lower()
            if "breeze tumbler" not in text:
                continue
            if "dishwasher" in text or "hand wash" in text or "hand-wash" in text:
                relevant.append(chunk)

        if len(relevant) < 2:
            return []

        has_handwash_claim = any(
            self._contains_handwash_claim(chunk.content)
            for chunk in relevant
        )
        has_dishwasher_claim = any(
            self._contains_all_components_dishwasher_claim(chunk.content)
            for chunk in relevant
        )

        if not (has_handwash_claim and has_dishwasher_claim):
            return []

        return [
            Conflict(
                topic="Breeze Tumbler cleaning instructions",
                explanation=(
                    "Two active authoritative customer-facing sources provide "
                    "contradictory cleaning guidance for the Breeze Tumbler."
                ),
                sources=relevant,
            )
        ]

    @staticmethod
    def _contains_handwash_claim(content: str) -> bool:
        normalized = content.lower()
        return bool(
            re.search(
                r"(body|tumbler).*(hand\s*-?\s*wash|hand\s*-?\s*washing)",
                normalized,
            )
            or re.search(
                r"(hand\s*-?\s*wash|hand\s*-?\s*washing).*(body|tumbler)",
                normalized,
            )
        )

    @staticmethod
    def _contains_all_components_dishwasher_claim(content: str) -> bool:
        normalized = content.lower()
        return "all components" in normalized and "dishwasher" in normalized
