from app.rag.conflicts import ConflictDetector
from app.rag.evidence import EvidenceSelector
from app.rag.retriever import HybridRetriever


class RAGPipeline:
    """Coordinate retrieval, authority filtering, evidence, and conflicts."""

    def __init__(
        self,
        retriever: HybridRetriever,
        evidence_selector: EvidenceSelector,
        conflict_detector: ConflictDetector,
    ):
        self.retriever = retriever
        self.evidence_selector = evidence_selector
        self.conflict_detector = conflict_detector

    def retrieve(
        self,
        query: str,
        chunks,
        candidate_k: int = 18,
        evidence_k: int = 7,
    ):
        retrieval_query = self._expand_query(query)

        retrieval_response = self.retriever.search(
            query=retrieval_query,
            chunks=chunks,
            top_k=candidate_k,
        )
        # Preserve the customer's/context-resolved query in observability.
        retrieval_response.query = query

        evidence = self.evidence_selector.select(
            query=query,
            results=retrieval_response.results,
            max_results=evidence_k,
        )

        conflict_analysis = self.conflict_detector.analyze(
            (result.chunk for result in evidence),
            query=query,
        )

        return retrieval_response, evidence, conflict_analysis

    @staticmethod
    def _expand_query(query: str) -> str:
        q = query.lower()
        additions: list[str] = []

        if any(term in q for term in ("international", "ship", "canada", "germany")):
            additions.append(
                "international shipping supported destinations Canada other countries delivery estimate duties taxes"
            )

        if (
            any(term in q for term in ("damaged", "broken", "defective", "wrong"))
            and any(term in q for term in ("final sale", "final-sale"))
        ):
            additions.append(
                "damaged defective wrong item final-sale reporting window available resolutions human review"
            )

        if "trailplus" in q:
            additions.append(
                "TrailPlus membership benefits 45 calendar day return window "
                "free standard United States shipping membership verification "
                "final sale restrictions international charges"
            )

        if ("regular" in q or "standard" in q) and "return" in q:
            additions.append("standard return window item condition delivery")

        if "warranty" in q:
            additions.append("warranty periods covered not covered")

        if "category bags backpacks" in q:
            additions.append("bags backpacks product care spot-clean mild soap machine wash")
        elif "category packing cubes" in q:
            additions.append("packing cubes product care hand-wash cool water")
        elif "breeze tumbler" in q or "capacity" in q:
            additions.append("Breeze Tumbler product details cleaning care")
        elif any(term in q for term in ("dishwasher", "cleaning")):
            additions.append("product care cleaning dishwasher")

        return query if not additions else query + "\n" + " ".join(additions)