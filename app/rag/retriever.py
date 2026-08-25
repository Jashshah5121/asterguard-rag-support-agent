from typing import Dict, List

from app.models.document import DocumentChunk
from app.models.retrieval import (
    RetrievalResponse,
    RetrievalResult,
)

from app.rag.index import VectorIndex
from app.rag.lexical import LexicalRetriever


class HybridRetriever:
    """
    Hybrid retrieval using dense semantic search and BM25.

    The retriever produces a candidate pool. It does not decide
    whether evidence is authoritative or safe for customer use.
    """

    RRF_K = 60

    def __init__(
        self,
        vector_index: VectorIndex,
        lexical_retriever: LexicalRetriever,
    ):
        self.vector_index = vector_index
        self.lexical_retriever = lexical_retriever

    def search(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 10,
    ) -> RetrievalResponse:

        candidate_k = max(
            top_k * 2,
            15,
        )

        semantic_results = self.vector_index.search(
            query,
            top_k=candidate_k,
        )

        lexical_results = self.lexical_retriever.search(
            query,
            chunks,
            top_k=candidate_k,
        )

        semantic_rank: Dict[str, int] = {
            chunk.chunk_id: rank
            for rank, (chunk, _) in enumerate(
                semantic_results,
                start=1,
            )
        }

        lexical_rank: Dict[str, int] = {
            chunk.chunk_id: rank
            for rank, (chunk, _) in enumerate(
                lexical_results,
                start=1,
            )
        }

        semantic_scores: Dict[str, float] = {
            chunk.chunk_id: score
            for chunk, score in semantic_results
        }

        lexical_scores: Dict[str, float] = {
            chunk.chunk_id: score
            for chunk, score in lexical_results
        }

        chunk_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        candidate_ids = (
            set(semantic_rank)
            | set(lexical_rank)
        )

        results = []

        for chunk_id in candidate_ids:
            semantic_position = semantic_rank.get(
                chunk_id
            )

            lexical_position = lexical_rank.get(
                chunk_id
            )

            rrf_score = 0.0

            if semantic_position is not None:
                rrf_score += 1.0 / (
                    self.RRF_K + semantic_position
                )

            if lexical_position is not None:
                rrf_score += 1.0 / (
                    self.RRF_K + lexical_position
                )

            chunk = chunk_by_id[chunk_id]

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    semantic_score=semantic_scores.get(
                        chunk_id,
                        0.0,
                    ),
                    lexical_score=lexical_scores.get(
                        chunk_id,
                        0.0,
                    ),
                    final_score=rrf_score,
                )
            )

        results.sort(
            key=lambda result: result.final_score,
            reverse=True,
        )

        return RetrievalResponse(
            query=query,
            results=results[:top_k],
        )