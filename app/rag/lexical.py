import math
import re
from collections import Counter
from typing import List, Tuple

try:
    from rank_bm25 import BM25Okapi as _ExternalBM25
except ImportError:  # pragma: no cover - depends on environment
    _ExternalBM25 = None

from app.models.document import DocumentChunk


class _SimpleBM25:
    """Small dependency-free BM25 fallback.

    It is intentionally limited to the interface used by this project. The
    fallback keeps clean-clone tests and local demos working even when the
    optional rank-bm25 dependency has not been installed yet.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

        document_frequency: Counter[str] = Counter()
        for doc in corpus:
            document_frequency.update(set(doc))

        total_docs = max(len(corpus), 1)
        self.idf = {
            term: math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []

        for frequencies, doc_len in zip(self.doc_freqs, self.doc_lengths):
            score = 0.0
            length_norm = 1.0 - self.b
            if self.avgdl > 0:
                length_norm += self.b * (doc_len / self.avgdl)

            for token in query_tokens:
                tf = frequencies.get(token, 0)
                if not tf:
                    continue

                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * length_norm
                score += self.idf.get(token, 0.0) * numerator / denominator

            scores.append(score)

        return scores


class LexicalRetriever:
    """BM25 lexical retrieval.

    Exact terminology, identifiers, product names, policy names, and section
    terminology matter heavily in support content. rank-bm25 is used when
    available; a deterministic local fallback is used otherwise.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: List[DocumentChunk] = []

    def build(
        self,
        chunks: List[DocumentChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build lexical index from empty chunks."
            )

        self._chunks = list(chunks)

        tokenized_documents = [
            self._tokenize(
                self._document_text(chunk)
            )
            for chunk in self._chunks
        ]

        bm25_cls = _ExternalBM25 or _SimpleBM25
        self._bm25 = bm25_cls(tokenized_documents)

    def search(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not query.strip():
            return []

        if (
            self._bm25 is None
            or len(self._chunks) != len(chunks)
            or {
                chunk.chunk_id
                for chunk in self._chunks
            }
            != {
                chunk.chunk_id
                for chunk in chunks
            }
        ):
            self.build(chunks)

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])

            if score <= 0:
                continue

            results.append(
                (
                    self._chunks[index],
                    score,
                )
            )

        return results

    @staticmethod
    def _document_text(
        chunk: DocumentChunk,
    ) -> str:
        metadata = chunk.metadata
        metadata_terms = " ".join(
            value
            for value in (
                metadata.title,
                metadata.status,
                metadata.audience,
                metadata.policy_authority,
            )
            if value
        )

        return " ".join(
            part
            for part in (
                chunk.heading or "",
                metadata_terms,
                chunk.content,
            )
            if part
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
