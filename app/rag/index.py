import hashlib
import re
from pathlib import Path
from typing import Any, List

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - environment dependent
    SentenceTransformer = None  # type: ignore

from app.models.document import DocumentChunk


class VectorIndex:
    """Semantic index with a graceful deterministic fallback.

    The normal path remains SentenceTransformers + FAISS. If either optional
    dependency is unavailable (or the model cannot be loaded), the class falls
    back to a small hashed bag-of-words cosine index. This prevents the entire
    application from failing at startup while preserving the project's hybrid
    retrieval architecture.
    """

    FALLBACK_DIMENSION = 1024
    FALLBACK_FILENAME = "semantic-fallback.npz"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model: Any = None
        self.index: Any = None
        self.chunks: List[DocumentChunk] = []
        self._fallback_matrix: np.ndarray | None = None
        self.backend = "uninitialized"

    def _get_model(self):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")

        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        return self.model

    def build(
        self,
        chunks: List[DocumentChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build vector index from empty chunks."
            )

        self.chunks = list(chunks)

        if faiss is not None and SentenceTransformer is not None:
            try:
                texts = [
                    self._document_text(chunk)
                    for chunk in chunks
                ]

                model = self._get_model()

                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                embeddings = np.asarray(
                    embeddings,
                    dtype="float32",
                )

                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)
                self._fallback_matrix = None
                self.backend = "faiss-transformer"
                return
            except Exception:
                # Model downloads or native FAISS initialization can fail in
                # restricted environments. Retrieval must still remain usable.
                self.model = None
                self.index = None

        self._build_fallback(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        if not query.strip():
            return []

        if not self.chunks:
            raise RuntimeError(
                "Vector index has not been built or loaded."
            )

        if self.backend == "faiss-transformer" and self.index is not None:
            try:
                model = self._get_model()
                query_embedding = model.encode(
                    [query],
                    normalize_embeddings=True,
                )
                query_embedding = np.asarray(
                    query_embedding,
                    dtype="float32",
                )

                scores, indices = self.index.search(
                    query_embedding,
                    min(top_k, len(self.chunks)),
                )

                return [
                    (self.chunks[index], float(score))
                    for score, index in zip(scores[0], indices[0])
                    if index >= 0
                ]
            except Exception:
                # Degrade locally instead of failing the customer request.
                self._build_fallback(self.chunks)

        if self._fallback_matrix is None:
            self._build_fallback(self.chunks)

        query_vector = self._fallback_encode([query])[0]
        scores = self._fallback_matrix @ query_vector
        ranked = np.argsort(-scores)[: min(top_k, len(self.chunks))]

        return [
            (self.chunks[int(index)], float(scores[int(index)]))
            for index in ranked
            if float(scores[int(index)]) > 0.0
        ]

    def save(
        self,
        directory: Path,
    ) -> None:
        if not self.chunks:
            raise RuntimeError(
                "Cannot save an unbuilt vector index."
            )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.backend == "faiss-transformer" and self.index is not None and faiss is not None:
            faiss.write_index(
                self.index,
                str(directory / "faiss.index"),
            )
            return

        if self._fallback_matrix is None:
            self._build_fallback(self.chunks)

        np.savez_compressed(
            directory / self.FALLBACK_FILENAME,
            vectors=self._fallback_matrix,
        )

    def load(
        self,
        directory: Path,
    ) -> None:
        from app.rag.store import ChunkStore

        self.chunks = ChunkStore().load(directory)
        fallback_file = directory / self.FALLBACK_FILENAME
        faiss_file = directory / "faiss.index"

        if fallback_file.exists():
            data = np.load(fallback_file)
            matrix = np.asarray(data["vectors"], dtype="float32")
            if len(matrix) != len(self.chunks):
                raise ValueError(
                    "Fallback semantic index and chunk store are out of sync."
                )
            self._fallback_matrix = matrix
            self.index = None
            self.backend = "hashed-fallback"
            return

        if faiss is not None and faiss_file.exists():
            self.index = faiss.read_index(str(faiss_file))
            if self.index.ntotal != len(self.chunks):
                raise ValueError(
                    "FAISS index and chunk store are out of sync: "
                    f"index contains {self.index.ntotal} vectors, "
                    f"but {len(self.chunks)} chunks were loaded."
                )
            self.backend = "faiss-transformer"
            return

        # A persisted chunk store is enough to reconstruct the lightweight
        # fallback. This is particularly useful on machines without FAISS.
        self._build_fallback(self.chunks)

    def _build_fallback(self, chunks: List[DocumentChunk]) -> None:
        texts = [self._document_text(chunk) for chunk in chunks]
        self._fallback_matrix = self._fallback_encode(texts)
        self.index = None
        self.backend = "hashed-fallback"

    @classmethod
    def _fallback_encode(cls, texts: list[str]) -> np.ndarray:
        matrix = np.zeros(
            (len(texts), cls.FALLBACK_DIMENSION),
            dtype="float32",
        )

        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            features = tokens + [
                f"{left}_{right}"
                for left, right in zip(tokens, tokens[1:])
            ]

            for feature in features:
                digest = hashlib.blake2b(
                    feature.encode("utf-8"),
                    digest_size=8,
                ).digest()
                index = int.from_bytes(digest, "little") % cls.FALLBACK_DIMENSION
                matrix[row, index] += 1.0

            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm

        return matrix

    @staticmethod
    def _document_text(chunk: DocumentChunk) -> str:
        return "\n".join(
            part
            for part in (
                chunk.metadata.title or "",
                chunk.heading or "",
                chunk.content,
            )
            if part
        )
