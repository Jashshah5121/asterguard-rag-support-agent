from app.config import settings
from app.rag.index import VectorIndex
from app.rag.lexical import LexicalRetriever
from app.rag.retriever import HybridRetriever
from app.rag.store import ChunkStore


class RAGRuntime:
    """
    Loads the persisted retrieval artifacts once and exposes
    a ready-to-use hybrid retriever.
    """

    def __init__(self) -> None:
        self.vector_index = VectorIndex()

        self.vector_index.load(
            settings.index_path
        )

        self.chunks = ChunkStore().load(
            settings.index_path
        )

        # FAISS vector positions correspond to chunk positions.
        if (
            self.vector_index.index is None
            or self.vector_index.index.ntotal
            != len(self.chunks)
        ):
            raise RuntimeError(
                "FAISS index and chunk store are inconsistent."
            )

        self.vector_index.chunks = self.chunks

        self.lexical_retriever = (
            LexicalRetriever()
        )

        self.retriever = HybridRetriever(
            vector_index=self.vector_index,
            lexical_retriever=self.lexical_retriever,
        )