from app.config import settings
from app.rag.authority import AuthorityResolver
from app.rag.conflicts import ConflictDetector
from app.rag.evidence import EvidenceSelector
from app.rag.index import VectorIndex
from app.rag.lexical import LexicalRetriever
from app.rag.parser import KnowledgeBaseParser
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import HybridRetriever


def build_pipeline():
    parser = KnowledgeBaseParser()

    chunks = parser.parse_directory(
        settings.knowledge_base_path
    )

    vector_index = VectorIndex()
    vector_index.build(chunks)

    lexical_retriever = LexicalRetriever()

    retriever = HybridRetriever(
        vector_index=vector_index,
        lexical_retriever=lexical_retriever,
    )

    evidence_selector = EvidenceSelector(
        AuthorityResolver()
    )

    conflict_detector = ConflictDetector()

    pipeline = RAGPipeline(
        retriever=retriever,
        evidence_selector=evidence_selector,
        conflict_detector=conflict_detector,
    )

    return pipeline, chunks


def test_pipeline_rejects_superseded_returns_policy():
    pipeline, chunks = build_pipeline()

    _, evidence, _ = pipeline.retrieve(
        query="How long can I return an item?",
        chunks=chunks,
    )

    filenames = {
        result.chunk.filename
        for result in evidence
    }

    assert (
        "02-returns-policy-legacy.md"
        not in filenames
    )


def test_pipeline_detects_breeze_conflict():
    pipeline, chunks = build_pipeline()

    _, evidence, conflict = pipeline.retrieve(
        query="How should I clean the Breeze Tumbler?",
        chunks=chunks,
    )

    filenames = {
        result.chunk.filename
        for result in evidence
    }

    assert (
        "11-product-care.md"
        in filenames
    )

    assert (
        "12-breeze-tumbler-product-card.md"
        in filenames
    )

    assert conflict.has_conflict is True