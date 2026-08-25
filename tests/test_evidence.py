from app.config import settings
from app.rag.evidence import EvidenceSelector
from app.rag.parser import KnowledgeBaseParser
from app.models.retrieval import RetrievalResult


def load_chunks():
    parser = KnowledgeBaseParser()

    return parser.parse_directory(
        settings.knowledge_base_path
    )


def get_chunk(chunks, filename, heading):
    for chunk in chunks:
        if (
            chunk.filename == filename
            and chunk.heading == heading
        ):
            return chunk

    raise AssertionError(
        f"Chunk not found: {filename} / {heading}"
    )


def test_evidence_selector_rejects_superseded_policy():
    chunks = load_chunks()

    legacy = get_chunk(
        chunks,
        "02-returns-policy-legacy.md",
        "Return window",
    )

    result = RetrievalResult(
        chunk=legacy,
        semantic_score=0.9,
        lexical_score=0.9,
        final_score=0.9,
    )

    selected = EvidenceSelector().select(
        query="How long can I return an item?",
        results=[result],
    )

    assert selected == []


def test_evidence_selector_rejects_internal_content():
    chunks = load_chunks()

    internal = get_chunk(
        chunks,
        "14-internal-content-migration-notes.md",
        "Unapproved legacy copy",
    )

    result = RetrievalResult(
        chunk=internal,
        semantic_score=0.9,
        lexical_score=0.9,
        final_score=0.9,
    )

    selected = EvidenceSelector().select(
        query="What is the return policy?",
        results=[result],
    )

    assert selected == []


def test_evidence_selector_keeps_current_policy():
    chunks = load_chunks()

    current = get_chunk(
        chunks,
        "01-returns-policy-current.md",
        "Standard return window",
    )

    result = RetrievalResult(
        chunk=current,
        semantic_score=0.9,
        lexical_score=0.9,
        final_score=0.9,
    )

    selected = EvidenceSelector().select(
        query="How long can I return an item?",
        results=[result],
    )

    assert len(selected) == 1
    assert (
        selected[0].chunk.filename
        == "01-returns-policy-current.md"
    )


def test_evidence_selector_prefers_return_window():
    chunks = load_chunks()

    shipping = get_chunk(
        chunks,
        "01-returns-policy-current.md",
        "Return shipping and refunds",
    )

    return_window = get_chunk(
        chunks,
        "01-returns-policy-current.md",
        "Standard return window",
    )

    shipping_result = RetrievalResult(
        chunk=shipping,
        semantic_score=0.8,
        lexical_score=0.8,
        final_score=0.0320,
    )

    return_window_result = RetrievalResult(
        chunk=return_window,
        semantic_score=0.6,
        lexical_score=0.6,
        final_score=0.0300,
    )

    selected = EvidenceSelector().select(
        query="How long can I return an item?",
        results=[
            shipping_result,
            return_window_result,
        ],
        max_results=1,
    )

    assert len(selected) == 1

    assert (
        selected[0].chunk.heading
        == "Standard return window"
    )