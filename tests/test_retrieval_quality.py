from app.config import settings
from app.rag.index import VectorIndex
from app.rag.lexical import LexicalRetriever
from app.rag.parser import KnowledgeBaseParser
from app.rag.retriever import HybridRetriever


def build_retriever():
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

    return retriever, chunks


def result_keys(results):
    return {
        (
            result.chunk.filename,
            result.chunk.heading,
        )
        for result in results
    }


def test_return_window_is_retrieved():
    retriever, chunks = build_retriever()

    response = retriever.search(
        "How long can I return an item?",
        chunks,
        top_k=10,
    )

    keys = result_keys(response.results)

    assert (
        "01-returns-policy-current.md",
        "Standard return window",
    ) in keys


def test_trailplus_return_window_is_retrieved():
    retriever, chunks = build_retriever()

    response = retriever.search(
        "How long does a TrailPlus member have to return an item?",
        chunks,
        top_k=10,
    )

    keys = result_keys(response.results)

    assert (
        "09-trailplus-membership.md",
        "Return window",
    ) in keys


def test_canada_destination_is_retrieved():
    retriever, chunks = build_retriever()

    response = retriever.search(
        "Do you ship to Canada?",
        chunks,
        top_k=10,
    )

    keys = result_keys(response.results)

    assert (
        "06-international-shipping.md",
        "Supported destinations",
    ) in keys


def test_breeze_sources_are_both_retrieved():
    retriever, chunks = build_retriever()

    response = retriever.search(
        "How should I clean the Breeze Tumbler?",
        chunks,
        top_k=10,
    )

    keys = result_keys(response.results)

    assert (
        "11-product-care.md",
        "Breeze Tumbler",
    ) in keys

    assert (
        "12-breeze-tumbler-product-card.md",
        "Cleaning",
    ) in keys


def test_warranty_coverage_is_retrieved():
    retriever, chunks = build_retriever()

    response = retriever.search(
        "What does the warranty cover?",
        chunks,
        top_k=10,
    )

    keys = result_keys(response.results)

    assert (
        "07-warranty.md",
        "What is covered",
    ) in keys