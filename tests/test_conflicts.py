from app.config import settings
from app.rag.authority import AuthorityResolver
from app.rag.conflicts import ConflictDetector
from app.rag.parser import KnowledgeBaseParser


def load_authoritative_chunks():
    parser = KnowledgeBaseParser()
    authority = AuthorityResolver()

    chunks = parser.parse_directory(
        settings.knowledge_base_path
    )

    return authority.filter_usable(chunks)


def get_document_chunks(chunks, filename):
    return [
        chunk
        for chunk in chunks
        if chunk.filename == filename
    ]


def test_breeze_tumbler_sources_conflict():
    chunks = load_authoritative_chunks()

    relevant = (
        get_document_chunks(
            chunks,
            "11-product-care.md",
        )
        + get_document_chunks(
            chunks,
            "12-breeze-tumbler-product-card.md",
        )
    )

    result = ConflictDetector().analyze(relevant)

    assert result.has_conflict is True
    assert len(result.conflicts) == 1

    conflict = result.conflicts[0]

    assert (
        conflict.topic
        == "Breeze Tumbler cleaning instructions"
    )

    source_files = {
        source.filename
        for source in conflict.sources
    }

    assert "11-product-care.md" in source_files
    assert "12-breeze-tumbler-product-card.md" in source_files


def test_single_breeze_tumbler_source_is_not_conflict():
    chunks = load_authoritative_chunks()

    relevant = get_document_chunks(
        chunks,
        "11-product-care.md",
    )

    result = ConflictDetector().analyze(relevant)

    assert result.has_conflict is False


def test_unrelated_documents_do_not_conflict():
    chunks = load_authoritative_chunks()

    relevant = (
        get_document_chunks(
            chunks,
            "05-domestic-shipping.md",
        )
        + get_document_chunks(
            chunks,
            "07-warranty.md",
        )
    )

    result = ConflictDetector().analyze(relevant)

    assert result.has_conflict is False