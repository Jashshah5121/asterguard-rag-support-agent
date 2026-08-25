from pathlib import Path

from app.config import settings
from app.rag.authority import AuthorityResolver
from app.rag.parser import KnowledgeBaseParser


def load_chunks():
    parser = KnowledgeBaseParser()

    return parser.parse_directory(
        settings.knowledge_base_path
    )


def get_document(chunks, filename):
    matches = [
        chunk
        for chunk in chunks
        if chunk.filename == filename
    ]

    assert matches, f"Document not found: {filename}"

    return matches[0]


def test_current_returns_policy_is_authoritative():
    chunks = load_chunks()

    chunk = get_document(
        chunks,
        "01-returns-policy-current.md",
    )

    decision = AuthorityResolver().evaluate(chunk)

    assert decision.usable is True
    assert decision.priority == 100


def test_legacy_returns_policy_is_not_customer_authority():
    chunks = load_chunks()

    chunk = get_document(
        chunks,
        "02-returns-policy-legacy.md",
    )

    decision = AuthorityResolver().evaluate(chunk)

    assert decision.usable is False


def test_migration_scratchpad_is_not_customer_authority():
    chunks = load_chunks()

    chunk = get_document(
        chunks,
        "14-internal-content-migration-notes.md",
    )

    decision = AuthorityResolver().evaluate(chunk)

    assert decision.usable is False


def test_internal_support_document_is_not_customer_policy():
    chunks = load_chunks()

    chunk = get_document(
        chunks,
        "13-support-escalation.md",
    )

    decision = AuthorityResolver().evaluate(chunk)

    assert decision.usable is False


def test_active_official_customer_documents_are_usable():
    chunks = load_chunks()

    resolver = AuthorityResolver()

    customer_documents = {
        "03-final-sale-and-promotions.md",
        "04-damaged-or-wrong-items.md",
        "05-domestic-shipping.md",
        "06-international-shipping.md",
        "07-warranty.md",
        "08-order-changes-and-cancellations.md",
        "09-trailplus-membership.md",
        "10-gift-cards-and-price-adjustments.md",
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
    }

    for filename in customer_documents:
        chunk = get_document(chunks, filename)

        decision = resolver.evaluate(chunk)

        assert decision.usable is True, filename