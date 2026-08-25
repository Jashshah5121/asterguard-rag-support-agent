from app.config import settings
from app.rag.parser import KnowledgeBaseParser


def main() -> None:
    parser = KnowledgeBaseParser()

    chunks = parser.parse_directory(
        settings.knowledge_base_path
    )

    print("=" * 80)
    print("KNOWLEDGE BASE INSPECTION")
    print("=" * 80)

    print(f"\nDocuments directory:")
    print(settings.knowledge_base_path)

    print(f"\nTotal chunks: {len(chunks)}")

    documents = {}

    for chunk in chunks:
        documents.setdefault(
            chunk.filename,
            [],
        ).append(chunk)

    print(f"Total documents: {len(documents)}")

    for filename, document_chunks in documents.items():
        first = document_chunks[0]
        metadata = first.metadata

        print("\n" + "-" * 80)
        print(filename)
        print("-" * 80)

        print(f"document_id       : {metadata.document_id}")
        print(f"title             : {metadata.title}")
        print(f"status            : {metadata.status}")
        print(f"effective_date    : {metadata.effective_date}")
        print(f"last_reviewed     : {metadata.last_reviewed}")
        print(f"superseded_date   : {metadata.superseded_date}")
        print(f"supersedes        : {metadata.supersedes}")
        print(f"superseded_by     : {metadata.superseded_by}")
        print(f"audience          : {metadata.audience}")
        print(f"policy_authority  : {metadata.policy_authority}")
        print(f"customer_answering: {metadata.customer_answering}")

        print("\nSections:")

        for index, chunk in enumerate(
            document_chunks,
            start=1,
        ):
            print(
                f"  {index:02d}. "
                f"{chunk.heading!r} "
                f"({len(chunk.content)} chars)"
            )


if __name__ == "__main__":
    main()