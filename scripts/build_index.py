from app.config import settings
from app.rag.index import VectorIndex
from app.rag.parser import KnowledgeBaseParser
from app.rag.store import ChunkStore


def main() -> None:
    parser = KnowledgeBaseParser()

    chunks = parser.parse_directory(
        settings.knowledge_base_path
    )

    print(f"Loaded {len(chunks)} chunks.")

    index = VectorIndex()

    index.build(chunks)

    index.save(
        settings.index_path
    )

    ChunkStore().save(
        chunks,
        settings.index_path,
    )

    print(
        "Persistent retrieval index built successfully."
    )

    print(
        f"Indexed chunks: {len(chunks)}"
    )

    print(
        f"Index directory: "
        f"{settings.index_path}"
    )


if __name__ == "__main__":
    main()