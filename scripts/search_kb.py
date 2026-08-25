import argparse

from app.config import settings
from app.rag.index import VectorIndex
from app.rag.lexical import LexicalRetriever
from app.rag.parser import KnowledgeBaseParser
from app.rag.retriever import HybridRetriever
from app.rag.runtime import RAGRuntime


def build_retriever():
    runtime = RAGRuntime()

    return (
        runtime.retriever,
        runtime.chunks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the knowledge base."
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display.",
    )

    args = parser.parse_args()

    query = " ".join(args.query)

    retriever, chunks = build_retriever()

    response = retriever.search(
        query=query,
        chunks=chunks,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("KNOWLEDGE BASE SEARCH")
    print("=" * 80)

    print(f"\nQuery:\n{query}")

    for index, result in enumerate(
        response.results,
        start=1,
    ):
        chunk = result.chunk
        metadata = chunk.metadata

        print("\n" + "-" * 80)
        print(f"RESULT {index}")
        print("-" * 80)

        print(f"File       : {chunk.filename}")
        print(f"Chunk ID    : {chunk.chunk_id}")
        print(f"Heading    : {chunk.heading}")
        print(f"Document ID: {chunk.document_id}")

        print("\nAuthority metadata:")
        print(f"  status           : {metadata.status}")
        print(
            f"  policy_authority : "
            f"{metadata.policy_authority}"
        )
        print(f"  audience         : {metadata.audience}")
        print(
            f"  customer_answering: "
            f"{metadata.customer_answering}"
        )

        print("\nScores:")
        print(
            f"  semantic : "
            f"{result.semantic_score:.4f}"
        )
        print(
            f"  lexical  : "
            f"{result.lexical_score:.4f}"
        )
        print(
            f"  final    : "
            f"{result.final_score:.4f}"
        )

        print("\nContent:")
        print(chunk.content)


if __name__ == "__main__":
    main()