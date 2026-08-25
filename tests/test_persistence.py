from pathlib import Path

from app.config import settings
from app.rag.index import VectorIndex
from app.rag.parser import KnowledgeBaseParser
from app.rag.store import ChunkStore


def test_vector_index_round_trip(tmp_path: Path):
    parser = KnowledgeBaseParser()

    chunks = parser.parse_directory(
        settings.knowledge_base_path
    )

    original_index = VectorIndex()
    original_index.build(chunks)

    original_results = original_index.search(
        "How long can I return an item?",
        top_k=5,
    )

    original_ids = [
        chunk.chunk_id
        for chunk, _ in original_results
    ]

    original_index.save(tmp_path)

    ChunkStore().save(
        chunks,
        tmp_path,
    )

    loaded_index = VectorIndex()
    loaded_index.load(tmp_path)

    loaded_chunks = ChunkStore().load(
        tmp_path
    )

    loaded_index.chunks = loaded_chunks

    loaded_results = loaded_index.search(
        "How long can I return an item?",
        top_k=5,
    )

    loaded_ids = [
        chunk.chunk_id
        for chunk, _ in loaded_results
    ]

    assert original_ids == loaded_ids


def test_chunk_store_round_trip(tmp_path: Path):
    parser = KnowledgeBaseParser()

    chunks = parser.parse_directory(
        settings.knowledge_base_path
    )

    store = ChunkStore()

    store.save(
        chunks,
        tmp_path,
    )

    loaded = store.load(
        tmp_path
    )

    assert len(loaded) == len(chunks)

    assert [
        chunk.chunk_id
        for chunk in loaded
    ] == [
        chunk.chunk_id
        for chunk in chunks
    ]