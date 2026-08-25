import json
from pathlib import Path
from typing import List

from app.models.document import DocumentChunk


class ChunkStore:
    """
    Persists document chunks separately from the vector index.
    """

    FILENAME = "chunks.json"

    def save(
        self,
        chunks: List[DocumentChunk],
        directory: Path,
    ) -> None:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = directory / self.FILENAME

        payload = [
            chunk.model_dump(mode="json")
            for chunk in chunks
        ]

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(
        self,
        directory: Path,
    ) -> List[DocumentChunk]:
        path = directory / self.FILENAME

        if not path.exists():
            raise FileNotFoundError(
                f"Chunk store not found: {path}"
            )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return [
            DocumentChunk.model_validate(item)
            for item in payload
        ]