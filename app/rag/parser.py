import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from app.models.document import DocumentChunk, DocumentMetadata


FRONT_MATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$",
    re.MULTILINE,
)


class KnowledgeBaseParser:
    """
    Parses markdown knowledge-base documents into structured chunks.

    Each level-2 heading (##) becomes a searchable chunk.
    The document-level metadata is preserved on every chunk.
    """

    def parse_file(self, path: Path) -> List[DocumentChunk]:
        text = path.read_text(encoding="utf-8")

        metadata_dict, body = self._parse_front_matter(text)

        metadata = DocumentMetadata.model_validate(metadata_dict)

        return self._chunk_document(
            path=path,
            body=body,
            metadata=metadata,
        )

    def parse_directory(self, directory: Path) -> List[DocumentChunk]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Knowledge-base directory does not exist: {directory}"
            )

        chunks: List[DocumentChunk] = []

        for path in sorted(directory.glob("*.md")):
            chunks.extend(self.parse_file(path))

        return chunks

    def _parse_front_matter(
        self,
        text: str,
    ) -> Tuple[Dict, str]:
        match = FRONT_MATTER_PATTERN.match(text)

        if not match:
            return {}, text.strip()

        raw_metadata = match.group(1)
        body = match.group(2).strip()

        metadata = yaml.safe_load(raw_metadata) or {}

        if not isinstance(metadata, dict):
            raise ValueError(
                "Document front matter must be a YAML mapping."
            )

        return metadata, body

    def _chunk_document(
        self,
        path: Path,
        body: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        lines = body.splitlines()

        document_title = self._extract_document_title(lines)

        sections = self._extract_sections(lines)

        chunks: List[DocumentChunk] = []

        for heading, content in sections:
            if not content.strip():
                continue

            chunk_content = self._build_chunk_content(
                document_title=document_title,
                heading=heading,
                content=content,
            )

            chunk_id = self._generate_chunk_id(
                document_id=metadata.document_id,
                filename=path.name,
                heading=heading,
                content=chunk_content,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=metadata.document_id,
                    filename=path.name,
                    heading=heading,
                    content=chunk_content,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _extract_document_title(
        lines: List[str],
    ) -> str | None:
        for line in lines:
            match = re.match(r"^#\s+(.+?)\s*$", line)

            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def _extract_sections(
        lines: List[str],
    ) -> List[Tuple[str | None, str]]:
        sections: List[Tuple[str | None, str]] = []

        current_heading: str | None = None
        current_content: List[str] = []

        for line in lines:
            match = re.match(r"^##\s+(.+?)\s*$", line)

            if match:
                if current_heading is not None:
                    sections.append(
                        (
                            current_heading,
                            "\n".join(current_content).strip(),
                        )
                    )

                current_heading = match.group(1).strip()
                current_content = []
                continue

            # Ignore the document-level "# Title".
            if re.match(r"^#\s+", line):
                continue

            # Ignore deeper headings as structural boundaries.
            # They remain part of their parent section.
            current_content.append(line)

        if current_heading is not None:
            sections.append(
                (
                    current_heading,
                    "\n".join(current_content).strip(),
                )
            )

        # Documents without ## headings still become one chunk.
        if not sections and any(line.strip() for line in lines):
            sections.append(
                (
                    None,
                    "\n".join(lines).strip(),
                )
            )

        return sections

    @staticmethod
    def _build_chunk_content(
        document_title: str | None,
        heading: str | None,
        content: str,
    ) -> str:
        parts: List[str] = []

        if document_title:
            parts.append(document_title)

        if heading:
            parts.append(heading)

        parts.append(content.strip())

        return "\n\n".join(parts)

    @staticmethod
    def _generate_chunk_id(
        document_id: str | None,
        filename: str,
        heading: str | None,
        content: str,
    ) -> str:
        raw = "|".join(
            [
                document_id or "",
                filename,
                heading or "",
                content,
            ]
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:16]

        return f"chunk_{digest}"