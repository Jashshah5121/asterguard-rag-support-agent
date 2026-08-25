from typing import List

from pydantic import BaseModel

from app.models.document import DocumentChunk


class RetrievalResult(BaseModel):
    chunk: DocumentChunk

    semantic_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0
    evidence_score: float = 0.0

    authority_priority: int = 0
    authority_usable: bool = False


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]