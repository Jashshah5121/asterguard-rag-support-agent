from typing import List

from pydantic import BaseModel, Field

from app.models.document import DocumentChunk


class Conflict(BaseModel):
    topic: str
    explanation: str
    sources: List[DocumentChunk] = Field(default_factory=list)


class ConflictAnalysis(BaseModel):
    has_conflict: bool = False
    conflicts: List[Conflict] = Field(default_factory=list)