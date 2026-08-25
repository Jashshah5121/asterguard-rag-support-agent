from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = None

    effective_date: Optional[date] = None
    last_reviewed: Optional[date] = None
    superseded_date: Optional[date] = None

    audience: Optional[str] = None
    policy_authority: Optional[str] = None
    customer_answering: Optional[bool] = None

    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: Optional[str] = None
    filename: str
    heading: Optional[str] = None
    content: str
    metadata: DocumentMetadata