from datetime import datetime

from pydantic import BaseModel, Field


class IndexRecord(BaseModel):
    id: str
    feature: str
    document_ids: list[str] = Field(default_factory=list)
    chunks_indexed: int
    status: str
    indexed_at: datetime
