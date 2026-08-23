from datetime import datetime

from pydantic import BaseModel, Field

from app.models.common import GenerationStatus
from app.models.test_case import TestCase


class Generation(BaseModel):
    id: str
    feature: str
    redmine_id: str
    description: str | None = None
    session_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    analysis_options: list[str] = Field(default_factory=list)
    status: GenerationStatus
    created_at: datetime
    updated_at: datetime
    stage_index: int = 0
    test_cases: list[TestCase] = Field(default_factory=list)
