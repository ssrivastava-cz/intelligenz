from datetime import datetime

from pydantic import AliasChoices, Field

from app.models.common import GenerationStatus, Priority, TestCaseSource, TestType
from app.schemas.common import CamelModel


class GenerateTestPlanRequest(CamelModel):
    feature: str
    redmine_id: str = Field(validation_alias=AliasChoices("redmineId", "ticketId", "ticket_id"))
    session_id: str | None = None
    description: str | None = None

    # Legacy: explicit document ids, from before /upload-session existed.
    # If sessionId is also provided, its documents are added to these.
    document_ids: list[str] = Field(default_factory=list)
    analysis_options: list[str] = Field(default_factory=list)


class GenerateTestPlanResponse(CamelModel):
    generation_id: str
    status: GenerationStatus
    message: str


class TestCaseOut(CamelModel):
    id: str
    feature: str
    test_case: str
    preconditions: str
    steps: list[str]
    expected_result: str
    priority: Priority
    type: TestType
    source: TestCaseSource
    automation_candidate: bool


class GenerationStatusResponse(CamelModel):
    generation_id: str
    status: GenerationStatus
    current_stage: str | None
    stage_index: int
    total_stages: int
    progress_percent: int
    test_cases: list[TestCaseOut] | None = None
    created_at: datetime
    updated_at: datetime


class HistoryItemOut(CamelModel):
    generation_id: str
    feature: str
    redmine_id: str
    generated_at: datetime
    test_case_count: int
    status: GenerationStatus


class HistoryResponse(CamelModel):
    items: list[HistoryItemOut]
    total: int
