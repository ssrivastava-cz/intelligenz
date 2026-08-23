from datetime import datetime

from pydantic import BaseModel

from app.models.cost import ActualUsage, EstimatedUsage, PricingSnapshot
from app.models.generated_test_case import GeneratedTestCase


class RetrievalSummary(BaseModel):
    """How many chunks of each kind fed this generation's prompt —
    mirrors `RetrievalResult`'s per-category counts, not the chunks
    themselves (the full prompt text already carries their content)."""

    workflow_chunks: int
    historical_test_cases: int
    historical_issues: int
    uploaded_documents: int


class GenerationMetadata(BaseModel):
    top_k: int | None
    upload_session_id: str | None
    generated_test_cases: int
    # Added alongside `generated_test_cases` so a generation's actual
    # output count can be judged against what was actually requested at
    # the time (MAX_GENERATED_TEST_CASES can change between
    # generations) rather than whatever the current configuration
    # happens to be. All three default to `None`/unset for backward
    # compatibility with generation history persisted before this
    # existed — never fabricated for an old record. `count_target_met`
    # is `generated_test_cases == requested_test_cases` (after
    # `GenerationService` truncates any overshoot), and `coverage_note`
    # is the model's own stated reason for a shortfall (see
    # `GeneratedTestCasesResponse.coverage_note`), `None` when the full
    # count was reached.
    requested_test_cases: int | None = None
    count_target_met: bool | None = None
    coverage_note: str | None = None


class GenerationHistoryDraft(BaseModel):
    """Everything `GenerationService.generate` knows before persisting —
    `HistoryService.save_generation_history` assigns `generation_id` and
    `timestamp` at save time (the same way `UploadService.create_session`
    self-assigns a session id), producing a full `GenerationHistoryEntry`.
    """

    feature: str
    redmine_ticket: str
    model: str
    prompt_version: str
    retrieval: RetrievalSummary
    estimated_usage: EstimatedUsage
    actual_usage: ActualUsage
    pricing: PricingSnapshot
    generation_time_ms: float
    prompt: str
    response: str
    test_cases: list[GeneratedTestCase]
    metadata: GenerationMetadata
    # Always "SUCCESS": a failed generation never reaches HistoryService
    # (see GenerationService.generate) — same reasoning as
    # IndexHistoryEntry.index_status. Defaulted (not required) so
    # generations persisted before this field existed still load fine.
    status: str = "SUCCESS"


class GenerationHistoryEntry(GenerationHistoryDraft):
    """One completed AI generation, persisted by
    `HistoryService.save_generation_history` as a single JSON file under
    `backend/data/history/generation/<timestamp>.json` — unlike indexing
    and upload history, generation history is one file per run, not a
    folder of files, since there's no separate embedding preview to
    persist alongside it.
    """

    generation_id: str
    timestamp: datetime
