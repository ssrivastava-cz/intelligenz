from datetime import datetime

from app.schemas.common import CamelModel
from app.schemas.cost import ActualUsageOut, EstimatedUsageOut, PricingSnapshotOut


class RetrievalSummaryOut(CamelModel):
    workflow_chunks: int
    historical_test_cases: int
    historical_issues: int
    uploaded_documents: int


class GenerationMetadataOut(CamelModel):
    top_k: int | None
    upload_session_id: str | None
    generated_test_cases: int
    # `None` for generation history persisted before these existed —
    # see `app.models.generation_history.GenerationMetadata`.
    requested_test_cases: int | None
    count_target_met: bool | None
    coverage_note: str | None


class GeneratedTestStepOut(CamelModel):
    step_no: int
    action: str
    expected_result: str


class GeneratedTestCaseOut(CamelModel):
    """`workItemType`/`automationStatus`/`state`/`areaPath`/`assignedTo`/
    `tags` are `None` only for a `GeneratedTestCase` persisted before
    ADO fields existed — every generation since always has them set
    (see `GenerationService._apply_ado_fields`).
    """

    requirement_id: str
    test_case_id: str
    test_case_title: str
    priority: str
    test_suite: str
    preconditions: str
    steps: list[GeneratedTestStepOut]
    post_conditions: str
    test_type: str
    test_classification: str
    work_item_type: str | None
    automation_status: str | None
    state: str | None
    area_path: str | None
    assigned_to: str | None
    tags: list[str] | None


class GenerationDebugResponse(CamelModel):
    """Exactly what `HistoryService.save_generation_history` persisted
    for one generation — loaded and returned as-is, never regenerated.
    """

    generation_id: str
    timestamp: datetime
    feature: str
    redmine_ticket: str
    model: str
    prompt_version: str
    retrieval: RetrievalSummaryOut
    estimated_usage: EstimatedUsageOut
    actual_usage: ActualUsageOut
    pricing: PricingSnapshotOut
    generation_time_ms: float
    prompt: str
    response: str
    test_cases: list[GeneratedTestCaseOut]
    metadata: GenerationMetadataOut


class GenerateRequest(CamelModel):
    """Request for `POST /generate` — feature is the only always-required
    input. A Redmine ticket reference and its description are optional:
    omit both to generate from the selected Feature (and whatever
    knowledge-base context retrieval finds for it) alone. `request_id`,
    if supplied, is an opaque client-generated token with no meaning to
    generation itself — it only lets the caller poll `GET
    /generate/progress/{request_id}` for this call's progress while it's
    still running.
    """

    feature: str
    redmine_id: str | None = None
    redmine_description: str | None = None
    optional_description: str | None = None
    upload_session_id: str | None = None
    top_k: int | None = None
    request_id: str | None = None


class GenerationProgressOut(CamelModel):
    """Read-only, best-effort progress for one in-flight `POST /generate`
    call — `stage` is `null` once the call finishes (or if `request_id`
    was never supplied to it, or is unrecognized).
    """

    stage: str | None


class GenerateResponse(CamelModel):
    """`requestedTestCases`/`countTargetMet`/`coverageNote` let a caller
    tell "the model generated fewer than configured because the
    evidence didn't support more" (`countTargetMet: false`, a populated
    `coverageNote`) apart from a real failure — `generatedTestCases`
    alone can't distinguish those. `generatedTestCases` is always
    `<= requestedTestCases`: `GenerationService` truncates any overshoot
    before this response is built.
    """

    generation_id: str
    model: str
    prompt_version: str
    generated_test_cases: int
    requested_test_cases: int
    count_target_met: bool
    coverage_note: str | None
    actual_usage: ActualUsageOut
    generation_time_ms: float
    test_cases: list[GeneratedTestCaseOut]


class GenerationSummaryOut(CamelModel):
    """One row of the `/generation/history` listing — used by both the
    History page's list view and the Usage Dashboard's Recent AI
    Generations and Cost Breakdown tables, not the full prompt/response/
    test cases (see `GenerationDebugResponse` for that).
    """

    generation_id: str
    created_at: datetime
    feature: str
    redmine_ticket: str
    model: str
    prompt_version: str
    generation_time_ms: float
    number_of_test_cases: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    actual_generation_cost_usd: float
    actual_generation_cost_inr: float
    estimated_prompt_tokens: int
    estimated_input_cost_usd: float
    estimated_input_cost_inr: float
    embedding_tokens: int | None
    embedding_cost_usd: float | None
    embedding_cost_inr: float | None
    total_ai_cost_usd: float
    total_ai_cost_inr: float
    status: str


class GenerationHistoryListResponse(CamelModel):
    items: list[GenerationSummaryOut]
    page: int
    page_size: int
    total_items: int
    total_pages: int


class GenerationStatisticsOut(CamelModel):
    total_generations: int
    successful_generations: int
    failed_generations: int
    average_generation_time_ms: float
    average_test_cases: float
    average_prompt_tokens: float
    average_completion_tokens: float
    average_cost_usd: float
    average_cost_inr: float
    most_used_feature: str | None
    most_used_model: str | None
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_spend_usd: float
    total_spend_inr: float
