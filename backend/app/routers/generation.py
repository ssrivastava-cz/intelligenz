"""Generation History & Downloads.

Every endpoint here — except `POST /generate`, which owns the OpenAI
Chat call — only ever reads (or deletes) persisted Generation History
via `HistoryService`. None of them regenerate AI output, rebuild a
prompt, perform retrieval, call OpenAI, or query ChromaDB.

Route ordering matters: FastAPI/Starlette matches routes in declaration
order, and a single dynamic segment like `/generation/{generation_id}`
would otherwise swallow a literal path like `/generation/history` (both
are exactly one segment past `/generation/`). The literal routes
(`/generation/history`, `/generation/history/stats`) are declared before
the bare `/generation/{generation_id}` route for exactly this reason —
see `tests/integration/test_generation_route_ordering.py`.
"""
from math import ceil

from fastapi import APIRouter, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.core.dependencies import CostCalculatorDep, ExcelExportServiceDep, GenerationServiceDep, HistoryServiceDep
from app.models.generation_history import GenerationHistoryEntry
from app.schemas.cost import ActualUsageOut
from app.schemas.generation import (
    GeneratedTestCaseOut,
    GenerateRequest,
    GenerateResponse,
    GenerationDebugResponse,
    GenerationHistoryListResponse,
    GenerationProgressOut,
    GenerationStatisticsOut,
    GenerationSummaryOut,
)
from app.services.excel_export_service import ExcelExportService
from app.services.history_service import HistoryService

_EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Maps the API's camelCase sort keys (matching GenerationSummaryOut's
# field names) onto GenerationSummary's Python attribute names.
_SORT_FIELDS = {
    "generationId": "generation_id",
    "createdAt": "created_at",
    "feature": "feature",
    "redmineTicket": "redmine_ticket",
    "model": "model",
    "promptVersion": "prompt_version",
    "generationTimeMs": "generation_time_ms",
    "numberOfTestCases": "number_of_test_cases",
    "promptTokens": "prompt_tokens",
    "completionTokens": "completion_tokens",
    "totalTokens": "total_tokens",
    "actualGenerationCostUsd": "actual_generation_cost_usd",
    "actualGenerationCostInr": "actual_generation_cost_inr",
    "totalAiCostUsd": "total_ai_cost_usd",
    "totalAiCostInr": "total_ai_cost_inr",
    "status": "status",
}

router = APIRouter(tags=["generation"])


@router.post("/generate", response_model=GenerateResponse)
async def generate_test_cases(payload: GenerateRequest, generation_service: GenerationServiceDep) -> GenerateResponse:
    """Runs the full AI Test Case Generation pipeline: retrieve context,
    build the prompt, call OpenAI Chat exactly once, validate its
    response, calculate actual usage, and persist generation history —
    all via `GenerationService`. History is only saved once every one of
    those steps has succeeded; a failure anywhere raises before anything
    is written. Redmine ticket fields are optional — a request may
    supply only `feature` and still generate successfully.

    `GenerationService.generate` is a blocking call (it makes a real
    OpenAI network request), so it's run in a worker thread rather than
    directly on the event loop — otherwise this request alone would
    block every other request the server is handling, including a
    concurrent `GET /generate/progress/{request_id}` poll for this exact
    call's progress.
    """
    entry = await run_in_threadpool(
        generation_service.generate,
        feature=payload.feature,
        redmine_id=payload.redmine_id,
        redmine_description=payload.redmine_description,
        optional_description=payload.optional_description,
        upload_session_id=payload.upload_session_id,
        top_k=payload.top_k,
        request_id=payload.request_id,
    )
    return GenerateResponse(
        generation_id=entry.generation_id,
        model=entry.model,
        prompt_version=entry.prompt_version,
        generated_test_cases=entry.metadata.generated_test_cases,
        requested_test_cases=entry.metadata.requested_test_cases,
        count_target_met=entry.metadata.count_target_met,
        coverage_note=entry.metadata.coverage_note,
        actual_usage=ActualUsageOut.model_validate(entry.actual_usage),
        generation_time_ms=entry.generation_time_ms,
        test_cases=[GeneratedTestCaseOut.model_validate(test_case) for test_case in entry.test_cases],
    )


@router.get("/generate/progress/{request_id}", response_model=GenerationProgressOut)
async def get_generation_progress(request_id: str, generation_service: GenerationServiceDep) -> GenerationProgressOut:
    """Read-only, best-effort peek at which internal step an in-flight
    `POST /generate` call (identified by the `requestId` it was given)
    has reached — for the frontend's progress UI to poll while that
    request is still running. Never starts, influences, or duplicates
    generation itself; `stage` is simply `null` once the call finishes,
    fails, or `request_id` is unrecognized.
    """
    return GenerationProgressOut(stage=generation_service.get_progress(request_id))


@router.get("/generation/history", response_model=GenerationHistoryListResponse)
async def list_generation_history(
    history_service: HistoryServiceDep,
    cost_calculator: CostCalculatorDep,
    page: int = 1,
    page_size: int = 20,
    sort: str = "createdAt",
    order: str = "desc",
) -> GenerationHistoryListResponse:
    """Paginated list of every persisted generation, summary fields
    only. Default sort is newest-first (`createdAt` descending). Reads
    only `HistoryService.list_generation_summaries()` — no prompt/
    response/test case bodies are loaded for the list view.
    `cost_calculator` is only used to convert each summary's already-
    persisted Embedding-stage USD cost into `{usd, inr}`, never to
    price anything itself.
    """
    summaries = history_service.list_generation_summaries(cost_calculator)

    sort_field = _SORT_FIELDS.get(sort, "created_at")
    summaries.sort(key=lambda summary: getattr(summary, sort_field), reverse=order.lower() != "asc")

    total_items = len(summaries)
    page_size = max(page_size, 1)
    start = (page - 1) * page_size
    page_items = summaries[start : start + page_size]

    return GenerationHistoryListResponse(
        items=[GenerationSummaryOut.model_validate(summary) for summary in page_items],
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=ceil(total_items / page_size) if total_items else 0,
    )


@router.get("/generation/history/stats", response_model=GenerationStatisticsOut)
async def get_generation_history_stats(
    history_service: HistoryServiceDep, cost_calculator: CostCalculatorDep
) -> GenerationStatisticsOut:
    """Dashboard aggregates computed directly from persisted history —
    see `HistoryService.generation_statistics`.
    """
    return GenerationStatisticsOut.model_validate(history_service.generation_statistics(cost_calculator))


@router.get("/generation/debug/{generation_id}", response_model=GenerationDebugResponse)
async def get_generation_debug(generation_id: str, history_service: HistoryServiceDep) -> GenerationDebugResponse:
    """Read-only: loads exactly what a prior generation persisted. Never
    regenerates a prompt, never calls RetrievalService or PromptBuilder,
    never calls OpenAI — it only reads the stored JSON.
    """
    return _load_generation_response(generation_id, history_service)


@router.get("/generation/{generation_id}/export/excel")
async def export_generation_excel(
    generation_id: str,
    history_service: HistoryServiceDep,
    excel_export_service: ExcelExportServiceDep,
) -> StreamingResponse:
    """Downloads an ADO-compatible Excel workbook built entirely from
    persisted generation history — never regenerates AI output, never
    performs retrieval, never calls OpenAI, never touches ChromaDB.
    `HistoryService.load_generation` is the only data source; formatting
    is delegated entirely to `ExcelExportService`.
    """
    return _excel_streaming_response(generation_id, history_service, excel_export_service)


@router.get("/generation/{generation_id}/download/excel")
async def download_generation_excel(
    generation_id: str,
    history_service: HistoryServiceDep,
    excel_export_service: ExcelExportServiceDep,
) -> StreamingResponse:
    """Same workbook as `/generation/{generation_id}/export/excel`,
    under the History page's "Download Excel" naming — both share
    `_excel_streaming_response` so the workbook-building logic exists in
    exactly one place.
    """
    return _excel_streaming_response(generation_id, history_service, excel_export_service)


@router.get("/generation/{generation_id}/prompt", response_class=PlainTextResponse)
async def get_generation_prompt(generation_id: str, history_service: HistoryServiceDep) -> PlainTextResponse:
    """The exact prompt text sent to GPT for this generation, verbatim
    from persisted history — never rebuilt.
    """
    entry = history_service.load_generation(generation_id)
    return PlainTextResponse(content=entry.prompt)


@router.get("/generation/{generation_id}/response")
async def get_generation_raw_response(generation_id: str, history_service: HistoryServiceDep) -> Response:
    """The raw AI response exactly as OpenAI returned it and exactly as
    persisted — never re-fetched from OpenAI, returned as-is rather than
    re-encoded (it's already valid JSON text).
    """
    entry = history_service.load_generation(generation_id)
    return Response(content=entry.response, media_type="application/json")


@router.get("/generation/{generation_id}", response_model=GenerationDebugResponse)
async def get_generation(generation_id: str, history_service: HistoryServiceDep) -> GenerationDebugResponse:
    """The complete persisted generation — prompt, raw AI response,
    generated test cases, retrieval summary, usage, costs, and metadata.
    Simply loads the stored JSON; never regenerates anything.
    """
    return _load_generation_response(generation_id, history_service)


@router.delete("/generation/{generation_id}", status_code=204)
async def delete_generation(generation_id: str, history_service: HistoryServiceDep) -> None:
    """Deletes only this generation's persisted history file. Never
    touches uploads, embeddings, ChromaDB collections, Redmine data, or
    Source of Truth data — `HistoryService.delete_generation` has no
    access to any of those in the first place.
    """
    history_service.delete_generation(generation_id)


def _load_generation_response(generation_id: str, history_service: HistoryService) -> GenerationDebugResponse:
    entry = history_service.load_generation(generation_id)
    return GenerationDebugResponse.model_validate(entry)


def _excel_streaming_response(
    generation_id: str, history_service: HistoryService, excel_export_service: ExcelExportService
) -> StreamingResponse:
    entry: GenerationHistoryEntry = history_service.load_generation(generation_id)
    workbook_bytes = excel_export_service.build_workbook(entry)

    filename = f"ADO_TestCases_{generation_id}.xlsx"
    return StreamingResponse(
        iter([workbook_bytes]),
        media_type=_EXCEL_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
