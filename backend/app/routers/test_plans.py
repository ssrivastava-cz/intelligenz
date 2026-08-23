from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.dependencies import TestPlanServiceDep
from app.core.exceptions import ValidationError
from app.models.common import GenerationStatus
from app.schemas.test_plan import (
    GenerateTestPlanRequest,
    GenerateTestPlanResponse,
    GenerationStatusResponse,
    HistoryItemOut,
    HistoryResponse,
)
from app.utils.csv_export import test_cases_to_csv
from app.utils.mock_data import PROGRESS_STAGES

router = APIRouter(tags=["test-plans"])


@router.post("/generate-test-plan", response_model=GenerateTestPlanResponse, status_code=202)
async def generate_test_plan(
    payload: GenerateTestPlanRequest,
    service: TestPlanServiceDep,
) -> GenerateTestPlanResponse:
    generation = service.create_generation(
        feature=payload.feature,
        redmine_id=payload.redmine_id,
        description=payload.description,
        session_id=payload.session_id,
        document_ids=payload.document_ids,
        analysis_options=payload.analysis_options,
    )
    return GenerateTestPlanResponse(
        generation_id=generation.id,
        status=generation.status,
        message="Test plan generation started.",
    )


@router.get("/status/{generation_id}", response_model=GenerationStatusResponse)
async def get_status(generation_id: str, service: TestPlanServiceDep) -> GenerationStatusResponse:
    generation = service.get_generation(generation_id)
    total_stages = len(PROGRESS_STAGES)
    is_processing = generation.status == GenerationStatus.PROCESSING
    current_stage = PROGRESS_STAGES[generation.stage_index] if is_processing else None
    progress_percent = 100 if not is_processing else round((generation.stage_index + 1) / total_stages * 100)

    return GenerationStatusResponse(
        generation_id=generation.id,
        status=generation.status,
        current_stage=current_stage,
        stage_index=generation.stage_index,
        total_stages=total_stages,
        progress_percent=progress_percent,
        test_cases=generation.test_cases or None,
        created_at=generation.created_at,
        updated_at=generation.updated_at,
    )


@router.get("/history", response_model=HistoryResponse)
async def get_history(service: TestPlanServiceDep, limit: int = 20, offset: int = 0) -> HistoryResponse:
    generations = sorted(service.list_generations(), key=lambda g: g.created_at, reverse=True)
    page = generations[offset : offset + limit]

    items = [
        HistoryItemOut(
            generation_id=g.id,
            feature=g.feature,
            redmine_id=g.redmine_id,
            generated_at=g.created_at,
            test_case_count=len(g.test_cases),
            status=g.status,
        )
        for g in page
    ]
    return HistoryResponse(items=items, total=len(generations))


@router.get("/download/{generation_id}")
async def download_test_plan(generation_id: str, service: TestPlanServiceDep) -> StreamingResponse:
    generation = service.get_generation(generation_id)
    if generation.status != GenerationStatus.COMPLETED:
        raise ValidationError(f"Generation '{generation_id}' is not ready for download yet.")

    csv_content = test_cases_to_csv(generation.test_cases)
    filename = f"test-plan-{generation_id}.csv"
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
