from fastapi import APIRouter

from app.core.dependencies import UsageServiceDep
from app.schemas.usage import UsageResponse

router = APIRouter(tags=["usage"])


@router.get("/usage", response_model=UsageResponse)
async def get_usage(service: UsageServiceDep) -> UsageResponse:
    """Unified Usage Dashboard view — combines Test Plan Generator and
    Knowledge Assistant usage, newest first. See `UsageService.get_usage`.
    """
    return UsageResponse.model_validate(service.get_usage())
