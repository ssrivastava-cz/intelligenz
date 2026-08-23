from fastapi import APIRouter

from app.core.dependencies import IndexingServiceDep
from app.schemas.indexing import IndexFeatureRequest, IndexFeatureResponse

router = APIRouter(tags=["indexing"])


@router.post("/index-feature", response_model=IndexFeatureResponse)
async def index_feature(
    payload: IndexFeatureRequest,
    service: IndexingServiceDep,
) -> IndexFeatureResponse:
    record = service.index_feature(payload.feature, payload.document_ids)
    return IndexFeatureResponse(
        index_id=record.id,
        feature=record.feature,
        status=record.status,
        chunks_indexed=record.chunks_indexed,
        message=f"Indexed {record.chunks_indexed} chunks for '{record.feature}'.",
    )
