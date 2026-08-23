from fastapi import APIRouter

from app.core.dependencies import UploadServiceDep
from app.schemas.upload_session import UploadSessionResponse

router = APIRouter(tags=["upload-session"])


@router.post("/upload-session", response_model=UploadSessionResponse)
async def create_upload_session(service: UploadServiceDep) -> UploadSessionResponse:
    """Starts a new upload session. Documents uploaded before a test plan
    generation exists are grouped under the returned `sessionId`, which
    `/generate-test-plan` later consumes.
    """
    session = service.create_session()
    return UploadSessionResponse(session_id=session.id)
