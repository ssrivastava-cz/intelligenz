from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.core.dependencies import UploadServiceDep
from app.core.exceptions import ValidationError
from app.schemas.documents import DocumentOut, UploadDocumentsResponse

router = APIRouter(tags=["documents"])


@router.post("/upload-documents", response_model=UploadDocumentsResponse)
async def upload_documents(
    service: UploadServiceDep,
    files: Annotated[list[UploadFile], File(...)],
    feature: Annotated[str, Form(...)],
    session_id: Annotated[str | None, Form(alias="sessionId")] = None,
    generation_id: Annotated[str | None, Form(alias="generationId")] = None,
) -> UploadDocumentsResponse:
    """User-uploaded documents attached while generating a test plan.

    Stored under `uploads/<session_id>/`. `generationId` is accepted as
    a deprecated alias for `sessionId` for backward compatibility with
    callers from before upload sessions existed.
    """
    resolved_session_id = session_id or generation_id
    if not resolved_session_id:
        raise ValidationError("sessionId is required.")

    documents = await service.upload_user_documents(
        session_id=resolved_session_id, feature=feature, files=files
    )
    return UploadDocumentsResponse(
        documents=[DocumentOut.from_document(d) for d in documents],
        message=f"{len(documents)} document(s) received.",
    )


@router.post("/source-of-truth/{feature}/documents", response_model=UploadDocumentsResponse)
async def upload_source_of_truth_documents(
    feature: str,
    service: UploadServiceDep,
    files: Annotated[list[UploadFile], File(...)],
) -> UploadDocumentsResponse:
    """Ingests approved knowledge-base documents for a feature.

    Stored under `source_of_truth/<feature>/`.
    """
    documents = await service.upload_source_of_truth_documents(feature=feature, files=files)
    return UploadDocumentsResponse(
        documents=[DocumentOut.from_document(d) for d in documents],
        message=f"{len(documents)} document(s) received.",
    )
