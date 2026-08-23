from fastapi import APIRouter

from app.core.dependencies import RedmineServiceDep
from app.schemas.redmine import (
    RedmineAttachmentDetailOut,
    RedmineAttachmentSummaryOut,
    RedmineDebugPathsOut,
    RedmineDebugResponse,
)

router = APIRouter(tags=["redmine"])


@router.get("/redmine/{ticket_id}/debug", response_model=RedmineDebugResponse)
async def get_redmine_debug(ticket_id: str, redmine_service: RedmineServiceDep) -> RedmineDebugResponse:
    """Read-only inspection of what was retrieved from Redmine and
    persisted locally for a ticket — a developer debugging tool. Reuses
    cached files on disk when present instead of calling Redmine again.
    Never touches OpenAI, ChromaDB, or the Retrieval Pipeline.
    """
    info = redmine_service.get_ticket_debug_info(ticket_id)

    counts = {"DOWNLOADED": 0, "FAILED": 0, "SKIPPED": 0}
    for entry in info.manifest.attachments:
        counts[entry.download_status] += 1

    return RedmineDebugResponse(
        ticket_id=info.ticket.ticket_id,
        feature=info.ticket.feature,
        title=info.ticket.title,
        status=info.ticket.status,
        author=info.ticket.author,
        created_at=info.ticket.created_at,
        updated_at=info.ticket.updated_at,
        description=info.ticket.description,
        attachment_count=len(info.manifest.attachments),
        attachments=RedmineAttachmentSummaryOut(
            downloaded=counts["DOWNLOADED"],
            failed=counts["FAILED"],
            skipped=counts["SKIPPED"],
        ),
        attachment_details=[
            RedmineAttachmentDetailOut.model_validate(entry) for entry in info.manifest.attachments
        ],
        paths=RedmineDebugPathsOut(
            ticket=str(info.ticket_path),
            normalized_ticket=str(info.normalized_ticket_path),
            manifest=str(info.manifest_path),
            attachments=str(info.attachments_dir),
        ),
    )
