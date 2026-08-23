from datetime import datetime

from app.models.redmine_manifest import DownloadStatus
from app.schemas.common import CamelModel


class RedmineAttachmentDetailOut(CamelModel):
    """One attachment's download outcome — mirrors
    `AttachmentManifestEntry`, exposed as its own API schema to keep the
    internal-model/API-schema separation consistent with the rest of the
    debugging endpoints.
    """

    attachment_id: int | None
    filename: str
    content_type: str | None
    file_size: int | None
    download_status: DownloadStatus
    local_path: str | None
    failure_reason: str | None


class RedmineAttachmentSummaryOut(CamelModel):
    downloaded: int
    failed: int
    skipped: int


class RedmineDebugPathsOut(CamelModel):
    ticket: str
    normalized_ticket: str
    manifest: str
    attachments: str


class RedmineDebugResponse(CamelModel):
    ticket_id: str
    feature: str
    title: str
    status: str
    author: str
    created_at: datetime
    updated_at: datetime
    description: str
    attachment_count: int
    attachments: RedmineAttachmentSummaryOut
    attachment_details: list[RedmineAttachmentDetailOut]
    paths: RedmineDebugPathsOut
