"""Persisted per-attachment download outcome —
`backend/data/redmine/<ticket_id>/download_manifest.json`.

camelCase on disk (unlike other internal models, which persist as plain
snake_case) to match this file's explicit, documented shape — a
debugging/inspection artifact meant to be read directly, and later
consumed by the Retrieval Pipeline.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

DownloadStatus = Literal["DOWNLOADED", "FAILED", "SKIPPED"]


class _ManifestCamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AttachmentManifestEntry(_ManifestCamelModel):
    attachment_id: int | None
    filename: str
    content_type: str | None = None
    file_size: int | None = None
    download_status: DownloadStatus
    local_path: str | None = None
    failure_reason: str | None = None


class DownloadManifest(_ManifestCamelModel):
    ticket_id: str
    generated_at: datetime
    attachments: list[AttachmentManifestEntry] = Field(default_factory=list)
