from datetime import datetime

from pydantic import BaseModel


class UploadSession(BaseModel):
    """A container for documents uploaded before a generation exists.

    Created via `POST /upload-session`, then referenced by `sessionId`
    on `/upload-documents`. `/generate-test-plan` consumes everything
    uploaded under it and records which generation consumed it.
    """

    id: str
    created_at: datetime
    consumed_by_generation_id: str | None = None
