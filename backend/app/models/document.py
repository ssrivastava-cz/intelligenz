from datetime import datetime

from pydantic import BaseModel

from app.models.common import DocumentCategory, DocumentSource, DocumentType


class Document(BaseModel):
    """A single ingested document, from either source.

    `storage_path` is the on-disk location a future parser service
    would read from — it's internal (not part of the API response
    contract in `app.schemas.documents`).
    """

    id: str
    filename: str
    document_type: DocumentType
    feature: str
    size: int
    source: DocumentSource
    session_id: str | None = None
    category: DocumentCategory | None = None
    """Set for Source of Truth documents discovered by SourceOfTruthIndexer
    (from their folder: workflows/TestCases/IssueSheets). None for
    documents from UploadService, which has no such folder convention.
    """
    storage_path: str
    uploaded_at: datetime
