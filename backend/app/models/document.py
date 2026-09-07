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
    source_relative_path: str | None = None
    """Portable path from Source of Truth, e.g.
    "source_of_truth/Service Model 1/workflows/example.pdf" — always
    forward-slashed, never a machine-specific absolute path. Set only by
    `SourceOfTruthIndexer._build_document` (from the actual discovered
    path); `None` for a `UploadService`-ingested document, which has no
    Source of Truth path at all.
    """
    source_folder: str | None = None
    """The folder directly under source_of_truth/ this document lives
    in — for the current folder layout, always equal to `feature`, but
    computed independently (not copied from it) so this stays correct
    if that ever changes. `None` for a `UploadService`-ingested document.
    """
