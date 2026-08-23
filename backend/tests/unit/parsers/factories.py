"""Shared test helper for building fake ingestion `Document` records
without touching the filesystem or the upload HTTP endpoints."""
from datetime import UTC, datetime

from app.models.common import DocumentSource, DocumentType
from app.models.document import Document


def make_document(**overrides) -> Document:
    defaults = {
        "id": "doc-test-1",
        "filename": "sample.txt",
        "document_type": DocumentType.TXT,
        "feature": "Appointments",
        "size": 0,
        "source": DocumentSource.SOURCE_OF_TRUTH,
        "session_id": None,
        "storage_path": "/tmp/sample.txt",
        "uploaded_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return Document(**defaults)
