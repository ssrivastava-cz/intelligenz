from datetime import UTC, datetime

import pytest

from app.core.exceptions import NotFoundError, ValidationError
from app.models.common import DocumentCategory, DocumentSource, DocumentType
from app.models.document import Document
from app.services.parser_service import ParserService
from app.services.upload_service import UploadService

_MAX_UPLOAD_SIZE_BYTES = 1024 * 1024


class _FakeUploadFile:
    """Duck-types just enough of fastapi.UploadFile for UploadService."""

    def __init__(self, filename: str, content: bytes, content_type: str = "text/plain"):
        self.filename = filename
        self.content_type = content_type
        self._content = content

    async def read(self) -> bytes:
        return self._content


async def test_parses_source_of_truth_document(tmp_path):
    upload_service = UploadService(storage_root=tmp_path, max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)
    documents = await upload_service.upload_source_of_truth_documents(
        feature="Appointments",
        files=[_FakeUploadFile("notes.md", b"# Notes\nBody")],
    )
    document_id = documents[0].id

    parsed = ParserService(upload_service=upload_service).parse_document(document_id)

    assert parsed.document_id == document_id
    assert parsed.metadata.document_source == "source_of_truth"
    assert parsed.metadata.feature == "Appointments"
    assert parsed.title == "Notes"


async def test_parses_user_uploaded_document(tmp_path):
    upload_service = UploadService(storage_root=tmp_path, max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)
    documents = await upload_service.upload_user_documents(
        session_id="sess-parse-1",
        feature="Appointments",
        files=[_FakeUploadFile("cases.csv", b"a,b\n1,2\n")],
    )
    document_id = documents[0].id

    parsed = ParserService(upload_service=upload_service).parse_document(document_id)

    assert parsed.document_id == document_id
    assert parsed.metadata.document_source == "user_upload"
    assert parsed.metadata.document_type == "CSV"


async def test_raises_when_document_id_is_unknown(tmp_path):
    upload_service = UploadService(storage_root=tmp_path, max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)

    with pytest.raises(NotFoundError):
        ParserService(upload_service=upload_service).parse_document("does-not-exist")


async def test_parses_txt_document(tmp_path):
    """TXT is a fully supported format, both sources — same as every other parser."""
    upload_service = UploadService(storage_root=tmp_path, max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)
    documents = await upload_service.upload_source_of_truth_documents(
        feature="Appointments",
        files=[_FakeUploadFile("notes.txt", b"Body text.")],
    )
    document_id = documents[0].id

    parsed = ParserService(upload_service=upload_service).parse_document(document_id)

    assert parsed.metadata.document_type == "TXT"
    assert parsed.metadata.parser_name == "TxtParser"
    assert parsed.metadata.parser_version == "1.0"


def test_parse_content_works_without_going_through_upload_service(tmp_path):
    """SourceOfTruthIndexer discovers files by scanning the filesystem
    itself — parse_content must work for a Document that was never
    registered via UploadService.upload_*_documents at all.
    """
    upload_service = UploadService(storage_root=tmp_path / "unused", max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)
    document = Document(
        id="doc-scanned-1",
        filename="onboarding.md",
        document_type=DocumentType.MARKDOWN,
        feature="Appointments",
        size=20,
        source=DocumentSource.SOURCE_OF_TRUTH,
        category=DocumentCategory.WORKFLOW,
        storage_path="unused-not-read",
        uploaded_at=datetime.now(UTC),
    )

    parsed = ParserService(upload_service=upload_service).parse_content(document, b"# Onboarding\nBody text.")

    assert parsed.document_id == "doc-scanned-1"
    assert parsed.title == "Onboarding"
    assert parsed.metadata.parser_name == "MarkdownParser"


async def test_raises_when_no_parser_registered_for_document_type(tmp_path):
    # Every real DocumentType has a registered parser now, so this
    # simulates an unregistered one directly rather than through a real
    # upload — exercising the defensive path in `_parser_for`.
    upload_service = UploadService(storage_root=tmp_path, max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)
    documents = await upload_service.upload_source_of_truth_documents(
        feature="Appointments",
        files=[_FakeUploadFile("data.csv", b"a,b\n1,2\n")],
    )
    document_id = documents[0].id

    parser_service = ParserService(upload_service=upload_service)
    del parser_service._parsers[DocumentType.CSV]

    with pytest.raises(ValidationError):
        parser_service.parse_document(document_id)
