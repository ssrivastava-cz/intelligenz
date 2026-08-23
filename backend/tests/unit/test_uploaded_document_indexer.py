"""Unit tests for UploadedDocumentIndexer — discovers and parses an
upload session's documents, the uploaded-document equivalent of
SourceOfTruthIndexer's discover-then-parse step.
"""
import io

import pytest
from fastapi import UploadFile

from app.core.exceptions import NotFoundError
from app.models.common import DocumentCategory, DocumentSource
from app.services.parser_service import ParserService
from app.services.upload_service import UploadService
from app.services.uploaded_document_indexer import UploadedDocumentIndexer

_MAX_UPLOAD_SIZE_BYTES = 1024 * 1024


def _make_indexer(tmp_path) -> tuple[UploadedDocumentIndexer, UploadService]:
    upload_service = UploadService(storage_root=tmp_path, max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)
    parser_service = ParserService(upload_service=upload_service)
    indexer = UploadedDocumentIndexer(upload_service=upload_service, parser_service=parser_service)
    return indexer, upload_service


def _upload_file(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def test_parse_session_raises_not_found_for_unknown_session(tmp_path):
    indexer, _ = _make_indexer(tmp_path)

    with pytest.raises(NotFoundError):
        indexer.parse_session("nonexistent-session")


async def test_parse_session_returns_empty_list_for_session_with_no_documents(tmp_path):
    indexer, upload_service = _make_indexer(tmp_path)
    session = upload_service.create_session()

    assert indexer.parse_session(session.id) == []


async def test_parse_session_parses_uploaded_documents_with_user_upload_category(tmp_path):
    indexer, upload_service = _make_indexer(tmp_path)
    session = upload_service.create_session()
    await upload_service.upload_user_documents(
        session_id=session.id,
        feature=session.id,
        files=[_upload_file("notes.md", b"# Role Permissions\n\nBody text.")],
    )

    [item] = indexer.parse_session(session.id)

    assert item.document_name == "notes.md"
    assert item.document_category == DocumentCategory.USER_UPLOAD
    assert item.parsed_document.metadata.document_source == DocumentSource.USER_UPLOAD
    assert "Body text." in item.parsed_document.content


async def test_parse_session_rescopes_metadata_feature_to_the_session_id(tmp_path):
    """Even if a document was uploaded under a real 'feature' name (the
    pre-existing /upload-documents flow for test-plan generation), the
    embedding pipeline must scope by session id, since that's what
    VectorStoreService.replace_feature_chunks uses to isolate/replace
    per session.
    """
    indexer, upload_service = _make_indexer(tmp_path)
    session = upload_service.create_session()
    await upload_service.upload_user_documents(
        session_id=session.id,
        feature="Some Real Feature Name",
        files=[_upload_file("notes.md", b"# Heading\n\nBody text.")],
    )

    [item] = indexer.parse_session(session.id)

    assert item.parsed_document.metadata.feature == session.id


async def test_parse_documents_parses_an_explicit_document_list(tmp_path):
    indexer, upload_service = _make_indexer(tmp_path)
    session = upload_service.create_session()
    documents = await upload_service.upload_user_documents(
        session_id=session.id,
        feature=session.id,
        files=[_upload_file("a.txt", b"ROLE PERMISSIONS\n\nBody one.")],
    )

    [item] = indexer.parse_documents(session.id, documents)

    assert item.document_name == "a.txt"
    assert item.document_category == DocumentCategory.USER_UPLOAD


async def test_parse_session_handles_multiple_documents(tmp_path):
    indexer, upload_service = _make_indexer(tmp_path)
    session = upload_service.create_session()
    await upload_service.upload_user_documents(
        session_id=session.id,
        feature=session.id,
        files=[
            _upload_file("a.md", b"# A\n\nBody A."),
            _upload_file("b.txt", b"ROLE PERMISSIONS\n\nBody B."),
        ],
    )

    parsed = indexer.parse_session(session.id)

    assert {item.document_name for item in parsed} == {"a.md", "b.txt"}
