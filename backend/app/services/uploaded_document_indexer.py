"""Discovers and parses an upload session's documents — the uploaded-
document equivalent of `SourceOfTruthIndexer`'s discover-then-parse
step. Documents here are already registered via `UploadService` (an
upload endpoint put them there) rather than found by scanning a folder
tree, so "discovery" is just `UploadService.list_session_documents`.

Kept independent of embeddings/ChromaDB/history — like
`SourceOfTruthIndexer`, this only produces parsed, chunkable documents.
"""
from pathlib import Path

from app.models.common import DocumentCategory
from app.models.document import Document
from app.models.parsed_feature_document import ParsedFeatureDocument
from app.services.parser_service import ParserService
from app.services.upload_service import UploadService


class UploadedDocumentIndexer:
    def __init__(self, upload_service: UploadService, parser_service: ParserService) -> None:
        self._upload_service = upload_service
        self._parser_service = parser_service

    def parse_session(self, upload_session_id: str) -> list[ParsedFeatureDocument]:
        """Raises `NotFoundError` (via `UploadService.get_session`) if the
        session was never created. A session with zero uploaded documents
        returns an empty list — not an error — mirroring
        `SourceOfTruthIndexer.parse_feature`.
        """
        self._upload_service.get_session(upload_session_id)
        documents = self._upload_service.list_session_documents(upload_session_id)
        return self.parse_documents(upload_session_id, documents)

    def parse_documents(self, upload_session_id: str, documents: list[Document]) -> list[ParsedFeatureDocument]:
        """Parses an already-known list of documents (the preview endpoint
        already has these in hand from `upload_user_documents`, so it
        calls this directly instead of re-discovering via `parse_session`).
        """
        results = []
        for document in documents:
            content = Path(document.storage_path).read_bytes()
            parsed = self._parser_service.parse_content(document, content)
            # Force the session id as the chunking/embedding scope key,
            # regardless of whatever "feature" the document was originally
            # tagged with (e.g. one uploaded via the pre-existing
            # /upload-documents flow for test-plan generation, which uses
            # a real feature name). VectorStoreService.replace_feature_chunks
            # scopes its delete-then-add by this exact value, so it must
            # always match upload_session_id for uploads to replace
            # correctly and stay isolated per session.
            rescoped_metadata = parsed.metadata.model_copy(update={"feature": upload_session_id})
            parsed = parsed.model_copy(update={"metadata": rescoped_metadata})
            results.append(
                ParsedFeatureDocument(
                    document_name=document.filename,
                    document_category=DocumentCategory.USER_UPLOAD,
                    parsed_document=parsed,
                )
            )
        return results
