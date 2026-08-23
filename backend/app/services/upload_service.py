"""Document ingestion.

Handles both document sources:

- Source of Truth: the org's approved knowledge base, stored under
  `<storage_root>/source_of_truth/<feature>/` (workflow docs, historical
  test cases/issue sheets, release notes).
- User uploads: ad-hoc documents attached while generating a test plan.
  Uploads happen *before* a generation exists, so they're grouped under
  an upload session instead: `<storage_root>/uploads/<session_id>/`.
  `/generate-test-plan` later consumes everything in the session.

This service only validates and stores files — no parsing, embedding,
or AI. `Document.storage_path` is the stable, predictable location
`ParserService` (`app/services/parser_service.py`) reads from;
`get_document`/`list_*` are the lookup surface it uses.
"""
from pathlib import Path

from fastapi import UploadFile

from app.core.exceptions import NotFoundError, ValidationError
from app.models.common import DocumentSource
from app.models.document import Document
from app.models.upload_session import UploadSession
from app.utils.datetime_utils import utcnow
from app.utils.file_validation import resolve_document_type, sanitize_filename, validate_size
from app.utils.ids import generate_id
from app.utils.slugify import safe_path_segment


class UploadService:
    def __init__(self, storage_root: Path, max_upload_size_bytes: int) -> None:
        self._source_of_truth_root = storage_root / "source_of_truth"
        self._uploads_root = storage_root / "uploads"
        self._max_upload_size_bytes = max_upload_size_bytes
        self._documents: dict[str, Document] = {}
        self._sessions: dict[str, UploadSession] = {}

    def create_session(self) -> UploadSession:
        session = UploadSession(id=generate_id("sess"), created_at=utcnow())
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> UploadSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError(f"No upload session found with id '{session_id}'.")
        return session

    def mark_session_consumed(self, session_id: str, generation_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.consumed_by_generation_id = generation_id

    async def upload_source_of_truth_documents(self, feature: str, files: list[UploadFile]) -> list[Document]:
        target_dir = self._source_of_truth_root / safe_path_segment(feature)
        return await self._ingest(files, target_dir, feature=feature, source=DocumentSource.SOURCE_OF_TRUTH)

    async def upload_user_documents(self, session_id: str, feature: str, files: list[UploadFile]) -> list[Document]:
        # A session created via /upload-session is used as-is; an unrecognized
        # id (e.g. a legacy caller that never called /upload-session) is
        # lazily registered so uploads still work without one.
        self._sessions.setdefault(session_id, UploadSession(id=session_id, created_at=utcnow()))

        target_dir = self._uploads_root / safe_path_segment(session_id)
        return await self._ingest(
            files,
            target_dir,
            feature=feature,
            source=DocumentSource.USER_UPLOAD,
            session_id=session_id,
        )

    def list_source_of_truth_documents(self, feature: str) -> list[Document]:
        return [
            d
            for d in self._documents.values()
            if d.source == DocumentSource.SOURCE_OF_TRUTH and d.feature == feature
        ]

    def list_session_documents(self, session_id: str) -> list[Document]:
        return [d for d in self._documents.values() if d.session_id == session_id]

    def get_document(self, document_id: str) -> Document:
        document = self._documents.get(document_id)
        if document is None:
            raise NotFoundError(f"No document found with id '{document_id}'.")
        return document

    async def _ingest(
        self,
        files: list[UploadFile],
        target_dir: Path,
        feature: str,
        source: DocumentSource,
        session_id: str | None = None,
    ) -> list[Document]:
        if not files:
            raise ValidationError("At least one file is required.")

        self._reject_duplicate_filenames_in_batch(files)

        target_dir.mkdir(parents=True, exist_ok=True)
        existing_names = {p.name.lower() for p in target_dir.iterdir() if p.is_file()}

        documents = []
        for file in files:
            filename = sanitize_filename(file.filename or "unnamed")
            document_type = resolve_document_type(filename)

            if filename.lower() in existing_names:
                source_label = source.value.replace("_", " ")
                raise ValidationError(f"A document named '{filename}' already exists for this {source_label}.")

            contents = await file.read()
            validate_size(filename, len(contents), self._max_upload_size_bytes)

            destination = target_dir / filename
            destination.write_bytes(contents)

            document = Document(
                id=generate_id("doc"),
                filename=filename,
                document_type=document_type,
                feature=feature,
                size=len(contents),
                source=source,
                session_id=session_id,
                storage_path=str(destination),
                uploaded_at=utcnow(),
            )
            self._documents[document.id] = document
            documents.append(document)
            existing_names.add(filename.lower())

        return documents

    @staticmethod
    def _reject_duplicate_filenames_in_batch(files: list[UploadFile]) -> None:
        seen: set[str] = set()
        for file in files:
            name = (file.filename or "unnamed").lower()
            if name in seen:
                raise ValidationError(f"Duplicate filename in upload: '{file.filename}'.")
            seen.add(name)
