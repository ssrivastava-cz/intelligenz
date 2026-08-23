"""File validation for the document ingestion subsystem.

Extension and size checks only — no content is parsed or inspected.
"""
from pathlib import Path

from app.core.exceptions import ValidationError
from app.models.common import DocumentType

_EXTENSION_TO_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".xlsx": DocumentType.XLSX,
    ".csv": DocumentType.CSV,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
}

SUPPORTED_EXTENSIONS = sorted(_EXTENSION_TO_TYPE)


def sanitize_filename(filename: str) -> str:
    """Strips any path components, keeping only the base filename.

    Prevents path traversal via a crafted filename (e.g. `../../evil.sh`
    or an absolute path).
    """
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise ValidationError(f"'{filename}' is not a valid filename.")
    return name


def resolve_document_type(filename: str) -> DocumentType:
    extension = Path(filename).suffix.lower()
    document_type = _EXTENSION_TO_TYPE.get(extension)
    if document_type is None:
        raise ValidationError(
            f"Unsupported file type '{extension or filename}'. "
            f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    return document_type


def validate_size(filename: str, size_bytes: int, max_size_bytes: int) -> None:
    if size_bytes > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValidationError(f"'{filename}' exceeds the maximum upload size of {max_mb:.0f} MB.")
