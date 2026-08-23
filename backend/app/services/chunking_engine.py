"""Splits a `ParsedDocument` into `Chunk` objects.

No embeddings, no ChromaDB — just text splitting, ready for whatever
indexes it later.

This engine is deliberately format- and category-agnostic: it doesn't
branch on `artifact_type` or try to figure out what kind of content
it's looking at. It just converts every section into one chunk,
splitting only a section that exceeds `chunk_size` into overlapping,
size-limited windows (`chunk_overlap` words of overlap). Getting "one
chunk per Test Case" or "one chunk per Issue" isn't this engine's
job — it's a consequence of the *parser* already exposing one section
per logical unit (see `CsvParser`, which produces one section per
row), so a normally-sized test case or issue section just doesn't
need splitting. `artifact_type` is still attached to every resulting
`Chunk` — it's part of the `Chunk` schema — but purely as metadata the
caller supplies, never used to pick a strategy.
"""
from app.models.chunk import Chunk
from app.models.common import DocumentCategory
from app.models.parsed_document import DocumentSection, ParsedDocument
from app.utils.ids import generate_id

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 50


class ChunkingEngine:
    def __init__(self, chunk_size: int = _DEFAULT_CHUNK_SIZE, chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_document(self, parsed_document: ParsedDocument, artifact_type: DocumentCategory) -> list[Chunk]:
        pieces = self._chunk_by_section(parsed_document.sections)

        metadata = parsed_document.metadata
        return [
            Chunk(
                chunk_id=generate_id("chunk"),
                document_id=parsed_document.document_id,
                feature=metadata.feature,
                artifact_type=artifact_type,
                document_source=metadata.document_source,
                source_filename=metadata.source_filename,
                page_number=page_number,
                chunk_number=chunk_number,
                chunk_text=text,
            )
            for chunk_number, (text, page_number) in enumerate(pieces, start=1)
        ]

    def _chunk_by_section(self, sections: list[DocumentSection]) -> list[tuple[str, int | None]]:
        pieces: list[tuple[str, int | None]] = []
        for section in sections:
            rendered = _render_section(section)
            windows = _split_into_windows(rendered, self._chunk_size, self._chunk_overlap)
            pieces.extend((window, section.page_number) for window in windows)
        return pieces


def _render_section(section: DocumentSection) -> str:
    if section.heading:
        return f"{section.heading}\n\n{section.content}".strip()
    return section.content.strip()


def _split_into_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into overlapping windows of `chunk_size` words."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    windows = []
    start = 0
    while start < len(words):
        windows.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return windows
