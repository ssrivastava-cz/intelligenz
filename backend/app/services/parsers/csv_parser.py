"""Extracts text from CSV files, one section per row.

Each row becomes its own `DocumentSection`, not one giant section for
the whole file — so a CSV of historical test cases or issues naturally
yields one *logical unit* per record, and `ChunkingEngine` (which
turns each section into a chunk) ends up producing one chunk per test
case / issue without needing to know that's what it's doing. Every
column in the row is rendered into the section's content, whatever
it's named — this parser doesn't assume a fixed schema.

The one piece of column-name awareness: a section's heading is built
from an "*ID" column and/or a "Title" column when the CSV has them
(matched case-insensitively — e.g. "Test Case ID" or "Issue ID" both
qualify), since a bare row of columns has no heading of its own
otherwise. A CSV without either just gets an unheaded section per row.

A completely blank row (every field empty/whitespace-only after
normalization — see `app.utils.text.is_blank_row`) is discarded before
it ever becomes a `DocumentSection`, so it never reaches chunking or
embedding. `0`, `False`, and similar falsy-but-meaningful values never
count as blank; a row is only dropped when *every* field genuinely
carries no content. A row with at least one meaningful value is
processed exactly as before.

Encoding: most Source of Truth CSVs are UTF-8 (with or without a BOM —
`utf-8-sig` handles both identically, stripping the BOM if present and
decoding as plain UTF-8 otherwise), but exports from Windows tools
(Excel "Save As CSV", scheduled/cron exports) are commonly Windows-1252
instead — e.g. a curly apostrophe encodes as byte 0x92, which is not
valid UTF-8 and fails to decode outright. `_decode_csv_bytes` tries
UTF-8 first and only falls back to CP1252 if that fails, never silently
replacing or dropping bytes (`errors="ignore"`/`"replace"` would corrupt
the very content the RAG knowledge base depends on) — a file that fails
both is a real parsing error, not a guess.
"""
import csv
import io

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.models.document import Document
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.services.parsers.base import DocumentParser
from app.utils.text import is_blank_row, title_from_filename

logger = get_logger(__name__)

# Tried in order — UTF-8 (BOM-aware) first, since that's the expected/
# preferred encoding for every new Source of Truth document; CP1252 only
# as a fallback for legacy Windows-exported files that were never UTF-8
# to begin with.
_ENCODINGS = ("utf-8-sig", "cp1252")


class CsvParser(DocumentParser):
    parser_name = "CsvParser"
    parser_version = "1.0"

    def parse(self, document: Document, content: bytes) -> ParsedDocument:
        text = _decode_csv_bytes(content, document.filename)
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except csv.Error as exc:
            raise ValidationError(f"Failed to parse CSV '{document.filename}': {exc}") from exc

        non_blank_rows = [row for row in rows if not is_blank_row(row.values())]
        blank_row_count = len(rows) - len(non_blank_rows)
        if blank_row_count:
            logger.info(
                "Skipped %d completely blank row(s) in CSV '%s' (%d of %d rows retained).",
                blank_row_count,
                document.filename,
                len(non_blank_rows),
                len(rows),
            )

        sections = [_row_to_section(row) for row in non_blank_rows] or [DocumentSection(heading=None, content="")]

        return ParsedDocument(
            document_id=document.id,
            title=title_from_filename(document.filename),
            sections=sections,
            content="\n\n".join(section.content for section in sections if section.content),
            metadata=ParsedDocumentMetadata(
                feature=document.feature,
                document_type=document.document_type,
                document_source=document.source,
                source_filename=document.filename,
                page_number=None,
                parser_name=self.parser_name,
                parser_version=self.parser_version,
                source_path=document.source_relative_path,
                source_folder=document.source_folder,
            ),
        )


def _decode_csv_bytes(content: bytes, filename: str) -> str:
    """Tries each encoding in `_ENCODINGS` in order, returning the text
    from the first one that decodes cleanly. Never falls back to
    `errors="ignore"`/`"replace"` — those would silently drop or
    mangle bytes `csv.DictReader` (and everything downstream: chunking,
    embedding, the RAG knowledge base) would then treat as legitimate
    content. If every encoding fails, raises `ValidationError` naming
    the file, every encoding attempted, and each one's original
    decoding failure, so the real cause is visible rather than a bare
    "failed to parse".
    """
    failures: list[str] = []
    for encoding in _ENCODINGS:
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc}")
            continue
        logger.info("Parsed CSV '%s' using '%s' encoding.", filename, encoding)
        return text

    attempted = ", ".join(_ENCODINGS)
    original_failures = "; ".join(failures)
    raise ValidationError(
        f"Failed to parse CSV '{filename}': could not decode using any supported encoding "
        f"(attempted: {attempted}). {original_failures}"
    )


def _row_to_section(row: dict[str, str]) -> DocumentSection:
    content = "\n".join(f"{key}: {value}" for key, value in row.items() if key)
    return DocumentSection(heading=_row_heading(row), content=content)


def _row_heading(row: dict[str, str]) -> str | None:
    id_value: str | None = None
    title_value: str | None = None

    for key, value in row.items():
        if not key or not value:
            continue
        words = key.strip().lower().split()
        if not words:
            continue
        if id_value is None and words[-1] == "id":
            id_value = value.strip()
        elif title_value is None and words == ["title"]:
            title_value = value.strip()

    if id_value and title_value:
        return f"{id_value} - {title_value}"
    return id_value or title_value
