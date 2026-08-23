"""Extracts text from PDF files.

Tries to recover logical sections rather than just handing back one
blob of text per page: within the text extracted from each page,
a line matching `looks_like_heading` (see `app.utils.text` — the same
heuristic TxtParser uses, since PDF text extraction gives us plain
lines with no font/style metadata to rely on) starts a new section,
which can continue across a page boundary if no further heading
appears. A section's `page_number` is the page its heading started on.

If no heading is found anywhere in the document, there's nothing to
split on — this falls back to the original behavior, one section per
page, headed "Page N".
"""
import io

from pypdf import PdfReader

from app.core.exceptions import ValidationError
from app.models.document import Document
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.services.parsers.base import DocumentParser
from app.utils.text import looks_like_heading, title_from_filename


class PdfParser(DocumentParser):
    parser_name = "PdfParser"
    parser_version = "1.0"

    def parse(self, document: Document, content: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [(index, (page.extract_text() or "").strip()) for index, page in enumerate(reader.pages, start=1)]
            title = _extract_title(reader)
        except Exception as exc:
            raise ValidationError(f"Failed to parse PDF '{document.filename}': {exc}") from exc

        sections = _split_into_sections(pages)

        return ParsedDocument(
            document_id=document.id,
            title=title or title_from_filename(document.filename),
            sections=sections,
            content="\n\n".join(section.content for section in sections if section.content),
            metadata=ParsedDocumentMetadata(
                feature=document.feature,
                document_type=document.document_type,
                document_source=document.source,
                source_filename=document.filename,
                page_number=len(pages),
                parser_name=self.parser_name,
                parser_version=self.parser_version,
            ),
        )


def _extract_title(reader: PdfReader) -> str | None:
    metadata = reader.metadata
    if metadata and metadata.title:
        return metadata.title.strip() or None
    return None


def _split_into_sections(pages: list[tuple[int, str]]) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    current_page: int | None = None
    any_heading_found = False

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if current_heading or body:
            sections.append(DocumentSection(heading=current_heading, content=body, page_number=current_page))

    for page_number, text in pages:
        for line in text.splitlines():
            if looks_like_heading(line):
                any_heading_found = True
                flush()
                current_heading = line.strip().rstrip(":")
                current_lines = []
                current_page = page_number
            else:
                if current_page is None:
                    current_page = page_number
                current_lines.append(line)

    flush()

    if not any_heading_found:
        return [
            DocumentSection(heading=f"Page {page_number}", content=text, page_number=page_number)
            for page_number, text in pages
        ]

    return sections
