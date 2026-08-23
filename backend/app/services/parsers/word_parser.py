"""Extracts text from DOCX files, sectioned by heading style."""
import io

from docx import Document as open_docx
from docx.document import Document as DocxDocument

from app.core.exceptions import ValidationError
from app.models.document import Document
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.services.parsers.base import DocumentParser
from app.utils.text import title_from_filename

_HEADING_STYLE_PREFIX = "Heading"


class WordParser(DocumentParser):
    parser_name = "WordParser"
    parser_version = "1.0"

    def parse(self, document: Document, content: bytes) -> ParsedDocument:
        try:
            docx = open_docx(io.BytesIO(content))
            sections = _split_into_sections(docx)
            core_title = (docx.core_properties.title or "").strip()
        except Exception as exc:
            raise ValidationError(f"Failed to parse DOCX '{document.filename}': {exc}") from exc

        return ParsedDocument(
            document_id=document.id,
            title=core_title or title_from_filename(document.filename),
            sections=sections,
            content="\n\n".join(section.content for section in sections if section.content),
            metadata=ParsedDocumentMetadata(
                feature=document.feature,
                document_type=document.document_type,
                document_source=document.source,
                source_filename=document.filename,
                page_number=None,  # DOCX has no fixed page count without rendering
                parser_name=self.parser_name,
                parser_version=self.parser_version,
            ),
        )


def _split_into_sections(docx: DocxDocument) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    current_heading: str | None = None
    current_paragraphs: list[str] = []

    def flush() -> None:
        body = "\n".join(p for p in current_paragraphs if p.strip())
        if current_heading or body:
            sections.append(DocumentSection(heading=current_heading, content=body))

    for paragraph in docx.paragraphs:
        style_name = paragraph.style.name if paragraph.style else ""
        if style_name.startswith(_HEADING_STYLE_PREFIX):
            flush()
            current_heading = paragraph.text.strip()
            current_paragraphs = []
        else:
            current_paragraphs.append(paragraph.text)

    flush()

    if not sections:
        sections.append(DocumentSection(heading=None, content=""))

    return sections
