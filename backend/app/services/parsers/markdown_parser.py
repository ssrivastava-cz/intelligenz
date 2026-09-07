"""Extracts structure from Markdown files, sectioned by heading."""
import re

from app.core.exceptions import ValidationError
from app.models.document import Document
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.services.parsers.base import DocumentParser
from app.utils.text import title_from_filename

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownParser(DocumentParser):
    parser_name = "MarkdownParser"
    parser_version = "1.0"

    def parse(self, document: Document, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"Failed to parse Markdown '{document.filename}': {exc}") from exc

        sections = _split_into_sections(text)
        title = _first_heading(sections) or title_from_filename(document.filename)

        return ParsedDocument(
            document_id=document.id,
            title=title,
            sections=sections,
            content=text.strip(),
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


def _split_into_sections(text: str) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if current_heading or body:
            sections.append(DocumentSection(heading=current_heading, content=body))

    for line in text.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match:
            flush()
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()

    if not sections:
        sections.append(DocumentSection(heading=None, content=text.strip()))

    return sections


def _first_heading(sections: list[DocumentSection]) -> str | None:
    return next((section.heading for section in sections if section.heading), None)
