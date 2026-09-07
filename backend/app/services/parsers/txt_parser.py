"""Extracts text from plain TXT files.

Sections are found the same way WordParser/MarkdownParser find theirs
— a heading line starts a new section, and everything after it (until
the next heading) is that section's body — except headings are found
with `looks_like_heading` (see `app.utils.text`) instead of style
names or `#` syntax, since plain text has no built-in structure:
blank lines (`\\n\\s*\\n`) separate the document into blocks, and a
block whose first line looks like a heading starts a new section. If
nothing in the document matches, there's nothing to split on, so the
whole document comes back as a single section.
"""
import re

from app.core.exceptions import ValidationError
from app.models.document import Document
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.services.parsers.base import DocumentParser
from app.utils.text import looks_like_heading, title_from_filename

_BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n")
_FALLBACK_ENCODING = "latin-1"  # maps every byte 0-255 to a code point — never raises


class TxtParser(DocumentParser):
    parser_name = "TxtParser"
    parser_version = "1.0"

    def parse(self, document: Document, content: bytes) -> ParsedDocument:
        try:
            text = _decode(content)
            sections = _split_into_sections(text)
        except Exception as exc:
            raise ValidationError(f"Failed to parse TXT '{document.filename}': {exc}") from exc

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


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode(_FALLBACK_ENCODING)


def _split_into_sections(text: str) -> list[DocumentSection]:
    blocks = [block.strip() for block in _BLOCK_SPLIT_PATTERN.split(text) if block.strip()]

    sections: list[DocumentSection] = []
    current_heading: str | None = None
    current_body_blocks: list[str] = []
    any_heading_found = False

    def flush() -> None:
        body = "\n\n".join(current_body_blocks).strip()
        if current_heading or body:
            sections.append(DocumentSection(heading=current_heading, content=body))

    for block in blocks:
        lines = block.splitlines()
        first_line = lines[0]
        if looks_like_heading(first_line):
            any_heading_found = True
            flush()
            current_heading = first_line.strip().rstrip(":")
            remainder = "\n".join(lines[1:]).strip()
            current_body_blocks = [remainder] if remainder else []
        else:
            current_body_blocks.append(block)

    flush()

    if not any_heading_found:
        return [DocumentSection(heading=None, content=text.strip())]

    return sections


def _first_heading(sections: list[DocumentSection]) -> str | None:
    return next((section.heading for section in sections if section.heading), None)
