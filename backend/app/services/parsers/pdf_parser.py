"""Extracts text from PDF files, and recovers ruled tables as structure.

Two passes over the document:

1. **Text** — pypdf `extract_text()` per page, exactly as before. Within
   a page's text, a line matching `looks_like_heading` (see
   `app.utils.text`, the same heuristic `TxtParser` uses — PDF text
   extraction gives plain lines with no font/style metadata) starts a
   new section, which can continue across a page boundary if no further
   heading appears. A section's `page_number` is the page its heading
   started on. If no heading is found anywhere, this falls back to the
   original behavior: one section per page, headed "Page N".

2. **Tables** — `pdfplumber` looks for *bordered* tables (ruling lines /
   rectangle edges only; whitespace alignment is never treated as a
   table). Each candidate must pass `pdf_tables.is_reliable_table`;
   one that does becomes its own `DocumentSection` rendered as
   "Header: value" blocks, and its text is removed from that page's
   pass-1 text so it is not indexed twice. A table that runs off the
   bottom of a page and resumes at the top of the next is stitched back
   together (`pdf_tables.continuation_rows`), repeating the header.

Pass 2 is best-effort and fully isolated: any failure (or a page with
no ruling lines at all) leaves pass 1 untouched, so a document with no
detectable tables — and every previously-working normal PDF — produces
exactly the sections it did before.
"""
import io
import time

import pdfplumber
from pypdf import PdfReader

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.models.document import Document
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.services.parsers import pdf_tables
from app.services.parsers.base import DocumentParser
from app.utils.text import looks_like_heading, title_from_filename

logger = get_logger(__name__)

# Wall-clock ceiling for the whole table pass. If exceeded, remaining
# pages keep their plain pypdf text — the document still indexes.
_TABLE_BUDGET_SECONDS = 20.0


class PdfParser(DocumentParser):
    parser_name = "PdfParser"
    # 1.1: adds bordered-table recovery on top of 1.0's flat-text +
    # heading sectioning. Text-only documents parse identically to 1.0.
    parser_version = "1.1"

    def parse(self, document: Document, content: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [(index, (page.extract_text() or "").strip()) for index, page in enumerate(reader.pages, start=1)]
            title = _extract_title(reader)
        except Exception as exc:
            raise ValidationError(f"Failed to parse PDF '{document.filename}': {exc}") from exc

        try:
            page_texts, table_sections = _extract_tables(content, pages)
        except Exception as exc:  # noqa: BLE001 - table recovery must never break parsing
            logger.warning(
                "PDF table extraction failed for '%s' (%s); using text-only extraction.",
                document.filename,
                exc,
            )
            page_texts, table_sections = pages, []

        text_sections = _split_into_sections(page_texts)
        sections = _merge_sections(text_sections, table_sections)

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
                source_path=document.source_relative_path,
                source_folder=document.source_folder,
            ),
        )


def _extract_title(reader: PdfReader) -> str | None:
    metadata = reader.metadata
    if metadata and metadata.title:
        return metadata.title.strip() or None
    return None


def _extract_tables(
    content: bytes, pypdf_pages: list[tuple[int, str]]
) -> tuple[list[tuple[int, str]], list[DocumentSection]]:
    """Runs the `pdfplumber` table pass.

    Returns `(page_texts, table_sections)` where `page_texts` is the
    per-page text for the heading splitter — identical to `pypdf_pages`
    for every page with no accepted table, and the table-region-removed
    remainder for pages that had one — and `table_sections` is the
    structured tables in document order.
    """
    page_texts: list[tuple[int, str]] = []
    table_sections: list[DocumentSection] = []
    table_number = 0
    carry: pdf_tables.TableCarry | None = None
    started_at = time.perf_counter()

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        plumber_pages = pdf.pages
        for position, (page_number, pypdf_text) in enumerate(pypdf_pages):
            if position >= len(plumber_pages):
                page_texts.append((page_number, pypdf_text))
                continue

            page = plumber_pages[position]

            over_budget = time.perf_counter() - started_at > _TABLE_BUDGET_SECONDS
            has_rules = bool(page.rects) or bool(page.lines)
            if over_budget or not page.chars or not has_rules:
                if over_budget:
                    logger.warning("PDF table extraction budget exceeded; page %s left as text.", page_number)
                page_texts.append((page_number, pypdf_text))
                carry = None
                continue

            accepted_bboxes: list[tuple[float, float, float, float]] = []
            new_carry: pdf_tables.TableCarry | None = None

            for candidate in page.find_tables(table_settings=pdf_tables.TABLE_SETTINGS):
                rows = pdf_tables.clean_rows(candidate.extract())
                if not rows:
                    continue

                if carry is not None and not accepted_bboxes:
                    stitched = pdf_tables.stitch_continuation(rows, candidate.bbox, page.height, carry)
                    if stitched is not None:
                        table_sections.append(
                            pdf_tables.render_table_section(
                                carry.number, carry.header, stitched, carry.start_page, continued=True
                            )
                        )
                        accepted_bboxes.append(candidate.bbox)
                        new_carry = pdf_tables.TableCarry(
                            number=carry.number,
                            header=carry.header,
                            column_count=carry.column_count,
                            start_page=carry.start_page,
                            ended_at_bottom=pdf_tables.ends_near_bottom(candidate.bbox, page.height),
                        )
                        continue

                if not pdf_tables.is_reliable_table(rows)[0]:
                    continue

                table_number += 1
                table_sections.append(
                    pdf_tables.render_table_section(table_number, rows[0], rows[1:], page_number, continued=False)
                )
                accepted_bboxes.append(candidate.bbox)
                new_carry = pdf_tables.TableCarry(
                    number=table_number,
                    header=rows[0],
                    column_count=pdf_tables.column_count(rows),
                    start_page=page_number,
                    ended_at_bottom=pdf_tables.ends_near_bottom(candidate.bbox, page.height),
                )

            if not accepted_bboxes:
                page_texts.append((page_number, pypdf_text))
                carry = None
                continue

            remainder = page
            for bbox in accepted_bboxes:
                remainder = remainder.outside_bbox(bbox)
            remainder_text = (remainder.extract_text() or "").strip()
            if remainder_text:
                page_texts.append((page_number, remainder_text))
            carry = new_carry

    return page_texts, table_sections


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


def _merge_sections(
    text_sections: list[DocumentSection], table_sections: list[DocumentSection]
) -> list[DocumentSection]:
    """Interleaves the two section lists by page — text before tables
    within a page, each list's own order preserved otherwise. Sections
    with no page number (only possible if the text pass produced none)
    sort to the end.
    """
    if not table_sections:
        return text_sections

    tagged = [(section.page_number, 0, index, section) for index, section in enumerate(text_sections)]
    tagged += [(section.page_number, 1, index, section) for index, section in enumerate(table_sections)]
    tagged.sort(key=lambda item: (item[0] is None, item[0] or 0, item[1], item[2]))
    return [item[3] for item in tagged]
