import io

import pytest
from reportlab.pdfgen import canvas

from app.core.exceptions import ValidationError
from app.models.common import DocumentSource, DocumentType
from app.services.parsers.pdf_parser import PdfParser
from tests.unit.parsers.factories import make_document


def _build_pdf(pages: list[str]) -> bytes:
    """One line per page — for the no-headings-detected fallback case."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(300, 300))
    pdf.setTitle("")  # reportlab defaults /Title to "untitled" otherwise
    for text in pages:
        pdf.drawString(10, 250, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _build_pdf_with_lines(pages: list[list[str]]) -> bytes:
    """Multiple lines per page, drawn top-to-bottom — for heading-detection
    tests, where line order within a page matters."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(300, 400))
    pdf.setTitle("")
    for lines in pages:
        y = 380
        for line in lines:
            pdf.drawString(10, y, line)
            y -= 20
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# --- fallback: no headings detected -> one section per page (original behavior) --


def test_falls_back_to_one_section_per_page_when_no_headings_detected():
    content = _build_pdf(["Hello from page one", "Hello from page two"])
    document = make_document(id="doc-1", filename="release-notes.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert parsed.document_id == "doc-1"
    assert len(parsed.sections) == 2
    assert parsed.sections[0].heading == "Page 1"
    assert "Hello from page one" in parsed.sections[0].content
    assert parsed.sections[0].page_number == 1
    assert parsed.sections[1].heading == "Page 2"
    assert "Hello from page two" in parsed.sections[1].content
    assert parsed.sections[1].page_number == 2
    assert "Hello from page one" in parsed.content
    assert "Hello from page two" in parsed.content


# --- logical section detection --------------------------------------------


def test_detects_title_case_headings_and_splits_into_sections():
    content = _build_pdf_with_lines(
        [
            [
                "Role Permissions",
                "Only admins can edit contacts.",
                "Validation Rules",
                "Email must be unique.",
            ]
        ]
    )
    document = make_document(filename="workflow.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert [s.heading for s in parsed.sections] == ["Role Permissions", "Validation Rules"]
    assert "Only admins can edit contacts." in parsed.sections[0].content
    assert "Email must be unique." in parsed.sections[1].content


def test_detects_all_caps_headings():
    content = _build_pdf_with_lines([["OVERVIEW", "This explains the process."]])
    document = make_document(filename="workflow.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert parsed.sections[0].heading == "OVERVIEW"


def test_detects_colon_terminated_headings():
    content = _build_pdf_with_lines([["Contact Update:", "Fields sync automatically."]])
    document = make_document(filename="workflow.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert parsed.sections[0].heading == "Contact Update"


def test_section_spanning_a_page_boundary_keeps_the_starting_page_number():
    content = _build_pdf_with_lines(
        [
            ["Role Permissions", "Line one on page one."],
            ["Line two, continuing on page two."],
        ]
    )
    document = make_document(filename="workflow.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading == "Role Permissions"
    assert parsed.sections[0].page_number == 1
    assert "Line one on page one." in parsed.sections[0].content
    assert "Line two, continuing on page two." in parsed.sections[0].content


def test_new_heading_on_a_later_page_starts_a_new_section():
    content = _build_pdf_with_lines(
        [
            ["Role Permissions", "Text A."],
            ["Validation Rules", "Text B."],
        ]
    )
    document = make_document(filename="workflow.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert [s.heading for s in parsed.sections] == ["Role Permissions", "Validation Rules"]
    assert [s.page_number for s in parsed.sections] == [1, 2]


# --- metadata / title -------------------------------------------------------


def test_metadata_reflects_source_document():
    content = _build_pdf(["Body text"])
    document = make_document(
        filename="notes.pdf",
        document_type=DocumentType.PDF,
        feature="Analytics",
        source=DocumentSource.USER_UPLOAD,
    )

    parsed = PdfParser().parse(document, content)

    assert parsed.metadata.feature == "Analytics"
    assert parsed.metadata.document_type == DocumentType.PDF
    assert parsed.metadata.document_source == DocumentSource.USER_UPLOAD
    assert parsed.metadata.source_filename == "notes.pdf"
    assert parsed.metadata.page_number == 1
    assert parsed.metadata.parser_name == "PdfParser"
    assert parsed.metadata.parser_version == "1.0"


def test_falls_back_to_filename_derived_title_when_no_pdf_title():
    content = _build_pdf(["Body text"])
    document = make_document(filename="release_notes_v2.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert parsed.title == "Release Notes V2"


def test_uses_embedded_pdf_title_when_present():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(300, 300))
    pdf.setTitle("Release Notes Q3")
    pdf.drawString(10, 250, "Body text")
    pdf.showPage()
    pdf.save()

    document = make_document(filename="release_notes_v2.pdf", document_type=DocumentType.PDF)
    parsed = PdfParser().parse(document, buffer.getvalue())

    assert parsed.title == "Release Notes Q3"


def test_raises_validation_error_for_corrupt_pdf():
    document = make_document(filename="broken.pdf", document_type=DocumentType.PDF)

    with pytest.raises(ValidationError):
        PdfParser().parse(document, b"not a real pdf")
