import io

import pytest
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.exceptions import ValidationError
from app.models.common import DocumentSource, DocumentType
from app.services.parsers import pdf_parser as pdf_parser_module
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


def test_id_value_lines_are_not_misclassified_as_headings():
    """Confirmed against the real Source of Truth PDFs, which contain
    exactly these lines today — this reproduces the bug at the parser
    level, not just in the underlying heuristic's own unit tests."""
    content = _build_pdf_with_lines(
        [
            [
                "Test Data Conventions",
                "Patient ID P10001",
                "Member ID M10001",
                "Plan ID PLAN-A",
            ]
        ]
    )
    document = make_document(filename="test-data.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert [s.heading for s in parsed.sections] == ["Test Data Conventions"]
    assert "Patient ID P10001" in parsed.sections[0].content
    assert "Member ID M10001" in parsed.sections[0].content
    assert "Plan ID PLAN-A" in parsed.sections[0].content


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
    assert parsed.metadata.parser_version == "1.1"


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


def test_metadata_carries_the_portable_source_path_and_folder():
    content = _build_pdf(["Body text"])
    document = make_document(
        filename="notes.pdf",
        document_type=DocumentType.PDF,
        source_relative_path="source_of_truth/Service Model 1/workflows/notes.pdf",
        source_folder="Service Model 1",
    )

    parsed = PdfParser().parse(document, content)

    assert parsed.metadata.source_path == "source_of_truth/Service Model 1/workflows/notes.pdf"
    assert parsed.metadata.source_folder == "Service Model 1"


# --- table extraction -----------------------------------------------------------


def _pdf(flowables: list) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(400, 600), title="")
    doc.build(flowables)
    return buffer.getvalue()


def _heading(text: str):
    return Paragraph(text, getSampleStyleSheet()["Heading1"])


def _para(text: str):
    return Paragraph(text, getSampleStyleSheet()["BodyText"])


_CELL_STYLE = ParagraphStyle("cell", parent=getSampleStyleSheet()["BodyText"], fontSize=8, leading=10)


def _cells(data: list[list[str]]) -> list[list[Paragraph]]:
    """Wrap every cell in a Paragraph so reportlab actually renders and
    word-wraps it (a bare string in a Table cell does neither)."""
    return [[Paragraph(str(value), _CELL_STYLE) for value in row] for row in data]


def _grid_table(data: list[list[str]], col_widths: list[int] | None = None) -> Table:
    """A bordered table — the GRID style draws the ruling lines pdfplumber's
    'lines' table strategy keys off."""
    table = Table(_cells(data), colWidths=col_widths)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    return table


def _table_sections(parsed) -> list:
    return [section for section in parsed.sections if (section.heading or "").startswith("Table ")]


def test_bordered_table_becomes_a_structured_key_value_section():
    content = _pdf(
        [
            _heading("Test Data Conventions"),
            _para("The values below are used throughout the plan."),
            Spacer(1, 12),
            _grid_table(
                [
                    ["Field", "Example value"],
                    ["Patient ID", "P10001"],
                    ["Member ID", "M10001"],
                    ["Plan ID", "PLAN-A"],
                ],
                col_widths=[140, 200],
            ),
        ]
    )
    document = make_document(filename="conventions.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)
    tables = _table_sections(parsed)

    assert len(tables) == 1
    body = tables[0].content
    assert "Field: Patient ID" in body
    assert "Example value: P10001" in body
    assert "Field: Plan ID" in body
    assert "Example value: PLAN-A" in body
    # the ambiguous flat form pypdf produces must not be what we ship
    assert "Plan ID PLAN-A" not in body


def test_table_preserves_row_order_and_every_column():
    content = _pdf(
        [
            _heading("Business Requirements"),
            _grid_table(
                [
                    ["ID", "Area", "Requirement"],
                    ["BR-001", "Intake", "Process appointments requiring evaluation."],
                    ["BR-002", "New", "Evaluate a new appointment."],
                    ["BR-003", "Cancel", "Skip a cancelled appointment."],
                ],
                col_widths=[70, 80, 200],
            ),
        ]
    )
    document = make_document(filename="brd.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)
    body = _table_sections(parsed)[0].content

    assert body.index("BR-001") < body.index("BR-002") < body.index("BR-003")
    assert "Area: Intake" in body
    assert "Requirement: Process appointments requiring evaluation." in body


def test_wrapped_cell_text_stays_within_its_row():
    wrapped = (
        "Provider is eligible when every configured condition is met, "
        "including active enrollment covering the visit date and a "
        "participating provider for the plan and site."
    )
    content = _pdf(
        [
            _heading("Provider Eligibility"),
            _grid_table(
                [
                    ["Provider ID", "Description"],
                    ["P10001", wrapped],
                    ["P10002", "Short description."],
                ],
                col_widths=[90, 250],
            ),
        ]
    )
    document = make_document(filename="providers.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)
    body = _table_sections(parsed)[0].content

    # exactly two data rows -> exactly two "Provider ID:" lines, not one per
    # wrapped physical line
    assert body.count("Provider ID: P") == 2
    assert "Provider ID: P10001" in body
    assert "Description: Provider is eligible when every configured condition is met" in body
    assert "Provider ID: P10002" in body


def test_borderless_aligned_text_is_not_promoted_to_a_table():
    # columns held together by whitespace alignment only — no ruling lines
    content = _build_pdf_with_lines(
        [
            [
                "Test Data Conventions",
                "Field           Example value",
                "Patient ID      P10001",
                "Member ID       M10001",
                "Plan ID         PLAN-A",
            ]
        ]
    )
    document = make_document(filename="borderless.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert _table_sections(parsed) == []
    # the content is still there, just as plain text
    assert "P10001" in parsed.content
    assert "PLAN-A" in parsed.content


def test_single_column_callout_box_is_not_a_table():
    content = _pdf(
        [
            _heading("Scope Notes"),
            _grid_table(
                [
                    ["PURPOSE"],
                    ["This document lays out the full set of scenarios the system must handle."],
                    ["It supersedes the earlier proposal."],
                ],
                col_widths=[340],
            ),
        ]
    )
    document = make_document(filename="callout.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert _table_sections(parsed) == []
    assert "supersedes the earlier proposal" in parsed.content


def test_data_like_lines_without_a_table_are_neither_headings_nor_a_table():
    content = _build_pdf_with_lines(
        [
            [
                "Test Data Conventions",
                "Patient ID P10001",
                "Member ID M10001",
                "Plan ID PLAN-A",
            ]
        ]
    )
    document = make_document(filename="test-data.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert _table_sections(parsed) == []
    assert [s.heading for s in parsed.sections] == ["Test Data Conventions"]
    assert "Patient ID P10001" in parsed.sections[0].content


def _two_page_table_pdf(second_page_first_row: list[str]) -> bytes:
    """Page 1: a heading, then a 3-column bordered table whose grid runs
    down to the bottom margin. Page 2: more grid rows starting hard against
    the top margin, then unrelated trailing prose. `second_page_first_row`
    lets a test choose whether the continuation fragment leads with a
    wrapped-prose crumb or with the repeated header."""
    width, height = 400, 320
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(width, height))
    pdf.setTitle("")
    col_x = [20, 90, 170, 380]
    row_h = 20

    def grid(y_top: float, rows: list[list[str]]) -> None:
        y_bottom = y_top - row_h * len(rows)
        for index in range(len(rows) + 1):
            y = y_top - index * row_h
            pdf.line(col_x[0], y, col_x[-1], y)
        for x in col_x:
            pdf.line(x, y_top, x, y_bottom)
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                pdf.drawString(col_x[col_index] + 3, y_top - row_index * row_h - row_h + 6, value)

    # page 1: heading, then a grid whose last line sits within a page
    # margin of the bottom edge (so it reads as "runs off the page")
    pdf.drawString(20, height - 16, "Business Requirements")
    grid(
        height - 26,
        [["ID", "Area", "Requirement"]]
        + [[f"BR-{n:03d}", "Area", f"Requirement number {n}."] for n in range(1, 12)],
    )
    pdf.showPage()
    # page 2: grid resumes hard against the top edge, then unrelated prose
    grid(
        height - 6,
        [
            second_page_first_row,
            ["BR-012", "Publish", "Eligibility changes emit an event."],
            ["BR-013", "Complete", "Processed appointments are completed."],
        ],
    )
    pdf.drawString(20, 120, "12. Closing Notes")
    pdf.drawString(20, 100, "Everything after the table is ordinary prose.")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_table_spanning_two_pages_is_stitched_with_the_header_repeated():
    # continuation fragment leads with a wrapped-cell crumb bleeding down
    # from page 1's last row
    content = _two_page_table_pdf(["", "", "supported for the plan and site."])
    document = make_document(filename="spanning.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)
    tables = _table_sections(parsed)

    assert [t.heading for t in tables] == ["Table 1", "Table 1 (continued)"]
    continued = tables[1]
    # header context is repeated on the continuation
    assert "ID: BR-012" in continued.content
    assert "Area: Publish" in continued.content
    assert "ID: BR-013" in continued.content
    # the wrapped-prose crumb is trimmed, never promoted to a data row
    assert continued.content.count("\nID: ") == 2
    # continuation keeps the starting page number
    assert continued.page_number == 1
    # trailing prose on page 2 is still its own text section
    assert any(s.heading == "12. Closing Notes" for s in parsed.sections)


def test_table_spanning_two_pages_handles_a_repeated_header_row():
    content = _two_page_table_pdf(["ID", "Area", "Requirement"])
    document = make_document(filename="spanning-hdr.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)
    tables = _table_sections(parsed)

    assert [t.heading for t in tables] == ["Table 1", "Table 1 (continued)"]
    assert "ID: BR-012" in tables[1].content
    assert "ID: BR-013" in tables[1].content
    # the repeated header row itself is not emitted as data
    assert "ID: ID" not in tables[1].content


def test_table_section_carries_page_number_and_document_metadata():
    content = _pdf(
        [
            _heading("Overview"),
            _para("Intro text on the first page."),
            PageBreak(),
            _heading("Requirements"),
            _grid_table(
                [
                    ["ID", "Requirement"],
                    ["BR-001", "Process new appointments."],
                    ["BR-002", "Evaluate eligibility."],
                ],
                col_widths=[80, 240],
            ),
        ]
    )
    document = make_document(
        filename="meta.pdf",
        document_type=DocumentType.PDF,
        feature="Service Model 1",
        source_relative_path="source_of_truth/Service Model 1/workflows/meta.pdf",
        source_folder="Service Model 1",
    )

    parsed = PdfParser().parse(document, content)
    tables = _table_sections(parsed)

    assert len(tables) == 1
    assert tables[0].page_number == 2
    assert parsed.metadata.parser_name == "PdfParser"
    assert parsed.metadata.parser_version == "1.1"
    assert parsed.metadata.page_number == 2
    assert parsed.metadata.source_path == "source_of_truth/Service Model 1/workflows/meta.pdf"
    assert parsed.metadata.source_folder == "Service Model 1"
    assert parsed.metadata.feature == "Service Model 1"


def test_table_pass_failure_falls_back_to_plain_text(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("pdfplumber exploded")

    monkeypatch.setattr(pdf_parser_module.pdfplumber, "open", boom)

    content = _pdf(
        [
            _heading("Business Requirements"),
            _grid_table(
                [["ID", "Requirement"], ["BR-001", "Process new appointments."]],
                col_widths=[80, 240],
            ),
        ]
    )
    document = make_document(filename="fallback.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    # no structured table, but the document still parses and keeps its text
    assert _table_sections(parsed) == []
    assert "Process new appointments." in parsed.content
    assert parsed.sections[0].heading == "Business Requirements"


def test_page_without_a_text_layer_does_not_crash_and_yields_no_table():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(300, 300))
    pdf.setTitle("")
    pdf.rect(20, 20, 260, 260, stroke=1, fill=0)  # a border, but no text at all
    pdf.showPage()
    pdf.save()
    document = make_document(filename="image-only.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, buffer.getvalue())

    assert _table_sections(parsed) == []
    assert parsed.metadata.page_number == 1


def test_normal_multi_section_pdf_is_unaffected_by_the_table_pass():
    content = _build_pdf_with_lines(
        [
            ["Role Permissions", "Only admins can edit contacts.", "Validation Rules", "Email must be unique."],
            ["Data Retention", "Records are kept for seven years."],
        ]
    )
    document = make_document(filename="policy.pdf", document_type=DocumentType.PDF)

    parsed = PdfParser().parse(document, content)

    assert _table_sections(parsed) == []
    assert [s.heading for s in parsed.sections] == ["Role Permissions", "Validation Rules", "Data Retention"]
    assert parsed.sections[2].page_number == 2
