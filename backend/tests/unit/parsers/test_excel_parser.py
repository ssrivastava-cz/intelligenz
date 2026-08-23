import io

import pytest
from openpyxl import Workbook

from app.core.exceptions import ValidationError
from app.models.common import DocumentSource, DocumentType
from app.services.parsers.excel_parser import ExcelParser
from tests.unit.parsers.factories import make_document


def _build_xlsx(sheets: dict[str, list[list]]) -> bytes:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_creates_one_section_per_sheet():
    content = _build_xlsx(
        {
            "Cases": [["ID", "Title"], [1, "Book appointment"]],
            "Notes": [["Remember to test cancellations"]],
        }
    )
    document = make_document(filename="test-cases.xlsx", document_type=DocumentType.XLSX)

    parsed = ExcelParser().parse(document, content)

    assert [s.heading for s in parsed.sections] == ["Cases", "Notes"]
    assert "Book appointment" in parsed.sections[0].content
    assert "Remember to test cancellations" in parsed.sections[1].content
    assert parsed.metadata.page_number is None


def test_metadata_reflects_user_upload_source():
    content = _build_xlsx({"Sheet1": [["a", "b"]]})
    document = make_document(
        filename="upload.xlsx",
        document_type=DocumentType.XLSX,
        source=DocumentSource.USER_UPLOAD,
        session_id="sess-1",
    )

    parsed = ExcelParser().parse(document, content)

    assert parsed.metadata.document_source == DocumentSource.USER_UPLOAD
    assert parsed.metadata.parser_name == "ExcelParser"
    assert parsed.metadata.parser_version == "1.0"


def test_title_derived_from_filename():
    content = _build_xlsx({"Sheet1": [["a"]]})
    document = make_document(filename="regression_matrix.xlsx", document_type=DocumentType.XLSX)

    parsed = ExcelParser().parse(document, content)

    assert parsed.title == "Regression Matrix"


def test_raises_validation_error_for_corrupt_xlsx():
    document = make_document(filename="broken.xlsx", document_type=DocumentType.XLSX)

    with pytest.raises(ValidationError):
        ExcelParser().parse(document, b"not a real xlsx")
