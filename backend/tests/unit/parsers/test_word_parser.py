import io

import pytest
from docx import Document as open_docx

from app.core.exceptions import ValidationError
from app.models.common import DocumentSource, DocumentType
from app.services.parsers.word_parser import WordParser
from tests.unit.parsers.factories import make_document


def _build_docx(sections: list[tuple[str, str]]) -> bytes:
    docx = open_docx()
    for heading, body in sections:
        docx.add_heading(heading, level=1)
        docx.add_paragraph(body)
    buffer = io.BytesIO()
    docx.save(buffer)
    return buffer.getvalue()


def test_splits_sections_by_heading():
    content = _build_docx([("Overview", "This describes the workflow."), ("Steps", "Do this then that.")])
    document = make_document(filename="workflow.docx", document_type=DocumentType.DOCX)

    parsed = WordParser().parse(document, content)

    assert [s.heading for s in parsed.sections] == ["Overview", "Steps"]
    assert "This describes the workflow." in parsed.sections[0].content
    assert "Do this then that." in parsed.sections[1].content
    assert "This describes the workflow." in parsed.content
    assert parsed.metadata.page_number is None


def test_metadata_reflects_user_upload_source():
    content = _build_docx([("Notes", "Body")])
    document = make_document(
        filename="notes.docx",
        document_type=DocumentType.DOCX,
        feature="Analytics",
        source=DocumentSource.USER_UPLOAD,
    )

    parsed = WordParser().parse(document, content)

    assert parsed.metadata.document_source == DocumentSource.USER_UPLOAD
    assert parsed.metadata.feature == "Analytics"
    assert parsed.metadata.parser_name == "WordParser"
    assert parsed.metadata.parser_version == "1.0"


def test_uses_document_core_title_when_present():
    docx = open_docx()
    docx.core_properties.title = "Workflow Guide"
    docx.add_paragraph("Body")
    buffer = io.BytesIO()
    docx.save(buffer)

    document = make_document(filename="workflow.docx", document_type=DocumentType.DOCX)
    parsed = WordParser().parse(document, buffer.getvalue())

    assert parsed.title == "Workflow Guide"


def test_falls_back_to_filename_when_no_title_or_headings():
    content = _build_docx([])
    document = make_document(filename="misc_notes.docx", document_type=DocumentType.DOCX)

    parsed = WordParser().parse(document, content)

    assert parsed.title == "Misc Notes"


def test_raises_validation_error_for_corrupt_docx():
    document = make_document(filename="broken.docx", document_type=DocumentType.DOCX)

    with pytest.raises(ValidationError):
        WordParser().parse(document, b"not a real docx")
