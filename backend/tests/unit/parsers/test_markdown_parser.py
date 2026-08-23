import pytest

from app.core.exceptions import ValidationError
from app.models.common import DocumentSource, DocumentType
from app.services.parsers.markdown_parser import MarkdownParser
from tests.unit.parsers.factories import make_document


def test_splits_sections_by_heading():
    text = "# Overview\nThis is the intro.\n\n## Steps\n1. Do this\n2. Do that\n"
    document = make_document(filename="spec.md", document_type=DocumentType.MARKDOWN)

    parsed = MarkdownParser().parse(document, text.encode("utf-8"))

    assert [s.heading for s in parsed.sections] == ["Overview", "Steps"]
    assert "This is the intro." in parsed.sections[0].content
    assert "Do this" in parsed.sections[1].content


def test_uses_first_heading_as_title():
    text = "# Feature Spec\nBody text.\n"
    document = make_document(filename="spec.md", document_type=DocumentType.MARKDOWN)

    parsed = MarkdownParser().parse(document, text.encode("utf-8"))

    assert parsed.title == "Feature Spec"


def test_falls_back_to_filename_when_no_headings():
    text = "Just plain text, no headings at all."
    document = make_document(filename="plain_notes.md", document_type=DocumentType.MARKDOWN)

    parsed = MarkdownParser().parse(document, text.encode("utf-8"))

    assert parsed.title == "Plain Notes"
    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading is None


def test_content_preserves_full_text():
    text = "# Title\nSome content here."
    document = make_document(filename="spec.md", document_type=DocumentType.MARKDOWN)

    parsed = MarkdownParser().parse(document, text.encode("utf-8"))

    assert "Some content here." in parsed.content


def test_metadata_reflects_user_upload_source():
    text = "# Title\nBody"
    document = make_document(
        filename="upload.md",
        document_type=DocumentType.MARKDOWN,
        source=DocumentSource.USER_UPLOAD,
        session_id="sess-3",
    )

    parsed = MarkdownParser().parse(document, text.encode("utf-8"))

    assert parsed.metadata.document_source == DocumentSource.USER_UPLOAD
    assert parsed.metadata.parser_name == "MarkdownParser"
    assert parsed.metadata.parser_version == "1.0"


def test_raises_validation_error_for_undecodable_bytes():
    document = make_document(filename="broken.md", document_type=DocumentType.MARKDOWN)

    with pytest.raises(ValidationError):
        MarkdownParser().parse(document, b"\xff\xfe\xfd\xfc")
