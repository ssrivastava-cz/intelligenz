from app.models.common import DocumentSource, DocumentType
from app.services.parsers.txt_parser import TxtParser
from tests.unit.parsers.factories import make_document


def test_returns_single_section_when_no_headings_exist():
    text = "This is the first paragraph.\n\nThis is the second paragraph, still no headings."
    document = make_document(filename="notes.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading is None
    assert "first paragraph" in parsed.sections[0].content
    assert "second paragraph" in parsed.sections[0].content


def test_splits_sections_by_all_caps_heading():
    text = "INTRODUCTION\n\nThis describes the workflow.\n\nSTEPS\n\nDo this.\nDo that."
    document = make_document(filename="workflow.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert [s.heading for s in parsed.sections] == ["INTRODUCTION", "STEPS"]
    assert "This describes the workflow." in parsed.sections[0].content
    assert "Do this." in parsed.sections[1].content


def test_splits_sections_by_colon_heading():
    text = "Overview:\n\nSome text here.\n\nDetails:\n\nMore text here."
    document = make_document(filename="spec.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert [s.heading for s in parsed.sections] == ["Overview", "Details"]
    assert "Some text here." in parsed.sections[0].content
    assert "More text here." in parsed.sections[1].content


def test_accumulates_multiple_paragraphs_under_one_heading():
    text = "NOTES\n\nFirst paragraph under Notes.\n\nSecond paragraph, still under Notes."
    document = make_document(filename="notes.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading == "NOTES"
    assert "First paragraph under Notes." in parsed.sections[0].content
    assert "Second paragraph, still under Notes." in parsed.sections[0].content


def test_content_preceding_first_heading_is_kept_as_a_headingless_section():
    text = "Some preamble text.\n\nSECTION ONE\n\nBody one."
    document = make_document(filename="doc.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert parsed.sections[0].heading is None
    assert "Some preamble text." in parsed.sections[0].content
    assert parsed.sections[1].heading == "SECTION ONE"


def test_uses_first_heading_as_title():
    text = "OVERVIEW\n\nBody text."
    document = make_document(filename="doc.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert parsed.title == "OVERVIEW"


def test_falls_back_to_filename_derived_title_when_no_headings():
    text = "Just a plain paragraph with no headings."
    document = make_document(filename="release_notes.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert parsed.title == "Release Notes"


def test_reads_valid_utf8_text():
    text = "Café résumé naïve — em dash and curly “quotes”."
    document = make_document(filename="unicode.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert "Café résumé naïve" in parsed.content


def test_falls_back_to_safe_encoding_when_utf8_decoding_fails():
    # 0xE9 0xE8 is not valid UTF-8 (0xE9 expects two continuation bytes in
    # 0x80-0xBF; 0xE8 isn't one), but is valid latin-1: b'\xe9\xe8' -> "éè".
    invalid_utf8 = b"\xe9\xe8 some more text"
    document = make_document(filename="legacy.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, invalid_utf8)

    assert "éè" in parsed.content
    assert "some more text" in parsed.content


def test_metadata_reflects_source_document():
    text = "Body text."
    document = make_document(
        filename="notes.txt",
        document_type=DocumentType.TXT,
        feature="Analytics",
        source=DocumentSource.USER_UPLOAD,
        session_id="sess-1",
    )

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert parsed.metadata.feature == "Analytics"
    assert parsed.metadata.document_type == DocumentType.TXT
    assert parsed.metadata.document_source == DocumentSource.USER_UPLOAD
    assert parsed.metadata.source_filename == "notes.txt"
    assert parsed.metadata.page_number is None


def test_metadata_includes_parser_name_and_version():
    document = make_document(filename="notes.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, b"Body text.")

    assert parsed.metadata.parser_name == "TxtParser"
    assert parsed.metadata.parser_version == "1.0"


def test_document_id_matches_source_document():
    document = make_document(id="doc-txt-1", filename="notes.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, b"Body text.")

    assert parsed.document_id == "doc-txt-1"


def test_sentence_ending_lines_are_not_treated_as_headings():
    # A short line ending in a period reads like a sentence, not a heading.
    text = "This is short.\n\nAnother paragraph entirely."
    document = make_document(filename="doc.txt", document_type=DocumentType.TXT)

    parsed = TxtParser().parse(document, text.encode("utf-8"))

    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading is None
