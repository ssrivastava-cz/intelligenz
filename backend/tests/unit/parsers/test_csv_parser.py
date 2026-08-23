import pytest

from app.core.exceptions import ValidationError
from app.models.common import DocumentSource, DocumentType
from app.services.parsers.csv_parser import CsvParser
from tests.unit.parsers.factories import make_document


def test_one_section_per_row_not_one_giant_section():
    content = b"id,title\n1,Book appointment\n2,Cancel appointment\n3,Reschedule appointment\n"
    document = make_document(filename="historical-cases.csv", document_type=DocumentType.CSV)

    parsed = CsvParser().parse(document, content)

    assert len(parsed.sections) == 3
    assert "Book appointment" in parsed.sections[0].content
    assert "Cancel appointment" in parsed.sections[1].content
    assert "Reschedule appointment" in parsed.sections[2].content


def test_heading_combines_id_and_title_columns():
    content = b"Test Case ID,Title,Steps\nTC-001,Book appointment,Open the form.\n"
    document = make_document(filename="cases.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert section.heading == "TC-001 - Book appointment"


def test_heading_works_for_issue_id_column_too():
    """The heading rule isn't hardcoded to "Test Case ID" — any *ID + Title
    column pair qualifies, so the same parser handles issue sheets."""
    content = b"Issue ID,Title,Status\nISSUE-42,Ticket sync lags,Open\n"
    document = make_document(filename="issues.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert section.heading == "ISSUE-42 - Ticket sync lags"


def test_heading_falls_back_to_id_only_when_no_title_column():
    content = b"Issue ID,Status\nISSUE-42,Open\n"
    document = make_document(filename="issues.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert section.heading == "ISSUE-42"


def test_heading_falls_back_to_title_only_when_no_id_column():
    content = b"Title,Status\nSync lag,Open\n"
    document = make_document(filename="issues.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert section.heading == "Sync lag"


def test_heading_is_none_without_id_or_title_columns():
    content = b"a,b\n1,2\n"
    document = make_document(filename="generic.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert section.heading is None
    assert "a: 1" in section.content
    assert "b: 2" in section.content


def test_section_content_includes_every_column_including_extras():
    content = (
        b"Test Case ID,Title,Objective,Preconditions,Steps,Expected Result,Priority,Type,Automation Notes\n"
        b"TC-001,Book appointment,Verify booking works,User is logged in,Click Book,"
        b"Appointment is created,High,Functional,Covered by e2e suite\n"
    )
    document = make_document(filename="cases.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    for expected in [
        "Test Case ID: TC-001",
        "Title: Book appointment",
        "Objective: Verify booking works",
        "Preconditions: User is logged in",
        "Steps: Click Book",
        "Expected Result: Appointment is created",
        "Priority: High",
        "Type: Functional",
        "Automation Notes: Covered by e2e suite",
    ]:
        assert expected in section.content


def test_issue_sheet_columns_all_preserved():
    content = (
        b"Issue ID,Title,Description,Root Cause,Resolution,Status\n"
        b"ISSUE-1,Sync lag,Contacts sync slowly,Batch job too infrequent,"
        b"Reduced batch interval,Resolved\n"
    )
    document = make_document(filename="issues.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    for expected in [
        "Issue ID: ISSUE-1",
        "Description: Contacts sync slowly",
        "Root Cause: Batch job too infrequent",
        "Resolution: Reduced batch interval",
        "Status: Resolved",
    ]:
        assert expected in section.content


def test_empty_csv_with_only_a_header_row_does_not_crash():
    content = b"id,title\n"
    document = make_document(filename="empty.csv", document_type=DocumentType.CSV)

    parsed = CsvParser().parse(document, content)

    assert len(parsed.sections) == 1
    assert parsed.sections[0].content == ""


def test_metadata_reflects_user_upload_source():
    content = b"a,b\n1,2\n"
    document = make_document(
        filename="upload.csv",
        document_type=DocumentType.CSV,
        source=DocumentSource.USER_UPLOAD,
        session_id="sess-2",
    )

    parsed = CsvParser().parse(document, content)

    assert parsed.metadata.document_source == DocumentSource.USER_UPLOAD
    assert parsed.metadata.parser_name == "CsvParser"
    assert parsed.metadata.parser_version == "1.0"


def test_title_derived_from_filename():
    content = b"a,b\n1,2\n"
    document = make_document(filename="release_notes.csv", document_type=DocumentType.CSV)

    parsed = CsvParser().parse(document, content)

    assert parsed.title == "Release Notes"


def test_raises_validation_error_for_undecodable_bytes():
    """0x81 is undefined in both UTF-8 and CP1252 — genuinely
    unreadable input, not just a file that needs the CP1252 fallback
    (unlike e.g. 0xff/0xfe/0xfd/0xfc, which CP1252 can decode)."""
    document = make_document(filename="broken.csv", document_type=DocumentType.CSV)

    with pytest.raises(ValidationError):
        CsvParser().parse(document, b"\x81\x8d\x8f\x90\x9d")


def test_undecodable_bytes_error_names_the_file_and_every_attempted_encoding():
    document = make_document(filename="broken.csv", document_type=DocumentType.CSV)

    with pytest.raises(ValidationError) as exc_info:
        CsvParser().parse(document, b"\x81\x8d\x8f\x90\x9d")

    message = str(exc_info.value)
    assert "broken.csv" in message
    assert "utf-8-sig" in message
    assert "cp1252" in message


# --- Encoding: UTF-8 (default), UTF-8 BOM, and CP1252 fallback ---


def test_plain_utf8_csv_parses_normally():
    content = "id,title\n1,Café appointment\n".encode()
    document = make_document(filename="utf8.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert "Café appointment" in section.content


def test_utf8_bom_csv_is_decoded_via_utf8_sig():
    content = "id,title\n1,Café appointment\n".encode("utf-8-sig")
    document = make_document(filename="utf8_bom.csv", document_type=DocumentType.CSV)

    parsed = CsvParser().parse(document, content)
    [section] = parsed.sections

    assert "Café appointment" in section.content
    # The BOM itself must never leak into a header/value.
    assert "﻿" not in parsed.content
    assert list(section.content)[0] != "﻿"


def test_cp1252_csv_with_curly_apostrophe_falls_back_successfully():
    """The reported bug: byte 0x92 (a Windows-1252 curly right single
    quote) is not valid UTF-8 and previously made the whole file fail
    to parse."""
    content = "id,title\n1,Contact’s record\n".encode("cp1252")
    assert b"\x92" in content  # sanity check the fixture actually exercises 0x92
    document = make_document(filename="Exporting List (Cron Export).csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert "Contact’s record" in section.content


def test_cp1252_csv_with_other_common_special_characters_falls_back_successfully():
    """En dash (0x96), em dash (0x97), and curly double quotes (0x93/0x94)
    — other bytes common in Windows-1252 exports that are invalid UTF-8."""
    content = "id,title\n1,“Status – Open” — review\n".encode("cp1252")
    document = make_document(filename="special_chars.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert "“Status – Open” — review" in section.content


def test_cp1252_fallback_never_uses_errors_ignore_or_replace():
    """A CP1252 fallback that silently dropped/replaced bytes could
    still "succeed" on genuinely corrupt input — proving real content
    round-trips exactly (not just that parsing didn't raise) is what
    actually rules that out."""
    original_text = "id,title\n1,Renewal – 100% success — confirmed’s satisfaction\n"
    content = original_text.encode("cp1252")
    document = make_document(filename="fidelity.csv", document_type=DocumentType.CSV)

    [section] = CsvParser().parse(document, content).sections

    assert "Renewal – 100% success — confirmed’s" in section.content


def test_logs_the_encoding_actually_used(caplog):
    content = "id,title\n1,Contact’s record\n".encode("cp1252")
    document = make_document(filename="Exporting List (Cron Export).csv", document_type=DocumentType.CSV)

    with caplog.at_level("INFO"):
        CsvParser().parse(document, content)

    assert any(
        "Exporting List (Cron Export).csv" in message and "cp1252" in message for message in caplog.messages
    )
