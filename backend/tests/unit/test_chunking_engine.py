from datetime import UTC, datetime

import pytest

from app.models.common import DocumentCategory, DocumentSource, DocumentType
from app.models.document import Document
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.services.chunking_engine import ChunkingEngine
from app.services.parsers.csv_parser import CsvParser
from app.services.parsers.markdown_parser import MarkdownParser


def _make_parsed_document(
    sections: list[DocumentSection],
    *,
    document_id: str = "doc-1",
    feature: str = "Appointments",
    document_source: DocumentSource = DocumentSource.SOURCE_OF_TRUTH,
    source_filename: str = "doc.md",
) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        title="Doc",
        sections=sections,
        content="\n\n".join(s.content for s in sections),
        metadata=ParsedDocumentMetadata(
            feature=feature,
            document_type=DocumentType.MARKDOWN,
            document_source=document_source,
            source_filename=source_filename,
            parser_name="MarkdownParser",
            parser_version="1.0",
        ),
    )


def _make_document(**overrides) -> Document:
    defaults = {
        "id": "doc-src-1",
        "filename": "source.csv",
        "document_type": DocumentType.CSV,
        "feature": "Appointments",
        "size": 0,
        "source": DocumentSource.SOURCE_OF_TRUTH,
        "category": DocumentCategory.TEST_CASE,
        "storage_path": "unused",
        "uploaded_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return Document(**defaults)


# --- construction / configuration validation -------------------------------


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkingEngine(chunk_size=0)


def test_chunk_overlap_cannot_be_negative():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingEngine(chunk_size=10, chunk_overlap=-1)


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingEngine(chunk_size=10, chunk_overlap=10)


# --- the engine is format/category-agnostic: it just converts sections -----


@pytest.mark.parametrize(
    "artifact_type",
    [
        DocumentCategory.WORKFLOW,
        DocumentCategory.RELEASE_NOTES,
        DocumentCategory.REQUIREMENT,
        DocumentCategory.TEST_CASE,
        DocumentCategory.ISSUE,
    ],
)
def test_one_chunk_per_section_regardless_of_artifact_type(artifact_type):
    """The engine doesn't branch on artifact_type at all — every type
    gets the same section -> chunk conversion. artifact_type is only
    ever attached to the resulting chunks as metadata."""
    document = _make_parsed_document(
        [
            DocumentSection(heading="Overview", content="This describes the workflow."),
            DocumentSection(heading="Steps", content="Do this then that."),
        ]
    )

    chunks = ChunkingEngine().chunk_document(document, artifact_type)

    assert len(chunks) == 2
    assert [c.chunk_number for c in chunks] == [1, 2]
    assert all(c.artifact_type == artifact_type for c in chunks)


def test_artifact_type_does_not_affect_chunk_count():
    """Regression guard: a large section produces the same number of
    chunks no matter which artifact_type is passed in — proving the
    engine really doesn't special-case TEST_CASE/ISSUE anymore.
    """
    words = [f"word{i}" for i in range(200)]
    document = _make_parsed_document([DocumentSection(heading="TC-001", content=" ".join(words))])
    engine = ChunkingEngine(chunk_size=50, chunk_overlap=5)

    counts = {
        artifact_type: len(engine.chunk_document(document, artifact_type))
        for artifact_type in DocumentCategory
    }

    assert len(set(counts.values())) == 1  # every artifact_type produced the same chunk count
    assert next(iter(counts.values())) > 1  # and it did actually get split


def test_chunk_text_includes_section_heading_when_present():
    document = _make_parsed_document([DocumentSection(heading="Overview", content="Body text.")])

    [chunk] = ChunkingEngine().chunk_document(document, DocumentCategory.WORKFLOW)

    assert chunk.chunk_text == "Overview\n\nBody text."


def test_chunk_text_omits_heading_when_none():
    document = _make_parsed_document([DocumentSection(heading=None, content="Just body text.")])

    [chunk] = ChunkingEngine().chunk_document(document, DocumentCategory.WORKFLOW)

    assert chunk.chunk_text == "Just body text."


def test_splits_long_section_into_overlapping_windows():
    words = [f"word{i}" for i in range(1, 26)]  # 25 words
    document = _make_parsed_document([DocumentSection(heading=None, content=" ".join(words))])

    chunks = ChunkingEngine(chunk_size=10, chunk_overlap=3).chunk_document(document, DocumentCategory.WORKFLOW)

    # step = 10 - 3 = 7 -> windows start at 0, 7, 14, 21 => 4 windows
    assert len(chunks) == 4
    assert chunks[0].chunk_text.split() == words[0:10]
    assert chunks[1].chunk_text.split() == words[7:17]
    # consecutive windows overlap by exactly `chunk_overlap` words
    assert chunks[0].chunk_text.split()[-3:] == chunks[1].chunk_text.split()[:3]
    assert chunks[3].chunk_text.split() == words[21:25]


def test_short_section_is_not_split():
    document = _make_parsed_document([DocumentSection(heading=None, content="Only five words right here.")])

    chunks = ChunkingEngine(chunk_size=500, chunk_overlap=50).chunk_document(document, DocumentCategory.WORKFLOW)

    assert len(chunks) == 1


def test_configurable_chunk_size_changes_chunk_count():
    words = [f"w{i}" for i in range(100)]
    document = _make_parsed_document([DocumentSection(heading=None, content=" ".join(words))])

    small_chunks = ChunkingEngine(chunk_size=20, chunk_overlap=0).chunk_document(document, DocumentCategory.WORKFLOW)
    large_chunks = ChunkingEngine(chunk_size=100, chunk_overlap=0).chunk_document(document, DocumentCategory.WORKFLOW)

    assert len(small_chunks) == 5
    assert len(large_chunks) == 1


def test_configurable_overlap_changes_chunk_count():
    words = [f"w{i}" for i in range(100)]
    document = _make_parsed_document([DocumentSection(heading=None, content=" ".join(words))])

    no_overlap = ChunkingEngine(chunk_size=20, chunk_overlap=0).chunk_document(document, DocumentCategory.WORKFLOW)
    with_overlap = ChunkingEngine(chunk_size=20, chunk_overlap=10).chunk_document(document, DocumentCategory.WORKFLOW)

    assert len(no_overlap) == 5
    assert len(with_overlap) == 9


def test_chunk_number_increments_across_sections_not_per_section():
    document = _make_parsed_document(
        [
            DocumentSection(heading="A", content="First."),
            DocumentSection(heading="B", content="Second."),
            DocumentSection(heading="C", content="Third."),
        ]
    )

    chunks = ChunkingEngine().chunk_document(document, DocumentCategory.WORKFLOW)

    assert [c.chunk_number for c in chunks] == [1, 2, 3]


# --- metadata preservation ---------------------------------------------


def test_preserves_all_required_metadata_on_every_chunk():
    document = _make_parsed_document(
        [DocumentSection(heading="Overview", content="Body.", page_number=2)],
        document_id="doc-xyz",
        feature="Analytics",
        document_source=DocumentSource.USER_UPLOAD,
        source_filename="release-notes.pdf",
    )

    [chunk] = ChunkingEngine().chunk_document(document, DocumentCategory.RELEASE_NOTES)

    assert chunk.document_id == "doc-xyz"
    assert chunk.feature == "Analytics"
    assert chunk.artifact_type == DocumentCategory.RELEASE_NOTES
    assert chunk.document_source == DocumentSource.USER_UPLOAD
    assert chunk.source_filename == "release-notes.pdf"
    assert chunk.page_number == 2
    assert chunk.chunk_number == 1
    assert chunk.chunk_id


def test_page_number_follows_the_section_it_came_from():
    document = _make_parsed_document(
        [
            DocumentSection(heading="Page 1", content="First page text.", page_number=1),
            DocumentSection(heading="Page 2", content="Second page text.", page_number=2),
        ]
    )

    chunks = ChunkingEngine().chunk_document(document, DocumentCategory.WORKFLOW)

    assert [c.page_number for c in chunks] == [1, 2]


def test_page_number_is_none_for_non_paginated_formats():
    document = _make_parsed_document([DocumentSection(heading=None, content="Body.", page_number=None)])

    [chunk] = ChunkingEngine().chunk_document(document, DocumentCategory.WORKFLOW)

    assert chunk.page_number is None


def test_chunk_ids_are_unique():
    document = _make_parsed_document(
        [DocumentSection(heading="A", content="One."), DocumentSection(heading="B", content="Two.")]
    )

    chunks = ChunkingEngine().chunk_document(document, DocumentCategory.WORKFLOW)

    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_empty_document_produces_no_chunks():
    document = _make_parsed_document([])

    assert ChunkingEngine().chunk_document(document, DocumentCategory.WORKFLOW) == []


# --- end-to-end with real parsers: this is where "one chunk per test case" /
# "one chunk per issue" actually comes from — the parser producing one small
# section per record, not any special handling in the engine -----------------


def test_one_chunk_per_test_case_via_csv_parser():
    source_document = _make_document(
        filename="cases.csv", document_type=DocumentType.CSV, category=DocumentCategory.TEST_CASE
    )
    content = (
        b"Test Case ID,Title,Steps\n"
        b"TC-001,Book appointment,Open the form and submit.\n"
        b"TC-002,Cancel appointment,Open the appointment and cancel.\n"
        b"TC-003,Reschedule appointment,Open the appointment and pick a new time.\n"
    )
    parsed = CsvParser().parse(source_document, content)

    chunks = ChunkingEngine().chunk_document(parsed, DocumentCategory.TEST_CASE)

    assert len(chunks) == 3
    assert [c.chunk_number for c in chunks] == [1, 2, 3]
    assert all(c.artifact_type == DocumentCategory.TEST_CASE for c in chunks)
    assert "TC-001 - Book appointment" in chunks[0].chunk_text
    assert "TC-002 - Cancel appointment" in chunks[1].chunk_text
    assert "TC-003 - Reschedule appointment" in chunks[2].chunk_text


def test_one_chunk_per_issue_via_csv_parser():
    source_document = _make_document(
        filename="issues.csv", document_type=DocumentType.CSV, category=DocumentCategory.ISSUE
    )
    content = (
        b"Issue ID,Title,Status\n"
        b"ISSUE-1,Contact sync lag,Resolved\n"
        b"ISSUE-2,Duplicate tickets,Open\n"
    )
    parsed = CsvParser().parse(source_document, content)

    chunks = ChunkingEngine().chunk_document(parsed, DocumentCategory.ISSUE)

    assert len(chunks) == 2
    assert all(c.artifact_type == DocumentCategory.ISSUE for c in chunks)
    assert "ISSUE-1 - Contact sync lag" in chunks[0].chunk_text
    assert "ISSUE-2 - Duplicate tickets" in chunks[1].chunk_text


def test_chunks_a_real_parsed_markdown_document():
    source_document = Document(
        id="doc-md-1",
        filename="onboarding.md",
        document_type=DocumentType.MARKDOWN,
        feature="Appointments",
        size=42,
        source=DocumentSource.SOURCE_OF_TRUTH,
        category=DocumentCategory.WORKFLOW,
        storage_path="unused",
        uploaded_at=datetime.now(UTC),
    )
    content = b"# Onboarding\nIntro text.\n\n## Steps\nDo this.\nDo that.\n"
    parsed = MarkdownParser().parse(source_document, content)

    chunks = ChunkingEngine().chunk_document(parsed, DocumentCategory.WORKFLOW)

    assert len(chunks) == 2
    assert chunks[0].document_id == "doc-md-1"
    assert "Intro text." in chunks[0].chunk_text
    assert "Do this." in chunks[1].chunk_text
