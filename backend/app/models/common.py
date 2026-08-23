"""Shared internal enums used across models."""
from enum import StrEnum


class Priority(StrEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TestType(StrEnum):
    FUNCTIONAL = "Functional"
    REGRESSION = "Regression"
    EDGE_CASE = "Edge Case"
    INTEGRATION = "Integration"
    NEGATIVE = "Negative"


class TestCaseSource(StrEnum):
    REDMINE_TICKET = "Redmine Ticket"
    HISTORICAL_TEST_CASE = "Historical Test Case"
    HISTORICAL_ISSUE_SHEET = "Historical Issue Sheet"
    RELEASE_NOTES = "Release Notes"
    GENERATED = "Generated"


class GenerationStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FeedbackRating(StrEnum):
    UP = "up"
    DOWN = "down"


class DocumentSource(StrEnum):
    """Where an ingested document came from."""

    SOURCE_OF_TRUTH = "source_of_truth"
    USER_UPLOAD = "user_upload"


class DocumentType(StrEnum):
    """Supported document *formats* for ingestion — determines which parser
    handles a file. Not to be confused with `DocumentCategory`, which is
    a different axis (what kind of Source of Truth content it is).
    """

    PDF = "PDF"
    DOCX = "DOCX"
    XLSX = "XLSX"
    CSV = "CSV"
    TXT = "TXT"
    MARKDOWN = "MARKDOWN"


class DocumentCategory(StrEnum):
    """What kind of content a document represents — independent of
    `DocumentType` (file format). For Source of Truth documents
    discovered by `SourceOfTruthIndexer`, this comes from the category
    folder it lives in (`workflows/`, `TestCases/`, `IssueSheets/`).
    Also reused as `Chunk.artifact_type` (see `app.models.chunk`) to
    pick a chunking strategy, which is why `RELEASE_NOTES` and
    `REQUIREMENT` exist here even though no folder produces them yet.

    `USER_UPLOAD` is for ad-hoc user-uploaded documents (see
    `UploadedDocumentIndexer`) — temporary knowledge, not part of any
    Source of Truth category folder.
    """

    WORKFLOW = "WORKFLOW"
    TEST_CASE = "TEST_CASE"
    ISSUE = "ISSUE"
    RELEASE_NOTES = "RELEASE_NOTES"
    REQUIREMENT = "REQUIREMENT"
    USER_UPLOAD = "USER_UPLOAD"
