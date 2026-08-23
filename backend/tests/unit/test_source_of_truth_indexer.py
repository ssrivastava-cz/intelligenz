import pytest

from app.core.exceptions import NotFoundError
from app.models.common import DocumentCategory, DocumentSource, DocumentType
from app.services.parser_service import ParserService
from app.services.source_of_truth_indexer import SourceOfTruthIndexer
from app.services.upload_service import UploadService

_MAX_UPLOAD_SIZE_BYTES = 1024 * 1024


def _make_indexer(root):
    # SourceOfTruthIndexer never uploads through this — it's only here to
    # satisfy ParserService's constructor, proving the indexer's scanning
    # is fully decoupled from UploadService's own storage.
    unused_upload_service = UploadService(storage_root=root / "unused", max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES)
    parser_service = ParserService(upload_service=unused_upload_service)
    return SourceOfTruthIndexer(source_of_truth_root=root / "source_of_truth", parser_service=parser_service)


def _write(root, feature: str, folder: str, filename: str, content: str = "content") -> None:
    path = root / "source_of_truth" / feature / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_list_features_discovers_feature_folders(tmp_path):
    _write(tmp_path, "Contact and Sticket Log", "workflows", "a.txt")
    _write(tmp_path, "Coding Tool", "TestCases", "b.csv")

    indexer = _make_indexer(tmp_path)

    assert indexer.list_features() == ["Coding Tool", "Contact and Sticket Log"]


def test_list_features_returns_empty_when_root_missing(tmp_path):
    indexer = _make_indexer(tmp_path)

    assert indexer.list_features() == []


def test_discover_documents_assigns_category_from_parent_folder(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "onboarding.md")
    _write(tmp_path, "Appointments", "TestCases", "cases.csv")
    _write(tmp_path, "Appointments", "IssueSheets", "issues.txt")

    indexer = _make_indexer(tmp_path)
    documents = indexer.discover_documents("Appointments")

    by_filename = {d.filename: d for d in documents}
    assert by_filename["onboarding.md"].category == DocumentCategory.WORKFLOW
    assert by_filename["cases.csv"].category == DocumentCategory.TEST_CASE
    assert by_filename["issues.txt"].category == DocumentCategory.ISSUE
    assert all(d.feature == "Appointments" for d in documents)
    assert all(d.source == DocumentSource.SOURCE_OF_TRUTH for d in documents)


def test_document_type_is_independent_of_category(tmp_path):
    """document_type (file format, for parser dispatch) and category
    (folder-derived) are independent axes — a CSV under workflows/ is
    still format CSV, categorized as WORKFLOW, not the other way around.
    """
    _write(tmp_path, "Appointments", "workflows", "data.csv")

    indexer = _make_indexer(tmp_path)
    [document] = indexer.discover_documents("Appointments")

    assert document.category == DocumentCategory.WORKFLOW
    assert document.document_type == DocumentType.CSV


def test_discover_documents_ignores_unrecognized_subfolders(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt")
    _write(tmp_path, "Appointments", "SomeOtherFolder", "b.txt")

    indexer = _make_indexer(tmp_path)
    documents = indexer.discover_documents("Appointments")

    assert [d.filename for d in documents] == ["a.txt"]


def test_discover_documents_skips_unsupported_extensions(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt")
    _write(tmp_path, "Appointments", "workflows", "diagram.png")

    indexer = _make_indexer(tmp_path)
    documents = indexer.discover_documents("Appointments")

    assert [d.filename for d in documents] == ["a.txt"]


def test_discover_documents_skips_hidden_files(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt")
    _write(tmp_path, "Appointments", "workflows", ".DS_Store")

    indexer = _make_indexer(tmp_path)
    documents = indexer.discover_documents("Appointments")

    assert [d.filename for d in documents] == ["a.txt"]


def test_discover_documents_returns_empty_for_unknown_feature(tmp_path):
    indexer = _make_indexer(tmp_path)

    assert indexer.discover_documents("Nonexistent") == []


def test_summarize_feature_counts_by_category(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt")
    _write(tmp_path, "Appointments", "workflows", "b.txt")
    _write(tmp_path, "Appointments", "TestCases", "c.csv")
    _write(tmp_path, "Appointments", "IssueSheets", "d.md")

    indexer = _make_indexer(tmp_path)
    summary = indexer.summarize_feature("Appointments")

    assert summary.feature == "Appointments"
    assert summary.documents == 4
    assert summary.workflows == 2
    assert summary.test_cases == 1
    assert summary.issues == 1


def test_summarize_all_features_returns_one_entry_per_feature(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt")
    _write(tmp_path, "Coding Tool", "TestCases", "b.csv")

    indexer = _make_indexer(tmp_path)
    summaries = indexer.summarize_all_features()

    assert {s.feature for s in summaries} == {"Appointments", "Coding Tool"}


def test_parse_feature_parses_every_discovered_document(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "onboarding.md", "# Onboarding\nSteps here.")
    _write(tmp_path, "Appointments", "TestCases", "cases.csv", "id,title\n1,Book\n")

    indexer = _make_indexer(tmp_path)
    results = indexer.parse_feature("Appointments")

    assert len(results) == 2
    by_name = {r.document_name: r for r in results}
    assert by_name["onboarding.md"].document_category == DocumentCategory.WORKFLOW
    assert by_name["onboarding.md"].parsed_document.title == "Onboarding"
    assert by_name["cases.csv"].document_category == DocumentCategory.TEST_CASE
    assert "Book" in by_name["cases.csv"].parsed_document.content


def test_parse_feature_output_uses_the_same_parser_service(tmp_path):
    """Confirms parsing goes through the shared ParserService/parsers
    unmodified — metadata carries the real parser_name/version.
    """
    _write(tmp_path, "Appointments", "workflows", "onboarding.md", "# Onboarding\nBody.")

    indexer = _make_indexer(tmp_path)
    [result] = indexer.parse_feature("Appointments")

    assert result.parsed_document.metadata.parser_name == "MarkdownParser"
    assert result.parsed_document.metadata.parser_version == "1.0"
    assert result.parsed_document.metadata.document_source == DocumentSource.SOURCE_OF_TRUTH


def test_parse_feature_raises_not_found_for_unknown_feature(tmp_path):
    indexer = _make_indexer(tmp_path)

    with pytest.raises(NotFoundError):
        indexer.parse_feature("Nonexistent")
