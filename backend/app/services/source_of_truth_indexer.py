"""Indexes the Source of Truth folder structure:

    source_of_truth/<feature>/workflows/...
    source_of_truth/<feature>/TestCases/...
    source_of_truth/<feature>/IssueSheets/...

The folder hierarchy — not the filename — determines `feature` and
`DocumentCategory` (WORKFLOW / TEST_CASE / ISSUE). This is a separate
axis from `Document.document_type` (the file *format* — PDF/DOCX/...),
which is still derived from the file extension exactly the way
`UploadService` does it, via the same `resolve_document_type` util.
Parsing itself stays completely unaware of any of this: a `Document`
built here looks just like one `UploadService` would build, plus the
one extra optional `category` field, and it's handed to the existing
`ParserService` exactly like any other document.
"""
from datetime import UTC, datetime
from pathlib import Path

from app.core.exceptions import NotFoundError, ValidationError
from app.models.common import DocumentCategory, DocumentSource
from app.models.document import Document
from app.models.feature_summary import FeatureSummary
from app.models.parsed_feature_document import ParsedFeatureDocument
from app.services.parser_service import ParserService
from app.utils.file_validation import resolve_document_type
from app.utils.ids import generate_id

# Exact, case-sensitive folder names as given by the Release Team
# Intelligenz Source of Truth layout.
_CATEGORY_FOLDERS: dict[str, DocumentCategory] = {
    "workflows": DocumentCategory.WORKFLOW,
    "TestCases": DocumentCategory.TEST_CASE,
    "IssueSheets": DocumentCategory.ISSUE,
}


class SourceOfTruthIndexer:
    def __init__(self, source_of_truth_root: Path, parser_service: ParserService) -> None:
        self._root = source_of_truth_root
        self._parser_service = parser_service

    def list_features(self) -> list[str]:
        """Every feature folder immediately under source_of_truth/."""
        if not self._root.is_dir():
            return []
        return sorted(path.name for path in self._root.iterdir() if path.is_dir())

    def discover_documents(self, feature: str) -> list[Document]:
        """Every recognized file under this feature's category folders,
        as `Document` records. Nothing is read or parsed here.
        """
        feature_dir = self._root / feature
        if not feature_dir.is_dir():
            return []

        documents: list[Document] = []
        for folder_name, category in _CATEGORY_FOLDERS.items():
            category_dir = feature_dir / folder_name
            if not category_dir.is_dir():
                continue

            for path in sorted(category_dir.iterdir()):
                document = _build_document(path, root=self._root, feature=feature, category=category)
                if document is not None:
                    documents.append(document)

        return documents

    def summarize_feature(self, feature: str) -> FeatureSummary:
        documents = self.discover_documents(feature)
        return FeatureSummary(
            feature=feature,
            documents=len(documents),
            workflows=sum(1 for d in documents if d.category == DocumentCategory.WORKFLOW),
            test_cases=sum(1 for d in documents if d.category == DocumentCategory.TEST_CASE),
            issues=sum(1 for d in documents if d.category == DocumentCategory.ISSUE),
        )

    def summarize_all_features(self) -> list[FeatureSummary]:
        return [self.summarize_feature(feature) for feature in self.list_features()]

    def parse_feature(self, feature: str) -> list[ParsedFeatureDocument]:
        """Discovers, reads, and parses every document for a feature —
        for the /parsed debugging endpoint. No chunking, no embeddings:
        just what the existing ParserService already returns.
        """
        feature_dir = self._root / feature
        if not feature_dir.is_dir():
            raise NotFoundError(f"No Source of Truth folder found for feature '{feature}'.")

        results = []
        for document in self.discover_documents(feature):
            content = Path(document.storage_path).read_bytes()
            parsed = self._parser_service.parse_content(document, content)
            results.append(
                ParsedFeatureDocument(
                    document_name=document.filename,
                    document_category=document.category,
                    parsed_document=parsed,
                )
            )
        return results


def _build_document(path: Path, root: Path, feature: str, category: DocumentCategory) -> Document | None:
    if not path.is_file() or path.name.startswith("."):
        return None

    try:
        document_type = resolve_document_type(path.name)
    except ValidationError:
        return None  # not a recognized/parseable format — skip it, don't fail the whole scan

    stat = path.stat()
    # Portable — always forward-slashed via `.as_posix()`, and rooted at
    # `root.name` ("source_of_truth") rather than `root`'s own absolute
    # path, so this never leaks a machine-specific location.
    source_relative_path = f"{root.name}/{path.relative_to(root).as_posix()}"
    return Document(
        id=generate_id("doc"),
        filename=path.name,
        document_type=document_type,
        feature=feature,
        size=stat.st_size,
        source=DocumentSource.SOURCE_OF_TRUTH,
        category=category,
        storage_path=str(path),
        uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        source_relative_path=source_relative_path,
        source_folder=feature,
    )
