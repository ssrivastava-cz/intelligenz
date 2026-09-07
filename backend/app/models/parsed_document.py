"""The common output shape every document parser returns.

Named `ParsedDocument` (not `Document`) to avoid colliding with
`app.models.document.Document` — the ingestion record every parser
takes as *input* alongside the raw file bytes.
"""
from pydantic import BaseModel, Field

from app.models.common import DocumentSource, DocumentType


class DocumentSection(BaseModel):
    """A structural chunk of a parsed document — a heading and its
    body, a PDF page, a spreadsheet sheet. The unit a future
    chunker/embedder would operate on, though this subsystem doesn't
    do that itself.
    """

    heading: str | None = None
    content: str
    page_number: int | None = None


class ParsedDocumentMetadata(BaseModel):
    feature: str
    document_type: DocumentType
    document_source: DocumentSource
    source_filename: str
    page_number: int | None = None
    """Total page count when the format has one (PDF); otherwise None."""
    parser_name: str
    """Which parser produced this — e.g. "PdfParser". Every parser populates this."""
    parser_version: str
    """The parser implementation's version, e.g. "1.0"."""
    source_path: str | None = None
    """Mirrors `Document.source_relative_path` — portable, forward-slashed,
    e.g. "source_of_truth/Service Model 1/workflows/example.pdf". `None`
    for a document with no Source of Truth path (a user upload)."""
    source_folder: str | None = None
    """Mirrors `Document.source_folder` — the folder directly under
    source_of_truth/. `None` for a user upload."""


class ParsedDocument(BaseModel):
    document_id: str
    title: str
    sections: list[DocumentSection] = Field(default_factory=list)
    content: str
    metadata: ParsedDocumentMetadata
