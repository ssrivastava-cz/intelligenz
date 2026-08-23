from app.models.common import DocumentCategory, DocumentSource, DocumentType
from app.models.parsed_feature_document import ParsedFeatureDocument
from app.schemas.common import CamelModel


class FeatureSummaryOut(CamelModel):
    feature: str
    documents: int
    workflows: int
    test_cases: int
    issues: int


class DocumentSectionOut(CamelModel):
    heading: str | None
    content: str
    page_number: int | None


class ParsedDocumentMetadataOut(CamelModel):
    feature: str
    document_type: DocumentType
    document_source: DocumentSource
    source_filename: str
    page_number: int | None
    parser_name: str
    parser_version: str


class ParsedDocumentOut(CamelModel):
    document_id: str
    title: str
    sections: list[DocumentSectionOut]
    content: str
    metadata: ParsedDocumentMetadataOut


class ParsedFeatureDocumentOut(CamelModel):
    document_name: str
    document_type: DocumentCategory
    parsed_document: ParsedDocumentOut
    section_count: int
    section_headings: list[str | None]

    @classmethod
    def from_parsed(cls, item: ParsedFeatureDocument) -> "ParsedFeatureDocumentOut":
        sections = item.parsed_document.sections
        return cls(
            document_name=item.document_name,
            document_type=item.document_category,
            parsed_document=ParsedDocumentOut.model_validate(item.parsed_document),
            section_count=len(sections),
            section_headings=[section.heading for section in sections],
        )


class SourceOfTruthParsedResponse(CamelModel):
    feature: str
    documents: list[ParsedFeatureDocumentOut]


class ChunkOut(CamelModel):
    """One chunk produced by `ChunkingEngine` — for the developer
    debugging endpoint `GET /source-of-truth/{feature}/chunks`, used to
    verify chunking output before Knowledge Base Indexing exists.
    """

    chunk_id: str
    artifact_type: DocumentCategory
    feature: str
    document_source: DocumentSource
    source_filename: str
    section_heading: str | None
    page_number: int | None
    chunk_number: int
    word_count: int
    chunk_text: str


class EmbeddingPreviewChunkOut(CamelModel):
    """One chunk's projected embedding cost — for the developer debugging
    endpoint `GET /source-of-truth/{feature}/embedding-preview`, used to
    check token usage and cost before actually indexing a feature.
    """

    chunk_id: str
    section_heading: str | None
    artifact_type: DocumentCategory
    source_filename: str
    page_number: int | None
    word_count: int
    embedding_tokens: int
    estimated_cost: float
    chunk_text: str


class EmbeddingPreviewResponse(CamelModel):
    feature: str
    embedding_model: str
    documents_found: int
    chunks_created: int
    total_embedding_tokens: int
    average_tokens_per_chunk: float
    estimated_embedding_cost: float
    chunks: list[EmbeddingPreviewChunkOut]
