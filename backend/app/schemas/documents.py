from app.models.common import DocumentSource, DocumentType
from app.models.document import Document
from app.schemas.common import CamelModel


class DocumentOut(CamelModel):
    document_id: str
    document_name: str
    document_type: DocumentType
    feature: str
    file_size: int
    document_source: DocumentSource

    @classmethod
    def from_document(cls, document: Document) -> "DocumentOut":
        return cls(
            document_id=document.id,
            document_name=document.filename,
            document_type=document.document_type,
            feature=document.feature,
            file_size=document.size,
            document_source=document.source,
        )


class UploadDocumentsResponse(CamelModel):
    documents: list[DocumentOut]
    message: str
