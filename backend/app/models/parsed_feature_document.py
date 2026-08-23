from pydantic import BaseModel

from app.models.common import DocumentCategory
from app.models.parsed_document import ParsedDocument


class ParsedFeatureDocument(BaseModel):
    """One Source of Truth document, parsed — for the debug/inspection
    endpoint `GET /source-of-truth/{feature}/parsed`.
    """

    document_name: str
    document_category: DocumentCategory
    parsed_document: ParsedDocument
