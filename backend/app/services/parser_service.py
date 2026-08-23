"""Dispatches to the correct format-specific parser based on document type.

Works uniformly for Source of Truth and user-uploaded documents — both
are just `Document` records distinguished by `.source`; parsing only
cares about `.document_type` and the file bytes at `.storage_path`.
"""
from pathlib import Path

from app.core.exceptions import ValidationError
from app.models.common import DocumentType
from app.models.document import Document
from app.models.parsed_document import ParsedDocument
from app.services.parsers.base import DocumentParser
from app.services.parsers.csv_parser import CsvParser
from app.services.parsers.excel_parser import ExcelParser
from app.services.parsers.markdown_parser import MarkdownParser
from app.services.parsers.pdf_parser import PdfParser
from app.services.parsers.txt_parser import TxtParser
from app.services.parsers.word_parser import WordParser
from app.services.upload_service import UploadService


class ParserService:
    def __init__(self, upload_service: UploadService) -> None:
        self._upload_service = upload_service
        self._parsers: dict[DocumentType, DocumentParser] = {
            DocumentType.PDF: PdfParser(),
            DocumentType.DOCX: WordParser(),
            DocumentType.XLSX: ExcelParser(),
            DocumentType.CSV: CsvParser(),
            DocumentType.MARKDOWN: MarkdownParser(),
            DocumentType.TXT: TxtParser(),
        }

    def parse_document(self, document_id: str) -> ParsedDocument:
        """Parses a previously-ingested document — Source of Truth or
        user upload, it makes no difference here — by id."""
        document = self._upload_service.get_document(document_id)
        content = Path(document.storage_path).read_bytes()
        return self.parse_content(document, content)

    def parse_content(self, document: Document, content: bytes) -> ParsedDocument:
        """Parses raw bytes directly against a `Document` record, without
        going through `UploadService.get_document`. For callers — like
        `SourceOfTruthIndexer` — that discover documents by scanning the
        filesystem themselves rather than via the upload endpoints.
        """
        parser = self._parser_for(document.document_type)
        return parser.parse(document, content)

    def _parser_for(self, document_type: DocumentType) -> DocumentParser:
        parser = self._parsers.get(document_type)
        if parser is None:
            raise ValidationError(f"No parser registered for document type '{document_type.value}'.")
        return parser
