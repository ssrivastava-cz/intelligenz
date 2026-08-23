"""Simulates indexing a feature's documents into a vector store. Mock only."""
import random

from app.models.indexing import IndexRecord
from app.utils.datetime_utils import utcnow
from app.utils.ids import generate_id


class IndexingService:
    def __init__(self) -> None:
        self._records: dict[str, IndexRecord] = {}

    def index_feature(self, feature: str, document_ids: list[str]) -> IndexRecord:
        record = IndexRecord(
            id=generate_id("idx"),
            feature=feature,
            document_ids=document_ids,
            chunks_indexed=max(len(document_ids), 1) * random.randint(8, 24),
            status="completed",
            indexed_at=utcnow(),
        )
        self._records[record.id] = record
        return record
