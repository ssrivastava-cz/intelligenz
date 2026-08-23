"""Interface-only stub. Retrieval from Redmine tickets isn't implemented
yet — this class exists purely so code can be written against the
`Retriever` abstraction for Redmine today, and start actually retrieving
later without any orchestration-level refactor.
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.base import Retriever


class RedmineRetriever(Retriever):
    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError("RedmineRetriever retrieval is not implemented yet.")
