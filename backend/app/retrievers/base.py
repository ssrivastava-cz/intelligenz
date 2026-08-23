"""The common interface every retrieval source implements, so
`RetrievalService` can orchestrate all of them identically regardless of
where their chunks actually live.
"""
from abc import ABC, abstractmethod

from app.models.retrieved_chunk import RetrievedChunk


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Returns up to `top_k` chunks most similar to `query_embedding`,
        narrowed by `filters` (metadata equality constraints, e.g.
        `{"artifactType": "WORKFLOW"}`) when given.
        """
        raise NotImplementedError
