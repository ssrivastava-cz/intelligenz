"""Builds ChromaDB `where` clauses from a flat metadata-filter dict —
shared by every retriever that queries `VectorStoreService`, so the
`$and` combination Chroma requires for multi-key filters exists in
exactly one place.
"""
from typing import Any


def build_where_clause(filters: dict[str, str] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    if len(filters) == 1:
        ((key, value),) = filters.items()
        return {key: value}
    return {"$and": [{key: value} for key, value in filters.items()]}
