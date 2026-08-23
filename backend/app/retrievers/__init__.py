"""The Retriever layer: every vector-search concern (collection lookup,
metadata filtering, similarity search, score mapping) lives here, behind
the `Retriever` interface, so `RetrievalService` never queries ChromaDB
directly and future retrieval enhancements (reranking, hybrid search,
score normalization, per-artifact-type top_k, semantic caching, ...)
can be added inside a retriever without touching orchestration.
"""
