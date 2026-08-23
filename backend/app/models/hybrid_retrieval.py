"""Diagnostics for the Knowledge Assistant's Hybrid Retrieval pipeline
(Vector Search + BM25 -> Reciprocal Rank Fusion -> Reranking ->
Relevance Threshold -> Deduplication -> Top N). Deliberately separate
from `app.models.retrieval_result` (the existing, per-source
candidate/final/merged shape `RetrievalService.retrieve()` returns,
still used unchanged by the Test Plan Generator) — this instead captures
*why* each candidate was or wasn't selected, across every knowledge
source together, for `GET /knowledge-assistant/debug/{generation_id}`
and for persistence (`GenerationRepository.save_retrieval_diagnostics`).

Nothing here holds an embedding vector or raw ChromaDB distance, mirroring
the same "never expose embeddings" rule the rest of the Knowledge
Assistant debug surface already follows.
"""
from pydantic import BaseModel

from app.models.common import DocumentCategory
from app.models.retrieval_result import RetrievalResult


class HybridCandidate(BaseModel):
    """One chunk's full journey through the hybrid pipeline — from
    however it was found (vector search, BM25, or both) through to
    whether it ultimately reached the prompt. `rrf_score` and
    `reranker_score` are always set once the candidate exists (every
    fused candidate gets reranked); `final_rank`/`selected` are only
    meaningful after the threshold + deduplication + top-N stage runs.
    """

    chunk_id: str
    document_id: str | None
    document_name: str
    artifact_type: DocumentCategory
    section_heading: str | None

    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0

    reranker_score: float | None = None
    final_rank: int | None = None
    selected: bool = False


class VectorQueryDiagnostic(BaseModel):
    """One `VectorStoreService.query_similar_chunks` call's outcome, per
    knowledge source — `available_chunks` and `returned_chunks` vary
    independently per `artifact_type` (e.g. a feature can have 1,069
    WORKFLOW chunks but 6,072 TEST_CASE chunks), which is exactly the
    detail needed to explain why a query for one source behaved
    differently than another for the same question.
    """

    artifact_type: DocumentCategory
    requested_n_results: int
    available_chunks: int
    returned_chunks: int


class FinalContextChunk(BaseModel):
    """One chunk that actually reached the prompt — the only place in
    the hybrid diagnostics that carries chunk text, since candidate-level
    diagnostics (`HybridCandidate`) are metadata/scores only.
    """

    chunk_id: str
    document_name: str
    section_heading: str | None
    text: str


class RetrievalDiagnostics(BaseModel):
    """Everything `GET /knowledge-assistant/debug/{generation_id}` needs
    to answer "what candidates existed, why was each one selected or
    rejected, and what actually reached the prompt" — read back verbatim
    from what `KnowledgeAssistantService.ask` persisted; retrieval is
    never re-run to produce this.
    """

    vector_candidate_count: int
    bm25_candidate_count: int
    hybrid_candidate_count: int
    reranked_candidate_count: int
    final_chunk_count: int
    duplicates_removed: int

    vector_queries: list[VectorQueryDiagnostic]
    candidates: list[HybridCandidate]
    final_context: list[FinalContextChunk]
    estimated_prompt_tokens: int

    vector_retrieval_ms: float
    bm25_retrieval_ms: float
    hybrid_fusion_ms: float
    reranking_ms: float
    total_retrieval_ms: float


class HybridRetrievalResult(BaseModel):
    """`RetrievalService.retrieve_hybrid`'s full result: the same
    per-source `RetrievalResult` shape `retrieve()` already produces
    (so everything downstream of retrieval — `KnowledgeAssistantPromptBuilder`,
    `AdjacentChunkMerger` output — needs no change at all), plus the
    diagnostics describing how that result was reached.
    """

    retrieval_result: RetrievalResult
    diagnostics: RetrievalDiagnostics
