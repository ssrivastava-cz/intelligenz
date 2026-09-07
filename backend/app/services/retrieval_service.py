"""Orchestrates the configurable, multi-stage Retrieval Pipeline. For
each knowledge source (Workflow, Historical Test Cases, Historical
Issues, Uploaded Documents), independently:

    Retrieve Candidate Chunks -> Final Chunk Selection
    -> Merge Adjacent Chunks -> Return Results

Every vector-search concern — collection lookup, metadata filtering,
similarity search, score mapping — lives inside the retrievers
(`app.retrievers`); this class never queries ChromaDB itself, only
`EmbeddingService` (to embed the query text) and whichever retrievers
the request needs. Ranking (`FinalChunkSelector`) and merging
(`AdjacentChunkMerger`) are injected collaborators rather than inline
logic, so a future Cross Encoder Reranker or Hybrid Search retriever
can be swapped in later without any change to this class, `PromptBuilder`,
or `GenerationService`. `RedmineRetriever` isn't wired in yet since
retrieval from Redmine isn't implemented — its interface already exists
so it can be added here later without changing how orchestration works.

No retriever ever calls another retriever, and nothing here compares
results across knowledge sources — each source's pipeline runs and
returns independently, exactly as `PromptBuilderInput` (four separate
chunk lists) already expects.

`retrieve_hybrid()` is a second, additive pipeline used only by the
Knowledge Assistant (`retrieve()` above is untouched — the Test Plan
Generator's `GenerationService` keeps calling it exactly as before, so
this class stays the single orchestrator for both without either one
affecting the other):

    Vector Search + BM25 (per source) -> Reciprocal Rank Fusion
    -> Hybrid Candidate Pool (all sources combined) -> Reranking
    -> Relevance Threshold -> Deduplication -> Top N
    -> Merge Adjacent Chunks -> Return Results (same RetrievalResult shape)

Selection happens once, globally, across every knowledge source's
candidates together (not a separate top-N per source) — a chunk's
category no longer buys it a reserved slot, which is exactly what fixes
a generic, high-embedding-similarity section from one category crowding
out a more relevant chunk from another. Retrieval (vector + BM25) is
still run per source, respecting each one's own `artifactType` filter,
so "preserve source-specific retrieval" still holds for candidate
generation — only the final ranking/selection step is global.

`retrieve_hybrid()` also searches across every `feature`, not just one
— the Knowledge Assistant's `SourceOfTruthRetriever`/`BM25Retriever`
instances are constructed with no bound feature at all, unlike
`retrieve()`'s (which always binds one). `feature` is still stored,
untouched, on every chunk's metadata and still comes back on every
`RetrievedChunk` as provenance; it simply no longer restricts which
chunks are eligible to match for the Knowledge Assistant.

The BM25 leg for Source of Truth is served from the *persistent* global
lexical index (`app.services.bm25_index_manager.BM25IndexManager`),
built once during indexing and loaded at app startup — `BM25Retriever`
here only queries it, it is never rebuilt per request. Uploaded
documents (per-session ChromaDB collections, not part of that global
index) keep the old rebuild-per-query lexical search via
`ephemeral_bm25_search`.
"""
import time

from chromadb.api import ClientAPI

from app.models.common import DocumentCategory
from app.models.hybrid_retrieval import (
    FinalContextChunk,
    HybridRetrievalResult,
    RetrievalDiagnostics,
    VectorQueryDiagnostic,
)
from app.models.retrieval_configuration import RetrievalConfiguration, SourceRetrievalConfig
from app.models.retrieval_result import RetrievalResult, SourceRetrievalResult
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.base import Retriever
from app.retrievers.bm25_retriever import BM25Match, BM25Retriever
from app.retrievers.chunk_deduplicator import ChunkDeduplicator
from app.retrievers.chunk_merger import AdjacentChunkMerger
from app.retrievers.ephemeral_bm25 import ephemeral_bm25_search
from app.retrievers.final_chunk_selector import FinalChunkSelector
from app.retrievers.rank_fusion import FusedCandidate, ReciprocalRankFusion
from app.retrievers.reranker import LexicalReranker, Reranker
from app.retrievers.reranking_selector import RerankingSelector
from app.retrievers.source_of_truth_retriever import SourceOfTruthRetriever
from app.retrievers.upload_retriever import UploadRetriever
from app.services.bm25_index_manager import BM25IndexManager
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        chroma_client: ClientAPI,
        bm25_index_manager: BM25IndexManager,
        upload_collection_prefix: str,
        default_top_k: int,
        default_configuration: RetrievalConfiguration | None = None,
        final_chunk_selector: FinalChunkSelector | None = None,
        chunk_merger: AdjacentChunkMerger | None = None,
        reranker: Reranker | None = None,
        rank_fusion: ReciprocalRankFusion | None = None,
        deduplicator: ChunkDeduplicator | None = None,
        hybrid_candidate_chunks: int = 20,
        reranked_final_chunks: int = 5,
        reranker_min_score: float = 0.0,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._chroma_client = chroma_client
        self._bm25_index_manager = bm25_index_manager
        self._upload_collection_prefix = upload_collection_prefix
        self._default_top_k = default_top_k
        self._default_configuration = default_configuration or RetrievalConfiguration.uniform(default_top_k)
        self._final_chunk_selector = final_chunk_selector or FinalChunkSelector()
        self._chunk_merger = chunk_merger or AdjacentChunkMerger()
        # `retrieve_hybrid()`-only collaborators — `retrieve()` above
        # never touches any of these, so leaving them at their defaults
        # has zero effect on the Test Plan Generator's retrieval.
        self._rank_fusion = rank_fusion or ReciprocalRankFusion()
        self._reranking_selector = RerankingSelector(
            reranker=reranker or LexicalReranker(), deduplicator=deduplicator, min_score=reranker_min_score
        )
        self._hybrid_candidate_chunks = hybrid_candidate_chunks
        self._reranked_final_chunks = reranked_final_chunks

    def retrieve(
        self,
        query_text: str,
        feature: str,
        upload_session_id: str | None = None,
        top_k: int | None = None,
        configuration: RetrievalConfiguration | None = None,
    ) -> RetrievalResult:
        """`top_k` (a single flat count applied to every source) is kept
        for backward compatibility with existing callers
        (`GenerationService`, both debug routers) that only know about a
        single knob — it's resolved into a uniform `RetrievalConfiguration`
        internally. Passing `configuration` directly is how a caller
        exercises independent per-source candidate/final counts.
        """
        resolved_configuration = self._resolve_configuration(top_k, configuration)

        embedding_batch = self._embedding_service.embed_texts([query_text])
        query_embedding = embedding_batch.embeddings[0]
        query_tokens = self._embedding_service.count_tokens([query_text])[0]

        source_of_truth_retriever = SourceOfTruthRetriever(self._vector_store, feature)

        workflow = self._retrieve_from_source(
            source_of_truth_retriever,
            query_embedding,
            resolved_configuration.workflow,
            filters={"artifactType": DocumentCategory.WORKFLOW.value},
        )
        historical_test_cases = self._retrieve_from_source(
            source_of_truth_retriever,
            query_embedding,
            resolved_configuration.historical_test_cases,
            filters={"artifactType": DocumentCategory.TEST_CASE.value},
        )
        historical_issues = self._retrieve_from_source(
            source_of_truth_retriever,
            query_embedding,
            resolved_configuration.historical_issues,
            filters={"artifactType": DocumentCategory.ISSUE.value},
        )

        if upload_session_id is not None:
            upload_retriever = UploadRetriever(
                self._chroma_client, self._upload_collection_prefix, upload_session_id
            )
            uploaded_documents = self._retrieve_from_source(
                upload_retriever, query_embedding, resolved_configuration.uploaded_documents
            )
        else:
            uploaded_documents = _empty_source_result(resolved_configuration.uploaded_documents)

        return RetrievalResult(
            query_text=query_text,
            query_embedding_dimension=len(query_embedding),
            query_embedding_tokens=query_tokens,
            workflow=workflow,
            historical_test_cases=historical_test_cases,
            historical_issues=historical_issues,
            uploaded_documents=uploaded_documents,
        )

    def retrieve_hybrid(
        self,
        query_text: str,
        upload_session_id: str | None = None,
        hybrid_candidate_chunks: int | None = None,
        final_chunks: int | None = None,
    ) -> HybridRetrievalResult:
        """The Knowledge Assistant's retrieval pipeline — see this
        module's docstring for the full stage sequence. Unlike
        `retrieve()`, this searches the *entire* Source of Truth corpus:
        `SourceOfTruthRetriever`/`BM25Retriever` are constructed with no
        `feature` at all (`feature=None`), so neither leg of candidate
        generation restricts by feature — a chunk from any feature folder
        is eligible, and two chunks from different features can both
        reach the same answer. `feature` remains on every resulting
        `RetrievedChunk` as provenance (it's stored metadata, untouched
        by this) — it just no longer narrows what's searched. Candidate
        generation still respects each knowledge source's own
        `artifactType` filter, exactly like `retrieve()`; only
        reranking/threshold/deduplication/top-N selection is global,
        across every source's candidates together. `hybrid_candidate_chunks`/
        `final_chunks` override this instance's configured defaults
        (`Settings.hybrid_candidate_chunks`/`reranked_final_chunks`) for
        this call only, mirroring `retrieve()`'s `top_k` override.
        """
        candidate_pool_size = hybrid_candidate_chunks or self._hybrid_candidate_chunks
        selection_size = final_chunks if final_chunks is not None else self._reranked_final_chunks

        embedding_batch = self._embedding_service.embed_texts([query_text])
        query_embedding = embedding_batch.embeddings[0]
        query_tokens = self._embedding_service.count_tokens([query_text])[0]

        source_of_truth_retriever = SourceOfTruthRetriever(self._vector_store)
        # Query-only: the persistent BM25 index is built during indexing
        # (`IndexService.index_feature`) and loaded at startup — never
        # rebuilt here, and never read from the whole ChromaDB corpus.
        bm25_retriever = BM25Retriever(self._bm25_index_manager)

        vector_ms = 0.0
        bm25_ms = 0.0
        all_vector_chunks: list[RetrievedChunk] = []
        all_bm25_matches: list[BM25Match] = []
        vector_query_diagnostics: list[VectorQueryDiagnostic] = []

        for artifact_type in (DocumentCategory.WORKFLOW, DocumentCategory.TEST_CASE, DocumentCategory.ISSUE):
            filters = {"artifactType": artifact_type.value}

            available = source_of_truth_retriever.count_available(filters)

            started = time.perf_counter()
            source_vector_chunks = source_of_truth_retriever.retrieve(
                query_embedding, candidate_pool_size, filters=filters
            )
            vector_ms += (time.perf_counter() - started) * 1000
            all_vector_chunks.extend(source_vector_chunks)

            vector_query_diagnostics.append(
                VectorQueryDiagnostic(
                    artifact_type=artifact_type,
                    requested_n_results=candidate_pool_size,
                    available_chunks=available,
                    returned_chunks=len(source_vector_chunks),
                )
            )

            started = time.perf_counter()
            all_bm25_matches.extend(bm25_retriever.retrieve(query_text, candidate_pool_size, filters=filters))
            bm25_ms += (time.perf_counter() - started) * 1000

        if upload_session_id is not None:
            upload_retriever = UploadRetriever(self._chroma_client, self._upload_collection_prefix, upload_session_id)

            available = upload_retriever.count_available()

            started = time.perf_counter()
            upload_vector_chunks = upload_retriever.retrieve(query_embedding, candidate_pool_size)
            vector_ms += (time.perf_counter() - started) * 1000
            all_vector_chunks.extend(upload_vector_chunks)

            vector_query_diagnostics.append(
                VectorQueryDiagnostic(
                    artifact_type=DocumentCategory.USER_UPLOAD,
                    requested_n_results=candidate_pool_size,
                    available_chunks=available,
                    returned_chunks=len(upload_vector_chunks),
                )
            )

            started = time.perf_counter()
            all_bm25_matches.extend(self._bm25_retrieve_uploads(query_text, upload_session_id, candidate_pool_size))
            bm25_ms += (time.perf_counter() - started) * 1000

        # Each retriever's own quota is per source (respecting source-
        # specific retrieval), but ranking for fusion must be global —
        # re-sorting the combined lists here is what makes RRF's rank
        # position meaningful across sources rather than just within
        # whichever source happened to be appended first.
        all_vector_chunks.sort(key=lambda chunk: chunk.similarity_score, reverse=True)
        all_bm25_matches.sort(key=lambda match: match.score, reverse=True)

        started = time.perf_counter()
        fused = self._rank_fusion.fuse(all_vector_chunks, all_bm25_matches)
        fusion_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        selection = self._reranking_selector.select(query_text, fused, selection_size)
        reranking_ms = (time.perf_counter() - started) * 1000

        merged_chunks = self._chunk_merger.merge(selection.selected_chunks)

        retrieval_result = RetrievalResult(
            query_text=query_text,
            query_embedding_dimension=len(query_embedding),
            query_embedding_tokens=query_tokens,
            workflow=_hybrid_source_result(
                fused,
                selection.selected_chunks,
                merged_chunks,
                DocumentCategory.WORKFLOW,
                candidate_pool_size,
                selection_size,
            ),
            historical_test_cases=_hybrid_source_result(
                fused,
                selection.selected_chunks,
                merged_chunks,
                DocumentCategory.TEST_CASE,
                candidate_pool_size,
                selection_size,
            ),
            historical_issues=_hybrid_source_result(
                fused,
                selection.selected_chunks,
                merged_chunks,
                DocumentCategory.ISSUE,
                candidate_pool_size,
                selection_size,
            ),
            uploaded_documents=_hybrid_source_result(
                fused,
                selection.selected_chunks,
                merged_chunks,
                DocumentCategory.USER_UPLOAD,
                candidate_pool_size,
                selection_size,
            ),
        )

        diagnostics = RetrievalDiagnostics(
            vector_candidate_count=len(all_vector_chunks),
            bm25_candidate_count=len(all_bm25_matches),
            hybrid_candidate_count=len(fused),
            reranked_candidate_count=selection.reranked_candidate_count,
            final_chunk_count=len(selection.selected_chunks),
            duplicates_removed=selection.duplicates_removed,
            vector_queries=vector_query_diagnostics,
            candidates=selection.all_candidates,
            final_context=[
                FinalContextChunk(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.source_filename,
                    section_heading=chunk.section_heading,
                    text=chunk.text,
                )
                for chunk in merged_chunks
            ],
            # Filled in by `KnowledgeAssistantService` once the prompt is
            # actually built — retrieval itself has no prompt to count
            # tokens for yet.
            estimated_prompt_tokens=0,
            vector_retrieval_ms=vector_ms,
            bm25_retrieval_ms=bm25_ms,
            hybrid_fusion_ms=fusion_ms,
            reranking_ms=reranking_ms,
            total_retrieval_ms=vector_ms + bm25_ms + fusion_ms + reranking_ms,
        )

        return HybridRetrievalResult(retrieval_result=retrieval_result, diagnostics=diagnostics)

    def _bm25_retrieve_uploads(self, query_text: str, upload_session_id: str, top_k: int) -> list[BM25Match]:
        """Uploaded documents live in short-lived, per-session ChromaDB
        collections that are not part of the persistent global BM25 index,
        so their lexical leg still builds a small BM25 index per query
        from the session's chunks (`ephemeral_bm25_search`) — unchanged
        behaviour, just no longer routed through `BM25Retriever`."""
        collection_name = f"{self._upload_collection_prefix}_{upload_session_id}"
        if not VectorStoreService.collection_exists(self._chroma_client, collection_name):
            return []
        upload_vector_store = VectorStoreService(client=self._chroma_client, collection_name=collection_name)
        return ephemeral_bm25_search(upload_vector_store, query_text, top_k, feature=upload_session_id)

    def _resolve_configuration(
        self, top_k: int | None, configuration: RetrievalConfiguration | None
    ) -> RetrievalConfiguration:
        if configuration is not None:
            return configuration
        if top_k is not None:
            return RetrievalConfiguration.uniform(top_k)
        return self._default_configuration

    def _retrieve_from_source(
        self,
        retriever: Retriever,
        query_embedding: list[float],
        config: SourceRetrievalConfig,
        filters: dict[str, str] | None = None,
    ) -> SourceRetrievalResult:
        candidate_chunks = retriever.retrieve(query_embedding, config.candidate_chunks, filters=filters)
        final_chunks = self._final_chunk_selector.select(candidate_chunks, config.final_chunks)
        merged_chunks = self._chunk_merger.merge(final_chunks)
        return SourceRetrievalResult(
            candidate_requested=config.candidate_chunks,
            candidate_chunks=candidate_chunks,
            final_requested=config.final_chunks,
            final_chunks=final_chunks,
            merged_chunks=merged_chunks,
        )


def _empty_source_result(config: SourceRetrievalConfig) -> SourceRetrievalResult:
    return SourceRetrievalResult(
        candidate_requested=config.candidate_chunks,
        candidate_chunks=[],
        final_requested=config.final_chunks,
        final_chunks=[],
        merged_chunks=[],
    )


def _hybrid_source_result(
    fused: list[FusedCandidate],
    selected_chunks: list[RetrievedChunk],
    merged_chunks: list[RetrievedChunk],
    artifact_type: DocumentCategory,
    candidate_pool_size: int,
    selection_size: int,
) -> SourceRetrievalResult:
    """Partitions the hybrid pipeline's *global* candidate pool and
    selection back into this one source's slice, by `artifact_type` —
    every chunk already carries its own category, so this is a pure
    filter, never a second retrieval. `candidate_requested`/
    `final_requested` report the configured hybrid pool size / global
    selection cap (there is no longer a per-source quota in hybrid
    mode — selection is global, see `retrieve_hybrid`).
    """
    return SourceRetrievalResult(
        candidate_requested=candidate_pool_size,
        candidate_chunks=[candidate.chunk for candidate in fused if candidate.chunk.artifact_type == artifact_type],
        final_requested=selection_size,
        final_chunks=[chunk for chunk in selected_chunks if chunk.artifact_type == artifact_type],
        merged_chunks=[chunk for chunk in merged_chunks if chunk.artifact_type == artifact_type],
    )
