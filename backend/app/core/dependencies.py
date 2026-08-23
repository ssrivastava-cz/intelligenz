"""Dependency-injection providers for FastAPI's `Depends`.

Each provider is memoized with `lru_cache` so every request shares the
same service instance (needed here since services hold their mock data
in memory). Services that need other services declare that service as
a `Depends`-typed parameter (not a direct call to another provider
function) so the whole graph — including test overrides via
`app.dependency_overrides` — resolves consistently at every level.
"""
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import chromadb
import httpx
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from fastapi import Depends
from openai import OpenAI

from app.config.settings import get_settings
from app.models.retrieval_configuration import RetrievalConfiguration, SourceRetrievalConfig
from app.repositories.filesystem_generation_repository import FileSystemGenerationRepository
from app.repositories.generation_repository import GenerationRepository
from app.retrievers.rank_fusion import ReciprocalRankFusion
from app.services.chunking_engine import ChunkingEngine
from app.services.cost_calculator import CostCalculator
from app.services.embedding_service import EmbeddingService
from app.services.excel_export_service import ExcelExportService
from app.services.feedback_service import FeedbackService
from app.services.generation_service import GenerationService
from app.services.history_service import HistoryService
from app.services.index_service import IndexService
from app.services.indexing_service import IndexingService
from app.services.knowledge_assistant_prompt_builder import KnowledgeAssistantPromptBuilder
from app.services.knowledge_assistant_service import KnowledgeAssistantService
from app.services.parser_service import ParserService
from app.services.prompt_builder import PromptBuilder
from app.services.redmine_service import RedmineService
from app.services.retrieval_service import RetrievalService
from app.services.source_of_truth_indexer import SourceOfTruthIndexer
from app.services.test_plan_service import TestPlanService
from app.services.upload_embedding_service import UploadEmbeddingService
from app.services.upload_service import UploadService
from app.services.uploaded_document_indexer import UploadedDocumentIndexer
from app.services.usage_service import UsageService
from app.services.vector_store_service import VectorStoreService


@lru_cache
def get_upload_service() -> UploadService:
    settings = get_settings()
    return UploadService(
        storage_root=Path(settings.storage_root),
        max_upload_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
    )


UploadServiceDep = Annotated[UploadService, Depends(get_upload_service)]


@lru_cache
def get_parser_service(upload_service: UploadServiceDep) -> ParserService:
    return ParserService(upload_service=upload_service)


ParserServiceDep = Annotated[ParserService, Depends(get_parser_service)]


@lru_cache
def get_source_of_truth_indexer(parser_service: ParserServiceDep) -> SourceOfTruthIndexer:
    settings = get_settings()
    # Mirrors UploadService's own storage_root/"source_of_truth" computation
    # (duplicated, not imported, since UploadService doesn't expose it and
    # we're not modifying UploadService) — both read/write the same tree.
    return SourceOfTruthIndexer(
        source_of_truth_root=Path(settings.storage_root) / "source_of_truth",
        parser_service=parser_service,
    )


SourceOfTruthIndexerDep = Annotated[SourceOfTruthIndexer, Depends(get_source_of_truth_indexer)]


@lru_cache
def get_chunking_engine() -> ChunkingEngine:
    settings = get_settings()
    return ChunkingEngine(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


ChunkingEngineDep = Annotated[ChunkingEngine, Depends(get_chunking_engine)]


@lru_cache
def get_indexing_service() -> IndexingService:
    return IndexingService()


IndexingServiceDep = Annotated[IndexingService, Depends(get_indexing_service)]


@lru_cache
def get_test_plan_service(upload_service: UploadServiceDep) -> TestPlanService:
    return TestPlanService(upload_service=upload_service)


TestPlanServiceDep = Annotated[TestPlanService, Depends(get_test_plan_service)]


@lru_cache
def get_feedback_service() -> FeedbackService:
    return FeedbackService()


FeedbackServiceDep = Annotated[FeedbackService, Depends(get_feedback_service)]


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


OpenAIClientDep = Annotated[OpenAI, Depends(get_openai_client)]


@lru_cache
def get_embedding_service(client: OpenAIClientDep) -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(client=client, model=settings.embedding_model)


EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]


@lru_cache
def get_chroma_client() -> ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


ChromaClientDep = Annotated[ClientAPI, Depends(get_chroma_client)]


@lru_cache
def get_vector_store_service(client: ChromaClientDep) -> VectorStoreService:
    settings = get_settings()
    return VectorStoreService(
        client=client,
        collection_name=settings.chroma_collection_name,
        write_batch_size=settings.chroma_write_batch_size,
    )


VectorStoreServiceDep = Annotated[VectorStoreService, Depends(get_vector_store_service)]


@lru_cache
def get_history_service() -> HistoryService:
    settings = get_settings()
    return HistoryService(history_root=Path(settings.history_root))


HistoryServiceDep = Annotated[HistoryService, Depends(get_history_service)]


@lru_cache
def get_index_service(
    indexer: SourceOfTruthIndexerDep,
    chunking_engine: ChunkingEngineDep,
    embedding_service: EmbeddingServiceDep,
    vector_store_service: VectorStoreServiceDep,
    history_service: HistoryServiceDep,
) -> IndexService:
    settings = get_settings()
    return IndexService(
        indexer=indexer,
        chunking_engine=chunking_engine,
        embedding_service=embedding_service,
        vector_store=vector_store_service,
        history_service=history_service,
        embedding_model=settings.embedding_model,
        price_per_1k_tokens=settings.embedding_price_per_1k_tokens,
    )


IndexServiceDep = Annotated[IndexService, Depends(get_index_service)]


@lru_cache
def get_redmine_http_client() -> httpx.Client:
    settings = get_settings()
    return httpx.Client(
        base_url=settings.redmine_url,
        headers={"X-Redmine-API-Key": settings.redmine_api_key},
        timeout=settings.redmine_timeout_seconds,
    )


RedmineHttpClientDep = Annotated[httpx.Client, Depends(get_redmine_http_client)]


@lru_cache
def get_redmine_service(client: RedmineHttpClientDep) -> RedmineService:
    settings = get_settings()
    return RedmineService(client=client, data_root=Path(settings.redmine_data_root))


RedmineServiceDep = Annotated[RedmineService, Depends(get_redmine_service)]


@lru_cache
def get_uploaded_document_indexer(
    upload_service: UploadServiceDep, parser_service: ParserServiceDep
) -> UploadedDocumentIndexer:
    return UploadedDocumentIndexer(upload_service=upload_service, parser_service=parser_service)


UploadedDocumentIndexerDep = Annotated[UploadedDocumentIndexer, Depends(get_uploaded_document_indexer)]


@lru_cache
def get_upload_embedding_service(
    indexer: UploadedDocumentIndexerDep,
    chunking_engine: ChunkingEngineDep,
    embedding_service: EmbeddingServiceDep,
    chroma_client: ChromaClientDep,
    history_service: HistoryServiceDep,
) -> UploadEmbeddingService:
    settings = get_settings()
    return UploadEmbeddingService(
        indexer=indexer,
        chunking_engine=chunking_engine,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        history_service=history_service,
        embedding_model=settings.embedding_model,
        price_per_1k_tokens=settings.embedding_price_per_1k_tokens,
        collection_prefix=settings.upload_chroma_collection_prefix,
    )


UploadEmbeddingServiceDep = Annotated[UploadEmbeddingService, Depends(get_upload_embedding_service)]


@lru_cache
def get_retrieval_service(
    embedding_service: EmbeddingServiceDep,
    vector_store_service: VectorStoreServiceDep,
    chroma_client: ChromaClientDep,
) -> RetrievalService:
    settings = get_settings()
    default_configuration = RetrievalConfiguration(
        workflow=SourceRetrievalConfig(
            candidate_chunks=settings.retrieval_workflow_candidate_chunks,
            final_chunks=settings.retrieval_workflow_final_chunks,
        ),
        historical_test_cases=SourceRetrievalConfig(
            candidate_chunks=settings.retrieval_historical_test_cases_candidate_chunks,
            final_chunks=settings.retrieval_historical_test_cases_final_chunks,
        ),
        historical_issues=SourceRetrievalConfig(
            candidate_chunks=settings.retrieval_historical_issues_candidate_chunks,
            final_chunks=settings.retrieval_historical_issues_final_chunks,
        ),
        uploaded_documents=SourceRetrievalConfig(
            candidate_chunks=settings.retrieval_uploaded_documents_candidate_chunks,
            final_chunks=settings.retrieval_uploaded_documents_final_chunks,
        ),
    )
    return RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store_service,
        chroma_client=chroma_client,
        upload_collection_prefix=settings.upload_chroma_collection_prefix,
        default_top_k=settings.retrieval_default_top_k,
        default_configuration=default_configuration,
        # `retrieve_hybrid()`-only settings — the Test Plan Generator's
        # `retrieve()` call path never reads any of these, so this
        # wiring has no effect on it.
        hybrid_candidate_chunks=settings.hybrid_candidate_chunks,
        reranked_final_chunks=settings.reranked_final_chunks,
        reranker_min_score=settings.reranker_min_score,
        rank_fusion=ReciprocalRankFusion(k=settings.rrf_k),
    )


RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    settings = get_settings()
    return PromptBuilder(max_test_cases=settings.max_generated_test_cases)


PromptBuilderDep = Annotated[PromptBuilder, Depends(get_prompt_builder)]


@lru_cache
def get_cost_calculator() -> CostCalculator:
    settings = get_settings()
    return CostCalculator(
        model=settings.llm_model,
        input_price_per_million_tokens=settings.llm_input_price_per_million_tokens,
        output_price_per_million_tokens=settings.llm_output_price_per_million_tokens,
        usd_to_inr_exchange_rate=settings.usd_to_inr_exchange_rate,
        cached_input_price_per_million_tokens=settings.llm_cached_input_price_per_million_tokens,
    )


CostCalculatorDep = Annotated[CostCalculator, Depends(get_cost_calculator)]


@lru_cache
def get_generation_repository() -> GenerationRepository:
    settings = get_settings()
    return FileSystemGenerationRepository(database_root=Path(settings.database_root))


GenerationRepositoryDep = Annotated[GenerationRepository, Depends(get_generation_repository)]


@lru_cache
def get_usage_service(
    history_service: HistoryServiceDep,
    generation_repository: GenerationRepositoryDep,
    cost_calculator: CostCalculatorDep,
) -> UsageService:
    return UsageService(
        history_service=history_service,
        generation_repository=generation_repository,
        cost_calculator=cost_calculator,
    )


UsageServiceDep = Annotated[UsageService, Depends(get_usage_service)]


@lru_cache
def get_knowledge_assistant_prompt_builder() -> KnowledgeAssistantPromptBuilder:
    return KnowledgeAssistantPromptBuilder()


KnowledgeAssistantPromptBuilderDep = Annotated[
    KnowledgeAssistantPromptBuilder, Depends(get_knowledge_assistant_prompt_builder)
]


@lru_cache
def get_knowledge_assistant_service(
    retrieval_service: RetrievalServiceDep,
    prompt_builder: KnowledgeAssistantPromptBuilderDep,
    embedding_service: EmbeddingServiceDep,
    openai_client: OpenAIClientDep,
    cost_calculator: CostCalculatorDep,
    generation_repository: GenerationRepositoryDep,
) -> KnowledgeAssistantService:
    return KnowledgeAssistantService(
        retrieval_service=retrieval_service,
        prompt_builder=prompt_builder,
        embedding_service=embedding_service,
        openai_client=openai_client,
        cost_calculator=cost_calculator,
        generation_repository=generation_repository,
    )


KnowledgeAssistantServiceDep = Annotated[KnowledgeAssistantService, Depends(get_knowledge_assistant_service)]


@lru_cache
def get_generation_service(
    retrieval_service: RetrievalServiceDep,
    prompt_builder: PromptBuilderDep,
    embedding_service: EmbeddingServiceDep,
    openai_client: OpenAIClientDep,
    cost_calculator: CostCalculatorDep,
    history_service: HistoryServiceDep,
) -> GenerationService:
    settings = get_settings()
    return GenerationService(
        retrieval_service=retrieval_service,
        prompt_builder=prompt_builder,
        embedding_service=embedding_service,
        openai_client=openai_client,
        cost_calculator=cost_calculator,
        history_service=history_service,
        chat_model=settings.llm_model,
        max_generated_test_cases=settings.max_generated_test_cases,
    )


GenerationServiceDep = Annotated[GenerationService, Depends(get_generation_service)]


@lru_cache
def get_excel_export_service() -> ExcelExportService:
    return ExcelExportService()


ExcelExportServiceDep = Annotated[ExcelExportService, Depends(get_excel_export_service)]
