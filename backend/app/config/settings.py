"""Application configuration loaded from environment variables."""
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Release Team Intelligenz"
    app_env: str = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"

    cors_origins: list[str] = ["http://localhost:5173"]

    # Document ingestion
    storage_root: str = "."
    max_upload_size_mb: int = 20

    # Chunking engine (word counts, not characters or tokens)
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Knowledge Base Indexing: OpenAI embeddings
    # `OPEN_API_KEY` (missing "AI") is the name already set in backend/.env —
    # accepted alongside the correctly-spelled name so either works.
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY", "OPEN_API_KEY"))
    embedding_model: str = "text-embedding-3-small"
    embedding_price_per_1k_tokens: float = 0.00002

    # Knowledge Base Indexing: ChromaDB
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "source_of_truth_chunks"
    # A large feature can produce more chunks than ChromaDB accepts in one
    # `collection.add()` call (its own hard limit, reported by
    # `client.get_max_batch_size()` — 5461 for the local Chroma version
    # this app uses). `VectorStoreService` writes in batches of this size,
    # clamped to whatever the client actually reports as its maximum, so
    # this value is always safe to raise without checking Chroma's limit
    # by hand.
    chroma_write_batch_size: int = 1000

    # Uploaded Document Embedding: same ChromaDB instance, but a separate
    # collection per upload session (named "<prefix>_<upload_session_id>"),
    # kept fully independent of the Source of Truth collection above.
    upload_chroma_collection_prefix: str = "uploaded_documents"

    # History: every real AI operation (indexing, generation) gets its
    # own timestamped folder here — the only history format the app
    # maintains, read by HistoryService.
    history_root: str = "data/history"

    # Knowledge Assistant: filesystem-backed GenerationRepository root —
    # one directory per generation under <database_root>/generations/,
    # read by FileSystemGenerationRepository. A future MongoDB-backed
    # repository would not use this setting at all.
    database_root: str = "database"

    # Redmine integration: every retrieved ticket is cached under
    # <redmine_data_root>/<ticket_id>/ — raw response, normalized model,
    # attachment download manifest, and the attachments themselves.
    redmine_url: str = ""
    redmine_api_key: str = ""
    redmine_timeout_seconds: float = 30.0
    redmine_data_root: str = "data/redmine"

    # Retrieval Pipeline: legacy single-knob fallback, used only when a
    # caller passes a flat `top_k` instead of a full RetrievalConfiguration
    # (applied uniformly as both candidate and final count, every source).
    retrieval_default_top_k: int = 5

    # Retrieval Pipeline: default RetrievalConfiguration — how many
    # candidate chunks each knowledge source's Retriever fetches, and how
    # many of those survive Final Chunk Selection, when a caller doesn't
    # supply its own configuration.
    retrieval_workflow_candidate_chunks: int = 20
    retrieval_workflow_final_chunks: int = 5
    retrieval_historical_test_cases_candidate_chunks: int = 20
    retrieval_historical_test_cases_final_chunks: int = 8
    retrieval_historical_issues_candidate_chunks: int = 15
    retrieval_historical_issues_final_chunks: int = 5
    retrieval_uploaded_documents_candidate_chunks: int = 10
    retrieval_uploaded_documents_final_chunks: int = 3

    # Hybrid Retrieval (Knowledge Assistant only — Test Plan Generator's
    # RetrievalService.retrieve() above is untouched by these). Vector
    # search and BM25 each contribute up to this many candidates per
    # knowledge source; the two rankings are fused via Reciprocal Rank
    # Fusion into one pool, then reranked/thresholded/deduplicated down
    # to `reranked_final_chunks` — see `RetrievalService.retrieve_hybrid`.
    hybrid_candidate_chunks: int = 20
    # Reciprocal Rank Fusion's `k` — the standard default from the
    # original RRF paper (Cormack et al., 2009), which found results
    # insensitive to k in a wide range around it; not re-tuned here.
    rrf_k: int = 60
    # Upper bound on how many chunks survive reranking to reach the
    # prompt — a ceiling, not a quota: fewer are returned whenever fewer
    # candidates are genuinely relevant (see `reranker_min_score`).
    reranked_final_chunks: int = 5
    # Minimum reranker score (0-1) a candidate must clear to be eligible
    # at all. Deliberately conservative: the reranker's score is
    # dominated by lexical/term-overlap signal (see
    # `app.retrievers.reranker.LexicalReranker`), and even a genuinely
    # relevant chunk may share only a couple of exact terms with the
    # question. Tune this up if noisy chunks are still reaching the
    # prompt in practice, or down if relevant chunks are being dropped.
    reranker_min_score: float = 0.08

    # AI Generation: OpenAI Chat model + pricing, read by CostCalculator
    # so pricing is never hardcoded anywhere in the app. `CHAT_MODEL` is
    # accepted as a fallback alias — it was already set in backend/.env
    # before this model/pricing config existed, and nothing else uses it.
    llm_model: str = Field(default="gpt-5", validation_alias=AliasChoices("LLM_MODEL", "CHAT_MODEL"))
    llm_input_price_per_million_tokens: float = 1.25
    llm_output_price_per_million_tokens: float = 10.00
    # Optional, for future use (prompt caching isn't implemented yet).
    llm_cached_input_price_per_million_tokens: float | None = None

    # AI Generation: USD is the source of truth for every cost
    # calculation; INR is a convenience conversion CostCalculator derives
    # from it using this rate — never hardcoded.
    usd_to_inr_exchange_rate: float = 87.00

    # AI Generation: the maximum number of test cases the model is
    # allowed/requested to generate per call — read by PromptBuilder (to
    # instruct the model) and GenerationService (to enforce the limit on
    # whatever the model actually returns). Never hardcoded elsewhere.
    max_generated_test_cases: int = 5

    # Logging
    logs_root: str = "logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
