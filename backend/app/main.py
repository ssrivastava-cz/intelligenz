"""FastAPI application entrypoint.

Document ingestion, Knowledge Base Indexing (real OpenAI + ChromaDB),
and Redmine ticket retrieval are real. Test Plan Generation (retrieval,
prompt building, chat completion) and MongoDB are still mock/not yet
implemented.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.core.dependencies import get_bm25_index_manager
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.routers.api import api_router

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Loads the persistent BM25 index into memory once, at startup — so
    a FastAPI restart never pays the index-build cost and the first user
    query is served from the ready index. If no index has been persisted
    yet, that's logged clearly (run indexing) and lexical retrieval
    returns nothing until then; the index is never rebuilt from Source of
    Truth here.
    """
    manager = get_bm25_index_manager()
    try:
        if manager.is_available():
            manager.load()
            if manager.is_stale():
                logger.warning(
                    "Persistent BM25 index at %s was built with a different tokenizer/chunking/BM25 "
                    "version than this process expects — re-run indexing for each feature to refresh it.",
                    manager.index_dir,
                )
        else:
            logger.warning(
                "No persistent BM25 index at %s. Knowledge Assistant lexical (BM25) retrieval will "
                "return nothing until indexing is run for each feature (POST /index-feature/{feature}). "
                "Vector retrieval is unaffected.",
                manager.index_dir,
            )
    except Exception:
        logger.exception("Failed to load the persistent BM25 index; lexical retrieval degraded until re-indexed.")
    yield


def create_app() -> FastAPI:
    # `DEBUG`-level logging (e.g. the Knowledge Assistant's raw OpenAI
    # usage object — see `KnowledgeAssistantService.ask`) only ever
    # prints when `Settings.debug` is on; production stays at `INFO`.
    configure_logging(level="DEBUG" if settings.debug else "INFO", logs_root=Path(settings.logs_root))

    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    return app


app = create_app()
