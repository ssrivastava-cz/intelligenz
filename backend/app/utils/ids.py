import uuid

from app.utils.datetime_utils import utcnow


def generate_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def generate_knowledge_assistant_generation_id() -> str:
    """`gen_<yyyymmdd>_<random>` — the Knowledge Assistant's
    `generation_id` format. Previously generated on the frontend (see
    the old `knowledgeAssistantMock.js`); now the backend is the sole
    authority, generated once at the very start of
    `KnowledgeAssistantService.ask` so it stays available for
    `error.json` even if every later stage fails.
    """
    timestamp = utcnow().strftime("%Y%m%d")
    random_suffix = uuid.uuid4().hex[:5]
    return f"gen_{timestamp}_{random_suffix}"
