"""Mock test-plan generation pipeline.

No AI, ChromaDB, or OpenAI integration. Progress is simulated
deterministically from wall-clock time elapsed since a generation was
created, so repeated polling of /status behaves realistically without
needing a background task runner.
"""
from app.core.exceptions import NotFoundError
from app.models.common import GenerationStatus
from app.models.generation import Generation
from app.services.upload_service import UploadService
from app.utils.datetime_utils import utcnow
from app.utils.ids import generate_id
from app.utils.mock_data import PROGRESS_STAGES, SEED_GENERATIONS, build_mock_test_cases

STAGE_DURATION_SECONDS = 1.5


class TestPlanService:
    def __init__(self, upload_service: UploadService) -> None:
        self._upload_service = upload_service
        self._generations: dict[str, Generation] = {
            seed["id"]: Generation(**seed) for seed in SEED_GENERATIONS
        }

    def create_generation(
        self,
        feature: str,
        redmine_id: str,
        description: str | None,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
        analysis_options: list[str] | None = None,
    ) -> Generation:
        resolved_document_ids = list(document_ids or [])
        if session_id:
            session_document_ids = [d.id for d in self._upload_service.list_session_documents(session_id)]
            # dict.fromkeys dedupes while preserving order (a plain set would not).
            resolved_document_ids = list(dict.fromkeys([*resolved_document_ids, *session_document_ids]))

        now = utcnow()
        generation = Generation(
            id=generate_id("gen"),
            feature=feature,
            redmine_id=redmine_id,
            description=description,
            session_id=session_id,
            document_ids=resolved_document_ids,
            analysis_options=list(analysis_options or []),
            status=GenerationStatus.PROCESSING,
            created_at=now,
            updated_at=now,
        )
        self._generations[generation.id] = generation

        if session_id:
            self._upload_service.mark_session_consumed(session_id, generation.id)

        return generation

    def get_generation(self, generation_id: str) -> Generation:
        generation = self._generations.get(generation_id)
        if generation is None:
            raise NotFoundError(f"No generation found with id '{generation_id}'.")
        return self._resolve_progress(generation)

    def list_generations(self) -> list[Generation]:
        return [self._resolve_progress(g) for g in self._generations.values()]

    def _resolve_progress(self, generation: Generation) -> Generation:
        """Advances a still-processing generation based on elapsed time."""
        if generation.status != GenerationStatus.PROCESSING:
            return generation

        total_stages = len(PROGRESS_STAGES)
        elapsed = (utcnow() - generation.created_at).total_seconds()

        if elapsed >= total_stages * STAGE_DURATION_SECONDS:
            generation.status = GenerationStatus.COMPLETED
            generation.stage_index = total_stages - 1
            generation.test_cases = build_mock_test_cases(generation.feature)
            generation.updated_at = utcnow()
        else:
            generation.stage_index = min(int(elapsed // STAGE_DURATION_SECONDS), total_stages - 1)

        return generation
