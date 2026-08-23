"""The only component allowed to write into `backend/data/history/`.

Every real AI operation gets its own timestamped folder there, so the
future Usage Dashboard can compute every statistic by reading these
folders. No other history format is maintained alongside it.

This phase implements indexing history (`history/index/`), uploaded-
document embedding history (`history/upload/`), and generation history
(`history/generation/`). `history/retrieval/` will be populated once the
Retrieval Pipeline needs its own persisted history — the private
`_next_available_timestamped_name` helper is deliberately generic (it
takes whatever root and filename format it should pick a unique,
timestamp-based name under) so a future `save_retrieval_history` can
reuse it without duplicating this naming/collision logic, exactly like
`save_generation_history` does.
"""
import json
import shutil
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import ExternalServiceError, NotFoundError
from app.models.embedding_preview import EmbeddingPreviewChunk
from app.models.generation_history import GenerationHistoryDraft, GenerationHistoryEntry
from app.models.generation_statistics import GenerationStatistics
from app.models.generation_summary import GenerationSummary
from app.models.index_history import IndexHistoryEntry, IndexHistoryRecord
from app.models.indexing_statistics import IndexingStatistics
from app.models.indexing_summary import IndexingSummary
from app.models.upload_history import UploadHistoryEntry, UploadHistoryRecord
from app.services.cost_calculator import CostCalculator
from app.utils.datetime_utils import utcnow

_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
_GENERATION_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"
_INDEX_SUMMARY_FILENAME = "index_summary.json"
_UPLOAD_SUMMARY_FILENAME = "uploaded_embedding_summary.json"
_EMBEDDING_PREVIEW_FILENAME = "embedding_preview.json"


class HistoryService:
    def __init__(self, history_root: Path) -> None:
        self._root = history_root
        self._index_root = history_root / "index"
        self._upload_root = history_root / "upload"
        self._generation_root = history_root / "generation"

    def save_index_history(
        self,
        entry: IndexHistoryEntry,
        embedding_preview: list[EmbeddingPreviewChunk],
    ) -> Path:
        """Creates a new timestamped folder under `history/index/` holding
        `index_summary.json` and `embedding_preview.json` (chunk text and
        token counts only — no embedding vectors, and nothing duplicated
        from ChromaDB). Callers must only invoke this after embeddings
        were generated *and* stored successfully — this method has no way
        to know that on its own, so it always writes `index_status =
        SUCCESS`.
        """
        folder = self._create_timestamped_folder(self._index_root)
        _write_json(folder / _INDEX_SUMMARY_FILENAME, entry.model_dump(mode="json"))
        _write_json(
            folder / _EMBEDDING_PREVIEW_FILENAME,
            [chunk.model_dump(mode="json") for chunk in embedding_preview],
        )
        return folder

    def list_index_history(self) -> list[IndexHistoryEntry]:
        """Every persisted, successful indexing run's summary, oldest
        first (folder names sort chronologically since they're
        timestamps). A folder with a missing or corrupted
        `index_summary.json` is skipped rather than failing the whole
        listing, exactly like `list_generations()` — one bad historical
        file must never break this or the Usage Dashboard.
        """
        return [entry for _, entry in self._list_index_history_with_id()]

    def list_indexing_summaries(self, cost_calculator: CostCalculator) -> list[IndexingSummary]:
        """Every persisted, successful indexing run as its Usage
        Dashboard projection, each carrying its own `history_id` (the
        run's folder name — there is no separate indexing id) and its
        embedding cost converted to `{usd, inr}` via `cost_calculator`,
        which only converts an already-computed USD amount here; it
        never prices anything itself. This cost is never added to, or
        derived from, any generation's cost — see `IndexingSummary`.
        """
        return [
            IndexingSummary.from_entry(
                history_id, entry, cost_calculator.convert_to_money(entry.estimated_embedding_cost)
            )
            for history_id, entry in self._list_index_history_with_id()
        ]

    def indexing_statistics(self, cost_calculator: CostCalculator) -> IndexingStatistics:
        """Usage Dashboard aggregates computed directly from
        `list_indexing_summaries()` — no separate statistics store, no
        regenerated data, and no mixing with `generation_statistics()`.
        """
        summaries = self.list_indexing_summaries(cost_calculator)
        total = len(summaries)
        if total == 0:
            return IndexingStatistics(
                total_indexing_runs=0,
                total_documents_indexed=0,
                total_chunks_indexed=0,
                total_embedding_tokens=0,
                total_embedding_cost_usd=0.0,
                total_embedding_cost_inr=0.0,
            )

        return IndexingStatistics(
            total_indexing_runs=total,
            total_documents_indexed=sum(summary.documents_indexed for summary in summaries),
            total_chunks_indexed=sum(summary.chunks_indexed for summary in summaries),
            total_embedding_tokens=sum(summary.embedding_tokens for summary in summaries),
            total_embedding_cost_usd=sum(summary.embedding_cost_usd for summary in summaries),
            total_embedding_cost_inr=sum(summary.embedding_cost_inr for summary in summaries),
        )

    def _list_index_history_with_id(self) -> list[tuple[str, IndexHistoryEntry]]:
        """Every persisted, successful indexing run paired with its
        `history_id` (its folder name), oldest first. Shared by
        `list_index_history()` and `list_indexing_summaries()` so the
        "skip a missing/corrupted file, only keep SUCCESS runs" rule
        exists in exactly one place. `index_status` is always
        `"SUCCESS"` in practice (`IndexService` never persists a failed
        run), but this filters explicitly anyway rather than trusting
        that invariant forever.
        """
        entries = []
        for folder in self._sorted_run_folders(self._index_root):
            summary_path = folder / _INDEX_SUMMARY_FILENAME
            if not summary_path.is_file():
                continue
            try:
                entry = IndexHistoryEntry.model_validate(_read_json(summary_path))
            except (json.JSONDecodeError, PydanticValidationError):
                continue
            if entry.index_status != "SUCCESS":
                continue
            entries.append((folder.name, entry))
        return entries

    def load_index_history(self, history_id: str) -> IndexHistoryRecord:
        """The full detail — summary plus embedding preview — for one
        indexing run, identified by its folder name (e.g.
        `2026-08-02_18-42-15`, as returned by `save_index_history`).
        """
        folder = self._index_root / history_id
        summary_path = folder / _INDEX_SUMMARY_FILENAME
        if not folder.is_dir() or not summary_path.is_file():
            raise NotFoundError(f"No index history found with id '{history_id}'.")

        preview_path = folder / _EMBEDDING_PREVIEW_FILENAME
        preview_raw = _read_json(preview_path) if preview_path.is_file() else []

        return IndexHistoryRecord(
            history_id=history_id,
            summary=IndexHistoryEntry.model_validate(_read_json(summary_path)),
            embedding_preview=[EmbeddingPreviewChunk.model_validate(chunk) for chunk in preview_raw],
        )

    def delete_index_history(self, history_id: str) -> None:
        """Removes one indexing run's history folder entirely."""
        folder = self._index_root / history_id
        if not folder.is_dir():
            raise NotFoundError(f"No index history found with id '{history_id}'.")
        shutil.rmtree(folder)

    def save_upload_history(
        self,
        entry: UploadHistoryEntry,
        embedding_preview: list[EmbeddingPreviewChunk],
    ) -> Path:
        """Creates a new timestamped folder under `history/upload/` holding
        `uploaded_embedding_summary.json` and `embedding_preview.json` —
        the exact same file-writing mechanism `save_index_history` uses,
        just pointed at a different root, so no logic is duplicated here.
        """
        folder = self._create_timestamped_folder(self._upload_root)
        _write_json(folder / _UPLOAD_SUMMARY_FILENAME, entry.model_dump(mode="json"))
        _write_json(
            folder / _EMBEDDING_PREVIEW_FILENAME,
            [chunk.model_dump(mode="json") for chunk in embedding_preview],
        )
        return folder

    def list_upload_history(self) -> list[UploadHistoryRecord]:
        """Every persisted upload embedding run's full record (summary
        plus embedding preview), oldest first. Unlike `list_index_history`
        (summaries only — its one caller never needed the preview), this
        returns full records because the upload debug endpoint needs the
        embedding preview too, and there's no other way to look one up
        without already knowing its `history_id`.
        """
        records = []
        for folder in self._sorted_run_folders(self._upload_root):
            summary_path = folder / _UPLOAD_SUMMARY_FILENAME
            if not summary_path.is_file():
                continue
            preview_path = folder / _EMBEDDING_PREVIEW_FILENAME
            preview_raw = _read_json(preview_path) if preview_path.is_file() else []
            records.append(
                UploadHistoryRecord(
                    history_id=folder.name,
                    summary=UploadHistoryEntry.model_validate(_read_json(summary_path)),
                    embedding_preview=[EmbeddingPreviewChunk.model_validate(chunk) for chunk in preview_raw],
                )
            )
        return records

    def save_generation_history(self, draft: GenerationHistoryDraft) -> GenerationHistoryEntry:
        """Persists one completed AI generation as a single JSON file
        under `history/generation/<timestamp>.json` — unlike indexing and
        upload history, there's no separate embedding preview to persist
        alongside it, so this writes exactly one file per generation.
        `generation_id` and `timestamp` are assigned here (from the same
        moment), not by the caller, so `generation_id` always matches the
        file it can be looked up by by `load_generation`.
        """
        self._generation_root.mkdir(parents=True, exist_ok=True)
        timestamp = utcnow()
        generation_id = _next_available_timestamped_name(
            timestamp,
            _GENERATION_TIMESTAMP_FORMAT,
            is_taken=lambda candidate: (self._generation_root / f"{candidate}.json").exists(),
        )
        entry = GenerationHistoryEntry(generation_id=generation_id, timestamp=timestamp, **draft.model_dump())
        _write_json(self._generation_root / f"{generation_id}.json", entry.model_dump(mode="json"))
        return entry

    def list_generations(self) -> list[GenerationHistoryEntry]:
        """Every persisted generation, oldest first (filenames sort
        chronologically since they're timestamps). A file that fails to
        parse or validate (e.g. truncated by a crash mid-write) is
        skipped rather than failing the whole listing — `load_generation`
        is where a caller asking for that *specific* generation finds
        out it's corrupted.
        """
        if not self._generation_root.is_dir():
            return []
        entries = []
        for path in sorted(self._generation_root.glob("*.json")):
            try:
                entries.append(GenerationHistoryEntry.model_validate(_read_json(path)))
            except (json.JSONDecodeError, PydanticValidationError):
                continue
        return entries

    def load_generation(self, generation_id: str) -> GenerationHistoryEntry:
        """One generation's full persisted detail, identified by its
        filename (e.g. `2026-08-03T18-24-10`, as returned by
        `save_generation_history`). Raises `ExternalServiceError` if the
        file exists but is corrupted (invalid JSON or doesn't match the
        schema) — a data-integrity problem, not a missing resource.
        """
        path = self._generation_root / f"{generation_id}.json"
        if not path.is_file():
            raise NotFoundError(f"No generation history found with id '{generation_id}'.")
        try:
            return GenerationHistoryEntry.model_validate(_read_json(path))
        except (json.JSONDecodeError, PydanticValidationError) as exc:
            raise ExternalServiceError(
                f"Generation history file for '{generation_id}' is corrupted: {exc}"
            ) from exc

    def delete_generation(self, generation_id: str) -> None:
        """Removes only the single persisted JSON file for one
        generation. Never touches uploads, embeddings, ChromaDB
        collections, Redmine data, or Source of Truth data — none of
        which this service has any access to in the first place.
        """
        path = self._generation_root / f"{generation_id}.json"
        if not path.is_file():
            raise NotFoundError(f"No generation history found with id '{generation_id}'.")
        path.unlink()

    def list_generation_summaries(self, cost_calculator: CostCalculator) -> list[GenerationSummary]:
        """Every persisted generation as its trimmed summary projection,
        each enriched with its Embedding stage cost — looked up by
        `metadata.upload_session_id` against the latest matching Upload
        Embedding run, never recomputed. `cost_calculator` only converts
        that already-computed embedding USD cost into the same
        `{usd, inr}` shape every other cost in the system uses; it never
        prices anything itself here.
        """
        embedding_by_session = self._latest_upload_history_by_session()
        summaries = []
        for entry in self.list_generations():
            embedding_entry = (
                embedding_by_session.get(entry.metadata.upload_session_id)
                if entry.metadata.upload_session_id
                else None
            )
            summaries.append(
                GenerationSummary.from_entry(
                    entry,
                    embedding_tokens=embedding_entry.embedding_tokens if embedding_entry else None,
                    embedding_cost=(
                        cost_calculator.convert_to_money(embedding_entry.estimated_embedding_cost)
                        if embedding_entry
                        else None
                    ),
                )
            )
        return summaries

    def _latest_upload_history_by_session(self) -> dict[str, UploadHistoryEntry]:
        """The most recently uploaded-and-embedded run for each upload
        session, mirroring `UploadEmbeddingService.get_latest_history`'s
        "latest for a session" rule — duplicated rather than imported
        since that method raises `NotFoundError` for an unknown session,
        which doesn't fit this method's "just return what's known" use.
        """
        latest: dict[str, UploadHistoryEntry] = {}
        for record in self.list_upload_history():
            summary = record.summary
            existing = latest.get(summary.upload_session_id)
            if existing is None or summary.uploaded_at > existing.uploaded_at:
                latest[summary.upload_session_id] = summary
        return latest

    def generation_statistics(self, cost_calculator: CostCalculator) -> GenerationStatistics:
        """Dashboard aggregates computed directly from
        `list_generation_summaries()` — no separate statistics store, and
        no regenerated data. Corrupted files are already excluded by
        `list_generations()`, so they don't skew these numbers.
        """
        summaries = self.list_generation_summaries(cost_calculator)
        total = len(summaries)
        if total == 0:
            return GenerationStatistics(
                total_generations=0,
                successful_generations=0,
                failed_generations=0,
                average_generation_time_ms=0.0,
                average_test_cases=0.0,
                average_prompt_tokens=0.0,
                average_completion_tokens=0.0,
                average_cost_usd=0.0,
                average_cost_inr=0.0,
                most_used_feature=None,
                most_used_model=None,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                total_spend_usd=0.0,
                total_spend_inr=0.0,
            )

        successful = sum(1 for summary in summaries if summary.status == "SUCCESS")
        total_spend_usd = sum(summary.total_ai_cost_usd for summary in summaries)
        total_spend_inr = sum(summary.total_ai_cost_inr for summary in summaries)
        total_prompt_tokens = sum(summary.prompt_tokens for summary in summaries)
        total_completion_tokens = sum(summary.completion_tokens for summary in summaries)
        feature_counts = Counter(summary.feature for summary in summaries)
        model_counts = Counter(summary.model for summary in summaries)

        return GenerationStatistics(
            total_generations=total,
            successful_generations=successful,
            failed_generations=total - successful,
            average_generation_time_ms=sum(s.generation_time_ms for s in summaries) / total,
            average_test_cases=sum(s.number_of_test_cases for s in summaries) / total,
            average_prompt_tokens=sum(s.prompt_tokens for s in summaries) / total,
            average_completion_tokens=sum(s.completion_tokens for s in summaries) / total,
            average_cost_usd=total_spend_usd / total,
            average_cost_inr=total_spend_inr / total,
            most_used_feature=feature_counts.most_common(1)[0][0],
            most_used_model=model_counts.most_common(1)[0][0],
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            total_spend_usd=total_spend_usd,
            total_spend_inr=total_spend_inr,
        )

    def _sorted_run_folders(self, root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        return sorted(path for path in root.iterdir() if path.is_dir())

    def _create_timestamped_folder(self, root: Path) -> Path:
        """Creates `root/<timestamp>/`, disambiguating with a numeric
        suffix on the rare collision (e.g. two runs completing within the
        same second). Generic over `root` so any future history subtype
        can reuse this without duplicating the naming/collision logic.
        """
        root.mkdir(parents=True, exist_ok=True)

        name = _next_available_timestamped_name(
            utcnow(), _TIMESTAMP_FORMAT, is_taken=lambda candidate: (root / candidate).exists()
        )
        folder = root / name
        folder.mkdir(parents=True)
        return folder


def _next_available_timestamped_name(
    timestamp: datetime, timestamp_format: str, is_taken: Callable[[str], bool]
) -> str:
    """Picks a timestamp-based name, disambiguating with a numeric suffix
    on the rare collision (e.g. two runs completing within the same
    second). `is_taken` decides whether a candidate name is already in
    use — a folder's own existence for `_create_timestamped_folder`, or
    `<name>.json`'s existence for `save_generation_history` — so this
    collision-avoidance logic exists in exactly one place regardless of
    whether the caller is naming a folder or a file.
    """
    base_name = timestamp.strftime(timestamp_format)
    candidate = base_name
    suffix = 2
    while is_taken(candidate):
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
