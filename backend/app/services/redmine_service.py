"""Talks to the Redmine REST API to retrieve a ticket, downloads its
attachments, and persists everything under
`backend/data/redmine/<ticket_id>/` so the future Retrieval Pipeline can
reuse it without another Redmine API call. Deliberately independent of
OpenAI, prompt building, embeddings, ChromaDB, and the retrieval
pipeline — this service only fetches raw Redmine data and stages it on
disk for later parsing.

`get_ticket`'s public contract — inputs, return type, and which
exceptions it raises for which failures — is unchanged from before this
refactor. Everything new here is an additional side effect: persisting
`ticket.json`, `normalized_ticket.json`, and `download_manifest.json`
alongside the attachments this service already downloaded.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.exceptions import ExternalServiceError, NotFoundError, UnauthorizedError
from app.core.logging import get_logger
from app.models.redmine_manifest import AttachmentManifestEntry, DownloadManifest
from app.models.redmine_ticket import RedmineAttachment, RedmineTicket
from app.utils.datetime_utils import utcnow

logger = get_logger(__name__)

# Redmine has no built-in "feature" field, so ticket.feature is read from
# a custom field with this name (matching the same feature taxonomy used
# by Source of Truth folders), falling back to the project name.
_FEATURE_CUSTOM_FIELD_NAME = "feature"

_TICKET_FILENAME = "ticket.json"
_NORMALIZED_TICKET_FILENAME = "normalized_ticket.json"
_MANIFEST_FILENAME = "download_manifest.json"


@dataclass
class RedmineDebugInfo:
    """Everything the `/redmine/{ticket_id}/debug` endpoint needs: the
    normalized ticket, its full download manifest (including failures),
    and the on-disk paths involved — returned by
    `RedmineService.get_ticket_debug_info`.
    """

    ticket: RedmineTicket
    manifest: DownloadManifest
    ticket_path: Path
    normalized_ticket_path: Path
    manifest_path: Path
    attachments_dir: Path


@dataclass
class _AttachmentOutcome:
    """One attachment's fate — downloaded, failed, or skipped outright
    (e.g. Redmine gave no filename/content_url to try). Feeds both
    `RedmineTicket.attachments` (successes only, same as before this
    refactor) and `download_manifest.json` (every outcome, including
    failures — a view `RedmineTicket` itself has never exposed).
    """

    attachment_id: int | None
    filename: str
    status: str
    content_type: str | None = None
    file_size: int | None = None
    local_path: str | None = None
    failure_reason: str | None = None


class RedmineService:
    def __init__(self, client: httpx.Client, data_root: Path) -> None:
        self._client = client
        self._data_root = data_root

    def get_ticket(self, ticket_id: str) -> RedmineTicket:
        """Raises `NotFoundError` if the ticket doesn't exist,
        `UnauthorizedError` if the API key is rejected, or
        `ExternalServiceError` for any other failure to reach Redmine or
        parse its response. Attachment download and local persistence
        failures never raise — they're logged and skipped, and the
        ticket is still returned.
        """
        issue = self._fetch_issue(ticket_id)
        logger.info("Ticket retrieved: %s", ticket_id)

        ticket_dir = self._ticket_dir(ticket_id)
        self._save_raw_ticket(ticket_dir, ticket_id, issue)

        outcomes = self._download_attachments(ticket_id, issue.get("attachments", []))
        ticket = _to_ticket(ticket_id, issue, outcomes)

        self._save_normalized_ticket(ticket_dir, ticket_id, ticket)
        self._save_manifest(ticket_dir, ticket_id, outcomes)

        return ticket

    def get_ticket_debug_info(self, ticket_id: str) -> RedmineDebugInfo:
        """Read-only inspection of what's been retrieved from Redmine and
        persisted locally for a ticket. Reuses `normalized_ticket.json`
        and `download_manifest.json` from a previous `get_ticket` call
        if both are already on disk — no Redmine API call, no attachment
        downloads. Otherwise, calls `get_ticket` first (the same real
        retrieval-and-persist path, not a separate implementation) to
        populate them, then reads the two files back. Never touches
        OpenAI, ChromaDB, or the retrieval pipeline.
        """
        ticket_dir = self._data_root / ticket_id
        ticket_path = ticket_dir / _TICKET_FILENAME
        normalized_path = ticket_dir / _NORMALIZED_TICKET_FILENAME
        manifest_path = ticket_dir / _MANIFEST_FILENAME

        if normalized_path.is_file() and manifest_path.is_file():
            logger.info("Using cached Redmine data for ticket %s", ticket_id)
        else:
            self.get_ticket(ticket_id)

        ticket = RedmineTicket.model_validate(_read_json(normalized_path))
        manifest = DownloadManifest.model_validate(_read_json(manifest_path))

        return RedmineDebugInfo(
            ticket=ticket,
            manifest=manifest,
            ticket_path=ticket_path,
            normalized_ticket_path=normalized_path,
            manifest_path=manifest_path,
            attachments_dir=ticket_dir / "attachments",
        )

    # --- Redmine API ---

    def _fetch_issue(self, ticket_id: str) -> dict:
        try:
            response = self._client.get(f"/issues/{ticket_id}.json", params={"include": "attachments"})
        except httpx.HTTPError as exc:
            logger.error("Failed to reach Redmine for ticket %s: %s", ticket_id, exc)
            raise ExternalServiceError(f"Could not reach Redmine for ticket '{ticket_id}': {exc}") from exc

        if response.status_code == 404:
            raise NotFoundError(f"No Redmine ticket found with id '{ticket_id}'.")
        if response.status_code in (401, 403):
            raise UnauthorizedError(f"Not authorized to access Redmine ticket '{ticket_id}'.")
        if response.is_error:
            logger.error("Redmine returned %s for ticket %s", response.status_code, ticket_id)
            raise ExternalServiceError(f"Redmine returned {response.status_code} for ticket '{ticket_id}'.")

        try:
            return response.json()["issue"]
        except (ValueError, KeyError) as exc:
            logger.error("Unexpected Redmine response shape for ticket %s: %s", ticket_id, exc)
            raise ExternalServiceError(f"Unexpected Redmine response for ticket '{ticket_id}'.") from exc

    # --- attachments ---

    def _download_attachments(self, ticket_id: str, attachments: list[dict]) -> list[_AttachmentOutcome]:
        return [self._download_one_attachment(ticket_id, attachment) for attachment in attachments]

    def _download_one_attachment(self, ticket_id: str, attachment: dict) -> _AttachmentOutcome:
        attachment_id = attachment.get("id")
        filename = attachment.get("filename")
        content_url = attachment.get("content_url")

        if not filename or not content_url:
            logger.warning(
                "Skipping attachment %s for ticket %s: missing filename or content_url",
                attachment_id,
                ticket_id,
            )
            return _AttachmentOutcome(
                attachment_id=attachment_id,
                filename=filename or "unknown",
                status="SKIPPED",
                failure_reason="Missing filename or content_url in Redmine response.",
            )

        logger.info("Attachment download started: %s", filename)
        try:
            response = self._client.get(content_url)
            response.raise_for_status()

            target_dir = self._data_root / ticket_id / "attachments"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename
            target_path.write_bytes(response.content)
        except Exception as exc:
            logger.warning("Attachment download failed: %s (%s)", filename, exc)
            return _AttachmentOutcome(
                attachment_id=attachment_id,
                filename=filename,
                status="FAILED",
                failure_reason=str(exc),
            )

        logger.info("Attachment downloaded: %s", filename)
        return _AttachmentOutcome(
            attachment_id=attachment_id,
            filename=filename,
            status="DOWNLOADED",
            content_type=attachment.get("content_type", "application/octet-stream"),
            file_size=len(response.content),
            local_path=str(target_path),
        )

    # --- persistence — the only place this service writes to backend/data/redmine/ ---

    def _ticket_dir(self, ticket_id: str) -> Path:
        ticket_dir = self._data_root / ticket_id
        ticket_dir.mkdir(parents=True, exist_ok=True)
        return ticket_dir

    def _save_raw_ticket(self, ticket_dir: Path, ticket_id: str, issue: dict) -> None:
        """Preserves the Redmine API response exactly as returned — no
        field renaming, no value normalization — for debugging/auditing.
        """
        try:
            _write_json(ticket_dir / _TICKET_FILENAME, issue)
        except OSError as exc:
            logger.error("Failed to save ticket.json for ticket %s: %s", ticket_id, exc)
        else:
            logger.info("ticket.json saved for ticket %s", ticket_id)

    def _save_normalized_ticket(self, ticket_dir: Path, ticket_id: str, ticket: RedmineTicket) -> None:
        """The application's internal model — what the future Retrieval
        Pipeline should read instead of reprocessing the raw response.
        """
        try:
            _write_json(ticket_dir / _NORMALIZED_TICKET_FILENAME, ticket.model_dump(mode="json"))
        except OSError as exc:
            logger.error("Failed to save normalized_ticket.json for ticket %s: %s", ticket_id, exc)
        else:
            logger.info("normalized_ticket.json saved for ticket %s", ticket_id)

    def _save_manifest(self, ticket_dir: Path, ticket_id: str, outcomes: list[_AttachmentOutcome]) -> None:
        manifest = DownloadManifest(
            ticket_id=ticket_id,
            generated_at=utcnow(),
            attachments=[
                AttachmentManifestEntry(
                    attachment_id=outcome.attachment_id,
                    filename=outcome.filename,
                    content_type=outcome.content_type,
                    file_size=outcome.file_size,
                    download_status=outcome.status,
                    local_path=outcome.local_path,
                    failure_reason=outcome.failure_reason,
                )
                for outcome in outcomes
            ],
        )
        try:
            _write_json(ticket_dir / _MANIFEST_FILENAME, manifest.model_dump(mode="json", by_alias=True))
        except OSError as exc:
            logger.error("Failed to save download_manifest.json for ticket %s: %s", ticket_id, exc)
        else:
            logger.info("download_manifest.json saved for ticket %s", ticket_id)


def _to_ticket(ticket_id: str, issue: dict, outcomes: list[_AttachmentOutcome]) -> RedmineTicket:
    return RedmineTicket(
        ticket_id=ticket_id,
        title=issue["subject"],
        description=issue.get("description") or "",
        status=issue["status"]["name"],
        feature=_extract_feature(issue),
        author=issue["author"]["name"],
        created_at=issue["created_on"],
        updated_at=issue["updated_on"],
        attachments=[
            RedmineAttachment(
                filename=outcome.filename,
                content_type=outcome.content_type or "application/octet-stream",
                file_size=outcome.file_size or 0,
                local_path=outcome.local_path,
            )
            for outcome in outcomes
            if outcome.status == "DOWNLOADED"
        ],
    )


def _extract_feature(issue: dict) -> str:
    for field in issue.get("custom_fields", []):
        if str(field.get("name", "")).strip().lower() == _FEATURE_CUSTOM_FIELD_NAME:
            value = field.get("value")
            if value:
                return value
            break
    return issue.get("project", {}).get("name", "")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
