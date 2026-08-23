"""Integration tests for GET /api/v1/redmine/{ticket_id}/debug — via
`httpx.MockTransport` standing in for the real Redmine server, same as
the RedmineService unit tests. No real network call is ever made.
"""
import httpx

from app.core.dependencies import get_redmine_service
from app.main import app
from app.services.redmine_service import RedmineService

_REDMINE_BASE_URL = "https://redmine.example.com"


def _issue_payload(ticket_id: str = "30463", **overrides) -> dict:
    payload = {
        "id": int(ticket_id),
        "project": {"id": 1, "name": "Cozeva Platform"},
        "status": {"id": 1, "name": "Resolved"},
        "author": {"id": 1, "name": "John Smith"},
        "subject": "Fix contact tracking bug",
        "description": "Steps to reproduce...",
        "created_on": "2026-08-01T10:00:00Z",
        "updated_on": "2026-08-02T12:00:00Z",
        "custom_fields": [{"id": 1, "name": "Feature", "value": "Contact and Sticket Log"}],
        "attachments": [],
    }
    payload.update(overrides)
    return payload


def _attachment(attachment_id: int, filename: str) -> dict:
    return {
        "id": attachment_id,
        "filename": filename,
        "content_type": "application/pdf",
        "content_url": f"{_REDMINE_BASE_URL}/attachments/download/{attachment_id}/{filename}",
    }


def _override_redmine_service(handler, tmp_path) -> RedmineService:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        base_url=_REDMINE_BASE_URL, headers={"X-Redmine-API-Key": "test-key"}, transport=transport
    )
    service = RedmineService(client=client, data_root=tmp_path / "redmine")
    app.dependency_overrides[get_redmine_service] = lambda: service
    return service


def _clear_redmine_override() -> None:
    app.dependency_overrides.pop(get_redmine_service, None)


async def test_get_redmine_debug_returns_full_response_for_successful_ticket(client, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    _override_redmine_service(handler, tmp_path)

    try:
        response = await client.get("/api/v1/redmine/30463/debug")

        assert response.status_code == 200
        body = response.json()
        assert body["ticketId"] == "30463"
        assert body["feature"] == "Contact and Sticket Log"
        assert body["title"] == "Fix contact tracking bug"
        assert body["status"] == "Resolved"
        assert body["author"] == "John Smith"
        assert body["description"] == "Steps to reproduce..."
        assert body["attachmentCount"] == 0
        assert body["attachments"] == {"downloaded": 0, "failed": 0, "skipped": 0}
        assert body["attachmentDetails"] == []
        assert "createdAt" in body
        assert "updatedAt" in body
    finally:
        _clear_redmine_override()


async def test_get_redmine_debug_404_for_missing_ticket(client, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errors": ["not found"]})

    _override_redmine_service(handler, tmp_path)

    try:
        response = await client.get("/api/v1/redmine/99999/debug")

        assert response.status_code == 404
    finally:
        _clear_redmine_override()


async def test_get_redmine_debug_returns_correct_file_paths(client, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    _override_redmine_service(handler, tmp_path)

    try:
        response = await client.get("/api/v1/redmine/30463/debug")

        assert response.status_code == 200
        paths = response.json()["paths"]
        ticket_dir = tmp_path / "redmine" / "30463"
        assert paths["ticket"] == str(ticket_dir / "ticket.json")
        assert paths["normalizedTicket"] == str(ticket_dir / "normalized_ticket.json")
        assert paths["manifest"] == str(ticket_dir / "download_manifest.json")
        assert paths["attachments"] == str(ticket_dir / "attachments")
    finally:
        _clear_redmine_override()


async def test_get_redmine_debug_reports_partial_attachment_failures(client, tmp_path):
    payload = _issue_payload(attachments=[_attachment(91, "Workflow.pdf"), _attachment(92, "Design.docx")])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/30463.json":
            return httpx.Response(200, json={"issue": payload})
        if "91" in request.url.path:
            return httpx.Response(200, content=b"fake pdf bytes")
        return httpx.Response(404)

    _override_redmine_service(handler, tmp_path)

    try:
        response = await client.get("/api/v1/redmine/30463/debug")

        assert response.status_code == 200
        body = response.json()
        assert body["attachmentCount"] == 2
        assert body["attachments"] == {"downloaded": 1, "failed": 1, "skipped": 0}
        details = {entry["filename"]: entry for entry in body["attachmentDetails"]}
        assert details["Workflow.pdf"]["downloadStatus"] == "DOWNLOADED"
        assert details["Workflow.pdf"]["localPath"]
        assert details["Design.docx"]["downloadStatus"] == "FAILED"
        assert details["Design.docx"]["failureReason"]
    finally:
        _clear_redmine_override()


async def test_get_redmine_debug_reports_complete_attachment_failure(client, tmp_path):
    payload = _issue_payload(attachments=[_attachment(91, "a.pdf"), _attachment(92, "b.docx")])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/30463.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(500)

    _override_redmine_service(handler, tmp_path)

    try:
        response = await client.get("/api/v1/redmine/30463/debug")

        assert response.status_code == 200
        body = response.json()
        assert body["attachments"] == {"downloaded": 0, "failed": 2, "skipped": 0}
        assert all(entry["downloadStatus"] == "FAILED" for entry in body["attachmentDetails"])
    finally:
        _clear_redmine_override()


async def test_get_redmine_debug_reuses_cached_data_on_second_call(client, tmp_path):
    call_count = {"total": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["total"] += 1
        return httpx.Response(200, json={"issue": _issue_payload()})

    _override_redmine_service(handler, tmp_path)

    try:
        first = await client.get("/api/v1/redmine/30463/debug")
        calls_after_first = call_count["total"]

        second = await client.get("/api/v1/redmine/30463/debug")

        assert first.status_code == second.status_code == 200
        assert call_count["total"] == calls_after_first  # no new HTTP call on the second request
        assert first.json() == second.json()
    finally:
        _clear_redmine_override()


async def test_get_redmine_debug_does_not_call_openai_or_chromadb(client, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    _override_redmine_service(handler, tmp_path)

    try:
        response = await client.get("/api/v1/redmine/30463/debug")

        assert response.status_code == 200
        body_text = response.text.lower()
        assert "embedding" not in body_text
        assert "chroma" not in body_text
    finally:
        _clear_redmine_override()
