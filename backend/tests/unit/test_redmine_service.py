"""Unit tests for RedmineService — ticket retrieval and attachment
downloads via `httpx.MockTransport` (a real httpx feature, not a mock
library), so no real network call is ever made and no mocking framework
is needed.
"""
import json
from pathlib import Path

import httpx
import pytest

from app.core.exceptions import ExternalServiceError, NotFoundError, UnauthorizedError
from app.models.redmine_ticket import RedmineTicket
from app.services.redmine_service import RedmineService

_BASE_URL = "https://redmine.example.com"


def _issue_payload(ticket_id: str = "12345", **overrides) -> dict:
    payload = {
        "id": int(ticket_id),
        "project": {"id": 1, "name": "Cozeva Platform"},
        "status": {"id": 1, "name": "New"},
        "author": {"id": 1, "name": "Jane Doe"},
        "subject": "Fix contact tracking bug",
        "description": "Steps to reproduce...",
        "created_on": "2026-08-01T10:00:00Z",
        "updated_on": "2026-08-02T12:00:00Z",
        "custom_fields": [{"id": 1, "name": "Feature", "value": "Contact and Sticket Log"}],
        "attachments": [],
    }
    payload.update(overrides)
    return payload


def _make_service(handler, tmp_path) -> RedmineService:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url=_BASE_URL, headers={"X-Redmine-API-Key": "test-key"}, transport=transport)
    return RedmineService(client=client, data_root=tmp_path / "redmine")


# --- successful ticket retrieval ---


def test_get_ticket_returns_a_strongly_typed_ticket(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/issues/12345.json"
        assert request.headers["X-Redmine-API-Key"] == "test-key"
        return httpx.Response(200, json={"issue": _issue_payload()})

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert isinstance(ticket, RedmineTicket)
    assert ticket.ticket_id == "12345"
    assert ticket.title == "Fix contact tracking bug"
    assert ticket.description == "Steps to reproduce..."
    assert ticket.status == "New"
    assert ticket.feature == "Contact and Sticket Log"
    assert ticket.author == "Jane Doe"
    assert ticket.attachments == []


def test_get_ticket_falls_back_to_project_name_when_no_feature_custom_field(tmp_path):
    payload = _issue_payload(custom_fields=[])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": payload})

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert ticket.feature == "Cozeva Platform"


def test_get_ticket_handles_missing_description(tmp_path):
    payload = _issue_payload(description=None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": payload})

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert ticket.description == ""


# --- ticket not found / authentication failures ---


def test_get_ticket_raises_not_found_for_missing_ticket(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errors": ["not found"]})

    service = _make_service(handler, tmp_path)

    with pytest.raises(NotFoundError):
        service.get_ticket("99999")


def test_get_ticket_raises_unauthorized_for_401(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"errors": ["unauthorized"]})

    service = _make_service(handler, tmp_path)

    with pytest.raises(UnauthorizedError):
        service.get_ticket("12345")


def test_get_ticket_raises_unauthorized_for_403(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"errors": ["forbidden"]})

    service = _make_service(handler, tmp_path)

    with pytest.raises(UnauthorizedError):
        service.get_ticket("12345")


def test_get_ticket_raises_external_service_error_for_server_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"errors": ["boom"]})

    service = _make_service(handler, tmp_path)

    with pytest.raises(ExternalServiceError):
        service.get_ticket("12345")


def test_get_ticket_raises_external_service_error_on_connection_failure(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = _make_service(handler, tmp_path)

    with pytest.raises(ExternalServiceError):
        service.get_ticket("12345")


def test_get_ticket_raises_external_service_error_on_timeout(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    service = _make_service(handler, tmp_path)

    with pytest.raises(ExternalServiceError):
        service.get_ticket("12345")


# --- attachments ---


def test_get_ticket_downloads_multiple_attachments(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "design.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/design.pdf",
            },
            {
                "id": 2,
                "filename": "workflow.docx",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "content_url": f"{_BASE_URL}/attachments/download/2/workflow.docx",
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        if request.url.path == "/attachments/download/1/design.pdf":
            return httpx.Response(200, content=b"%PDF-1.4 fake pdf bytes")
        if request.url.path == "/attachments/download/2/workflow.docx":
            return httpx.Response(200, content=b"fake docx bytes")
        raise AssertionError(f"unexpected request to {request.url}")

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert len(ticket.attachments) == 2
    filenames = {attachment.filename for attachment in ticket.attachments}
    assert filenames == {"design.pdf", "workflow.docx"}
    for attachment in ticket.attachments:
        assert Path(attachment.local_path).is_file()
        assert attachment.file_size > 0


def test_get_ticket_saves_attachments_under_ticket_id_folder(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "design.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/design.pdf",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(200, content=b"fake pdf bytes")

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    expected_path = tmp_path / "redmine" / "12345" / "attachments" / "design.pdf"
    assert Path(ticket.attachments[0].local_path) == expected_path
    assert expected_path.read_bytes() == b"fake pdf bytes"
    assert ticket.attachments[0].content_type == "application/pdf"
    assert ticket.attachments[0].file_size == len(b"fake pdf bytes")


def test_get_ticket_continues_after_one_attachment_fails(tmp_path, caplog):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "design.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/design.pdf",
            },
            {
                "id": 2,
                "filename": "workflow.docx",
                "content_type": "application/octet-stream",
                "content_url": f"{_BASE_URL}/attachments/download/2/workflow.docx",
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        if request.url.path == "/attachments/download/1/design.pdf":
            return httpx.Response(404)
        return httpx.Response(200, content=b"fake docx bytes")

    service = _make_service(handler, tmp_path)

    with caplog.at_level("WARNING"):
        ticket = service.get_ticket("12345")

    assert isinstance(ticket, RedmineTicket)
    assert len(ticket.attachments) == 1
    assert ticket.attachments[0].filename == "workflow.docx"
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert any("design.pdf" in record.message for record in warnings)


def test_get_ticket_returns_ticket_successfully_when_all_attachments_fail(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "design.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/design.pdf",
            },
            {
                "id": 2,
                "filename": "workflow.docx",
                "content_type": "application/octet-stream",
                "content_url": f"{_BASE_URL}/attachments/download/2/workflow.docx",
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(500)

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert isinstance(ticket, RedmineTicket)
    assert ticket.ticket_id == "12345"
    assert ticket.attachments == []


def test_get_ticket_treats_attachment_timeout_as_a_recoverable_failure(tmp_path, caplog):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "slow.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/slow.pdf",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        raise httpx.TimeoutException("timed out", request=request)

    service = _make_service(handler, tmp_path)

    with caplog.at_level("WARNING"):
        ticket = service.get_ticket("12345")

    assert ticket.attachments == []
    assert any("slow.pdf" in record.message for record in caplog.records)


# --- logging behavior ---


def test_get_ticket_logs_info_on_success(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    service = _make_service(handler, tmp_path)

    with caplog.at_level("INFO"):
        service.get_ticket("12345")

    info_records = [record for record in caplog.records if record.levelname == "INFO"]
    assert any("12345" in record.message for record in info_records)


def test_get_ticket_logs_info_for_each_successful_attachment_download(tmp_path, caplog):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "design.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/design.pdf",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(200, content=b"fake pdf bytes")

    service = _make_service(handler, tmp_path)

    with caplog.at_level("INFO"):
        service.get_ticket("12345")

    info_records = [record for record in caplog.records if record.levelname == "INFO"]
    assert any("design.pdf" in record.message for record in info_records)


def test_get_ticket_logs_info_when_ticket_json_is_saved(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    service = _make_service(handler, tmp_path)

    with caplog.at_level("INFO"):
        service.get_ticket("12345")

    info_records = [record for record in caplog.records if record.levelname == "INFO"]
    assert any("ticket.json" in record.message for record in info_records)
    assert any("normalized_ticket.json" in record.message for record in info_records)
    assert any("download_manifest.json" in record.message for record in info_records)


def test_get_ticket_logs_attachment_download_started(tmp_path, caplog):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "design.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/design.pdf",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(200, content=b"fake pdf bytes")

    service = _make_service(handler, tmp_path)

    with caplog.at_level("INFO"):
        service.get_ticket("12345")

    info_records = [record for record in caplog.records if record.levelname == "INFO"]
    assert any("started" in record.message.lower() and "design.pdf" in record.message for record in info_records)


# --- raw ticket persistence (ticket.json) ---


def test_get_ticket_saves_raw_ticket_json_exactly_as_returned(tmp_path):
    payload = _issue_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": payload})

    service = _make_service(handler, tmp_path)

    service.get_ticket("12345")

    raw = json.loads((tmp_path / "redmine" / "12345" / "ticket.json").read_text(encoding="utf-8"))
    assert raw == payload
    # Untouched Redmine field names/casing — not renamed to the app's own conventions.
    assert "created_on" in raw
    assert "subject" in raw
    assert "created_at" not in raw


def test_get_ticket_does_not_save_raw_ticket_json_when_ticket_fetch_fails(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errors": ["not found"]})

    service = _make_service(handler, tmp_path)

    with pytest.raises(NotFoundError):
        service.get_ticket("99999")

    assert not (tmp_path / "redmine" / "99999").exists()


# --- normalized ticket persistence (normalized_ticket.json) ---


def test_get_ticket_saves_normalized_ticket_json(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    normalized = json.loads(
        (tmp_path / "redmine" / "12345" / "normalized_ticket.json").read_text(encoding="utf-8")
    )
    assert normalized["ticket_id"] == "12345"
    assert normalized["feature"] == "Contact and Sticket Log"
    assert normalized["title"] == ticket.title
    assert normalized["description"] == ticket.description
    assert normalized["status"] == ticket.status
    assert normalized["author"] == ticket.author
    assert "created_at" in normalized
    assert "updated_at" in normalized
    assert normalized["attachments"] == []


def test_get_ticket_normalized_json_includes_local_attachment_paths(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 1,
                "filename": "design.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/1/design.pdf",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(200, content=b"fake pdf bytes")

    service = _make_service(handler, tmp_path)

    service.get_ticket("12345")

    normalized = json.loads(
        (tmp_path / "redmine" / "12345" / "normalized_ticket.json").read_text(encoding="utf-8")
    )
    [attachment] = normalized["attachments"]
    assert attachment["filename"] == "design.pdf"
    expected_path = str(tmp_path / "redmine" / "12345" / "attachments" / "design.pdf")
    assert attachment["local_path"] == expected_path


# --- directory creation ---


def test_get_ticket_creates_the_ticket_directory_even_with_no_attachments(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload(attachments=[])})

    service = _make_service(handler, tmp_path)

    service.get_ticket("12345")

    ticket_dir = tmp_path / "redmine" / "12345"
    assert ticket_dir.is_dir()
    assert (ticket_dir / "ticket.json").is_file()
    assert (ticket_dir / "normalized_ticket.json").is_file()
    assert (ticket_dir / "download_manifest.json").is_file()
    assert not (ticket_dir / "attachments").exists()  # never created when there's nothing to download


# --- download manifest (download_manifest.json) ---


def test_manifest_records_a_downloaded_attachment(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 91,
                "filename": "Workflow.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/91/Workflow.pdf",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(200, content=b"fake pdf bytes")

    service = _make_service(handler, tmp_path)

    service.get_ticket("12345")

    manifest = json.loads(
        (tmp_path / "redmine" / "12345" / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["ticketId"] == "12345"
    assert "generatedAt" in manifest
    [entry] = manifest["attachments"]
    assert entry["attachmentId"] == 91
    assert entry["filename"] == "Workflow.pdf"
    assert entry["contentType"] == "application/pdf"
    assert entry["fileSize"] == len(b"fake pdf bytes")
    assert entry["downloadStatus"] == "DOWNLOADED"
    assert entry["localPath"] == str(tmp_path / "redmine" / "12345" / "attachments" / "Workflow.pdf")
    assert entry["failureReason"] is None


def test_manifest_records_a_failed_attachment(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 92,
                "filename": "Design.docx",
                "content_type": "application/octet-stream",
                "content_url": f"{_BASE_URL}/attachments/download/92/Design.docx",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(404, text="Not Found")

    service = _make_service(handler, tmp_path)

    service.get_ticket("12345")

    manifest = json.loads(
        (tmp_path / "redmine" / "12345" / "download_manifest.json").read_text(encoding="utf-8")
    )
    [entry] = manifest["attachments"]
    assert entry["attachmentId"] == 92
    assert entry["filename"] == "Design.docx"
    assert entry["downloadStatus"] == "FAILED"
    assert entry["localPath"] is None
    assert entry["failureReason"]
    assert "404" in entry["failureReason"]


def test_manifest_records_a_skipped_attachment_missing_content_url(tmp_path):
    payload = _issue_payload(
        attachments=[{"id": 93, "filename": "mystery.zip", "content_type": "application/zip"}]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": payload})

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert ticket.attachments == []
    manifest = json.loads(
        (tmp_path / "redmine" / "12345" / "download_manifest.json").read_text(encoding="utf-8")
    )
    [entry] = manifest["attachments"]
    assert entry["downloadStatus"] == "SKIPPED"
    assert entry["localPath"] is None
    assert entry["failureReason"]


def test_manifest_is_created_even_with_zero_attachments(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload(attachments=[])})

    service = _make_service(handler, tmp_path)

    service.get_ticket("12345")

    manifest = json.loads(
        (tmp_path / "redmine" / "12345" / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["attachments"] == []


def test_manifest_records_mixed_outcomes_for_partial_failure(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 91,
                "filename": "Workflow.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/91/Workflow.pdf",
            },
            {
                "id": 92,
                "filename": "Design.docx",
                "content_type": "application/octet-stream",
                "content_url": f"{_BASE_URL}/attachments/download/92/Design.docx",
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        if "91" in request.url.path:
            return httpx.Response(200, content=b"fake pdf bytes")
        return httpx.Response(404)

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert len(ticket.attachments) == 1  # unchanged public behavior: only successes on the ticket
    manifest = json.loads(
        (tmp_path / "redmine" / "12345" / "download_manifest.json").read_text(encoding="utf-8")
    )
    statuses = {entry["filename"]: entry["downloadStatus"] for entry in manifest["attachments"]}
    assert statuses == {"Workflow.pdf": "DOWNLOADED", "Design.docx": "FAILED"}


def test_manifest_records_all_failed_when_every_attachment_download_fails(tmp_path):
    payload = _issue_payload(
        attachments=[
            {
                "id": 91,
                "filename": "a.pdf",
                "content_type": "application/pdf",
                "content_url": f"{_BASE_URL}/attachments/download/91/a.pdf",
            },
            {
                "id": 92,
                "filename": "b.docx",
                "content_type": "application/octet-stream",
                "content_url": f"{_BASE_URL}/attachments/download/92/b.docx",
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(500)

    service = _make_service(handler, tmp_path)

    ticket = service.get_ticket("12345")

    assert ticket.attachments == []  # ticket retrieval still succeeds
    manifest = json.loads(
        (tmp_path / "redmine" / "12345" / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert all(entry["downloadStatus"] == "FAILED" for entry in manifest["attachments"])
    assert len(manifest["attachments"]) == 2


# --- get_ticket_debug_info ---


def _attachment(attachment_id: int, filename: str, content_type: str = "application/pdf") -> dict:
    return {
        "id": attachment_id,
        "filename": filename,
        "content_type": content_type,
        "content_url": f"{_BASE_URL}/attachments/download/{attachment_id}/{filename}",
    }


def test_get_ticket_debug_info_fetches_live_when_nothing_is_cached(tmp_path):
    fetch_count = {"issues": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            fetch_count["issues"] += 1
            return httpx.Response(200, json={"issue": _issue_payload()})
        return httpx.Response(200, content=b"bytes")

    service = _make_service(handler, tmp_path)

    info = service.get_ticket_debug_info("12345")

    assert fetch_count["issues"] == 1
    assert info.ticket.ticket_id == "12345"
    assert info.ticket.feature == "Contact and Sticket Log"
    assert info.manifest.ticket_id == "12345"


def test_get_ticket_debug_info_returns_correct_paths(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    service = _make_service(handler, tmp_path)

    info = service.get_ticket_debug_info("12345")

    ticket_dir = tmp_path / "redmine" / "12345"
    assert info.ticket_path == ticket_dir / "ticket.json"
    assert info.normalized_ticket_path == ticket_dir / "normalized_ticket.json"
    assert info.manifest_path == ticket_dir / "download_manifest.json"
    assert info.attachments_dir == ticket_dir / "attachments"


def test_get_ticket_debug_info_reuses_cached_files_without_calling_redmine_again(tmp_path):
    call_count = {"total": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["total"] += 1
        return httpx.Response(200, json={"issue": _issue_payload()})

    service = _make_service(handler, tmp_path)
    service.get_ticket("12345")
    calls_after_first_fetch = call_count["total"]

    info = service.get_ticket_debug_info("12345")

    assert call_count["total"] == calls_after_first_fetch  # no new HTTP calls at all
    assert info.ticket.ticket_id == "12345"


def test_get_ticket_debug_info_logs_when_using_cached_data(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    service = _make_service(handler, tmp_path)
    service.get_ticket("12345")

    with caplog.at_level("INFO"):
        service.get_ticket_debug_info("12345")

    info_records = [record for record in caplog.records if record.levelname == "INFO"]
    assert any("cached" in record.message.lower() and "12345" in record.message for record in info_records)


def test_get_ticket_debug_info_raises_not_found_for_missing_uncached_ticket(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errors": ["not found"]})

    service = _make_service(handler, tmp_path)

    with pytest.raises(NotFoundError):
        service.get_ticket_debug_info("99999")


def test_get_ticket_debug_info_reports_partial_attachment_failures(tmp_path):
    payload = _issue_payload(
        attachments=[_attachment(91, "Workflow.pdf"), _attachment(92, "Design.docx")]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        if "91" in request.url.path:
            return httpx.Response(200, content=b"fake pdf bytes")
        return httpx.Response(404)

    service = _make_service(handler, tmp_path)

    info = service.get_ticket_debug_info("12345")

    statuses = {entry.filename: entry.download_status for entry in info.manifest.attachments}
    assert statuses == {"Workflow.pdf": "DOWNLOADED", "Design.docx": "FAILED"}
    downloaded = [e for e in info.manifest.attachments if e.download_status == "DOWNLOADED"]
    failed = [e for e in info.manifest.attachments if e.download_status == "FAILED"]
    assert downloaded[0].local_path is not None
    assert failed[0].failure_reason is not None


def test_get_ticket_debug_info_reports_complete_attachment_failure(tmp_path):
    payload = _issue_payload(
        attachments=[_attachment(91, "a.pdf"), _attachment(92, "b.docx")]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        return httpx.Response(500)

    service = _make_service(handler, tmp_path)

    info = service.get_ticket_debug_info("12345")

    assert info.ticket.attachments == []
    assert all(entry.download_status == "FAILED" for entry in info.manifest.attachments)
    assert len(info.manifest.attachments) == 2


def test_get_ticket_debug_info_on_cached_ticket_reflects_original_failures(tmp_path):
    payload = _issue_payload(attachments=[_attachment(91, "a.pdf"), _attachment(92, "b.docx")])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/issues/12345.json":
            return httpx.Response(200, json={"issue": payload})
        if "91" in request.url.path:
            return httpx.Response(200, content=b"fake pdf bytes")
        return httpx.Response(404)

    service = _make_service(handler, tmp_path)
    service.get_ticket("12345")

    info = service.get_ticket_debug_info("12345")  # served from cache

    statuses = {entry.filename: entry.download_status for entry in info.manifest.attachments}
    assert statuses == {"a.pdf": "DOWNLOADED", "b.docx": "FAILED"}
