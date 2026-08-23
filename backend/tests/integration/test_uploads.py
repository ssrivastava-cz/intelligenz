"""Integration tests for the Uploaded Document pipeline, now split into
four dedicated steps mirroring Source of Truth's architecture:

    POST /api/v1/uploads                                  (upload)
    POST /api/v1/uploads/{upload_session_id}/embedding-preview
    POST /api/v1/uploads/{upload_session_id}/embed
    GET  /api/v1/uploads/{upload_session_id}/debug

`upload_service`/`source_of_truth_indexer` are already tmp_path-scoped
by the `client` fixture (see conftest.py); only the OpenAI/Chroma/
History boundaries need overriding here, same pattern as
tests/integration/test_index_service.py.
"""
import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.dependencies import get_chroma_client, get_history_service, get_openai_client
from app.main import app
from app.services.history_service import HistoryService
from tests.fakes import FakeOpenAIClient


def _override_upload_dependencies(tmp_path) -> FakeOpenAIClient:
    fake_openai_client = FakeOpenAIClient()
    app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
    app.dependency_overrides[get_chroma_client] = lambda: chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    app.dependency_overrides[get_history_service] = lambda: HistoryService(history_root=tmp_path / "history")
    return fake_openai_client


def _clear_upload_dependency_overrides() -> None:
    app.dependency_overrides.pop(get_openai_client, None)
    app.dependency_overrides.pop(get_chroma_client, None)
    app.dependency_overrides.pop(get_history_service, None)


def _markdown_file(filename: str = "notes.md", body: str = "# Role Permissions\n\nBody text.") -> tuple:
    return ("files", (filename, body.encode("utf-8"), "text/markdown"))


async def _upload(client, *files) -> str:
    response = await client.post("/api/v1/uploads", files=list(files) or [_markdown_file()])
    assert response.status_code == 200
    return response.json()["uploadSessionId"]


# --- OpenAPI contract: files must be a real multipart file upload, not list[str] ---


def test_upload_openapi_schema_accepts_multipart_file_uploads():
    """Regression test for a previously reported Swagger UI issue:
    `files` rendering as a plain string array instead of a file picker.
    """
    schema = app.openapi()
    assert schema["openapi"].startswith("3.1")

    request_body = schema["paths"]["/api/v1/uploads"]["post"]["requestBody"]
    assert "multipart/form-data" in request_body["content"]

    body_schema_ref = request_body["content"]["multipart/form-data"]["schema"]["$ref"]
    body_schema_name = body_schema_ref.rsplit("/", 1)[-1]
    body_schema = schema["components"]["schemas"][body_schema_name]

    files_field = body_schema["properties"]["files"]
    assert files_field["type"] == "array"
    assert files_field["items"]["type"] == "string"
    assert files_field["items"]["contentMediaType"] == "application/octet-stream"
    assert body_schema["required"] == ["files"]


def test_upload_files_field_matches_the_working_upload_documents_endpoint():
    """This endpoint's file field must be represented identically to the
    known-working `/upload-documents` endpoint's `files` field.
    """
    schema = app.openapi()

    def _files_field_schema(path: str, operation_id_fragment: str) -> dict:
        request_body = schema["paths"][path]["post"]["requestBody"]
        ref = request_body["content"]["multipart/form-data"]["schema"]["$ref"]
        name = ref.rsplit("/", 1)[-1]
        assert operation_id_fragment in name
        return schema["components"]["schemas"][name]["properties"]["files"]

    upload_files_field = _files_field_schema("/api/v1/uploads", "upload_files")
    working_files_field = _files_field_schema("/api/v1/upload-documents", "upload_documents")

    assert upload_files_field == working_files_field


def test_embedding_preview_no_longer_accepts_a_request_body():
    """The preview endpoint moved from POST /uploads/embedding-preview
    (accepted files) to POST /uploads/{upload_session_id}/embedding-preview
    (accepts nothing — it reads what /uploads already saved).
    """
    schema = app.openapi()

    assert "/api/v1/uploads/embedding-preview" not in schema["paths"]
    operation = schema["paths"]["/api/v1/uploads/{upload_session_id}/embedding-preview"]["post"]
    assert "requestBody" not in operation


# --- POST /uploads ---


async def test_upload_files_persists_them_and_returns_session_id(client, tmp_path):
    response = await client.post("/api/v1/uploads", files=[_markdown_file()])

    assert response.status_code == 200
    body = response.json()
    assert body["uploadSessionId"]
    assert body["documentsUploaded"] == 1
    [uploaded] = body["uploadedFiles"]
    assert uploaded["documentName"] == "notes.md"
    assert uploaded["fileSize"] == len(b"# Role Permissions\n\nBody text.")

    session_dir = tmp_path / "uploads" / body["uploadSessionId"]
    assert session_dir.is_dir()
    assert (session_dir / "notes.md").read_bytes() == b"# Role Permissions\n\nBody text."


async def test_upload_files_handles_multiple_files(client, tmp_path):
    response = await client.post(
        "/api/v1/uploads",
        files=[_markdown_file("a.md", "# A\n\nBody A."), _markdown_file("b.md", "# B\n\nBody B.")],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["documentsUploaded"] == 2
    names = {uploaded["documentName"] for uploaded in body["uploadedFiles"]}
    assert names == {"a.md", "b.md"}


async def test_upload_files_does_not_create_history(client, tmp_path):
    response = await client.post("/api/v1/uploads", files=[_markdown_file()])

    assert response.status_code == 200
    assert not (tmp_path / "history").exists()


async def test_upload_files_response_contains_no_chunk_or_embedding_data(client, tmp_path):
    """POST /uploads must not parse, chunk, count tokens, or embed."""
    response = await client.post("/api/v1/uploads", files=[_markdown_file()])

    body_text = response.text.lower()
    assert "chunk" not in body_text
    assert "embedding" not in body_text
    assert "token" not in body_text


# --- POST /uploads/{upload_session_id}/embedding-preview ---


async def test_preview_upload_embeddings_returns_full_response(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)

        response = await client.post(f"/api/v1/uploads/{upload_session_id}/embedding-preview")

        assert response.status_code == 200
        body = response.json()
        assert body["uploadSessionId"] == upload_session_id
        assert body["documentsFound"] == 1
        assert body["chunksCreated"] == 1
        assert body["embeddingModel"] == "text-embedding-3-small"
        assert body["totalEmbeddingTokens"] > 0
        assert body["averageTokensPerChunk"] > 0
        assert body["estimatedEmbeddingCost"] >= 0
        assert body["estimatedEmbeddingCostInr"] >= 0
        assert body["estimatedEmbeddingCostInr"] == pytest.approx(body["estimatedEmbeddingCost"] * 87.0, abs=0.01)

        [chunk] = body["chunks"]
        assert chunk["documentName"] == "notes.md"
        assert chunk["sectionHeading"] == "Role Permissions"
        assert chunk["artifactType"] == "USER_UPLOAD"
        assert chunk["wordCount"] > 0
        assert chunk["embeddingTokens"] > 0
        assert chunk["estimatedCost"] >= 0
        assert "Body text." in chunk["chunkText"]
    finally:
        _clear_upload_dependency_overrides()


async def test_preview_upload_embeddings_404_for_unknown_session(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/uploads/nonexistent-session/embedding-preview")

        assert response.status_code == 404
    finally:
        _clear_upload_dependency_overrides()


async def test_preview_upload_embeddings_does_not_call_openai(client, tmp_path):
    fake_client = _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)

        response = await client.post(f"/api/v1/uploads/{upload_session_id}/embedding-preview")

        assert response.status_code == 200
        assert fake_client.calls == []
    finally:
        _clear_upload_dependency_overrides()


async def test_preview_upload_embeddings_does_not_create_history(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)

        response = await client.post(f"/api/v1/uploads/{upload_session_id}/embedding-preview")

        assert response.status_code == 200
        assert not (tmp_path / "history").exists()
    finally:
        _clear_upload_dependency_overrides()


async def test_preview_upload_embeddings_does_not_write_chromadb(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)

        response = await client.post(f"/api/v1/uploads/{upload_session_id}/embedding-preview")

        assert response.status_code == 200
        # Debug reads persisted history only, so a feature never embedded
        # (only previewed) must still show no history — proof nothing
        # was written to Chroma or history by the preview call.
        debug_response = await client.get(f"/api/v1/uploads/{upload_session_id}/debug")
        assert debug_response.status_code == 404
    finally:
        _clear_upload_dependency_overrides()


async def test_preview_upload_embeddings_handles_multiple_documents(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(
            client, _markdown_file("a.md", "# A\n\nBody A."), _markdown_file("b.md", "# B\n\nBody B.")
        )

        response = await client.post(f"/api/v1/uploads/{upload_session_id}/embedding-preview")

        assert response.status_code == 200
        body = response.json()
        assert body["documentsFound"] == 2
        assert body["chunksCreated"] == 2
        names = {chunk["documentName"] for chunk in body["chunks"]}
        assert names == {"a.md", "b.md"}
    finally:
        _clear_upload_dependency_overrides()


# --- POST /uploads/{upload_session_id}/embed ---


async def test_embed_uploaded_documents_returns_summary_and_creates_history(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)

        response = await client.post(f"/api/v1/uploads/{upload_session_id}/embed")

        assert response.status_code == 200
        body = response.json()
        assert body["uploadSessionId"] == upload_session_id
        assert body["documentsIndexed"] == 1
        assert body["chunksIndexed"] == 1
        assert body["embeddingTokens"] > 0
        assert body["chromaCollectionName"] == f"uploaded_documents_{upload_session_id}"
        assert body["indexStatus"] == "SUCCESS"

        upload_history_root = tmp_path / "history" / "upload"
        assert upload_history_root.is_dir()
        [run_folder] = [p for p in upload_history_root.iterdir() if p.is_dir()]
        assert (run_folder / "uploaded_embedding_summary.json").is_file()
        assert (run_folder / "embedding_preview.json").is_file()
    finally:
        _clear_upload_dependency_overrides()


async def test_embed_uploaded_documents_404_for_unknown_session(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/uploads/nonexistent-session/embed")

        assert response.status_code == 404
    finally:
        _clear_upload_dependency_overrides()


async def test_embed_uploaded_documents_reembedding_replaces_history_run(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)

        await client.post(f"/api/v1/uploads/{upload_session_id}/embed")
        second = await client.post(f"/api/v1/uploads/{upload_session_id}/embed")

        assert second.status_code == 200
        upload_history_root = tmp_path / "history" / "upload"
        # Two separate history runs recorded (mirrors index re-run behavior)...
        assert len(list(upload_history_root.iterdir())) == 2
    finally:
        _clear_upload_dependency_overrides()


async def test_embed_uploaded_documents_does_not_affect_source_of_truth_collection(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)

        response = await client.post(f"/api/v1/uploads/{upload_session_id}/embed")

        assert response.status_code == 200
        assert response.json()["chromaCollectionName"] != "source_of_truth_chunks"
    finally:
        _clear_upload_dependency_overrides()


async def test_embed_uploaded_documents_does_not_upload_new_files(client, tmp_path):
    """POST /embed takes no file input — it only operates on what /uploads
    already saved. Passing extra query/body data should have no effect;
    this just confirms the endpoint accepts no request body at all.
    """
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/uploads/{upload_session_id}/embed"]["post"]
    assert "requestBody" not in operation


# --- GET /uploads/{upload_session_id}/debug ---


async def test_get_upload_debug_404_when_never_embedded(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/uploads/nonexistent-session/debug")

        assert response.status_code == 404
    finally:
        _clear_upload_dependency_overrides()


async def test_get_upload_debug_returns_persisted_summary_and_preview(client, tmp_path):
    fake_client = _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)
        await client.post(f"/api/v1/uploads/{upload_session_id}/embed")
        calls_after_embed = len(fake_client.calls)

        response = await client.get(f"/api/v1/uploads/{upload_session_id}/debug")

        assert response.status_code == 200
        body = response.json()
        assert body["uploadSessionId"] == upload_session_id
        assert body["documentsIndexed"] == 1
        assert body["chunksIndexed"] == 1
        assert body["embeddingTokens"] > 0
        assert body["chromaCollectionName"] == f"uploaded_documents_{upload_session_id}"
        assert body["indexStatus"] == "SUCCESS"

        [chunk] = body["chunks"]
        assert chunk["sectionHeading"] == "Role Permissions"
        assert chunk["sourceFilename"] == "notes.md"
        assert "embedding" not in chunk

        # Read-only: no new OpenAI calls from the debug read itself.
        assert len(fake_client.calls) == calls_after_embed
    finally:
        _clear_upload_dependency_overrides()


async def test_get_upload_debug_does_not_reembed_or_reparse(client, tmp_path):
    """GET /debug must be read-only: no requestBody in its own contract,
    and repeated calls must not change the persisted history at all.
    """
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/uploads/{upload_session_id}/debug"]["get"]
    assert "requestBody" not in operation

    fake_client = _override_upload_dependencies(tmp_path)

    try:
        upload_session_id = await _upload(client)
        await client.post(f"/api/v1/uploads/{upload_session_id}/embed")
        calls_after_embed = len(fake_client.calls)

        first = await client.get(f"/api/v1/uploads/{upload_session_id}/debug")
        second = await client.get(f"/api/v1/uploads/{upload_session_id}/debug")

        assert first.json() == second.json()
        assert len(fake_client.calls) == calls_after_embed
    finally:
        _clear_upload_dependency_overrides()


# --- end-to-end workflow ---


async def test_full_upload_preview_embed_debug_workflow(client, tmp_path):
    _override_upload_dependencies(tmp_path)

    try:
        upload_response = await client.post("/api/v1/uploads", files=[_markdown_file()])
        assert upload_response.status_code == 200
        upload_session_id = upload_response.json()["uploadSessionId"]

        preview_response = await client.post(f"/api/v1/uploads/{upload_session_id}/embedding-preview")
        assert preview_response.status_code == 200
        assert preview_response.json()["chunksCreated"] == 1

        embed_response = await client.post(f"/api/v1/uploads/{upload_session_id}/embed")
        assert embed_response.status_code == 200
        assert embed_response.json()["chunksIndexed"] == 1

        debug_response = await client.get(f"/api/v1/uploads/{upload_session_id}/debug")
        assert debug_response.status_code == 200
        assert debug_response.json()["chunksIndexed"] == 1
    finally:
        _clear_upload_dependency_overrides()
