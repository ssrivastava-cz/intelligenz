from pathlib import Path


async def test_create_upload_session(client):
    response = await client.post("/api/v1/upload-session")

    assert response.status_code == 200
    assert response.json()["sessionId"]


async def test_upload_user_document(client):
    session = await client.post("/api/v1/upload-session")
    session_id = session.json()["sessionId"]

    content = b"%PDF-1.4 mock content"
    files = [("files", ("plan.pdf", content, "application/pdf"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session_id, "feature": "Appointments"},
    )

    assert response.status_code == 200
    doc = response.json()["documents"][0]
    assert doc["documentName"] == "plan.pdf"
    assert doc["documentType"] == "PDF"
    assert doc["feature"] == "Appointments"
    assert doc["fileSize"] == len(content)
    assert doc["documentSource"] == "user_upload"
    assert doc["documentId"]


async def test_upload_multiple_user_documents(client):
    session = await client.post("/api/v1/upload-session")
    session_id = session.json()["sessionId"]

    files = [
        ("files", ("notes.txt", b"release notes", "text/plain")),
        ("files", ("cases.csv", b"id,title\n1,foo", "text/csv")),
        ("files", ("spec.md", b"# spec", "text/markdown")),
    ]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session_id, "feature": "Appointments"},
    )

    assert response.status_code == 200
    types = {d["documentType"] for d in response.json()["documents"]}
    assert types == {"TXT", "CSV", "MARKDOWN"}


async def test_upload_without_calling_upload_session_first_still_works(client):
    """Backward compatibility: an arbitrary sessionId that was never
    registered via POST /upload-session is lazily accepted."""
    files = [("files", ("adhoc.txt", b"content", "text/plain"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": "sess-never-registered", "feature": "Appointments"},
    )

    assert response.status_code == 200


async def test_legacy_generation_id_field_still_accepted(client):
    """Backward compatibility: the pre-session field name still works."""
    files = [("files", ("legacy.txt", b"content", "text/plain"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"generationId": "legacy-id-123", "feature": "Appointments"},
    )

    assert response.status_code == 200


async def test_upload_documents_requires_a_session_identifier(client):
    files = [("files", ("no-session.txt", b"content", "text/plain"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"feature": "Appointments"},
    )

    assert response.status_code == 422


async def test_upload_source_of_truth_document(client):
    files = [
        (
            "files",
            (
                "workflow.docx",
                b"mock docx bytes",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        )
    ]
    response = await client.post("/api/v1/source-of-truth/Appointments/documents", files=files)

    assert response.status_code == 200
    doc = response.json()["documents"][0]
    assert doc["documentSource"] == "source_of_truth"
    assert doc["feature"] == "Appointments"
    assert doc["documentType"] == "DOCX"


async def test_reject_unsupported_extension(client):
    session = await client.post("/api/v1/upload-session")
    files = [("files", ("malware.exe", b"bad", "application/octet-stream"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session.json()["sessionId"], "feature": "Appointments"},
    )

    assert response.status_code == 422


async def test_reject_oversized_file(client):
    session = await client.post("/api/v1/upload-session")
    content = b"x" * (2 * 1024 * 1024)  # exceeds the 1 MB test cap set in conftest
    files = [("files", ("big.txt", content, "text/plain"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session.json()["sessionId"], "feature": "Appointments"},
    )

    assert response.status_code == 422


async def test_reject_duplicate_filenames_within_batch(client):
    session = await client.post("/api/v1/upload-session")
    files = [
        ("files", ("dup.txt", b"one", "text/plain")),
        ("files", ("dup.txt", b"two", "text/plain")),
    ]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session.json()["sessionId"], "feature": "Appointments"},
    )

    assert response.status_code == 422


async def test_reject_duplicate_filename_against_existing_upload(client):
    session_id = (await client.post("/api/v1/upload-session")).json()["sessionId"]

    files = [("files", ("notes.txt", b"first", "text/plain"))]
    first = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session_id, "feature": "Appointments"},
    )
    assert first.status_code == 200

    files_again = [("files", ("notes.txt", b"second", "text/plain"))]
    second = await client.post(
        "/api/v1/upload-documents",
        files=files_again,
        data={"sessionId": session_id, "feature": "Appointments"},
    )
    assert second.status_code == 422


async def test_same_filename_allowed_across_different_sessions(client):
    session_a = (await client.post("/api/v1/upload-session")).json()["sessionId"]
    session_b = (await client.post("/api/v1/upload-session")).json()["sessionId"]

    files_a = [("files", ("notes.txt", b"first", "text/plain"))]
    response_a = await client.post(
        "/api/v1/upload-documents",
        files=files_a,
        data={"sessionId": session_a, "feature": "Appointments"},
    )
    files_b = [("files", ("notes.txt", b"second", "text/plain"))]
    response_b = await client.post(
        "/api/v1/upload-documents",
        files=files_b,
        data={"sessionId": session_b, "feature": "Appointments"},
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200


async def test_rejects_path_traversal_in_session_id(client):
    files = [("files", ("note.txt", b"x", "text/plain"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": "../../etc", "feature": "Appointments"},
    )

    assert response.status_code == 422


async def test_uploaded_file_is_written_to_disk_for_future_parsers(client, upload_service):
    session_id = (await client.post("/api/v1/upload-session")).json()["sessionId"]

    files = [("files", ("saved.txt", b"persisted content", "text/plain"))]
    response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session_id, "feature": "Appointments"},
    )
    document_id = response.json()["documents"][0]["documentId"]

    document = upload_service.get_document(document_id)
    storage_path = Path(document.storage_path)

    assert storage_path.read_bytes() == b"persisted content"
    assert storage_path.parent.name == session_id


async def test_list_session_documents(client, upload_service):
    session_id = (await client.post("/api/v1/upload-session")).json()["sessionId"]

    files = [("files", ("a.txt", b"a", "text/plain")), ("files", ("b.txt", b"b", "text/plain"))]
    await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session_id, "feature": "Appointments"},
    )

    documents = upload_service.list_session_documents(session_id)
    assert {d.filename for d in documents} == {"a.txt", "b.txt"}


async def test_list_source_of_truth_documents(client, upload_service):
    files = [("files", ("release-notes.md", b"# notes", "text/markdown"))]
    await client.post("/api/v1/source-of-truth/Analytics/documents", files=files)

    documents = upload_service.list_source_of_truth_documents("Analytics")
    assert len(documents) == 1
    assert documents[0].filename == "release-notes.md"
