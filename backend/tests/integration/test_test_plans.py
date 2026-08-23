async def test_generate_status_history_flow(client):
    generate_response = await client.post(
        "/api/v1/generate-test-plan",
        json={
            "feature": "Appointments",
            "redmineId": "12345",
            "description": "Focus on booking flows.",
        },
    )
    assert generate_response.status_code == 202
    generation_id = generate_response.json()["generationId"]
    assert generate_response.json()["status"] == "processing"

    status_response = await client.get(f"/api/v1/status/{generation_id}")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "processing"
    assert status_body["currentStage"] == "Reading Redmine"
    assert status_body["testCases"] is None

    history_response = await client.get("/api/v1/history")
    assert history_response.status_code == 200
    items = history_response.json()["items"]
    ids = [item["generationId"] for item in items]
    assert generation_id in ids
    assert next(item for item in items if item["generationId"] == generation_id)["redmineId"] == "12345"


async def test_generate_test_plan_accepts_legacy_ticket_id_field(client):
    """Backward compatibility: the pre-rename field name still works."""
    response = await client.post(
        "/api/v1/generate-test-plan",
        json={"feature": "Appointments", "ticketId": "54321"},
    )

    assert response.status_code == 202


async def test_generate_test_plan_consumes_upload_session_documents(client, upload_service, test_plan_service):
    session_id = (await client.post("/api/v1/upload-session")).json()["sessionId"]

    files = [("files", ("requirements.pdf", b"mock pdf bytes", "application/pdf"))]
    upload_response = await client.post(
        "/api/v1/upload-documents",
        files=files,
        data={"sessionId": session_id, "feature": "Appointments"},
    )
    document_id = upload_response.json()["documents"][0]["documentId"]

    generate_response = await client.post(
        "/api/v1/generate-test-plan",
        json={"feature": "Appointments", "redmineId": "77777", "sessionId": session_id},
    )
    assert generate_response.status_code == 202
    generation_id = generate_response.json()["generationId"]

    # The session records which generation consumed it...
    session = upload_service.get_session(session_id)
    assert session.consumed_by_generation_id == generation_id

    # ...and the uploaded document ended up attached to the generation.
    stored_generation = test_plan_service.get_generation(generation_id)
    assert document_id in stored_generation.document_ids


async def test_generate_test_plan_without_a_session_still_works(client):
    """A test plan can still be generated with no documents at all."""
    response = await client.post(
        "/api/v1/generate-test-plan",
        json={"feature": "Analytics", "redmineId": "11111"},
    )

    assert response.status_code == 202


async def test_status_not_found(client):
    response = await client.get("/api/v1/status/does-not-exist")
    assert response.status_code == 404


async def test_download_requires_completed_generation(client):
    generate_response = await client.post(
        "/api/v1/generate-test-plan",
        json={"feature": "Analytics", "redmineId": "99999"},
    )
    generation_id = generate_response.json()["generationId"]

    response = await client.get(f"/api/v1/download/{generation_id}")
    assert response.status_code == 422


async def test_download_completed_seed_generation(client):
    response = await client.get("/api/v1/download/gen-seed-1042")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "TC-001" in response.text
