async def test_index_feature(client):
    response = await client.post(
        "/api/v1/index-feature",
        json={"feature": "Appointments", "documentIds": ["doc-1", "doc-2"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["feature"] == "Appointments"
    assert body["status"] == "completed"
    assert body["chunksIndexed"] > 0
