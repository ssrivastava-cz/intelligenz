async def test_submit_feedback(client):
    response = await client.post(
        "/api/v1/feedback",
        json={"generationId": "gen-seed-1042", "rating": "up", "comment": "Looks great"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["generationId"] == "gen-seed-1042"
    assert body["message"] == "Thanks for your feedback!"
