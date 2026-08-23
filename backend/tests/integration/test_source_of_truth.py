import pytest
import tiktoken

from app.core.dependencies import get_openai_client
from app.main import app
from tests.fakes import FakeOpenAIClient

_EMBEDDING_PRICE_PER_1K_TOKENS = 0.00002  # matches Settings.embedding_price_per_1k_tokens default


def _override_openai_client() -> FakeOpenAIClient:
    fake_client = FakeOpenAIClient()
    app.dependency_overrides[get_openai_client] = lambda: fake_client
    return fake_client


def _clear_openai_override() -> None:
    app.dependency_overrides.pop(get_openai_client, None)


def _write_feature_files(tmp_path, feature: str) -> None:
    root = tmp_path / "source_of_truth" / feature

    (root / "workflows").mkdir(parents=True)
    (root / "workflows" / "onboarding.md").write_text("# Onboarding\nSteps here.", encoding="utf-8")

    (root / "TestCases").mkdir(parents=True)
    (root / "TestCases" / "cases.csv").write_text("id,title\n1,Book appointment\n", encoding="utf-8")

    (root / "IssueSheets").mkdir(parents=True)
    (root / "IssueSheets" / "issues.txt").write_text("KNOWN ISSUES\n\nSomething broke once.", encoding="utf-8")


async def test_list_features_returns_empty_when_nothing_indexed(client):
    response = await client.get("/api/v1/source-of-truth")

    assert response.status_code == 200
    assert response.json() == []


async def test_list_features_returns_counts_per_category(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")

    response = await client.get("/api/v1/source-of-truth")

    assert response.status_code == 200
    assert response.json() == [
        {"feature": "Appointments", "documents": 3, "workflows": 1, "testCases": 1, "issues": 1}
    ]


async def test_list_features_covers_multiple_features(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")
    (tmp_path / "source_of_truth" / "Coding Tool" / "workflows").mkdir(parents=True)
    (tmp_path / "source_of_truth" / "Coding Tool" / "workflows" / "setup.txt").write_text("Setup steps.")

    response = await client.get("/api/v1/source-of-truth")

    assert response.status_code == 200
    features = {item["feature"] for item in response.json()}
    assert features == {"Appointments", "Coding Tool"}


async def test_get_parsed_feature_documents(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")

    response = await client.get("/api/v1/source-of-truth/Appointments/parsed")

    assert response.status_code == 200
    body = response.json()
    assert body["feature"] == "Appointments"
    assert len(body["documents"]) == 3

    types = {d["documentType"] for d in body["documents"]}
    assert types == {"WORKFLOW", "TEST_CASE", "ISSUE"}

    for doc in body["documents"]:
        assert doc["documentName"]
        assert doc["sectionCount"] == len(doc["parsedDocument"]["sections"])
        assert doc["sectionHeadings"] == [s["heading"] for s in doc["parsedDocument"]["sections"]]
        assert doc["parsedDocument"]["metadata"]["documentSource"] == "source_of_truth"
        assert doc["parsedDocument"]["metadata"]["parserName"]

    workflow_doc = next(d for d in body["documents"] if d["documentType"] == "WORKFLOW")
    assert workflow_doc["parsedDocument"]["title"] == "Onboarding"


async def test_get_parsed_feature_documents_404_for_unknown_feature(client):
    response = await client.get("/api/v1/source-of-truth/Nonexistent/parsed")

    assert response.status_code == 404


async def test_get_parsed_feature_documents_does_not_include_chunks_or_embeddings(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")

    response = await client.get("/api/v1/source-of-truth/Appointments/parsed")

    body_text = response.text.lower()
    assert "chunk" not in body_text
    assert "embedding" not in body_text


async def test_get_feature_chunks_404_for_unknown_feature(client):
    response = await client.get("/api/v1/source-of-truth/Nonexistent/chunks")

    assert response.status_code == 404


async def test_get_feature_chunks_returns_empty_array_when_feature_has_no_documents(client, tmp_path):
    (tmp_path / "source_of_truth" / "EmptyFeature").mkdir(parents=True)

    response = await client.get("/api/v1/source-of-truth/EmptyFeature/chunks")

    assert response.status_code == 200
    assert response.json() == []


async def test_get_feature_chunks_covers_workflow_test_case_and_issue_documents(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")

    response = await client.get("/api/v1/source-of-truth/Appointments/chunks")

    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 3
    assert {chunk["artifactType"] for chunk in body} == {"WORKFLOW", "TEST_CASE", "ISSUE"}


async def test_get_feature_chunks_covers_multiple_documents_in_one_category(client, tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments" / "workflows"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("First workflow content.", encoding="utf-8")
    (root / "b.txt").write_text("Second workflow content.", encoding="utf-8")

    response = await client.get("/api/v1/source-of-truth/Appointments/chunks")

    assert response.status_code == 200
    filenames = {chunk["sourceFilename"] for chunk in response.json()}
    assert filenames == {"a.txt", "b.txt"}


async def test_get_feature_chunks_metadata_is_correct(client, tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments" / "workflows"
    root.mkdir(parents=True)
    (root / "Contact_Workflow.txt").write_text(
        "Role Permissions\n\nUsers with CU role can edit Contact Tracking.", encoding="utf-8"
    )

    response = await client.get("/api/v1/source-of-truth/Appointments/chunks")

    assert response.status_code == 200
    [chunk] = response.json()
    assert chunk["artifactType"] == "WORKFLOW"
    assert chunk["feature"] == "Appointments"
    assert chunk["documentSource"] == "source_of_truth"
    assert chunk["sourceFilename"] == "Contact_Workflow.txt"
    assert chunk["sectionHeading"] == "Role Permissions"
    assert chunk["pageNumber"] is None
    assert chunk["chunkNumber"] == 1
    assert chunk["wordCount"] == len(chunk["chunkText"].split())
    assert "Users with CU role" in chunk["chunkText"]
    assert chunk["chunkId"]


async def test_get_feature_chunks_ordering_matches_section_order(client, tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments" / "workflows"
    root.mkdir(parents=True)
    (root / "onboarding.md").write_text(
        "# First Section\nFirst body.\n\n# Second Section\nSecond body.", encoding="utf-8"
    )

    response = await client.get("/api/v1/source-of-truth/Appointments/chunks")

    assert response.status_code == 200
    body = response.json()
    assert [chunk["sectionHeading"] for chunk in body] == ["First Section", "Second Section"]
    assert [chunk["chunkNumber"] for chunk in body] == [1, 2]


async def test_get_feature_chunks_does_not_include_embeddings(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")

    response = await client.get("/api/v1/source-of-truth/Appointments/chunks")

    assert "embedding" not in response.text.lower()


async def test_get_embedding_preview_404_for_unknown_feature(client):
    fake_client = _override_openai_client()

    try:
        response = await client.get("/api/v1/source-of-truth/Nonexistent/embedding-preview")

        assert response.status_code == 404
        assert fake_client.calls == []
    finally:
        _clear_openai_override()


async def test_get_embedding_preview_returns_zeroed_stats_when_feature_has_no_documents(client, tmp_path):
    (tmp_path / "source_of_truth" / "EmptyFeature").mkdir(parents=True)
    _override_openai_client()

    try:
        response = await client.get("/api/v1/source-of-truth/EmptyFeature/embedding-preview")

        assert response.status_code == 200
        body = response.json()
        assert body["documentsFound"] == 0
        assert body["chunksCreated"] == 0
        assert body["totalEmbeddingTokens"] == 0
        assert body["averageTokensPerChunk"] == 0
        assert body["estimatedEmbeddingCost"] == 0
        assert body["chunks"] == []
    finally:
        _clear_openai_override()


async def test_get_embedding_preview_reports_correct_tokens_and_cost(client, tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments" / "workflows"
    root.mkdir(parents=True)
    (root / "Workflow.md").write_text(
        "# Role Permissions\n\nUsers with CU role can edit Contact Tracking records today.", encoding="utf-8"
    )
    _override_openai_client()

    try:
        response = await client.get("/api/v1/source-of-truth/Appointments/embedding-preview")

        assert response.status_code == 200
        body = response.json()
        assert body["feature"] == "Appointments"
        assert body["embeddingModel"] == "text-embedding-3-small"
        assert body["documentsFound"] == 1
        assert body["chunksCreated"] == 1

        [chunk] = body["chunks"]
        expected_tokens = len(tiktoken.encoding_for_model("text-embedding-3-small").encode(chunk["chunkText"]))
        assert chunk["embeddingTokens"] == expected_tokens
        assert body["totalEmbeddingTokens"] == expected_tokens
        assert body["averageTokensPerChunk"] == expected_tokens

        expected_cost = (expected_tokens / 1000) * _EMBEDDING_PRICE_PER_1K_TOKENS
        assert chunk["estimatedCost"] == pytest.approx(expected_cost)
        assert body["estimatedEmbeddingCost"] == pytest.approx(expected_cost)

        assert chunk["chunkId"]
        assert chunk["artifactType"] == "WORKFLOW"
        assert chunk["sourceFilename"] == "Workflow.md"
        assert chunk["sectionHeading"] == "Role Permissions"
        assert chunk["wordCount"] == len(chunk["chunkText"].split())
    finally:
        _clear_openai_override()


async def test_get_embedding_preview_computes_average_across_multiple_chunks(client, tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments" / "workflows"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("SHORT\n\nBrief.", encoding="utf-8")
    (root / "b.txt").write_text(
        "LONG\n\nThis section has quite a lot more words in its body than the other file does.",
        encoding="utf-8",
    )
    _override_openai_client()

    try:
        response = await client.get("/api/v1/source-of-truth/Appointments/embedding-preview")

        assert response.status_code == 200
        body = response.json()
        assert body["documentsFound"] == 2
        assert body["chunksCreated"] == len(body["chunks"]) == 2
        assert sum(chunk["embeddingTokens"] for chunk in body["chunks"]) == body["totalEmbeddingTokens"]
        assert body["averageTokensPerChunk"] == pytest.approx(body["totalEmbeddingTokens"] / 2)
        token_counts = {chunk["sourceFilename"]: chunk["embeddingTokens"] for chunk in body["chunks"]}
        assert token_counts["a.txt"] < token_counts["b.txt"]
    finally:
        _clear_openai_override()


async def test_get_embedding_preview_does_not_call_openai(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")
    fake_client = _override_openai_client()

    try:
        response = await client.get("/api/v1/source-of-truth/Appointments/embedding-preview")

        assert response.status_code == 200
        assert fake_client.calls == []
    finally:
        _clear_openai_override()


async def test_get_embedding_preview_covers_workflow_test_case_and_issue_documents(client, tmp_path):
    _write_feature_files(tmp_path, "Appointments")
    _override_openai_client()

    try:
        response = await client.get("/api/v1/source-of-truth/Appointments/embedding-preview")

        assert response.status_code == 200
        body = response.json()
        assert body["documentsFound"] == 3
        assert {chunk["artifactType"] for chunk in body["chunks"]} == {"WORKFLOW", "TEST_CASE", "ISSUE"}
    finally:
        _clear_openai_override()
