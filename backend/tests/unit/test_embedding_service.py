import tiktoken

from app.services.embedding_service import EmbeddingService
from tests.fakes import FakeOpenAIClient


def _make_service(model: str = "text-embedding-3-small") -> EmbeddingService:
    # The fake client is never touched by count_tokens — it's a purely
    # local computation — but EmbeddingService still requires one.
    return EmbeddingService(client=FakeOpenAIClient(), model=model)


def test_model_property_reflects_configured_model():
    service = _make_service(model="text-embedding-3-large")

    assert service.model == "text-embedding-3-large"


def test_count_tokens_returns_empty_list_for_no_texts():
    service = _make_service()

    assert service.count_tokens([]) == []


def test_count_tokens_makes_no_openai_api_call():
    fake_client = FakeOpenAIClient()
    service = EmbeddingService(client=fake_client, model="text-embedding-3-small")

    service.count_tokens(["Users with CU role can edit Contact Tracking records."])

    assert fake_client.calls == []


def test_count_tokens_matches_the_real_tiktoken_encoding_for_the_model():
    service = _make_service()
    text = "Users with CU role can edit Contact Tracking records."

    [count] = service.count_tokens([text])

    expected = len(tiktoken.encoding_for_model("text-embedding-3-small").encode(text))
    assert count == expected


def test_count_tokens_returns_one_count_per_input_text_in_order():
    service = _make_service()
    texts = ["Short.", "A slightly longer sentence with more words in it."]

    counts = service.count_tokens(texts)

    assert len(counts) == 2
    assert counts[0] < counts[1]
    assert all(count > 0 for count in counts)


def test_count_tokens_falls_back_to_cl100k_base_for_an_unknown_model_name():
    service = _make_service(model="some-future-embedding-model")

    counts = service.count_tokens(["Some text to tokenize."])

    assert counts[0] > 0
