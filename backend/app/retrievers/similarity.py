"""Score mapping is a retriever-layer concern, not `RetrievalService`'s,
so future changes to how a raw distance becomes a similarity score
(a different formula, score normalization across retrievers, per-metric
handling) stay isolated here instead of leaking into orchestration.
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.models.similarity_statistics import SimilarityStatistics


def distance_to_similarity(distance: float) -> float:
    """Every collection here uses ChromaDB's default L2 (squared
    Euclidean) space, where 0 means identical and larger means less
    similar with no fixed upper bound. This maps that onto a bounded
    (0, 1] score that decreases monotonically with distance, without
    assuming any particular distance metric.
    """
    return 1.0 / (1.0 + distance)


def compute_similarity_statistics(chunks: list[RetrievedChunk]) -> SimilarityStatistics:
    """Aggregates `similarity_score`s already sitting on each chunk — a
    pure diagnostic for `GET /retrieval/debug` and `GET /prompt/debug`,
    never a second similarity calculation (`distance_to_similarity`
    still runs exactly once per chunk, back in `chunk_mapping`).
    """
    if not chunks:
        return SimilarityStatistics(average=0.0, minimum=0.0, maximum=0.0)

    scores = [chunk.similarity_score for chunk in chunks]
    return SimilarityStatistics(average=sum(scores) / len(scores), minimum=min(scores), maximum=max(scores))
