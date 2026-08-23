"""Final Chunk Selection — the pipeline stage between Retrieve Candidate
Chunks and Merge Adjacent Chunks. Today this simply keeps the top
`final_chunks` candidates by similarity score. This is the plug point
for a future Cross Encoder Reranker: same `select()` signature, a
different internal ranking — `RetrievalService` would only need a
different `FinalChunkSelector` implementation injected, never a change
to itself.
"""
from app.models.retrieved_chunk import RetrievedChunk


class FinalChunkSelector:
    def select(self, candidates: list[RetrievedChunk], final_chunks: int) -> list[RetrievedChunk]:
        """Candidates already arrive ranked by similarity (vector search
        returns nearest-first), but sorting explicitly here makes that
        guarantee this stage's own responsibility rather than an
        assumption borrowed from an earlier stage.
        """
        ranked = sorted(candidates, key=lambda chunk: chunk.similarity_score, reverse=True)
        return ranked[:final_chunks]
