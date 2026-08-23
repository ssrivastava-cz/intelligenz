from pydantic import BaseModel


class SourceRetrievalConfig(BaseModel):
    """How many chunks one knowledge source contributes at each stage of
    its own pipeline — Retrieve Candidate Chunks, then narrow down to
    Final Chunk Selection. `candidate_chunks` is always >= `final_chunks`
    in practice, though nothing enforces that here (a `FinalChunkSelector`
    asked for more final chunks than it was given candidates simply
    returns every candidate it has).
    """

    candidate_chunks: int
    final_chunks: int


class RetrievalConfiguration(BaseModel):
    """Replaces a single flat `top_k` with one `SourceRetrievalConfig`
    per knowledge source, so each source's candidate pool and final
    selection size can be tuned independently — e.g. casting a wide net
    over Workflow documents but keeping only a handful, while Historical
    Test Cases keep more of what's retrieved.
    """

    workflow: SourceRetrievalConfig
    historical_test_cases: SourceRetrievalConfig
    historical_issues: SourceRetrievalConfig
    uploaded_documents: SourceRetrievalConfig

    @classmethod
    def uniform(cls, chunks: int) -> "RetrievalConfiguration":
        """Applies the same candidate/final count to every source —
        exactly what a single flat `top_k` meant before this
        configuration existed, so callers that only ever set `top_k`
        (e.g. `GenerationService`) keep behaving identically.
        """
        return cls(
            workflow=SourceRetrievalConfig(candidate_chunks=chunks, final_chunks=chunks),
            historical_test_cases=SourceRetrievalConfig(candidate_chunks=chunks, final_chunks=chunks),
            historical_issues=SourceRetrievalConfig(candidate_chunks=chunks, final_chunks=chunks),
            uploaded_documents=SourceRetrievalConfig(candidate_chunks=chunks, final_chunks=chunks),
        )
