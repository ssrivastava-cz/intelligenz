from pydantic import BaseModel

from app.models.retrieved_chunk import RetrievedChunk


class SourceRetrievalResult(BaseModel):
    """One knowledge source's own pipeline output, captured at every
    stage — Retrieve Candidate Chunks, Final Chunk Selection, Merge
    Adjacent Chunks — so `GET /retrieval/debug` and `GET /prompt/debug`
    can show exactly how much each stage narrowed things down, without
    re-running any part of the pipeline. `merged_chunks` is what
    actually reaches the Prompt Builder.
    """

    candidate_requested: int
    candidate_chunks: list[RetrievedChunk]
    final_requested: int
    final_chunks: list[RetrievedChunk]
    merged_chunks: list[RetrievedChunk]


class RetrievalResult(BaseModel):
    """Everything `RetrievalService.retrieve` produced: query embedding
    stats, plus one `SourceRetrievalResult` per knowledge source.

    `workflow_results`/`test_case_results`/`issue_results`/
    `upload_results` are read-only properties, not stored fields, kept
    only so existing callers (`GenerationService`, `GET /prompt/debug`)
    that were written against the old flat-list shape keep working
    unchanged — they now transparently receive each source's final,
    post-merge chunks (`merged_chunks`), which is exactly what should
    feed the Prompt Builder.
    """

    query_text: str
    query_embedding_dimension: int
    query_embedding_tokens: int
    workflow: SourceRetrievalResult
    historical_test_cases: SourceRetrievalResult
    historical_issues: SourceRetrievalResult
    uploaded_documents: SourceRetrievalResult

    @property
    def workflow_results(self) -> list[RetrievedChunk]:
        return self.workflow.merged_chunks

    @property
    def test_case_results(self) -> list[RetrievedChunk]:
        return self.historical_test_cases.merged_chunks

    @property
    def issue_results(self) -> list[RetrievedChunk]:
        return self.historical_issues.merged_chunks

    @property
    def upload_results(self) -> list[RetrievedChunk]:
        return self.uploaded_documents.merged_chunks
