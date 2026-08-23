"""Merge Adjacent Chunks — the pipeline stage between Final Chunk
Selection and returning a knowledge source's results. Dedicated
component rather than logic inside a retriever, since merging is a
cross-cutting concern identical for every source (Workflow, Historical
Test Cases, Historical Issues, Uploaded Documents) — no retriever needs
to know about this or about any other retriever.
"""
from app.models.retrieved_chunk import RetrievedChunk

_TEXT_JOIN = "\n\n"


class AdjacentChunkMerger:
    def merge(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Combines consecutive chunks (by `chunk_number`) from the same
        source document (`collection_name` + `document_id`) into one
        chunk, so the prompt gets continuous context instead of
        fragmented windows. A chunk missing `document_id` or
        `chunk_number` — indexed before that metadata existed — can't
        have its adjacency verified, so it's always kept on its own
        rather than guessed into a group.
        """
        groups: list[list[RetrievedChunk]] = []
        group_by_key: dict[tuple[str, str], list[RetrievedChunk]] = {}
        for chunk in chunks:
            if chunk.document_id is None or chunk.chunk_number is None:
                groups.append([chunk])
                continue
            key = (chunk.collection_name, chunk.document_id)
            group = group_by_key.get(key)
            if group is None:
                group = []
                group_by_key[key] = group
                groups.append(group)
            group.append(chunk)

        merged = [combined for group in groups for combined in self._merge_group(group)]
        return sorted(merged, key=lambda chunk: chunk.similarity_score, reverse=True)

    def _merge_group(self, group: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if len(group) == 1:
            return group

        ordered = sorted(group, key=lambda chunk: chunk.chunk_number)
        runs: list[list[RetrievedChunk]] = [[ordered[0]]]
        for chunk in ordered[1:]:
            if chunk.chunk_number == runs[-1][-1].chunk_number + 1:
                runs[-1].append(chunk)
            else:
                runs.append([chunk])

        return [run[0] if len(run) == 1 else self._combine(run) for run in runs]

    def _combine(self, run: list[RetrievedChunk]) -> RetrievedChunk:
        first = run[0]
        return first.model_copy(
            update={
                "text": _TEXT_JOIN.join(chunk.text for chunk in run),
                # The run "deserves" to rank at its best-matching member's
                # score, not an average diluted by weaker neighbours.
                "similarity_score": max(chunk.similarity_score for chunk in run),
                # `run` is already sorted by chunk_number (see
                # `_merge_group`), so this is already ascending — the
                # debug endpoint's sole source for "Merged: Yes, Range,
                # Count", derived rather than tracked separately.
                "merged_chunk_numbers": [chunk.chunk_number for chunk in run],
            }
        )
