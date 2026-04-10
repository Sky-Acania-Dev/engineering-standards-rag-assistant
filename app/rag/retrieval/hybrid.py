from __future__ import annotations

from dataclasses import dataclass

from app.stores.docstore import JsonlChunkStore, StoredChunk
from app.stores.faiss_store import FaissStore


@dataclass(frozen=True)
class RetrievalHit:
    chunk: StoredChunk
    score: float


def retrieve_top_k(
    *,
    index: FaissStore,
    docstore: JsonlChunkStore,
    query_vector: list[float],
    k: int = 5,
    excluded_content_types: set[str] | None = None,
) -> list[RetrievalHit]:
    """Retrieve and hydrate top-k chunks from vector index + docstore."""
    scored_chunks = index.search(query_vector, k=max(k, 0))
    hydrated: list[RetrievalHit] = []

    excluded = excluded_content_types or set()

    for scored in scored_chunks:
        stored = docstore.get(scored.chunk_uid)
        if stored is None or stored.content_type in excluded:
            continue
        hydrated.append(RetrievalHit(chunk=stored, score=scored.score))

    return hydrated
