from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.api.schemas.query import Citation, QueryRequest, QueryResponse
from app.llm.embeddings import Embedder, build_embedder
from app.rag.retrieval.hybrid import retrieve_top_k
from app.stores.docstore import JsonlChunkStore
from app.stores.faiss_store import FaissStore


@dataclass(frozen=True)
class QueryArtifacts:
    index: FaissStore
    docstore: JsonlChunkStore


class QueryService:
    """Vertical-slice query service: embed -> retrieve -> extractive answer + citations."""

    def __init__(self, artifacts: QueryArtifacts, embedder: Embedder) -> None:
        self._artifacts = artifacts
        self._embedder = embedder

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str | Path,
        *,
        embedder_provider: str = "hash",
        embedding_dimension: int | None = None,
        embedding_model: str | None = None,
    ) -> QueryService:
        artifact_dir = Path(index_dir)
        index_path = artifact_dir / "chunk_index.json"
        store_path = artifact_dir / "chunk_store.jsonl"

        index = FaissStore.load(index_path)
        docstore = JsonlChunkStore.load(store_path)
        embedder = build_embedder(
            provider=embedder_provider,
            dimension=embedding_dimension or index.dimension,
            model_name=embedding_model,
        )
        return cls(QueryArtifacts(index=index, docstore=docstore), embedder)

    def query(self, request: QueryRequest) -> QueryResponse:
        query_embedding = self._embedder.embed_texts([request.question])
        if not query_embedding:
            return QueryResponse(answer="No query embedding generated.", citations=[], retrieved_chunks=0)

        hits = retrieve_top_k(
            index=self._artifacts.index,
            docstore=self._artifacts.docstore,
            query_vector=query_embedding[0],
            k=request.top_k,
        )

        if not hits:
            return QueryResponse(
                answer="I could not find supporting evidence in the indexed documents.",
                citations=[],
                retrieved_chunks=0,
            )

        citations = [
            Citation(
                chunk_uid=hit.chunk.chunk_uid,
                doc_id=hit.chunk.doc_id,
                title=hit.chunk.title,
                section=hit.chunk.section,
                chunk_id=hit.chunk.chunk_id,
                page=hit.chunk.page,
                score=hit.score,
            )
            for hit in hits
        ]
        answer = self._build_extractive_answer(hits)

        return QueryResponse(answer=answer, citations=citations, retrieved_chunks=len(hits))

    @staticmethod
    def _build_extractive_answer(hits: list) -> str:
        lines = ["Evidence-based answer (extractive preview):"]
        for position, hit in enumerate(hits[:3], start=1):
            snippet = " ".join(hit.chunk.text.split())[:220]
            lines.append(f"{position}. {snippet} (source: {hit.chunk.chunk_uid})")
        return "\n".join(lines)
