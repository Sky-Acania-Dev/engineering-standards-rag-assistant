from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Literal
from urllib import error, request

from app.api.schemas.query import Citation, QueryRequest, QueryResponse
from app.llm.embeddings import Embedder, build_embedder
from app.rag.retrieval.hybrid import RetrievalHit, retrieve_top_k
from app.stores.docstore import JsonlChunkStore
from app.stores.faiss_store import FaissStore


@dataclass(frozen=True)
class QueryArtifacts:
    index: FaissStore
    docstore: JsonlChunkStore


@dataclass(frozen=True)
class GenerationConfig:
    enabled: bool = False
    model: str | None = None
    endpoint: str = "http://localhost:11434/api/generate"
    timeout_seconds: float = 15.0
    temperature: float = 0.0


@dataclass(frozen=True)
class QueryServiceConfig:
    default_mode: Literal["extractive", "generative"] = "extractive"
    min_score: float = 0.2
    min_chunks: int = 1
    retrieval_top_k: int = 5
    generation: GenerationConfig = GenerationConfig()


class QueryService:
    """Vertical-slice query service: embed -> retrieve -> answer + citations."""

    def __init__(
        self,
        artifacts: QueryArtifacts,
        embedder: Embedder,
        *,
        config: QueryServiceConfig | None = None,
    ) -> None:
        self._artifacts = artifacts
        self._embedder = embedder
        self._config = config or QueryServiceConfig()

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str | Path,
        *,
        embedder_provider: str = "hash",
        embedding_dimension: int | None = None,
        embedding_model: str | None = None,
        config: QueryServiceConfig | None = None,
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
        return cls(QueryArtifacts(index=index, docstore=docstore), embedder, config=config)

    def query(self, request_payload: QueryRequest) -> QueryResponse:
        timings: dict[str, float] = {}
        started_at = perf_counter()

        request_mode = request_payload.mode if request_payload.mode else self._config.default_mode
        top_k = request_payload.top_k if request_payload.top_k is not None else self._config.retrieval_top_k

        embed_start = perf_counter()
        query_embedding = self._embedder.embed_texts([request_payload.question])
        timings["embedding_ms"] = self._elapsed_ms(embed_start)

        if not query_embedding:
            return self._refusal_response(
                reason="no_query_embedding",
                debug=request_payload.debug,
                debug_context={"timings_ms": timings, "mode": request_mode, "top_k": top_k},
            )

        retrieve_start = perf_counter()
        hits = retrieve_top_k(
            index=self._artifacts.index,
            docstore=self._artifacts.docstore,
            query_vector=query_embedding[0],
            k=top_k,
        )
        timings["retrieval_ms"] = self._elapsed_ms(retrieve_start)

        sufficiency = self._evaluate_evidence(hits)
        if sufficiency["refusal_reason"] is not None:
            timings["total_ms"] = self._elapsed_ms(started_at)
            debug_context = {
                "timings_ms": timings,
                "mode": request_mode,
                "top_k": top_k,
                "retrieval": sufficiency,
            }
            return self._refusal_response(
                reason=str(sufficiency["refusal_reason"]),
                debug=request_payload.debug,
                debug_context=debug_context,
            )

        citations = self._build_citations(hits)
        answer_start = perf_counter()
        if request_mode == "generative":
            answer, used_chunk_uids = self._build_generative_answer(request_payload.question, hits)
        else:
            answer, used_chunk_uids = self._build_extractive_answer(hits)
        timings["answer_ms"] = self._elapsed_ms(answer_start)
        timings["total_ms"] = self._elapsed_ms(started_at)

        confidence = self._compute_confidence(hits)
        response_debug = None
        if request_payload.debug:
            response_debug = {
                "mode": request_mode,
                "top_k": top_k,
                "timings_ms": timings,
                "retrieval": {
                    "hit_count": len(hits),
                    "min_score_threshold": self._config.min_score,
                    "min_chunks_threshold": self._config.min_chunks,
                    "scores": [round(hit.score, 4) for hit in hits],
                    "selected_chunk_uids": [hit.chunk.chunk_uid for hit in hits],
                    "used_chunk_uids": used_chunk_uids,
                },
            }

        citation_lookup = {citation.chunk_uid: citation for citation in citations}
        ordered_citations = [citation_lookup[uid] for uid in used_chunk_uids if uid in citation_lookup]
        if not ordered_citations:
            ordered_citations = citations

        return QueryResponse(
            answer=answer,
            citations=ordered_citations,
            retrieved_chunks=len(hits),
            confidence=confidence,
            refusal_reason=None,
            debug_info=response_debug,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 2)

    def _evaluate_evidence(self, hits: list[RetrievalHit]) -> dict[str, object]:
        if len(hits) < self._config.min_chunks:
            return {
                "hit_count": len(hits),
                "best_score": hits[0].score if hits else None,
                "refusal_reason": "insufficient_chunks",
            }

        best_score = hits[0].score if hits else -1.0
        if best_score < self._config.min_score:
            return {
                "hit_count": len(hits),
                "best_score": best_score,
                "refusal_reason": "low_retrieval_score",
            }

        return {
            "hit_count": len(hits),
            "best_score": best_score,
            "refusal_reason": None,
        }

    def _build_generative_answer(self, question: str, hits: list[RetrievalHit]) -> tuple[str, list[str]]:
        usable_hits = hits[: min(5, len(hits))]
        used_chunk_uids = [hit.chunk.chunk_uid for hit in usable_hits]

        if not self._config.generation.enabled or not self._config.generation.model:
            fallback_answer, _ = self._build_extractive_answer(usable_hits)
            return (
                "Generative mode is unavailable because no model configuration is enabled. "
                "Returning extractive evidence summary instead.\n"
                f"{fallback_answer}",
                used_chunk_uids,
            )

        evidence_lines = [
            f"[{idx}] {hit.chunk.text.strip()} (chunk_uid={hit.chunk.chunk_uid})"
            for idx, hit in enumerate(usable_hits, start=1)
        ]
        evidence_block = "\n".join(evidence_lines)
        prompt = (
            "Answer the question using only the evidence snippets below. "
            "If the evidence is not enough, explicitly say so.\n\n"
            f"Question: {question}\n\n"
            "Evidence:\n"
            f"{evidence_block}\n\n"
            "Return a concise answer and include references in the form [n]."
        )

        payload = {
            "model": self._config.generation.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._config.generation.temperature},
        }

        http_request = request.Request(
            self._config.generation.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self._config.generation.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                generated = str(body.get("response", "")).strip()
        except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            generated = ""

        if not generated:
            fallback_answer, _ = self._build_extractive_answer(usable_hits)
            return (
                "Generative answer failed to render from the configured model. "
                "Returning extractive evidence summary instead.\n"
                f"{fallback_answer}",
                used_chunk_uids,
            )

        return generated, used_chunk_uids

    @staticmethod
    def _build_citations(hits: list[RetrievalHit]) -> list[Citation]:
        return [
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

    @staticmethod
    def _build_extractive_answer(hits: list[RetrievalHit]) -> tuple[str, list[str]]:
        lines = ["Evidence-based answer (extractive preview):"]
        used_chunk_uids: list[str] = []
        for position, hit in enumerate(hits[:3], start=1):
            snippet = " ".join(hit.chunk.text.split())[:220]
            lines.append(f"{position}. {snippet} (source: {hit.chunk.chunk_uid})")
            used_chunk_uids.append(hit.chunk.chunk_uid)
        return "\n".join(lines), used_chunk_uids

    def _compute_confidence(self, hits: list[RetrievalHit]) -> float:
        if not hits:
            return 0.0

        top_scores = [max(0.0, hit.score) for hit in hits[:3]]
        if not top_scores:
            return 0.0

        average_score = sum(top_scores) / len(top_scores)
        score_component = min(1.0, average_score)
        chunk_component = min(1.0, len(hits) / max(1, self._config.min_chunks * 2))
        return round((0.7 * score_component) + (0.3 * chunk_component), 3)

    def _refusal_response(
        self,
        *,
        reason: str,
        debug: bool,
        debug_context: dict[str, object],
    ) -> QueryResponse:
        return QueryResponse(
            answer="I don't have enough reliable evidence in the index to answer this question safely.",
            citations=[],
            retrieved_chunks=0,
            confidence=0.0,
            refusal_reason=reason,
            debug_info=debug_context if debug else None,
        )
