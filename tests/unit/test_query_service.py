from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.schemas.query import QueryRequest
from app.llm.generation import GenerationRequest, GenerationResult
from app.llm.embeddings import EmbedderSpec
from app.rag.retrieval.hybrid import RetrievalHit
from app.services.query_service import QueryService, QueryServiceConfig
from app.services.query_service import QueryArtifacts
from app.stores.docstore import JsonlChunkStore, StoredChunk
from app.stores.faiss_store import FaissStore
from scripts.build_index import build_index


class _FailingGenerator:
    provider = "ollama"

    def generate(self, payload: GenerationRequest) -> GenerationResult:  # noqa: ARG002
        return GenerationResult(text="", used_generator=False, fallback_reason="ollama_unreachable")


class _WorkingGenerator:
    provider = "ollama"

    def generate(self, payload: GenerationRequest) -> GenerationResult:
        return GenerationResult(text=f"Generated: {payload.question}", used_generator=True)


class _StubSentenceTransformerEmbedder:
    @property
    def spec(self) -> EmbedderSpec:
        return EmbedderSpec(provider="sentence_transformer", model_name="stub", dimension=3)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        return [[0.1, 0.2, 0.3]]


class QueryServiceTests(unittest.TestCase):
    def test_query_returns_extractive_answer_with_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "index"
            input_dir.mkdir(parents=True)
            (input_dir / "electrical.txt").write_text(
                "Chapter 1 Intro\n\nGrounding conductors must be bonded to the main service panel.",
                encoding="utf-8",
            )
            build_index(input_dir=str(input_dir), output_dir=str(output_dir), embedding_dimension=64)

            service = QueryService.from_index_dir(output_dir)
            response = service.query(QueryRequest(question="What should be bonded to the main service panel?", top_k=3))

            self.assertGreaterEqual(response.retrieved_chunks, 1)
            self.assertTrue(response.citations)
            self.assertIn("extractive preview", response.answer.lower())

    def test_query_handles_empty_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "index"
            output_dir.mkdir(parents=True)
            (output_dir / "chunk_index.json").write_text(
                '{"dimension": 8, "ids": [], "vectors": []}',
                encoding="utf-8",
            )
            (output_dir / "chunk_store.jsonl").write_text("", encoding="utf-8")
            (output_dir / "index_manifest.json").write_text(
                json.dumps(
                    {
                        "documents": 0,
                        "chunks": 0,
                        "embedding_dimension": 8,
                        "embedder_provider": "hash",
                        "embedding_model": None,
                    }
                ),
                encoding="utf-8",
            )

            service = QueryService.from_index_dir(output_dir)
            response = service.query(QueryRequest(question="Any requirement?", top_k=5))

            self.assertEqual(0, response.retrieved_chunks)
            self.assertIn("don't have enough reliable evidence", response.answer.lower())
            self.assertEqual("insufficient_chunks", response.refusal_reason)
            self.assertEqual([], response.citations)

    def test_embedder_manifest_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "index"
            input_dir.mkdir(parents=True)
            (input_dir / "sample.txt").write_text("Grounding is required for service entrances.", encoding="utf-8")
            build_index(input_dir=str(input_dir), output_dir=str(output_dir), embedding_dimension=32)

            with self.assertRaises(ValueError):
                QueryService.from_index_dir(output_dir, embedder_provider="sentence_transformer")

    def test_generative_mode_falls_back_when_generator_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "index"
            input_dir.mkdir(parents=True)
            (input_dir / "sample.txt").write_text("Bonding conductors are required.", encoding="utf-8")
            build_index(input_dir=str(input_dir), output_dir=str(output_dir), embedding_dimension=64)

            service = QueryService.from_index_dir(
                output_dir,
                config=QueryServiceConfig(
                    default_mode="generative",
                    min_score_thresholds={"default": -1.0, "sentence_transformer": -1.0},
                ),
                generator=_FailingGenerator(),
            )
            response = service.query(QueryRequest(question="What is required?", mode="generative", debug=True))

            self.assertIsNone(response.refusal_reason)
            assert response.debug_info is not None
            generation_debug = response.debug_info["generation"]
            self.assertEqual("ollama_unreachable", generation_debug["fallback_reason"])
            self.assertTrue(generation_debug["fallback_used"])
            self.assertFalse(generation_debug["used"])

    def test_generative_mode_uses_generator_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "index"
            input_dir.mkdir(parents=True)
            (input_dir / "sample.txt").write_text("Switchgear requires labeling.", encoding="utf-8")
            build_index(input_dir=str(input_dir), output_dir=str(output_dir), embedding_dimension=64)

            service = QueryService.from_index_dir(
                output_dir,
                config=QueryServiceConfig(
                    default_mode="generative",
                    min_score_thresholds={"default": -1.0, "sentence_transformer": -1.0},
                ),
                generator=_WorkingGenerator(),
            )
            response = service.query(QueryRequest(question="What requires labeling?", mode="generative", debug=True))

            self.assertIn("Generated:", response.answer)
            assert response.debug_info is not None
            self.assertTrue(response.debug_info["generation"]["used"])
            self.assertFalse(response.debug_info["generation"]["fallback_used"])

    def test_sentence_transformer_uses_stricter_min_score_for_refusal(self) -> None:
        artifacts = QueryArtifacts(index=FaissStore(dimension=3), docstore=JsonlChunkStore(), manifest={})
        service = QueryService(
            artifacts=artifacts,
            embedder=_StubSentenceTransformerEmbedder(),
            generator=_WorkingGenerator(),
            config=QueryServiceConfig(min_score_thresholds={"default": 0.2, "sentence_transformer": 0.4}),
        )

        low_score_hit = RetrievalHit(
            chunk=StoredChunk(
                chunk_uid="doc:1",
                text="Section 1 wiring requirements",
                doc_id="doc",
                title="doc",
                section="Section 1",
                chunk_id=1,
            ),
            score=0.35,
        )

        with patch("app.services.query_service.retrieve_top_k", return_value=[low_score_hit]):
            response = service.query(
                QueryRequest(question="What is the population of Mars City in 2040?", top_k=5, debug=True)
            )

        self.assertEqual("low_retrieval_score", response.refusal_reason)
        self.assertEqual([], response.citations)
        self.assertEqual(0.0, response.confidence)
        assert response.debug_info is not None
        self.assertEqual(0.4, response.debug_info["retrieval"]["min_score_threshold"])


if __name__ == "__main__":
    unittest.main()
