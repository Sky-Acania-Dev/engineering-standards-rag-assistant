from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.api.schemas.query import QueryRequest
from app.llm.generation import GenerationRequest, GenerationResult
from app.services.query_service import QueryService, QueryServiceConfig
from scripts.build_index import build_index


class _FailingGenerator:
    provider = "ollama"

    def generate(self, payload: GenerationRequest) -> GenerationResult:  # noqa: ARG002
        return GenerationResult(text="", used_generator=False, fallback_reason="ollama_unreachable")


class _WorkingGenerator:
    provider = "ollama"

    def generate(self, payload: GenerationRequest) -> GenerationResult:
        return GenerationResult(text=f"Generated: {payload.question}", used_generator=True)


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
                config=QueryServiceConfig(default_mode="generative", min_score=-1.0),
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
                config=QueryServiceConfig(default_mode="generative", min_score=-1.0),
                generator=_WorkingGenerator(),
            )
            response = service.query(QueryRequest(question="What requires labeling?", mode="generative", debug=True))

            self.assertIn("Generated:", response.answer)
            assert response.debug_info is not None
            self.assertTrue(response.debug_info["generation"]["used"])
            self.assertFalse(response.debug_info["generation"]["fallback_used"])


if __name__ == "__main__":
    unittest.main()
