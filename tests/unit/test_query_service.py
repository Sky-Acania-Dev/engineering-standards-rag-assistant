from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.api.schemas.query import QueryRequest
from app.services.query_service import QueryService
from scripts.build_index import build_index


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

            service = QueryService.from_index_dir(output_dir)
            response = service.query(QueryRequest(question="Any requirement?", top_k=5))

            self.assertEqual(0, response.retrieved_chunks)
            self.assertIn("could not find supporting evidence", response.answer)
            self.assertEqual([], response.citations)


if __name__ == "__main__":
    unittest.main()
