from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.stores.docstore import JsonlChunkStore, StoredChunk
from app.stores.faiss_store import FaissStore
from scripts.build_index import build_index
from app.ingestion.normalize import normalize_ingested_text


class FaissStoreTests(unittest.TestCase):
    def test_search_returns_most_similar_chunk(self) -> None:
        store = FaissStore(dimension=4)
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        store.add(["c1", "c2", "c3"], vectors)

        results = store.search([0.9, 0.1, 0.0, 0.0], k=2)

        self.assertEqual([r.chunk_uid for r in results], ["c1", "c2"])

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "chunk_index.json"
            store = FaissStore(dimension=3)
            store.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            store.save(index_path)

            restored = FaissStore.load(index_path)
            results = restored.search([0.99, 0.01, 0.0], k=1)

            self.assertEqual("a", results[0].chunk_uid)


class DocStoreTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "chunk_store.jsonl"
            store = JsonlChunkStore()
            store.upsert_many(
                [
                    StoredChunk(
                        chunk_uid="doc-1:0",
                        text="alpha",
                        doc_id="doc-1",
                        title="Doc 1",
                        section="Section 1",
                        chunk_id=0,
                    )
                ]
            )
            store.save(store_path)

            restored = JsonlChunkStore.load(store_path)
            record = restored.get("doc-1:0")

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual("alpha", record.text)

    def test_save_orders_by_numeric_chunk_id_not_uid_lexicographic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "chunk_store.jsonl"
            store = JsonlChunkStore()
            store.upsert_many(
                [
                    StoredChunk(chunk_uid="doc-1:10", text="ten", doc_id="doc-1", title="Doc", section="S", chunk_id=10),
                    StoredChunk(chunk_uid="doc-1:2", text="two", doc_id="doc-1", title="Doc", section="S", chunk_id=2),
                    StoredChunk(chunk_uid="doc-1:1", text="one", doc_id="doc-1", title="Doc", section="S", chunk_id=1),
                ]
            )
            store.save(store_path)

            lines = [json.loads(line) for line in store_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([1, 2, 10], [line["chunk_id"] for line in lines])


class NormalizeTests(unittest.TestCase):
    def test_normalize_ingested_text_removes_repeated_headers_and_page_artifacts(self) -> None:
        raw = """## Page 1

ACME STANDARD HEADER

1 | P a g e

Body line one

## Page 2

ACME STANDARD HEADER

2 | P a g e

Body line two

## Page 3

ACME STANDARD HEADER

3 | P a g e

Body line three
"""
        normalized = normalize_ingested_text(raw)

        self.assertNotIn("ACME STANDARD HEADER", normalized)
        self.assertNotIn("P a g e", normalized)
        self.assertIn("Body line one", normalized)
        self.assertIn("Body line two", normalized)
        self.assertIn("Body line three", normalized)


class BuildIndexTests(unittest.TestCase):
    def test_build_index_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir(parents=True)
            (input_dir / "sample.txt").write_text(
                "Chapter 1 Intro\n\nSafety requirements for wiring and grounding.",
                encoding="utf-8",
            )

            stats = build_index(input_dir=str(input_dir), output_dir=str(output_dir), embedding_dimension=32)

            self.assertEqual(1, stats["documents"])
            self.assertGreaterEqual(stats["chunks"], 1)
            self.assertEqual(32, stats["embedding_dimension"])
            self.assertEqual("hash", stats["embedder_provider"])
            self.assertIsNone(stats["embedding_model"])

            self.assertTrue((output_dir / "chunk_index.json").exists())
            self.assertTrue((output_dir / "chunk_store.jsonl").exists())
            self.assertTrue((output_dir / "index_manifest.json").exists())

            manifest = json.loads((output_dir / "index_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stats, manifest)

    def test_build_index_persists_table_chunks_in_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir(parents=True)
            (input_dir / "table_case.txt").write_text(
                "\n".join(
                    [
                        "## Page 7",
                        "",
                        "Chapter 10: Windows",
                        "",
                        "10.5 Windows",
                        "",
                        "Performance Measure  CZ2  CZ3  CZ4",
                        "U-Factor             0.40 0.35 0.30",
                        "SHGC                 0.25 0.25 0.25",
                    ]
                ),
                encoding="utf-8",
            )

            build_index(input_dir=str(input_dir), output_dir=str(output_dir), embedding_dimension=16)

            rows = [
                json.loads(line)
                for line in (output_dir / "chunk_store.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            table_rows = [row for row in rows if row["content_type"] == "table"]

            self.assertEqual(1, len(table_rows))
            self.assertEqual("table:p7", table_rows[0]["table_id"])
            self.assertIn("| Performance Measure | CZ2 | CZ3 | CZ4 |", table_rows[0]["text"])


if __name__ == "__main__":
    unittest.main()
