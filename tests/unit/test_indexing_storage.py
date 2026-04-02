from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.stores.docstore import JsonlChunkStore, StoredChunk
from app.stores.faiss_store import FaissStore
from scripts.build_index import build_index


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
            self.assertEqual("hash", stats["embedder_type"])

            self.assertTrue((output_dir / "chunk_index.json").exists())
            self.assertTrue((output_dir / "chunk_store.jsonl").exists())
            self.assertTrue((output_dir / "index_manifest.json").exists())

            manifest = json.loads((output_dir / "index_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stats, manifest)


if __name__ == "__main__":
    unittest.main()
