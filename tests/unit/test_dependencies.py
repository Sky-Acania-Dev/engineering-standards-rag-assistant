from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.api.dependencies import get_dependencies, get_query_service
from scripts.build_index import build_index


class DependenciesTests(unittest.TestCase):
    def setUp(self) -> None:
        get_dependencies.cache_clear()
        get_query_service.cache_clear()

    def tearDown(self) -> None:
        get_dependencies.cache_clear()
        get_query_service.cache_clear()
        for key in [
            "INDEX_DIR",
            "EMBEDDER_PROVIDER",
            "EMBEDDING_MODEL",
            "QUERY_GENERATION_PROVIDER",
            "QUERY_GENERATION_ENABLED",
            "QUERY_GENERATION_MODEL",
        ]:
            os.environ.pop(key, None)

    def test_startup_dependency_fails_on_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "index"
            input_dir.mkdir(parents=True)
            (input_dir / "sample.txt").write_text("Arc flash labeling is required.", encoding="utf-8")
            build_index(input_dir=str(input_dir), output_dir=str(output_dir), embedding_dimension=32, embedder_type="hash")

            os.environ["INDEX_DIR"] = str(output_dir)
            os.environ["EMBEDDER_PROVIDER"] = "sentence_transformer"

            with self.assertRaises(ValueError):
                get_dependencies()


if __name__ == "__main__":
    unittest.main()
