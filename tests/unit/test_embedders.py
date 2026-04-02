from __future__ import annotations

import unittest

from app.llm.embeddings import (
    EmbedderConfig,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)


class EmbedderFactoryTests(unittest.TestCase):
    def test_factory_defaults_to_hash_embedder(self) -> None:
        embedder = build_embedder()
        self.assertIsInstance(embedder, HashingEmbedder)

    def test_factory_explicit_hash_embedder(self) -> None:
        embedder = build_embedder(provider="hash", dimension=64)
        self.assertIsInstance(embedder, HashingEmbedder)
        self.assertEqual(64, embedder.dimension)

    def test_factory_from_config_hash_embedder(self) -> None:
        config = EmbedderConfig(provider="hash", dimension=32)
        embedder = build_embedder(config)
        self.assertIsInstance(embedder, HashingEmbedder)
        self.assertEqual(32, embedder.dimension)

    def test_factory_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_embedder(provider="unknown")

    def test_sentence_transformer_provider_missing_dependency_error(self) -> None:
        try:
            embedder = build_embedder(provider="sentence_transformer")
            self.assertIsInstance(embedder, SentenceTransformerEmbedder)
        except ImportError as exc:
            self.assertIn("sentence-transformers", str(exc))


if __name__ == "__main__":
    unittest.main()
