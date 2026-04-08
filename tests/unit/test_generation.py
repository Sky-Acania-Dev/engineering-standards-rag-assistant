from __future__ import annotations

import unittest

from app.llm.generation import ExtractiveGenerator, GeneratorConfig, OllamaGenerator, build_generator


class GeneratorFactoryTests(unittest.TestCase):
    def test_disabled_generation_uses_extractive_generator(self) -> None:
        generator = build_generator(GeneratorConfig(enabled=False, provider="ollama"))
        self.assertIsInstance(generator, ExtractiveGenerator)

    def test_ollama_generation_requires_model(self) -> None:
        with self.assertRaises(ValueError):
            build_generator(GeneratorConfig(enabled=True, provider="ollama", model=None))

    def test_ollama_generation_builder(self) -> None:
        generator = build_generator(
            GeneratorConfig(enabled=True, provider="ollama", model="llama3.1", endpoint="http://localhost:11434/api/generate")
        )
        self.assertIsInstance(generator, OllamaGenerator)


if __name__ == "__main__":
    unittest.main()
