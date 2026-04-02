from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Protocol


class Embedder(Protocol):
    """Minimal embedder contract used by indexing and retrieval pipelines."""

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class EmbedderConfig:
    provider: str = "hash"
    dimension: int = 256
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class HashingEmbedder:
    """Dependency-free baseline embedding for local indexing workflows."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dimension)]
        for token in text.split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vector[bucket] += sign
        return vector


class SentenceTransformerEmbedder:
    """Embedder backed by sentence-transformers with lazy dependency loading."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "sentence-transformer embedder requires 'sentence-transformers'. "
                "Install it with `pip install sentence-transformers`."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        embeddings = self._model.encode(text_list, convert_to_numpy=True, normalize_embeddings=False)
        return [[float(value) for value in row] for row in embeddings.tolist()]



def build_embedder(
    config: EmbedderConfig | None = None,
    *,
    provider: str | None = None,
    dimension: int | None = None,
    model_name: str | None = None,
) -> Embedder:
    """Factory for constructing embedders from a provider key/config."""
    resolved = config or EmbedderConfig()

    selected_provider = provider or resolved.provider
    selected_dimension = dimension if dimension is not None else resolved.dimension
    selected_model = model_name or resolved.model_name

    if selected_provider == "hash":
        return HashingEmbedder(dimension=selected_dimension)

    if selected_provider == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name=selected_model)

    raise ValueError(
        "Unknown embedder provider "
        f"'{selected_provider}'. Supported providers: 'hash', 'sentence_transformer'."
    )
