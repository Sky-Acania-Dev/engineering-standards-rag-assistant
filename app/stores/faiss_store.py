from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ScoredChunk:
    chunk_uid: str
    score: float


class FaissStore:
    """Vector index API with pure-Python cosine similarity backend."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids length must match vectors rows")

        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(f"vectors must have shape (n, {self.dimension})")

        self._ids.extend(chunk_ids)
        self._vectors.extend(self._normalize([float(v) for v in vector]) for vector in vectors)

    def search(self, query_vector: list[float], *, k: int = 5) -> list[ScoredChunk]:
        if not self._ids:
            return []
        if len(query_vector) != self.dimension:
            raise ValueError(f"query_vector must have shape ({self.dimension},)")

        normalized_query = self._normalize([float(v) for v in query_vector])
        scored = []
        for index, vector in enumerate(self._vectors):
            score = sum(a * b for a, b in zip(vector, normalized_query))
            scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            ScoredChunk(chunk_uid=self._ids[index], score=float(score))
            for index, score in scored[: max(k, 0)]
        ]

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimension": self.dimension,
            "ids": self._ids,
            "vectors": self._vectors,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, input_path: str | Path) -> FaissStore:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Index file does not exist: {input_path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        store = cls(dimension=int(payload["dimension"]))
        ids = [str(chunk_uid) for chunk_uid in payload["ids"]]
        vectors = [[float(value) for value in vector] for vector in payload["vectors"]]

        if len(ids) != len(vectors):
            raise ValueError("Stored ids count does not match vectors rows")

        store._ids = ids
        store._vectors = vectors
        return store
