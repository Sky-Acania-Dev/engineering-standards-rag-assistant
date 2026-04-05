from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class QueryRequest:
    question: str
    top_k: int | None = None
    mode: Literal["extractive", "generative"] = "extractive"
    debug: bool = False


@dataclass(frozen=True)
class Citation:
    chunk_uid: str
    doc_id: str
    title: str
    section: str
    chunk_id: int
    page: int | None
    score: float


@dataclass(frozen=True)
class QueryResponse:
    answer: str
    citations: list[Citation]
    retrieved_chunks: int
    confidence: float
    refusal_reason: str | None = None
    debug_info: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
            "retrieved_chunks": self.retrieved_chunks,
            "confidence": self.confidence,
            "refusal_reason": self.refusal_reason,
        }
        if self.debug_info is not None:
            payload["debug_info"] = self.debug_info
        return payload
