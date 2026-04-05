from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QueryRequest:
    question: str
    top_k: int = 5


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

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
            "retrieved_chunks": self.retrieved_chunks,
        }
