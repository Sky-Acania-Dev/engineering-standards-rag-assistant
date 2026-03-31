from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.rag.chunking import TextChunk


@dataclass(frozen=True)
class ChunkMetadata:
    doc_id: str
    title: str
    page: int | None
    section: str
    chunk_id: int


def build_chunk_metadata(
    *,
    doc_id: str,
    title: str,
    chunks: Iterable[TextChunk],
    page: int | None = None,
) -> list[ChunkMetadata]:
    return [
        ChunkMetadata(
            doc_id=doc_id,
            title=title,
            page=page,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
        )
        for chunk in chunks
    ]
