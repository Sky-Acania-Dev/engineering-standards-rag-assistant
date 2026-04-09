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
    content_type: str = "body_text"
    section_path: tuple[str, ...] = ()
    table_id: str | None = None
    figure_id: str | None = None
    figure_ref: str | None = None
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None


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
            page=chunk.page if chunk.page is not None else page,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
            content_type=chunk.content_type,
            section_path=chunk.section_path,
            table_id=chunk.table_id,
            figure_id=chunk.figure_id,
            figure_ref=chunk.figure_ref,
            prev_chunk_id=chunk.prev_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
        )
        for chunk in chunks
    ]
