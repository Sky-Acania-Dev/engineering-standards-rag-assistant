from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.rag.chunking import TextChunk


@dataclass(frozen=True)
class ChunkMetadata:
    doc_id: str
    title: str
    page_start: int | None
    page_end: int | None
    section: str
    chunk_id: int
    content_type: str = "body_text"
    section_path: tuple[str, ...] = ()
    table_id: str | None = None
    figure_id: str | None = None
    figure_ref: str | None = None
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None
    footnotes: tuple[dict[str, Any], ...] = ()


def build_chunk_metadata(
    *,
    doc_id: str,
    title: str,
    chunks: Iterable[TextChunk],
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[ChunkMetadata]:
    return [
        ChunkMetadata(
            doc_id=doc_id,
            title=title,
            page_start=chunk.page_start if chunk.page_start is not None else page_start,
            page_end=chunk.page_end if chunk.page_end is not None else page_end,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
            content_type=chunk.content_type,
            section_path=chunk.section_path,
            table_id=chunk.table_id,
            figure_id=chunk.figure_id,
            figure_ref=chunk.figure_ref,
            prev_chunk_id=chunk.prev_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
            footnotes=chunk.footnotes,
        )
        for chunk in chunks
    ]
