from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from app.rag.chunking import TextChunk, chunk_document_by_section
from app.stores.metadata_store import ChunkMetadata, build_chunk_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionDocument:
    doc_id: str
    title: str
    raw_text: str


@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    chunks: list[TextChunk]
    metadata: list[ChunkMetadata]


def ingest_documents(
    documents: list[IngestionDocument],
    *,
    parser: Callable[[str], str],
    normalizer: Callable[[str], str],
    fail_fast: bool,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[IngestionResult]:
    results: list[IngestionResult] = []

    for document in documents:
        try:
            parsed_text = parser(document.raw_text)
            normalized_text = normalizer(parsed_text)
            chunks = chunk_document_by_section(normalized_text, chunk_size=chunk_size, overlap=overlap)
            metadata = build_chunk_metadata(doc_id=document.doc_id, title=document.title, chunks=chunks)
            results.append(IngestionResult(doc_id=document.doc_id, chunks=chunks, metadata=metadata))
        except Exception:
            if fail_fast:
                raise
            logger.warning("Skipping document due to ingestion error: doc_id=%s", document.doc_id, exc_info=True)

    return results
