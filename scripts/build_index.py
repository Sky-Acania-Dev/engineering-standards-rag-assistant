from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.parsers.html import ingest_html_folder
from app.ingestion.parsers.pdf import ingest_pdf_folder
from app.ingestion.parsers.txt import ingest_txt_folder
from app.ingestion.pipeline import IngestionDocument, ingest_documents
from app.stores.docstore import JsonlChunkStore, StoredChunk
from app.stores.faiss_store import FaissStore


def _normalize_text(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def _stable_chunk_uid(doc_id: str, chunk_id: int) -> str:
    return f"{doc_id}:{chunk_id}"


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


def _collect_documents(input_dir: str) -> list[IngestionDocument]:
    txt_docs = ingest_txt_folder(input_dir)
    html_docs = ingest_html_folder(input_dir)
    pdf_docs = ingest_pdf_folder(input_dir)

    raw_docs = txt_docs + html_docs + pdf_docs
    return [
        IngestionDocument(
            doc_id=doc["filename"],
            title=doc["filename"],
            raw_text=doc["content"],
        )
        for doc in raw_docs
    ]


def build_index(
    *,
    input_dir: str,
    output_dir: str,
    chunk_size: int = 800,
    overlap: int = 150,
    embedding_dimension: int = 256,
) -> dict[str, int]:
    documents = _collect_documents(input_dir)
    ingestion_results = ingest_documents(
        documents,
        parser=lambda text: text,
        normalizer=_normalize_text,
        fail_fast=True,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    stored_chunks: list[StoredChunk] = []

    for result in ingestion_results:
        metadata_by_chunk_id = {entry.chunk_id: entry for entry in result.metadata}
        for chunk in result.chunks:
            metadata = metadata_by_chunk_id[chunk.chunk_id]
            chunk_uid = _stable_chunk_uid(result.doc_id, chunk.chunk_id)
            chunk_ids.append(chunk_uid)
            chunk_texts.append(chunk.text)
            stored_chunks.append(
                StoredChunk(
                    chunk_uid=chunk_uid,
                    text=chunk.text,
                    doc_id=metadata.doc_id,
                    title=metadata.title,
                    section=metadata.section,
                    chunk_id=metadata.chunk_id,
                    page=metadata.page,
                )
            )

    embedder = HashingEmbedder(dimension=embedding_dimension)
    vectors = embedder.embed_texts(chunk_texts)

    index_store = FaissStore(dimension=embedding_dimension)
    index_store.add(chunk_ids, vectors)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_store.save(output_path / "chunk_index.json")

    docstore = JsonlChunkStore()
    docstore.upsert_many(stored_chunks)
    docstore.save(output_path / "chunk_store.jsonl")

    stats = {
        "documents": len(ingestion_results),
        "chunks": len(chunk_ids),
        "embedding_dimension": embedding_dimension,
    }
    with (output_path / "index_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk index and store artifacts.")
    parser.add_argument("--input", required=True, help="Input folder containing txt/html/pdf files")
    parser.add_argument("--output", required=True, help="Output folder for index artifacts")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--embedding-dim", type=int, default=256)
    args = parser.parse_args()

    stats = build_index(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_dimension=args.embedding_dim,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
