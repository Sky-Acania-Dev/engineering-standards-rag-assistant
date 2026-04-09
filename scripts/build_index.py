from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.parsers.html import ingest_html_folder
from app.ingestion.parsers.pdf import ingest_pdf_folder
from app.ingestion.parsers.txt import ingest_txt_folder
from app.ingestion.pipeline import IngestionDocument, ingest_documents
from app.ingestion.normalize import normalize_ingested_text
from app.llm.embeddings import Embedder, build_embedder
from app.stores.docstore import JsonlChunkStore, StoredChunk
from app.stores.faiss_store import FaissStore


def _stable_chunk_uid(doc_id: str, chunk_id: int) -> str:
    return f"{doc_id}:{chunk_id}"


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
    embedder_type: str = "hash",
    embedding_model: str | None = None,
    embedder: Embedder | None = None,
) -> dict[str, int | str]:
    documents = _collect_documents(input_dir)
    ingestion_results = ingest_documents(
        documents,
        parser=lambda text: text,
        normalizer=normalize_ingested_text,
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
            if chunk.content_type != "image_artifact":
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
                    page_start=metadata.page_start,
                    page_end=metadata.page_end,
                    content_type=metadata.content_type,
                    section_path=metadata.section_path,
                    table_id=metadata.table_id,
                    figure_id=metadata.figure_id,
                    figure_ref=metadata.figure_ref,
                    prev_chunk_id=metadata.prev_chunk_id,
                    next_chunk_id=metadata.next_chunk_id,
                )
            )

    resolved_embedder = embedder or build_embedder(
        provider=embedder_type,
        dimension=embedding_dimension,
        model_name=embedding_model,
    )
    vectors = resolved_embedder.embed_texts(chunk_texts)

    vector_dimension = len(vectors[0]) if vectors else embedding_dimension
    index_store = FaissStore(dimension=vector_dimension)
    index_store.add(chunk_ids, vectors)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_store.save(output_path / "chunk_index.json")

    docstore = JsonlChunkStore()
    docstore.upsert_many(stored_chunks)
    docstore.save(output_path / "chunk_store.jsonl")

    stats: dict[str, int | str | None] = {
        "documents": len(ingestion_results),
        "chunks": len(chunk_ids),
        "stored_chunks": len(stored_chunks),
        "embedding_dimension": vector_dimension,
        "embedder_provider": resolved_embedder.spec.provider,
        "embedding_model": resolved_embedder.spec.model_name,
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
    parser.add_argument("--embedder", default="hash", help="Embedder provider (hash | sentence_transformer)")
    parser.add_argument("--embedding-model", default=None, help="Optional model name for model-based embedders")
    args = parser.parse_args()

    stats = build_index(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_dimension=args.embedding_dim,
        embedder_type=args.embedder,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
