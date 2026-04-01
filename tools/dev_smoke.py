from app.ingestion.parsers.pdf import ingest_pdf_folder
from app.ingestion.pipeline import IngestionDocument, ingest_documents
from app.rag.chunking import chunk_document_by_section


def run_chunking_smoke() -> None:
    print("\n[chunking smoke]")
    prefix = " ".join(f"p{i}" for i in range(790))
    numbered_list = "\n".join([
        "1. first requirement",
        "2. second requirement",
        "3. third requirement",
    ])
    suffix = " ".join(f"s{i}" for i in range(80))
    document = f"Section 5.2 Controls\n\n{prefix}\n\n{numbered_list}\n\n{suffix}"

    chunks = chunk_document_by_section(document, chunk_size=800, overlap=150)
    list_chunks = [chunk for chunk in chunks if "1. first requirement" in chunk.text]

    print(f"chunk_count={len(chunks)}")
    print(f"list_chunk_count={len(list_chunks)}")
    if list_chunks:
        print("list_chunk_preview:", list_chunks[0].text.replace("\n", " | "))


# 1) Parse PDF files into text payloads
work_dir = "C:\\Personal Folder\\Work\\WorkRepo\\engineering-standards-rag-assistant"
temp_dir = work_dir + "\\temp\\my_pdfs"
docs = ingest_pdf_folder(temp_dir)

# 2) Convert parsed docs into pipeline input
ingestion_docs = [
    IngestionDocument(
        doc_id=d["filename"],
        title=d["filename"],
        raw_text=d["content"],
    )
    for d in docs
]

# 3) Run full ingestion pipeline
results = ingest_documents(
    ingestion_docs,
    parser=lambda text: text,      # already parsed by ingest_pdf_folder
    normalizer=lambda text: text,  # swap in real normalizer when available
    fail_fast=True,
    chunk_size=800,
    overlap=150,
)

print(f"Temp docs dir: {temp_dir}")
print(f"Parsed docs: {len(docs)}")
print(f"Ingestion results: {len(results)}")
for r in results:
    print(f"--- {r.doc_id} ---")
    print(f"chunks={len(r.chunks)} metadata={len(r.metadata)}")
    if r.chunks:
        print("first chunk preview:", r.chunks[0].text[:300].replace('\\n', ' '))

run_chunking_smoke()
