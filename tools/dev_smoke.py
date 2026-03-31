from app.ingestion.parsers.pdf import ingest_pdf_folder
from app.ingestion.pipeline import IngestionDocument, ingest_documents

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