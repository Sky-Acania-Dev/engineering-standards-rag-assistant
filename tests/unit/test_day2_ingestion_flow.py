import unittest

from app.ingestion.normalize import normalize_ingested_text
from app.ingestion.pipeline import IngestionDocument, ingest_documents
from app.rag.chunking import chunk_document_by_section


class Day2IngestionFlowTests(unittest.TestCase):
    def test_ingestion_fail_fast_raises_on_parser_error(self) -> None:
        documents = [
            IngestionDocument(doc_id="ok", title="OK", raw_text="Section 1\n\nalpha beta"),
            IngestionDocument(doc_id="bad", title="BAD", raw_text="BAD_PDF"),
        ]

        def parser(raw_text: str) -> str:
            if raw_text == "BAD_PDF":
                raise ValueError("parse failed")
            return raw_text

        with self.assertRaises(ValueError):
            ingest_documents(documents, parser=parser, normalizer=lambda t: t, fail_fast=True)

    def test_ingestion_log_and_skip_continues_on_parser_error(self) -> None:
        documents = [
            IngestionDocument(doc_id="bad", title="BAD", raw_text="BAD_PDF"),
            IngestionDocument(doc_id="ok", title="OK", raw_text="Section 1\n\nalpha beta gamma"),
        ]

        def parser(raw_text: str) -> str:
            if raw_text == "BAD_PDF":
                raise ValueError("parse failed")
            return raw_text

        with self.assertLogs("app.ingestion.pipeline", level="WARNING") as logs:
            results = ingest_documents(documents, parser=parser, normalizer=lambda t: t, fail_fast=False)

        self.assertEqual([result.doc_id for result in results], ["ok"])
        self.assertIn("Skipping document due to ingestion error", "\n".join(logs.output))

    def test_chunking_uses_section_boundaries_target_and_overlap(self) -> None:
        sec1_tokens = [f"a{i}" for i in range(900)]
        sec2_tokens = [f"b{i}" for i in range(900)]
        document = (
            "Chapter 1 Intro\n\n" + " ".join(sec1_tokens) + "\n\n" + "Section 2.0 Limits\n\n" + " ".join(sec2_tokens)
        )

        chunks = chunk_document_by_section(document, chunk_size=800, overlap=150)

        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(all(chunk.section in {"Chapter 1 Intro", "Section 2.0 Limits"} for chunk in chunks))
        self.assertTrue(all(chunk.token_count <= 800 for chunk in chunks))

        chapter_chunks = [c for c in chunks if c.section == "Chapter 1 Intro"]
        self.assertGreaterEqual(len(chapter_chunks), 2)
        first_tokens = chapter_chunks[0].text.split()
        second_tokens = chapter_chunks[1].text.split()
        self.assertEqual(first_tokens[-150:], second_tokens[:150])

    def test_chunking_avoids_splitting_mid_table_or_list(self) -> None:
        prefix = " ".join(f"p{i}" for i in range(780))
        table_block = "\n".join([
            "| col_a | col_b |",
            "| --- | --- |",
            "| value_1 | value_2 |",
            "| value_3 | value_4 |",
        ])
        suffix = " ".join(f"s{i}" for i in range(100))
        document = f"Section 1\n\n{prefix}\n\n{table_block}\n\n{suffix}"

        chunks = chunk_document_by_section(document, chunk_size=800, overlap=150)

        table_holders = [chunk for chunk in chunks if "| col_a | col_b |" in chunk.text]
        self.assertEqual(len(table_holders), 1)
        holder_text = table_holders[0].text
        self.assertIn("| value_1 | value_2 |", holder_text)
        self.assertIn("| value_3 | value_4 |", holder_text)

    def test_ingestion_metadata_includes_page_and_content_type(self) -> None:
        documents = [
            IngestionDocument(
                doc_id="doc-1",
                title="Doc 1",
                raw_text="""## Page 3

# Title

Body text here.

[IMAGE] Figure asset (src=figs/f1.png)

[IMAGE_CAPTION] Figure 1. Sample""",
            )
        ]

        results = ingest_documents(documents, parser=lambda t: t, normalizer=lambda t: t, fail_fast=True, chunk_size=80, overlap=10)
        self.assertEqual(1, len(results))
        metadata = results[0].metadata

        caption_meta = [m for m in metadata if m.content_type == "figure_caption"]
        self.assertEqual(1, len(caption_meta))
        self.assertEqual(3, caption_meta[0].page)
        self.assertEqual("Figure 1", caption_meta[0].figure_id)
        self.assertIsNotNone(caption_meta[0].figure_ref)

    def test_chunking_treats_numbered_lists_as_single_protected_chunk(self) -> None:
        prefix = " ".join(f"x{i}" for i in range(790))
        numbered_list = "\n".join([
            "1. first requirement",
            "2. second requirement",
            "3. third requirement",
        ])
        suffix = " ".join(f"y{i}" for i in range(60))
        document = f"Section 7.1 Rules\n\n{prefix}\n\n{numbered_list}\n\n{suffix}"

        chunks = chunk_document_by_section(document, chunk_size=800, overlap=150)

        list_chunks = [chunk for chunk in chunks if "1. first requirement" in chunk.text]
        self.assertEqual(len(list_chunks), 1)
        self.assertIn("2. second requirement", list_chunks[0].text)
        self.assertIn("3. third requirement", list_chunks[0].text)

    def test_ingestion_keeps_image_only_pdf_pages_chunkable(self) -> None:
        documents = [
            IngestionDocument(
                doc_id="scan.pdf",
                title="scan.pdf",
                raw_text="## Page 1\n\n[IMAGES] 2 embedded image(s)",
            )
        ]

        results = ingest_documents(
            documents,
            parser=lambda t: t,
            normalizer=normalize_ingested_text,
            fail_fast=True,
            chunk_size=80,
            overlap=10,
        )

        self.assertEqual(1, len(results))
        self.assertGreaterEqual(len(results[0].chunks), 1)
        self.assertTrue(any("[IMAGES]" in chunk.text for chunk in results[0].chunks))

    def test_ingestion_keeps_repeated_image_markers_across_pages(self) -> None:
        documents = [
            IngestionDocument(
                doc_id="scan-multipage.pdf",
                title="scan-multipage.pdf",
                raw_text=(
                    "## Page 1\n\n[IMAGES] 1 embedded image(s)\n\n"
                    "## Page 2\n\n[IMAGES] 1 embedded image(s)\n\n"
                    "## Page 3\n\n[IMAGES] 1 embedded image(s)"
                ),
            )
        ]

        results = ingest_documents(
            documents,
            parser=lambda t: t,
            normalizer=normalize_ingested_text,
            fail_fast=True,
            chunk_size=80,
            overlap=10,
        )

        self.assertEqual(1, len(results))
        self.assertGreaterEqual(len(results[0].chunks), 1)
        self.assertTrue(any("[IMAGES]" in chunk.text for chunk in results[0].chunks))

    def test_chunking_extracts_page_content_type_and_paths(self) -> None:
        document = """## Page 1

# Standard A

Intro paragraph about controls.

- first bullet
- second bullet

[IMAGE] System Diagram (src=figs/diag-1.png)

[IMAGE_CAPTION] Figure 2. Layout overview

[TABLE]
[TABLE_CAPTION] Table 3. Test limits
| Parameter | Limit |
| --- | --- |
| Voltage | 120V |
[/TABLE]

## Page 2

## Section 2

Note: Keep spacing.
"""

        chunks = chunk_document_by_section(document, chunk_size=120, overlap=20)

        figure_chunks = [c for c in chunks if c.content_type == "figure_caption"]
        self.assertEqual(1, len(figure_chunks))
        self.assertEqual(1, figure_chunks[0].page)
        self.assertEqual("Figure 2", figure_chunks[0].figure_id)
        self.assertIn("src=figs/diag-1.png", figure_chunks[0].figure_ref or "")

        table_chunks = [c for c in chunks if c.content_type == "table"]
        self.assertEqual(1, len(table_chunks))
        self.assertIn("Table caption: Table 3. Test limits", table_chunks[0].text)
        self.assertIn("Table columns: Parameter; Limit", table_chunks[0].text)
        self.assertIn("Row 2: Parameter=Voltage; Limit=120V", table_chunks[0].text)

        note_chunks = [c for c in chunks if c.content_type == "note"]
        self.assertEqual(1, len(note_chunks))
        self.assertEqual(2, note_chunks[0].page)

        body_chunks = [c for c in chunks if c.content_type == "body_text"]
        self.assertTrue(any(c.section_path for c in body_chunks))
        self.assertTrue(any(c.prev_chunk_id is not None or c.next_chunk_id is not None for c in chunks))


if __name__ == "__main__":
    unittest.main()
