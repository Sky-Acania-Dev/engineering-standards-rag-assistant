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
        self.assertTrue(all(chunk.section in {"Chapter 1: Intro", "Section 2.0: Limits"} for chunk in chunks))
        self.assertTrue(all(chunk.token_count <= 800 for chunk in chunks))

        chapter_chunks = [c for c in chunks if c.section == "Chapter 1: Intro"]
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
        self.assertEqual(3, caption_meta[0].page_start)
        self.assertEqual(3, caption_meta[0].page_end)
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
        artifacts = [chunk for chunk in results[0].chunks if chunk.content_type == "image_artifact"]
        self.assertGreaterEqual(len(artifacts), 1)

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
        artifacts = [chunk for chunk in results[0].chunks if chunk.content_type == "image_artifact"]
        self.assertGreaterEqual(len(artifacts), 1)

    def test_chunking_keeps_body_when_page_and_heading_share_block(self) -> None:
        document = "## Page 1\n# Inline Heading\nBody line one.\nBody line two."

        chunks = chunk_document_by_section(document, chunk_size=120, overlap=20)

        self.assertGreaterEqual(len(chunks), 1)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]
        self.assertGreaterEqual(len(body_chunks), 1)
        self.assertEqual(1, body_chunks[0].page_start)
        self.assertEqual(1, body_chunks[0].page_end)
        self.assertIn("Body line one.", body_chunks[0].text)
        self.assertIn("Body line two.", body_chunks[0].text)

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
        self.assertEqual(1, figure_chunks[0].page_start)
        self.assertEqual(1, figure_chunks[0].page_end)
        self.assertEqual("Figure 2", figure_chunks[0].figure_id)
        self.assertIn("src=figs/diag-1.png", figure_chunks[0].figure_ref or "")

        table_chunks = [c for c in chunks if c.content_type == "table"]
        self.assertEqual(1, len(table_chunks))
        self.assertIn("Table caption: Table 3. Test limits", table_chunks[0].text)
        self.assertIn("Table columns: Parameter; Limit", table_chunks[0].text)
        self.assertIn("Row 2: Parameter=Voltage; Limit=120V", table_chunks[0].text)

        note_chunks = [c for c in chunks if c.content_type == "note"]
        self.assertEqual(1, len(note_chunks))
        self.assertEqual(2, note_chunks[0].page_start)
        self.assertEqual(2, note_chunks[0].page_end)

        body_chunks = [c for c in chunks if c.content_type == "body_text"]
        self.assertTrue(any(c.section_path for c in body_chunks))
        self.assertTrue(any(c.prev_chunk_id is not None or c.next_chunk_id is not None for c in chunks))

    def test_multi_page_toc_is_preserved_and_not_mixed_with_body(self) -> None:
        document = """## Page 2

Contents
Chapter 1: Administration .................................................................. 5
1.1 Definitions ............................................................................... 5

## Page 3

2.9 Landscaping for New Construction and Additions ............................. 13
Chapter 3: Foundations .................................................................... 13

## Page 5

Chapter 1: Administration and General Requirements

1.1 Definitions

The capitalized terms used herein are defined below.
"""
        chunks = chunk_document_by_section(document, chunk_size=120, overlap=20)

        toc_chunks = [chunk for chunk in chunks if chunk.content_type == "toc"]
        self.assertEqual(1, len(toc_chunks))
        self.assertIn("Contents", toc_chunks[0].text)
        self.assertIn("2.9 Landscaping for New Construction and Additions", toc_chunks[0].text)
        self.assertIn("Chapter 3: Foundations", toc_chunks[0].text)

        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]
        self.assertTrue(body_chunks)
        self.assertTrue(all("Contents" not in chunk.text for chunk in body_chunks))
        self.assertEqual(2, toc_chunks[0].page_start)
        self.assertEqual(3, toc_chunks[0].page_end)

    def test_chunking_deduplicates_repeated_cover_artifacts(self) -> None:
        document = (
            "## Page 1\n\nTexas Minimum Construction Standards [IMAGES] 1 embedded image(s)\n\n"
            "## Page 2\n\nTexas Minimum Construction Standards [IMAGES] 1 embedded image(s)\n\n"
            "## Page 3\n\nTexas Minimum Construction Standards [IMAGES] 1 embedded image(s)"
        )

        chunks = chunk_document_by_section(document, chunk_size=80, overlap=10)

        artifacts = [chunk for chunk in chunks if chunk.content_type == "image_artifact"]
        self.assertEqual(1, len(artifacts))

    def test_chunking_reconstructs_hierarchical_decimal_headings(self) -> None:
        document = """## Page 5

Chapter 1: Administration and General Requirements

1.1 Definitions

Capitalized terms are defined by program rules.
"""

        chunks = chunk_document_by_section(document, chunk_size=120, overlap=20)

        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]
        self.assertGreaterEqual(len(body_chunks), 1)
        self.assertIn("Chapter 1: Administration and General Requirements", body_chunks[0].section_path)
        self.assertIn("1.1 Definitions", body_chunks[0].section_path)
        self.assertIn("Chapter 1: Administration and General Requirements", body_chunks[0].text)
        self.assertIn("1.1 Definitions", body_chunks[0].text)

    def test_page_start_recomputes_across_pages_when_chunks_slide(self) -> None:
        page1 = " ".join(f"p1_{i}" for i in range(12))
        page2 = " ".join(f"p2_{i}" for i in range(12))
        document = f"""## Page 1

Chapter 1 Intro

{page1}

## Page 2

{page2}
"""

        chunks = chunk_document_by_section(document, chunk_size=10, overlap=4)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]

        self.assertGreaterEqual(len(body_chunks), 3)
        self.assertEqual(1, body_chunks[0].page_start)
        self.assertEqual(2, body_chunks[-1].page_end)
        self.assertTrue(any(chunk.page_start == 2 for chunk in body_chunks[1:]))

    def test_section_metadata_changes_after_heading_change(self) -> None:
        document = """## Page 1

Chapter 1 First

alpha beta gamma delta epsilon zeta eta theta iota kappa

Chapter 2 Second

lambda mu nu xi omicron pi rho sigma tau upsilon
"""

        chunks = chunk_document_by_section(document, chunk_size=8, overlap=2)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]

        self.assertTrue(any(chunk.section == "Chapter 1: First" for chunk in body_chunks))
        self.assertTrue(any(chunk.section == "Chapter 2: Second" for chunk in body_chunks))
        self.assertTrue(any("Chapter 2: Second" in chunk.section_path for chunk in body_chunks))
        chapter1_chunks = [chunk for chunk in body_chunks if chunk.section == "Chapter 1: First"]
        chapter2_chunks = [chunk for chunk in body_chunks if chunk.section == "Chapter 2: Second"]
        self.assertTrue(chapter1_chunks)
        self.assertTrue(chapter2_chunks)
        self.assertTrue(all("Chapter 2" not in chunk.text for chunk in chapter1_chunks))

    def test_new_chapter_heading_forces_chunk_boundary(self) -> None:
        document = """## Page 1

Chapter 1: Alpha

one two three four five six seven eight nine ten

Chapter 2: Beta

eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
"""
        chunks = chunk_document_by_section(document, chunk_size=9, overlap=3)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]
        chapter1_chunks = [chunk for chunk in body_chunks if chunk.section == "Chapter 1: Alpha"]
        chapter2_chunks = [chunk for chunk in body_chunks if chunk.section == "Chapter 2: Beta"]

        self.assertTrue(chapter1_chunks)
        self.assertTrue(chapter2_chunks)
        self.assertTrue(all("Chapter 2: Beta" not in chunk.text for chunk in chapter1_chunks))
        self.assertTrue(all("Chapter 1: Alpha" not in chunk.text for chunk in chapter2_chunks))

    def test_malformed_heading_text_is_normalized(self) -> None:
        document = """## Page 1

5.4 Water Supply5

All fixtures must be pressure tested.
"""
        chunks = chunk_document_by_section(document, chunk_size=60, overlap=10)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]

        self.assertTrue(body_chunks)
        self.assertEqual("5.4 Water Supply", body_chunks[0].section)
        self.assertIn("5.4 Water Supply", body_chunks[0].section_path)

    def test_footnote_contaminated_heading_is_normalized(self) -> None:
        document = """## Page 13

Chapter 3: Foundations9

3.1 General Requirements

Foundation requirements begin here.
"""
        chunks = chunk_document_by_section(document, chunk_size=80, overlap=10)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]

        self.assertTrue(body_chunks)
        self.assertEqual("3.1 General Requirements", body_chunks[0].section)
        self.assertIn("Chapter 3: Foundations", body_chunks[0].section_path)
        self.assertNotIn("Foundations9", body_chunks[0].text)

    def test_overlap_preserved_without_cross_section_metadata_leak(self) -> None:
        sec1 = " ".join(f"s1_{i}" for i in range(16))
        sec2 = " ".join(f"s2_{i}" for i in range(16))
        document = f"""## Page 1

Chapter 1 First

{sec1}

## Page 2

Chapter 2 Second

{sec2}
"""

        chunks = chunk_document_by_section(document, chunk_size=10, overlap=4)
        sec1_chunks = [chunk for chunk in chunks if chunk.section == "Chapter 1: First" and chunk.content_type == "body_text"]
        sec2_chunks = [chunk for chunk in chunks if chunk.section == "Chapter 2: Second" and chunk.content_type == "body_text"]

        self.assertGreaterEqual(len(sec1_chunks), 2)
        self.assertGreaterEqual(len(sec2_chunks), 2)
        self.assertEqual(sec1_chunks[0].text.split()[-4:], sec1_chunks[1].text.split()[:4])
        self.assertEqual(sec2_chunks[0].text.split()[-4:], sec2_chunks[1].text.split()[:4])
        self.assertTrue(all(not token.startswith("s1_") for token in sec2_chunks[0].text.split()))

    def test_image_marker_with_running_text_is_not_misclassified_as_artifact(self) -> None:
        document = """## Page 1

Chapter 1 Intro

This paragraph has substantial running text and ends with [IMAGES] 2 embedded image(s)
but it should still be handled as body text.
"""

        chunks = chunk_document_by_section(document, chunk_size=80, overlap=10)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]
        artifact_chunks = [chunk for chunk in chunks if chunk.content_type == "image_artifact"]

        self.assertGreaterEqual(len(body_chunks), 1)
        self.assertEqual([], artifact_chunks)

    def test_large_single_section_splits_by_paragraph_before_token_fallback(self) -> None:
        paragraph1 = " ".join(f"p1_{i}" for i in range(320))
        paragraph2 = " ".join(f"p2_{i}" for i in range(320))
        paragraph3 = " ".join(f"p3_{i}" for i in range(320))
        document = f"""## Page 8

Chapter 6: Heating, Ventilation, and Air Conditioning Systems (HVAC)

6.1 General Requirements

{paragraph1}

{paragraph2}

{paragraph3}
"""
        chunks = chunk_document_by_section(document, chunk_size=700, overlap=50)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]

        self.assertGreaterEqual(len(body_chunks), 2)
        self.assertTrue(any("p1_0" in chunk.text and "p2_0" in chunk.text for chunk in body_chunks))
        self.assertTrue(any("p3_0" in chunk.text for chunk in body_chunks))
        # Section metadata must remain stable for child chunks of the same large section.
        self.assertTrue(all(chunk.section == "6.1 General Requirements" for chunk in body_chunks))

    def test_soft_chunk_policy_prefers_paragraph_boundary(self) -> None:
        paragraph1 = " ".join(f"a{i}" for i in range(360))
        paragraph2 = " ".join(f"b{i}" for i in range(360))
        paragraph3 = " ".join(f"c{i}" for i in range(360))
        document = f"""## Page 10

Chapter 7: Roofing Systems and Attics

7.1 General Requirements

{paragraph1}

{paragraph2}

{paragraph3}
"""
        chunks = chunk_document_by_section(document, chunk_size=700, overlap=50)
        body_chunks = [chunk for chunk in chunks if chunk.content_type == "body_text"]

        self.assertGreaterEqual(len(body_chunks), 2)
        # Soft split should happen at paragraph boundary after first two paragraphs.
        self.assertTrue(any("a0" in chunk.text and "b359" in chunk.text for chunk in body_chunks))
        self.assertTrue(any("c0" in chunk.text for chunk in body_chunks))
        self.assertTrue(all(chunk.token_count <= 875 for chunk in body_chunks))


if __name__ == "__main__":
    unittest.main()
