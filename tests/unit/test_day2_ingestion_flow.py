import unittest

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


if __name__ == "__main__":
    unittest.main()
