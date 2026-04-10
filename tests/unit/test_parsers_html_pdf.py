from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.parsers.html import ingest_html_folder, parse_html_content
from app.ingestion.parsers.pdf import parse_pdf_file


class HTMLParserTests(unittest.TestCase):
    def test_parse_html_content_extracts_headings_tables_and_images(self) -> None:
        html = """
        <html><body>
            <h1>Safety Standard</h1>
            <section><h2>Scope</h2><p>Applies to all devices.</p></section>
            <table>
                <caption>Table 1. Scope matrix</caption>
                <tr><th>Clause</th><th>Requirement</th></tr>
                <tr><td>1.2</td><td>Use PPE</td></tr>
            </table>
            <img src="diagram.png" alt="Safety Diagram" />
            <figure><figcaption>Diagram A</figcaption></figure>
        </body></html>
        """

        parsed = parse_html_content(html)

        self.assertIn("# Safety Standard", parsed)
        self.assertIn("## Scope", parsed)
        self.assertIn("[TABLE]", parsed)
        self.assertIn("[TABLE_CAPTION] Table 1. Scope matrix", parsed)
        self.assertIn("Clause | Requirement", parsed)
        self.assertIn("[IMAGE] Safety Diagram", parsed)
        self.assertIn("[IMAGE_CAPTION] Diagram A", parsed)

    def test_ingest_html_folder_reads_html_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / "doc.html"
            html_path.write_text("<h1>Title</h1><p>Body</p>", encoding="utf-8")

            docs = ingest_html_folder(temp_dir)

            self.assertEqual(1, len(docs))
            self.assertEqual("doc.html", docs[0]["filename"])
            self.assertIn("# Title", docs[0]["content"])


class PDFParserTests(unittest.TestCase):
    def test_parse_pdf_file_uses_pdfplumber_primary_extraction(self) -> None:
        class _FakePage:
            images = []

            def extract_text(self) -> str:
                return "Section 10.5 Windows\nPerformance table follows"

            def extract_tables(self) -> list[list[list[str]]]:
                return [[["Performance Measure", "CZ2", "CZ3", "CZ4"], ["U-Factor", "0.65", "0.50", "0.35"]]]

        class _FakeDoc:
            pages = [_FakePage()]

            def __enter__(self) -> "_FakeDoc":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        class _FakePdfPlumber:
            @staticmethod
            def open(_: str) -> _FakeDoc:
                return _FakeDoc()

        with patch("app.ingestion.parsers.pdf._require_pdfplumber", return_value=_FakePdfPlumber):
            parsed = parse_pdf_file("dummy.pdf")

        self.assertIn("## Page 1", parsed)
        self.assertIn("[TABLE]", parsed)
        self.assertIn("| Performance Measure | CZ2 | CZ3 | CZ4 |", parsed)

    def test_parse_pdf_file_falls_back_to_pypdf_when_pdfplumber_unusable(self) -> None:
        class _FakePage:
            images = []

            def extract_text(self) -> str:
                return "Chapter 1 Intro"

        class _FakeReader:
            pages = [_FakePage()]

        with patch(
            "app.ingestion.parsers.pdf._extract_with_pdfplumber",
            return_value="## Page 1",
        ), patch(
            "app.ingestion.parsers.pdf._require_pypdf",
            return_value=lambda _: _FakeReader(),
        ):
            parsed = parse_pdf_file("dummy.pdf")

        self.assertIn("Chapter 1 Intro", parsed)

    def test_parse_pdf_file_missing_dependency_has_actionable_error(self) -> None:
        with patch("app.ingestion.parsers.pdf._extract_with_pdfplumber", side_effect=ImportError("missing pdfplumber")), patch(
            "app.ingestion.parsers.pdf._extract_with_pypdf",
            side_effect=ImportError("PDF ingestion requires 'pypdf'. Install it with `pip install pypdf`."),
        ):
            with self.assertRaises(RuntimeError) as context:
                parse_pdf_file("dummy.pdf")

        self.assertIn("pdfplumber", str(context.exception))
        self.assertIn("pip install pypdf", str(context.exception))


if __name__ == "__main__":
    unittest.main()
