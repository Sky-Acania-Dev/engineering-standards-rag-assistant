from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.parsers.html import ingest_html_folder, parse_html_content
from app.ingestion.parsers.pdf import _build_anchor_debug_for_page, parse_pdf_file


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
    def test_phase1_detects_heading_and_punctuation_superscript_anchors(self) -> None:
        class _FakePage:
            def __init__(self, chars: list[dict[str, object]]) -> None:
                self.chars = chars

        chars = [
            # "Chapter 1: Foundations" + superscript 1
            {"text": "C", "x0": 10, "x1": 14, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "h", "x0": 14, "x1": 18, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "a", "x0": 18, "x1": 22, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "p", "x0": 22, "x1": 26, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "t", "x0": 26, "x1": 30, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "e", "x0": 30, "x1": 34, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "r", "x0": 34, "x1": 38, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": " ", "x0": 38, "x1": 40, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "1", "x0": 40, "x1": 44, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": ":", "x0": 44, "x1": 46, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": " ", "x0": 46, "x1": 48, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "F", "x0": 48, "x1": 52, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "o", "x0": 52, "x1": 56, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "u", "x0": 56, "x1": 60, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "n", "x0": 60, "x1": 64, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "d", "x0": 64, "x1": 68, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "a", "x0": 68, "x1": 72, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "t", "x0": 72, "x1": 76, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "i", "x0": 76, "x1": 80, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "o", "x0": 80, "x1": 84, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "n", "x0": 84, "x1": 88, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "s", "x0": 88, "x1": 92, "top": 20, "bottom": 30, "doctop": 20, "size": 12},
            {"text": "1", "x0": 93, "x1": 95, "top": 14, "bottom": 20, "doctop": 14, "size": 8},
            # "40 CFR Part 745." + superscript 2
            {"text": "4", "x0": 10, "x1": 14, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "0", "x0": 14, "x1": 18, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": " ", "x0": 18, "x1": 20, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "C", "x0": 20, "x1": 24, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "F", "x0": 24, "x1": 28, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "R", "x0": 28, "x1": 32, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": " ", "x0": 32, "x1": 34, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "P", "x0": 34, "x1": 38, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "a", "x0": 38, "x1": 42, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "r", "x0": 42, "x1": 46, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "t", "x0": 46, "x1": 50, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": " ", "x0": 50, "x1": 52, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "7", "x0": 52, "x1": 56, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "4", "x0": 56, "x1": 60, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "5", "x0": 60, "x1": 64, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": ".", "x0": 64, "x1": 66, "top": 50, "bottom": 60, "doctop": 50, "size": 12},
            {"text": "2", "x0": 67, "x1": 69, "top": 44, "bottom": 50, "doctop": 44, "size": 8},
            # two-digit anchor 10
            {"text": "R", "x0": 10, "x1": 14, "top": 80, "bottom": 90, "doctop": 80, "size": 12},
            {"text": "e", "x0": 14, "x1": 18, "top": 80, "bottom": 90, "doctop": 80, "size": 12},
            {"text": "f", "x0": 18, "x1": 22, "top": 80, "bottom": 90, "doctop": 80, "size": 12},
            {"text": "1", "x0": 23, "x1": 25, "top": 74, "bottom": 80, "doctop": 74, "size": 8},
            {"text": "0", "x0": 25, "x1": 27, "top": 74, "bottom": 80, "doctop": 74, "size": 8},
        ]
        debug = _build_anchor_debug_for_page(_FakePage(chars), page_number=5)
        ids = [entry["anchor_id"] for entry in debug["detected_anchors"]]
        self.assertIn("1", ids)
        self.assertIn("2", ids)
        self.assertIn("10", ids)
        punctuation_anchor = [entry for entry in debug["detected_anchors"] if entry["anchor_id"] == "2"][0]
        self.assertTrue(punctuation_anchor["flags"]["punctuation_adjacent"])
        heading_anchor = [entry for entry in debug["detected_anchors"] if entry["anchor_id"] == "1"][0]
        self.assertTrue(heading_anchor["flags"]["heading_like"])

    def test_phase1_ignores_url_reference_lines(self) -> None:
        class _FakePage:
            def __init__(self, chars: list[dict[str, object]]) -> None:
                self.chars = chars

        chars = [
            # "2 http://www."
            {"text": "2", "x0": 10, "x1": 12, "top": 10, "bottom": 16, "doctop": 10, "size": 8},
            {"text": " ", "x0": 12, "x1": 14, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "h", "x0": 14, "x1": 18, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "t", "x0": 18, "x1": 20, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "t", "x0": 20, "x1": 22, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "p", "x0": 22, "x1": 26, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": ":", "x0": 26, "x1": 27, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "/", "x0": 27, "x1": 28, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "/", "x0": 28, "x1": 29, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "w", "x0": 29, "x1": 33, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "w", "x0": 33, "x1": 37, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": "w", "x0": 37, "x1": 41, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
            {"text": ".", "x0": 41, "x1": 42, "top": 15, "bottom": 25, "doctop": 15, "size": 11},
        ]
        debug = _build_anchor_debug_for_page(_FakePage(chars), page_number=7)
        self.assertEqual([], debug["detected_anchors"])

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
        self.assertIn("| Performance Measure | CZ2 | CZ3 | CZ4 |", parsed)
        self.assertNotIn("[TABLE]", parsed)

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
