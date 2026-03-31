from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ingestion.parsers.html import ingest_html_folder, parse_html_content
from app.ingestion.parsers.pdf import parse_pdf_file


class HTMLParserTests(unittest.TestCase):
    def test_parse_html_content_extracts_headings_tables_and_images(self) -> None:
        html = """
        <html><body>
            <h1>Safety Standard</h1>
            <section><h2>Scope</h2><p>Applies to all devices.</p></section>
            <table>
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
    def test_parse_pdf_file_missing_dependency_has_actionable_error(self) -> None:
        # A non-PDF path is sufficient because dependency check happens before parsing.
        with self.assertRaises(ImportError) as context:
            parse_pdf_file("dummy.pdf")

        self.assertIn("pip install pypdf", str(context.exception))


if __name__ == "__main__":
    unittest.main()
