from __future__ import annotations

import unittest
from unittest.mock import patch

from app.ingestion.parsers.char_superscript_detector import detect_superscript_anchors
from app.ingestion.parsers.footnote_linker import link_anchors_to_bodies
from app.ingestion.parsers.footnote_region_detector import detect_footnote_bodies
from app.ingestion.parsers.pdf import parse_pdf_file
from app.ingestion.parsers.pdfplumber_layout import CharInfo, build_visual_lines
from app.ingestion.parsers.text_reconstructor import reconstruct_page_text
from app.rag.chunking import chunk_document_by_section


class FootnotePipelineTests(unittest.TestCase):
    def _chars_for_text(self, text: str, *, y_top: float, size: float, page: int = 1, x_start: float = 10.0, order_start: int = 0) -> list[CharInfo]:
        chars: list[CharInfo] = []
        x = x_start
        order = order_start
        for ch in text:
            width = 4.0 if ch != " " else 2.0
            chars.append(CharInfo(text=ch, x0=x, x1=x + width, top=y_top, bottom=y_top + size, size=size, fontname="Times", page_number=page, order=order))
            x += width
            order += 1
        return chars

    def test_detect_split_superscripts_char_level(self) -> None:
        chars: list[CharInfo] = []
        order = 0
        base_cases = [
            "40 CFR Part 745",
            "Rule",
            "36 CFR Part 67,",
            "§2306.51",
            "10 TAC Chapter 21",
            "10 TAC 21.62",
            "R802",
            "10 degrees F",
        ]
        supers = ["2", "3", "4", "4", "23", "4", "19", "2"]
        for i, body in enumerate(base_cases):
            y = 20.0 + (i * 14)
            line = self._chars_for_text(body, y_top=y, size=10.0, order_start=order)
            order = line[-1].order + 1
            sup = self._chars_for_text(supers[i], y_top=y - 2.5, size=7.0, order_start=order, x_start=line[-1].x1 + 0.6)
            order = sup[-1].order + 1
            chars.extend(line + sup)

        lines = build_visual_lines(chars)
        anchors = detect_superscript_anchors(lines)
        self.assertGreaterEqual(len(anchors), 8)
        self.assertIn("23", {a.label for a in anchors})

    def test_preserve_ordinary_numbers_without_superscript_geometry(self) -> None:
        text = "1978 2009 2306 2306.51 UL 2075 ASSE 1051-2009 R802.4(1) https://example.com/a?x=1"
        chars = self._chars_for_text(text, y_top=40.0, size=10.0)
        lines = build_visual_lines(chars)
        anchors = detect_superscript_anchors(lines)
        self.assertEqual([], anchors)

    def test_detect_footer_region_and_resolved_link(self) -> None:
        body = self._chars_for_text("Rule", y_top=20.0, size=10.0, order_start=0)
        sup = self._chars_for_text("3", y_top=17.5, size=7.0, order_start=100, x_start=body[-1].x1 + 0.4)
        foot = self._chars_for_text("3 Footnote body line one", y_top=250.0, size=8.0, order_start=200)
        foot2 = self._chars_for_text("continued line", y_top=261.0, size=8.0, order_start=300)
        lines = build_visual_lines(body + sup + foot + foot2)
        anchors = detect_superscript_anchors(lines)
        bodies, foot_lines = detect_footnote_bodies(lines, page_height=300.0)
        resolved, unresolved = link_anchors_to_bodies(anchors, bodies)
        page = reconstruct_page_text(lines, resolved=resolved, unresolved=unresolved, footnote_line_indexes=foot_lines)
        self.assertIn("Rule[fn:3]", "\n".join(page.lines))
        self.assertNotIn("Footnote body", "\n".join(page.lines))
        self.assertEqual(1, len(page.footnotes))

    def test_unresolved_anchor_drops_digit_without_corruption(self) -> None:
        body = self._chars_for_text("10 degrees F", y_top=20.0, size=10.0, order_start=0)
        sup = self._chars_for_text("2", y_top=17.0, size=7.0, order_start=100, x_start=body[-1].x1 + 0.4)
        lines = build_visual_lines(body + sup)
        anchors = detect_superscript_anchors(lines)
        page = reconstruct_page_text(lines, resolved=[], unresolved=anchors, footnote_line_indexes=set())
        self.assertIn("10 degrees F", " ".join(page.lines))
        self.assertNotIn("102", " ".join(page.lines))



    def test_no_dangling_marker_when_definition_missing(self) -> None:
        text = "## Page 1\nRule[fn:7] applies"
        chunks = chunk_document_by_section(text)
        merged = "\n".join(c.text for c in chunks)
        self.assertNotIn("[fn:7]", merged)

    def test_debug_lines_never_leak_into_chunks(self) -> None:
        text = "## Page 1\n[DEBUG] line_count=5\nRule text"
        chunks = chunk_document_by_section(text)
        self.assertTrue(all("[DEBUG]" not in c.text for c in chunks))

    def test_marker_implies_metadata_consistency(self) -> None:
        text = "## Page 1\nRule[fn:3] applies\n[FNDEF page=1 id=3 anchor=Rule] linked note"
        chunks = chunk_document_by_section(text)
        body = [c for c in chunks if c.content_type == "body_text"]
        self.assertTrue(any("[fn:3]" in c.text for c in body))
        self.assertTrue(any(any(note.id == "3" for note in c.footnotes) for c in body))

    def test_preserve_citation_drop_unresolved_anchor_fragment(self) -> None:
        body = self._chars_for_text("§2306.51", y_top=40.0, size=10.0, order_start=0)
        sup = self._chars_for_text("4", y_top=37.0, size=7.0, x_start=body[-1].x1 + 0.4, order_start=100)
        lines = build_visual_lines(body + sup)
        anchors = detect_superscript_anchors(lines)
        page = reconstruct_page_text(lines, resolved=[], unresolved=anchors, footnote_line_indexes=set())
        text = " ".join(page.lines)
        self.assertIn("§2306.51", text)
        self.assertNotIn("§2306.514", text)

    def test_no_unresolved_http_number_fragment_leakage(self) -> None:
        class _FakePage:
            images = []
            height = 300.0

            def __init__(self) -> None:
                helper = FootnotePipelineTests()
                self.chars = []
                self.chars.extend(helper._chars_for_text("Main paragraph body.", y_top=80.0, size=10.0, order_start=0))
                self.chars.extend(helper._chars_for_text("10 http://example.org/ref", y_top=190.0, size=10.0, order_start=200))

            def extract_text(self) -> str:
                return ""

            def extract_tables(self):
                return []

        class _FakeDoc:
            pages = [_FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        class _FakePdfPlumber:
            @staticmethod
            def open(_: str):
                return _FakeDoc()

        with patch("app.ingestion.parsers.pdf._require_pdfplumber", return_value=_FakePdfPlumber):
            parsed = parse_pdf_file("dummy.pdf")
        chunks = chunk_document_by_section(parsed)
        merged = "\n".join(c.text for c in chunks if c.content_type == "body_text")
        self.assertNotIn("10 http", merged)

    def test_realistic_same_size_footer_removed_and_metadata_attached(self) -> None:
        class _FakePage:
            images = []
            height = 300.0

            def __init__(self) -> None:
                helper = FootnotePipelineTests()
                self.chars = []
                self.chars.extend(helper._chars_for_text("1.3 Qualifications and Workmanship", y_top=30.0, size=10.0, order_start=0))
                body = helper._chars_for_text("If the nature of the work", y_top=80.0, size=10.0, order_start=100)
                self.chars.extend(body)
                self.chars.extend(helper._chars_for_text("1", y_top=77.0, size=7.0, x_start=body[-1].x1 + 0.4, order_start=200))
                self.chars.extend(helper._chars_for_text("requires an individual to be licensed.", y_top=80.0, size=10.0, order_start=230, x_start=body[-1].x1 + 8.0))
                self.chars.extend(helper._chars_for_text("1 Code references throughout this document are provided as guidance.", y_top=182.0, size=10.0, order_start=400))
                self.chars.extend(helper._chars_for_text("All code references are to the 2009 IRC found at: https://example.org/ref", y_top=194.0, size=10.0, order_start=520))

            def extract_text(self) -> str:
                return ""

            def extract_tables(self):
                return []

        class _FakeDoc:
            pages = [_FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        class _FakePdfPlumber:
            @staticmethod
            def open(_: str):
                return _FakeDoc()

        with patch("app.ingestion.parsers.pdf._require_pdfplumber", return_value=_FakePdfPlumber):
            parsed = parse_pdf_file("dummy.pdf")

        chunks = chunk_document_by_section(parsed)
        body_chunks = [c for c in chunks if c.content_type == "body_text"]
        merged = "\n".join(c.text for c in body_chunks)
        self.assertIn("[fn:1]", merged)
        self.assertNotIn("Code references throughout this document", merged)
        self.assertTrue(any(any(note.id == "1" for note in c.footnotes) for c in body_chunks))

    def test_parser_and_chunk_metadata_include_resolved_footnotes(self) -> None:
        class _FakePage:
            images = []
            height = 300.0

            def __init__(self) -> None:
                self.chars = []
                self.chars.extend(FootnotePipelineTests()._chars_for_text("Section 1 Scope", y_top=20.0, size=10.0, order_start=0))
                line = FootnotePipelineTests()._chars_for_text("Rule", y_top=40.0, size=10.0, order_start=100)
                self.chars.extend(line)
                self.chars.extend(FootnotePipelineTests()._chars_for_text("3", y_top=37.0, size=7.0, x_start=line[-1].x1 + 0.4, order_start=200))
                self.chars.extend(FootnotePipelineTests()._chars_for_text("3 linked note", y_top=250.0, size=8.0, order_start=300))

            def extract_text(self) -> str:
                return ""

            def extract_tables(self):
                return []

        class _FakeDoc:
            pages = [_FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        class _FakePdfPlumber:
            @staticmethod
            def open(_: str):
                return _FakeDoc()

        with patch("app.ingestion.parsers.pdf._require_pdfplumber", return_value=_FakePdfPlumber):
            parsed = parse_pdf_file("dummy.pdf")

        chunks = chunk_document_by_section(parsed)
        body_chunks = [c for c in chunks if c.content_type == "body_text"]
        self.assertTrue(any("[fn:3]" in c.text for c in body_chunks))
        self.assertTrue(any(c.footnotes for c in body_chunks))


if __name__ == "__main__":
    unittest.main()
