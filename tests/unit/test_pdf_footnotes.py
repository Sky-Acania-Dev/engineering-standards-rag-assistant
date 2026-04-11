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




    def test_first_footer_body_not_dropped(self) -> None:
        chars = []
        chars.extend(self._chars_for_text("1 http://example.com/one", y_top=130.0, size=9.0, order_start=0))
        chars.extend(self._chars_for_text("2 http://example.com/two", y_top=144.0, size=9.0, order_start=80))
        lines = build_visual_lines(chars)
        bodies, _ = detect_footnote_bodies(lines, page_height=300.0)
        self.assertEqual(["1", "2"], [b.label for b in bodies])

    def test_multidigit_inline_anchor_grouping(self) -> None:
        body = self._chars_for_text("10 TAC Chapter 21", y_top=40.0, size=10.0, order_start=0)
        sup = self._chars_for_text("23", y_top=37.0, size=7.0, x_start=body[-1].x1 + 0.2, order_start=100)
        lines = build_visual_lines(body + sup)
        anchors = detect_superscript_anchors(lines)
        self.assertTrue(any(a.label == "23" for a in anchors))

    def test_footer_detection_not_limited_to_extreme_bottom(self) -> None:
        chars = self._chars_for_text("1 Code references throughout this document", y_top=130.0, size=8.0, order_start=0)
        lines = build_visual_lines(chars)
        bodies, _ = detect_footnote_bodies(lines, page_height=300.0)
        self.assertEqual("1", bodies[0].label)

    def test_leading_label_parsing_for_url_line(self) -> None:
        line = self._chars_for_text("2 http://example.com/x", y_top=220.0, size=8.0, order_start=0)
        lines = build_visual_lines(line)
        bodies, _ = detect_footnote_bodies(lines, page_height=300.0)
        self.assertEqual("2", bodies[0].label)
        self.assertTrue(bodies[0].content.startswith("http://"))


    def test_numbered_list_near_bottom_not_footnote_block(self) -> None:
        chars = []
        chars.extend(self._chars_for_text("1 General requirement", y_top=190.0, size=10.0, order_start=0))
        chars.extend(self._chars_for_text("2 Installer shall verify", y_top=204.0, size=10.0, order_start=60))
        chars.extend(self._chars_for_text("3 Contractor must comply", y_top=218.0, size=10.0, order_start=140))
        lines = build_visual_lines(chars)
        bodies, _ = detect_footnote_bodies(lines, page_height=300.0)
        self.assertEqual([], bodies)

    def test_table_rows_not_classified_as_footnotes(self) -> None:
        chars = []
        chars.extend(self._chars_for_text("1  0.63  0.94", y_top=220.0, size=9.0, order_start=0))
        chars.extend(self._chars_for_text("2  0.61  0.93", y_top=234.0, size=9.0, order_start=40))
        lines = build_visual_lines(chars)
        bodies, _ = detect_footnote_bodies(lines, page_height=300.0)
        self.assertEqual([], bodies)

    def test_footer_filtering_excludes_page_artifacts(self) -> None:
        lines = build_visual_lines(
            self._chars_for_text("6 | Page", y_top=286.0, size=8.0, order_start=0)
            + self._chars_for_text("Page 7", y_top=294.0, size=8.0, order_start=40)
        )
        bodies, _ = detect_footnote_bodies(lines, page_height=300.0)
        self.assertEqual([], bodies)




    def test_large_heading_font_anchor_detection(self) -> None:
        title = self._chars_for_text("Chapter 1 Administration", y_top=20.0, size=16.0, order_start=0)
        sup = self._chars_for_text("1", y_top=16.5, size=14.5, x_start=title[-1].x1 + 0.3, order_start=120)
        body = self._chars_for_text("1 http://example.com/heading-footnote-body", y_top=220.0, size=8.0, order_start=220)
        lines = build_visual_lines(title + sup + body)
        anchors = detect_superscript_anchors(lines)
        self.assertTrue(any(a.label == "1" for a in anchors))

    def test_period_adjacent_745_anchor_detected(self) -> None:
        line = self._chars_for_text("40 CFR Part 745.", y_top=40.0, size=10.0, order_start=0)
        sup = self._chars_for_text("2", y_top=37.5, size=8.5, x_start=line[-1].x1 + 0.2, order_start=200)
        footer = self._chars_for_text("2 http://example.com/two", y_top=220.0, size=8.0, order_start=300)
        lines = build_visual_lines(line + sup + footer)
        anchors = detect_superscript_anchors(lines)
        bodies, idx = detect_footnote_bodies(lines, page_height=300.0)
        resolved, unresolved, _ = link_anchors_to_bodies(anchors, bodies)
        page = reconstruct_page_text(lines, resolved=resolved, unresolved=unresolved, footnote_line_indexes=idx)
        self.assertIn("40 CFR Part 745.[footnote: 2]", "\n".join(page.lines))

    def test_line_final_anchor_detected(self) -> None:
        body = self._chars_for_text("Section R802", y_top=40.0, size=10.0, order_start=0)
        sup = self._chars_for_text("19", y_top=37.0, size=7.0, x_start=body[-1].x1 + 0.4, order_start=100)
        footer = self._chars_for_text("19 http://example.com/line-final", y_top=220.0, size=8.0, order_start=200)
        lines = build_visual_lines(body + sup + footer)
        anchors = detect_superscript_anchors(lines)
        bodies, idx = detect_footnote_bodies(lines, page_height=300.0)
        resolved, unresolved, _ = link_anchors_to_bodies(anchors, bodies)
        page = reconstruct_page_text(lines, resolved=resolved, unresolved=unresolved, footnote_line_indexes=idx)
        self.assertIn("Section R802[footnote: 19]", "\n".join(page.lines))

    def test_heading_title_anchor_detected(self) -> None:
        title = self._chars_for_text("Chapter 1: Intro", y_top=30.0, size=12.0, order_start=0)
        sup = self._chars_for_text("1", y_top=27.5, size=9.5, x_start=title[-1].x1 + 0.3, order_start=100)
        bodies = self._chars_for_text("1 http://example.com/heading-footnote", y_top=220.0, size=8.0, order_start=200)
        lines = build_visual_lines(title + sup + bodies)
        anchors = detect_superscript_anchors(lines)
        foot_bodies, idx = detect_footnote_bodies(lines, page_height=300.0)
        resolved, unresolved, _ = link_anchors_to_bodies(anchors, foot_bodies)
        page = reconstruct_page_text(lines, resolved=resolved, unresolved=unresolved, footnote_line_indexes=idx)
        self.assertIn("[footnote: 1]", "\n".join(page.lines))

    def test_two_digit_label_grouping(self) -> None:
        labels = ["10", "11", "14", "19", "22", "23"]
        chars = []
        order = 0
        for idx, label in enumerate(labels):
            y = 220.0 + idx * 10.0
            chars.extend(self._chars_for_text(f"{label} http://example.com/{label}", y_top=y, size=8.0, order_start=order))
            order += 80
        lines = build_visual_lines(chars)
        bodies, _ = detect_footnote_bodies(lines, page_height=300.0)
        self.assertEqual(labels, [b.label for b in bodies])

    def test_orphan_end_of_chunk_cleanup(self) -> None:
        text = "## Page 1\nBody text [footnote: 2] 1 2"
        chunks = chunk_document_by_section(text)
        merged = "\n".join(c.text for c in chunks)
        self.assertNotIn(" 1 2", merged)

    def test_period_comma_and_citation_preservation(self) -> None:
        chars = []
        l1 = self._chars_for_text("40 CFR Part 745.", y_top=30.0, size=10.0, order_start=0)
        s1 = self._chars_for_text("2", y_top=27.0, size=7.0, x_start=l1[-1].x1 + 0.4, order_start=100)
        l2 = self._chars_for_text("36 CFR Part 67,", y_top=44.0, size=10.0, order_start=200)
        s2 = self._chars_for_text("4", y_top=41.0, size=7.0, x_start=l2[-1].x1 + 0.4, order_start=300)
        l3 = self._chars_for_text("§2306.51", y_top=58.0, size=10.0, order_start=350)
        s3 = self._chars_for_text("8", y_top=55.0, size=7.0, x_start=l3[-1].x1 + 0.4, order_start=380)
        b2 = self._chars_for_text("2 http://example.com/two", y_top=220.0, size=8.0, order_start=400)
        b4 = self._chars_for_text("4 http://example.com/four", y_top=232.0, size=8.0, order_start=500)
        b8 = self._chars_for_text("8 http://example.com/eight", y_top=244.0, size=8.0, order_start=600)
        chars.extend(l1 + s1 + l2 + s2 + l3 + s3 + b2 + b4 + b8)

        lines = build_visual_lines(chars)
        anchors = detect_superscript_anchors(lines)
        bodies, indexes = detect_footnote_bodies(lines, page_height=300.0)
        resolved, unresolved, _ = link_anchors_to_bodies(anchors, bodies)
        page = reconstruct_page_text(lines, resolved=resolved, unresolved=unresolved, footnote_line_indexes=indexes)
        text = "\n".join(page.lines)
        self.assertIn("40 CFR Part 745.[footnote: 2]", text)
        self.assertIn("36 CFR Part 67,[footnote: 4]", text)
        self.assertIn("§2306.51[footnote: 8]", text)
        self.assertNotIn("§2306.514", text)

    def test_decimal_preservation(self) -> None:
        body = self._chars_for_text("10 TAC 21.62", y_top=40.0, size=10.0, order_start=0)
        sup = self._chars_for_text("24", y_top=37.0, size=7.0, x_start=body[-1].x1 + 0.4, order_start=100)
        lines = build_visual_lines(body + sup)
        anchors = detect_superscript_anchors(lines)
        page = reconstruct_page_text(lines, resolved=[], unresolved=anchors, footnote_line_indexes=set())
        text = " ".join(page.lines)
        self.assertIn("10 TAC 21.62", text)
        self.assertNotIn("21.624", text)

    def test_no_dangling_marker_and_no_debug_leak(self) -> None:
        text = "## Page 1\n[DEBUG] line_count=3\nRule[footnote: 9] applies"
        chunks = chunk_document_by_section(text)
        merged = "\n".join(c.text for c in chunks)
        self.assertNotIn("[DEBUG]", merged)
        self.assertNotIn("[footnote: 9]", merged)

    def test_unlinked_fallback_metadata_and_no_leakage(self) -> None:
        text = (
            "## Page 5\n"
            "Chapter 1 Title[footnote: 1]\n"
            "Body line\n"
            "[FNDEF page=5 id=1 anchor=Chapter 1 Title] chapter note\n"
            "[FNUNLINK page=5 id=2 reason=anchor_missing] 2 http://example.org/a\n"
            "[FNUNLINK page=5 id=3 reason=anchor_missing] 3 http://example.org/b"
        )
        chunks = chunk_document_by_section(text)
        body = [c for c in chunks if c.content_type == "body_text"]
        merged = "\n".join(c.text for c in body)
        self.assertIn("[footnote: 1]", merged)
        self.assertNotIn("2 http://", merged)
        self.assertNotIn("3 http://", merged)
        notes = [n for c in body for n in c.footnotes]
        self.assertTrue(any(n.id == "2" and n.linked is False for n in notes))
        self.assertTrue(any(n.id == "3" and n.linked is False for n in notes))

    def test_page_bottom_map_not_shifted_or_merged(self) -> None:
        class _FakePage:
            images = []
            height = 300.0

            def __init__(self) -> None:
                helper = FootnotePipelineTests()
                self.chars = []
                self.chars.extend(helper._chars_for_text("Rule", y_top=60.0, size=10.0, order_start=0))
                self.chars.extend(helper._chars_for_text("3", y_top=57.0, size=7.0, x_start=30.0, order_start=100))
                self.chars.extend(helper._chars_for_text("2 http://example.com/two", y_top=220.0, size=8.0, order_start=200))
                self.chars.extend(helper._chars_for_text("3 http://example.com/three", y_top=232.0, size=8.0, order_start=260))
                self.chars.extend(helper._chars_for_text("4 http://example.com/four", y_top=244.0, size=8.0, order_start=320))
                self.chars.extend(helper._chars_for_text("5 http://example.com/five", y_top=256.0, size=8.0, order_start=380))

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
        notes = [n for c in chunks for n in c.footnotes]
        contents = {n.id: n.content for n in notes}
        self.assertEqual("http://example.com/two", contents.get("2"))
        self.assertEqual("http://example.com/three", contents.get("3"))
        self.assertEqual("http://example.com/four", contents.get("4"))
        self.assertEqual("http://example.com/five", contents.get("5"))

    def test_parser_does_not_duplicate_unmatched_tables_inline(self) -> None:
        class _FakePage:
            images = []
            height = 300.0

            def __init__(self) -> None:
                helper = FootnotePipelineTests()
                self.chars = []
                self.chars.extend(helper._chars_for_text("Storage Size Gas EF Electric EF", y_top=80.0, size=10.0, order_start=0))
                self.chars.extend(helper._chars_for_text("30 0.63 0.94", y_top=94.0, size=10.0, order_start=50))

            def extract_text(self) -> str:
                return ""

            def extract_tables(self):
                return [[["Storage Size", "Gas EF", "Electric EF"], ["30", "0.63", "0.94"]]]

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
        # table rows should not appear duplicated twice
        self.assertEqual(1, parsed.count("| Storage Size | Gas EF | Electric EF |"))



if __name__ == "__main__":
    unittest.main()
