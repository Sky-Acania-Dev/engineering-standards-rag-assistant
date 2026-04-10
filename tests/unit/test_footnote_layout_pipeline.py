from __future__ import annotations

import unittest

from app.ingestion.footnote_detector import detect_footnote_bodies, detect_superscript_anchors
from app.ingestion.footnote_linker import link_anchors_to_bodies
from app.ingestion.pdf_layout_extractor import LayoutToken
from app.ingestion.text_normalizer import render_page_text_with_footnotes


class FootnoteLayoutPipelineTests(unittest.TestCase):
    def test_detects_real_superscript_anchor_and_links_body(self) -> None:
        tokens = [
            LayoutToken(page=1, text="Requirements", x0=10, x1=70, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=0),
            LayoutToken(page=1, text="1", x0=71, x1=73, top=96, bottom=103, size=8, fontname="A", line_id=100, reading_order=1),
            LayoutToken(page=1, text="1", x0=10, x1=12, top=760, bottom=768, size=8, fontname="A", line_id=760, reading_order=2),
            LayoutToken(page=1, text="https://example.org/spec", x0=14, x1=200, top=760, bottom=768, size=8, fontname="A", line_id=760, reading_order=3),
        ]
        anchors = detect_superscript_anchors(tokens)
        bodies = detect_footnote_bodies(tokens, page_height=800)
        links = link_anchors_to_bodies(anchors, bodies)
        lines, metadata = render_page_text_with_footnotes(tokens, links)

        self.assertTrue(any("[fn:1]" in line for line in lines))
        self.assertEqual(1, len(metadata))
        self.assertEqual("Requirements", metadata[0]["anchor_text"])
        self.assertEqual("https://example.org/spec", metadata[0]["content"])

    def test_plain_numeric_tokens_are_not_treated_as_superscripts(self) -> None:
        tokens = [
            LayoutToken(page=1, text="Texas", x0=10, x1=30, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=0),
            LayoutToken(page=1, text="Government", x0=32, x1=80, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=1),
            LayoutToken(page=1, text="Code", x0=82, x1=102, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=2),
            LayoutToken(page=1, text="2306", x0=104, x1=130, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=3),
            LayoutToken(page=1, text="2009", x0=132, x1=154, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=4),
            LayoutToken(page=1, text="IRC", x0=156, x1=172, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=5),
            LayoutToken(page=1, text="R802", x0=174, x1=194, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=6),
            LayoutToken(page=1, text="UL", x0=196, x1=205, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=7),
            LayoutToken(page=1, text="2075", x0=207, x1=227, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=8),
            LayoutToken(page=1, text="ASSE", x0=229, x1=255, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=9),
            LayoutToken(page=1, text="1051-2009", x0=257, x1=300, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=10),
        ]
        anchors = detect_superscript_anchors(tokens)
        self.assertEqual([], anchors)


if __name__ == "__main__":
    unittest.main()
