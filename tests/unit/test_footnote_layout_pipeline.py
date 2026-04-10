from __future__ import annotations

import unittest

from app.ingestion.footnote_detector import analyze_page_layout
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
        analysis = analyze_page_layout(tokens, page_height=800)
        links = link_anchors_to_bodies(list(analysis.anchor_candidates), analysis.footnote_bodies)
        lines, metadata = render_page_text_with_footnotes(tokens, links, analysis=analysis)

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
        analysis = analyze_page_layout(tokens, page_height=800)
        self.assertEqual((), analysis.anchor_candidates)

    def test_page_analysis_separates_body_and_footnote_regions(self) -> None:
        tokens = [
            LayoutToken(page=1, text="Chapter", x0=10, x1=40, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=0),
            LayoutToken(page=1, text="1", x0=42, x1=44, top=96, bottom=103, size=8, fontname="A", line_id=100, reading_order=1),
            LayoutToken(page=1, text="Body", x0=10, x1=30, top=120, bottom=132, size=12, fontname="A", line_id=120, reading_order=2),
            LayoutToken(page=1, text="1", x0=10, x1=12, top=760, bottom=768, size=8, fontname="A", line_id=760, reading_order=3),
            LayoutToken(page=1, text="footnote", x0=14, x1=50, top=760, bottom=768, size=8, fontname="A", line_id=760, reading_order=4),
        ]
        analysis = analyze_page_layout(tokens, page_height=800)
        self.assertIn(3, analysis.footnote_token_indexes)
        self.assertIn(4, analysis.footnote_token_indexes)
        self.assertIn(0, analysis.body_token_indexes)
        self.assertTrue(analysis.anchor_candidates)

    def test_unlinked_anchor_is_removed_from_normalized_text(self) -> None:
        tokens = [
            LayoutToken(page=1, text="Foundations", x0=10, x1=60, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=0),
            LayoutToken(page=1, text="9", x0=61, x1=63, top=96, bottom=103, size=8, fontname="A", line_id=100, reading_order=1),
        ]
        analysis = analyze_page_layout(tokens, page_height=800)
        lines, _ = render_page_text_with_footnotes(tokens, [], analysis=analysis)
        self.assertEqual("Foundations", lines[0])

    def test_unlinked_footnote_region_is_not_emitted_inline(self) -> None:
        tokens = [
            LayoutToken(page=1, text="Body", x0=10, x1=30, top=120, bottom=132, size=12, fontname="A", line_id=120, reading_order=0),
            LayoutToken(page=1, text="1", x0=10, x1=12, top=760, bottom=768, size=8, fontname="A", line_id=760, reading_order=1),
            LayoutToken(page=1, text="https://example.org/footnote", x0=14, x1=120, top=760, bottom=768, size=8, fontname="A", line_id=760, reading_order=2),
        ]
        analysis = analyze_page_layout(tokens, page_height=800)
        lines, _ = render_page_text_with_footnotes(tokens, [], analysis=analysis)
        joined = " ".join(lines)
        self.assertIn("Body", joined)
        self.assertNotIn("https://example.org/footnote", joined)

    def test_numeric_legal_tokens_remain_intact(self) -> None:
        tokens = [
            LayoutToken(page=1, text="40", x0=10, x1=18, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=0),
            LayoutToken(page=1, text="CFR", x0=20, x1=34, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=1),
            LayoutToken(page=1, text="Part", x0=36, x1=54, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=2),
            LayoutToken(page=1, text="745", x0=56, x1=72, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=3),
            LayoutToken(page=1, text="§2306.51", x0=74, x1=110, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=4),
            LayoutToken(page=1, text="10", x0=112, x1=120, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=5),
            LayoutToken(page=1, text="TAC", x0=122, x1=138, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=6),
            LayoutToken(page=1, text="Chapter", x0=140, x1=170, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=7),
            LayoutToken(page=1, text="21", x0=172, x1=182, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=8),
            LayoutToken(page=1, text="102", x0=184, x1=198, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=9),
            LayoutToken(page=1, text="degrees", x0=200, x1=232, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=10),
            LayoutToken(page=1, text="F", x0=234, x1=238, top=100, bottom=112, size=12, fontname="A", line_id=100, reading_order=11),
        ]
        analysis = analyze_page_layout(tokens, page_height=800)
        lines, _ = render_page_text_with_footnotes(tokens, [], analysis=analysis)
        joined = " ".join(lines)
        self.assertIn("40 CFR Part 745", joined)
        self.assertIn("§2306.51", joined)
        self.assertIn("10 TAC Chapter 21", joined)
        self.assertIn("102 degrees F", joined)


if __name__ == "__main__":
    unittest.main()
