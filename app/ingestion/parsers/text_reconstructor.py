from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers.char_superscript_detector import AnchorCandidate
from app.ingestion.parsers.footnote_linker import ResolvedFootnote
from app.ingestion.parsers.pdfplumber_layout import LineInfo


@dataclass(frozen=True)
class ReconstructedPage:
    lines: list[str]
    footnotes: list[ResolvedFootnote]


def reconstruct_page_text(
    lines: list[LineInfo],
    *,
    resolved: list[ResolvedFootnote],
    unresolved: list[AnchorCandidate],
    footnote_line_indexes: set[int],
) -> ReconstructedPage:
    marker_by_order: dict[int, str] = {}
    unresolved_orders: set[int] = set()

    for anchor in unresolved:
        unresolved_orders.update(anchor.char_orders)
    for item in resolved:
        marker = f"[footnote: {item.label}]"
        for line in lines:
            for char in line.chars:
                if _in_bbox(char, item.anchor_bbox):
                    marker_by_order[char.order] = marker

    rendered: list[str] = []
    for idx, line in enumerate(lines):
        if idx in footnote_line_indexes:
            continue
        parts: list[str] = []
        inserted_markers: set[str] = set()
        for char in line.chars:
            if char.order in unresolved_orders:
                continue
            marker = marker_by_order.get(char.order)
            if marker:
                if marker not in inserted_markers:
                    parts.append(marker)
                    inserted_markers.add(marker)
                continue
            parts.append(char.text)
        text = "".join(parts).strip()
        if text:
            rendered.append(" ".join(text.split()))
    return ReconstructedPage(lines=rendered, footnotes=resolved)


def _in_bbox(char, bbox: tuple[float, float, float, float]) -> bool:
    return char.x0 >= bbox[0] - 0.3 and char.x1 <= bbox[2] + 0.3 and char.top >= bbox[1] - 0.3 and char.bottom <= bbox[3] + 0.3
