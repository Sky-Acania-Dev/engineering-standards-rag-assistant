from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers.char_superscript_detector import AnchorCandidate
from app.ingestion.parsers.footnote_region_detector import FootnoteBody


@dataclass(frozen=True)
class ResolvedFootnote:
    page_number: int
    label: str
    content: str
    anchor_text: str
    anchor_bbox: tuple[float, float, float, float]
    footnote_bbox: tuple[float, float, float, float]


def link_anchors_to_bodies(
    anchors: list[AnchorCandidate],
    bodies: list[FootnoteBody],
) -> tuple[list[ResolvedFootnote], list[AnchorCandidate]]:
    by_label: dict[str, list[FootnoteBody]] = {}
    for body in bodies:
        by_label.setdefault(body.label, []).append(body)

    resolved: list[ResolvedFootnote] = []
    unresolved: list[AnchorCandidate] = []
    used_body_ids: set[tuple[int, str]] = set()

    for anchor in anchors:
        matches = by_label.get(anchor.label, [])
        if len(matches) != 1:
            unresolved.append(anchor)
            continue
        body = matches[0]
        body_key = (body.page_number, body.label)
        if body_key in used_body_ids:
            unresolved.append(anchor)
            continue
        used_body_ids.add(body_key)
        resolved.append(
            ResolvedFootnote(
                page_number=anchor.page_number,
                label=anchor.label,
                content=body.content,
                anchor_text=anchor.anchor_text,
                anchor_bbox=anchor.bbox,
                footnote_bbox=body.bbox,
            )
        )

    return resolved, unresolved
