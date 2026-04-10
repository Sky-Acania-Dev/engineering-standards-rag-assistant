from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.footnote_detector import AnchorCandidate, FootnoteBody


@dataclass(frozen=True)
class LinkedFootnote:
    id: int
    token_index: int
    anchor_text: str
    content: str
    confidence: float


def link_anchors_to_bodies(
    anchors: list[AnchorCandidate],
    bodies: dict[int, FootnoteBody],
) -> list[LinkedFootnote]:
    linked: list[LinkedFootnote] = []
    for anchor in anchors:
        body = bodies.get(anchor.id)
        if body is None:
            continue
        linked.append(
            LinkedFootnote(
                id=anchor.id,
                token_index=anchor.token_index,
                anchor_text=anchor.anchor_text,
                content=body.content,
                confidence=anchor.confidence,
            )
        )
    return linked
