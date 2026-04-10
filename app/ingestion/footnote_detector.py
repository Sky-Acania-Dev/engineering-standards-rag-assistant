from __future__ import annotations

from dataclasses import dataclass
import re

from app.ingestion.pdf_layout_extractor import LayoutToken


@dataclass(frozen=True)
class AnchorCandidate:
    id: int
    token_index: int
    anchor_text: str
    confidence: float


@dataclass(frozen=True)
class FootnoteBody:
    id: int
    content: str
    top: float
    bottom: float


def detect_superscript_anchors(tokens: list[LayoutToken]) -> list[AnchorCandidate]:
    anchors: list[AnchorCandidate] = []
    for idx in range(1, len(tokens)):
        token = tokens[idx]
        prev = tokens[idx - 1]
        if token.line_id != prev.line_id:
            continue
        if not re.fullmatch(r"\d{1,3}", token.text.strip()):
            continue
        if prev.text.lower().startswith(("http://", "https://", "www.")):
            continue
        gap = token.x0 - prev.x1
        size_ratio = (token.size / prev.size) if prev.size > 0 else 1.0
        raised = token.top + 0.5 < prev.top
        score = 0.0
        if size_ratio <= 0.82:
            score += 0.45
        if raised:
            score += 0.35
        if 0 <= gap <= 3.5:
            score += 0.2
        if score < 0.8:
            continue
        anchors.append(
            AnchorCandidate(
                id=int(token.text),
                token_index=idx,
                anchor_text=prev.text,
                confidence=round(score, 3),
            )
        )
    return anchors


def detect_footnote_bodies(tokens: list[LayoutToken], *, page_height: float) -> dict[int, FootnoteBody]:
    if not tokens:
        return {}
    by_line: dict[int, list[LayoutToken]] = {}
    for token in tokens:
        if token.top < page_height * 0.78:
            continue
        by_line.setdefault(token.line_id, []).append(token)

    bodies: dict[int, FootnoteBody] = {}
    for line_id in sorted(by_line):
        line = sorted(by_line[line_id], key=lambda t: t.x0)
        if not line:
            continue
        lead = line[0].text.strip().rstrip(").")
        if not re.fullmatch(r"\d{1,3}", lead):
            continue
        note_id = int(lead)
        content_tokens = [tok.text for tok in line[1:] if tok.text.strip()]
        if not content_tokens:
            continue
        content = " ".join(content_tokens).strip()
        if note_id not in bodies:
            bodies[note_id] = FootnoteBody(id=note_id, content=content, top=line[0].top, bottom=line[0].bottom)
        else:
            previous = bodies[note_id]
            merged = f"{previous.content} {content}".strip()
            bodies[note_id] = FootnoteBody(id=note_id, content=merged, top=previous.top, bottom=line[0].bottom)
    return bodies
