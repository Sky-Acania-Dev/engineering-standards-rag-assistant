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


@dataclass(frozen=True)
class RejectedAnchor:
    token_index: int
    reason: str


@dataclass(frozen=True)
class PageAnalysis:
    body_token_indexes: tuple[int, ...]
    footnote_token_indexes: tuple[int, ...]
    anchor_candidates: tuple[AnchorCandidate, ...]
    footnote_bodies: dict[int, FootnoteBody]
    rejected_anchors: tuple[RejectedAnchor, ...]


def detect_superscript_anchors(tokens: list[LayoutToken], *, body_token_indexes: set[int] | None = None) -> tuple[list[AnchorCandidate], list[RejectedAnchor]]:
    anchors: list[AnchorCandidate] = []
    rejected: list[RejectedAnchor] = []
    for idx in range(1, len(tokens)):
        if body_token_indexes is not None and idx not in body_token_indexes:
            continue
        token = tokens[idx]
        prev = tokens[idx - 1]
        if token.line_id != prev.line_id:
            continue
        if not re.fullmatch(r"\d{1,3}", token.text.strip()):
            continue
        prev_clean = prev.text.strip().strip(".,;:()[]")
        if re.fullmatch(r"\d{4}", prev_clean):
            rejected.append(RejectedAnchor(token_index=idx, reason="year_neighbor"))
            continue
        if prev_clean.lower().startswith(("http://", "https://", "www.")):
            rejected.append(RejectedAnchor(token_index=idx, reason="url_neighbor"))
            continue
        if (
            re.fullmatch(r"\d+(?:\.\d+)+", prev_clean)
            or re.fullmatch(r"[A-Z]{1,4}\d+(?:\.\d+)*(?:\(\d+\))?", prev_clean)
            or re.fullmatch(r"[A-Z]{2,}\d{3,}(?:-\d+)?", prev_clean)
        ):
            rejected.append(RejectedAnchor(token_index=idx, reason="protected_token_neighbor"))
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
        if score < 0.65:
            rejected.append(RejectedAnchor(token_index=idx, reason="low_score"))
            continue
        anchors.append(
            AnchorCandidate(
                id=int(token.text),
                token_index=idx,
                anchor_text=prev.text,
                confidence=round(score, 3),
            )
        )
    return anchors, rejected


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


def analyze_page_layout(tokens: list[LayoutToken], *, page_height: float) -> PageAnalysis:
    if not tokens:
        return PageAnalysis((), (), (), {}, ())
    # Footnote region: numeric-labeled lines near bottom with smaller fonts.
    body_size_candidates = [t.size for t in tokens if t.size > 0 and t.top < page_height * 0.75]
    if not body_size_candidates:
        body_size_candidates = [t.size for t in tokens if t.size > 0]
    median_size = sorted(body_size_candidates)[len(body_size_candidates) // 2] if body_size_candidates else 0
    line_map: dict[int, list[tuple[int, LayoutToken]]] = {}
    for idx, tok in enumerate(tokens):
        line_map.setdefault(tok.line_id, []).append((idx, tok))
    ordered_lines = sorted(
        ((line_id, sorted(indexed, key=lambda x: x[1].x0)) for line_id, indexed in line_map.items()),
        key=lambda item: item[1][0][1].top if item[1] else 0.0,
    )
    line_top_index = {line_id: pos for pos, (line_id, _) in enumerate(ordered_lines)}

    footnote_token_indexes: set[int] = set()
    for line_id, indexed in ordered_lines:
        line = [tok for _, tok in indexed]
        if not line:
            continue
        first = line[0].text.strip().rstrip(").")
        content = " ".join(tok.text for tok in line[1:]).strip()
        avg_size = sum(tok.size for tok in line) / max(1, len(line))
        prev_gap = 0.0
        pos = line_top_index.get(line_id, 0)
        if pos > 0:
            prev_line = ordered_lines[pos - 1][1]
            if prev_line:
                prev_gap = line[0].top - prev_line[0][1].top
        is_footer = bool(re.search(r"\bP\s*a\s*g\s*e\b", content, flags=re.IGNORECASE))
        if (
            line[0].top >= page_height * 0.78
            and re.fullmatch(r"\d{1,3}", first)
            and not is_footer
            and (median_size == 0 or avg_size <= median_size * 0.95 or prev_gap >= 12.0)
        ):
            for idx, _ in indexed:
                footnote_token_indexes.add(idx)

    bodies = detect_footnote_bodies([tok for idx, tok in enumerate(tokens) if idx in footnote_token_indexes], page_height=page_height)
    body_indexes = tuple(idx for idx in range(len(tokens)) if idx not in footnote_token_indexes)
    anchors, rejected = detect_superscript_anchors(tokens, body_token_indexes=set(body_indexes))

    return PageAnalysis(
        body_token_indexes=body_indexes,
        footnote_token_indexes=tuple(sorted(footnote_token_indexes)),
        anchor_candidates=tuple(anchors),
        footnote_bodies=bodies,
        rejected_anchors=tuple(rejected),
    )
