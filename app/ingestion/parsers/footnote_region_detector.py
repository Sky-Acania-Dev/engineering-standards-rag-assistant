from __future__ import annotations

from dataclasses import dataclass
import re

from app.ingestion.parsers.pdfplumber_layout import LineInfo


@dataclass(frozen=True)
class FootnoteBody:
    page_number: int
    label: str
    content: str
    bbox: tuple[float, float, float, float]
    line_indexes: tuple[int, ...]


def detect_footnote_bodies(lines: list[LineInfo], page_height: float) -> tuple[list[FootnoteBody], set[int]]:
    if not lines:
        return [], set()
    region_start = page_height * 0.72
    region = [
        (idx, line)
        for idx, line in enumerate(lines)
        if line.top >= region_start and line.body_size <= _page_body_size(lines) * 1.05
    ]
    if not region:
        return [], set()

    bodies: list[FootnoteBody] = []
    consumed_lines: set[int] = set()
    current_label: str | None = None
    current_text: list[str] = []
    current_bbox: tuple[float, float, float, float] | None = None
    current_lines: list[int] = []

    for idx, line in region:
        text = "".join(c.text for c in line.chars).strip()
        if not text:
            continue
        label_match = re.match(r"^(\d{1,3})[\).\-:]?\s+(.*)$", text)
        if label_match:
            if current_label and current_text:
                bodies.append(
                    FootnoteBody(
                        page_number=line.page_number,
                        label=current_label,
                        content=" ".join(current_text).strip(),
                        bbox=current_bbox or (0.0, 0.0, 0.0, 0.0),
                        line_indexes=tuple(current_lines),
                    )
                )
                consumed_lines.update(current_lines)
            current_label = label_match.group(1)
            current_text = [label_match.group(2).strip()]
            current_bbox = (min(c.x0 for c in line.chars), line.top, max(c.x1 for c in line.chars), line.bottom)
            current_lines = [idx]
        elif current_label:
            current_text.append(text)
            left = min(c.x0 for c in line.chars)
            right = max(c.x1 for c in line.chars)
            current_bbox = (
                min(current_bbox[0], left) if current_bbox else left,
                min(current_bbox[1], line.top) if current_bbox else line.top,
                max(current_bbox[2], right) if current_bbox else right,
                max(current_bbox[3], line.bottom) if current_bbox else line.bottom,
            )
            current_lines.append(idx)

    if current_label and current_text:
        bodies.append(
            FootnoteBody(
                page_number=lines[0].page_number,
                label=current_label,
                content=" ".join(current_text).strip(),
                bbox=current_bbox or (0.0, 0.0, 0.0, 0.0),
                line_indexes=tuple(current_lines),
            )
        )
        consumed_lines.update(current_lines)

    return bodies, consumed_lines


def _page_body_size(lines: list[LineInfo]) -> float:
    sizes = sorted(line.body_size for line in lines if line.body_size > 0)
    if not sizes:
        return 0.0
    return sizes[len(sizes) // 2]
