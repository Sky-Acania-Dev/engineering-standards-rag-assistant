from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class CharInfo:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    fontname: str
    page_number: int
    order: int


@dataclass(frozen=True)
class LineInfo:
    page_number: int
    top: float
    bottom: float
    baseline: float
    body_size: float
    chars: tuple[CharInfo, ...]


def extract_page_chars(page: Any, page_number: int) -> list[CharInfo]:
    chars: list[CharInfo] = []
    for idx, raw in enumerate(getattr(page, "chars", []) or []):
        if isinstance(raw, CharInfo):
            chars.append(raw)
            continue
        text = str(raw.get("text", ""))
        if not text:
            continue
        chars.append(
            CharInfo(
                text=text,
                x0=float(raw.get("x0", 0.0)),
                x1=float(raw.get("x1", 0.0)),
                top=float(raw.get("top", 0.0)),
                bottom=float(raw.get("bottom", 0.0)),
                size=float(raw.get("size", 0.0) or 0.0),
                fontname=str(raw.get("fontname", "")),
                page_number=page_number,
                order=idx,
            )
        )
    return chars


def build_visual_lines(chars: list[CharInfo], y_tolerance: float = 6.5) -> list[LineInfo]:
    if not chars:
        return []
    sorted_chars = sorted(chars, key=lambda c: (c.top, c.x0, c.order))
    groups: list[list[CharInfo]] = []
    for char in sorted_chars:
        best_idx: int | None = None
        best_delta = 1e9
        for idx, group in enumerate(groups):
            avg_bottom = sum(c.bottom for c in group) / len(group)
            delta = abs(char.bottom - avg_bottom)
            if delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is not None and best_delta <= y_tolerance:
            groups[best_idx].append(char)
        else:
            groups.append([char])

    lines: list[LineInfo] = []
    for group in groups:
        ordered = tuple(sorted(group, key=lambda c: (c.x0, c.order)))
        sizes = [c.size for c in ordered if c.size > 0]
        bottoms = [c.bottom for c in ordered]
        body_size = median(sizes) if sizes else 0.0
        baseline = median(bottoms) if bottoms else 0.0
        lines.append(
            LineInfo(
                page_number=ordered[0].page_number,
                top=min(c.top for c in ordered),
                bottom=max(c.bottom for c in ordered),
                baseline=baseline,
                body_size=body_size,
                chars=ordered,
            )
        )
    return lines
