from __future__ import annotations

from dataclasses import dataclass
import re

from app.ingestion.parsers.pdfplumber_layout import CharInfo, LineInfo


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

    lines_by_top = sorted(enumerate(lines), key=lambda item: item[1].top)
    median_size = _page_body_size(lines)
    min_top = page_height * 0.5

    first_candidate_pos: int | None = None
    for pos, (_, line) in enumerate(lines_by_top):
        parsed = _parse_label_from_line(line)
        text = "".join(c.text for c in line.chars).strip()
        if line.top < min_top or parsed is None:
            continue
        if _is_footer_artifact(text, line, page_height=page_height):
            continue
        size_ok = line.body_size <= median_size * 1.25 if median_size > 0 else True
        if size_ok and (_looks_like_footnote_lexical(text) or _has_vertical_gap(lines_by_top, pos)):
            first_candidate_pos = pos
            break

    if first_candidate_pos is None:
        return [], set()

    bodies: list[FootnoteBody] = []
    consumed_lines: set[int] = set()
    current_label: str | None = None
    current_text: list[str] = []
    current_bbox: tuple[float, float, float, float] | None = None
    current_lines: list[int] = []

    for idx, line in lines_by_top[first_candidate_pos:]:
        raw_text = "".join(c.text for c in line.chars).strip()
        if not raw_text:
            continue
        if _is_footer_artifact(raw_text, line, page_height=page_height):
            continue

        parsed = _parse_label_from_line(line)
        if parsed is not None:
            label, remainder = parsed
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
            current_label = label
            current_text = [remainder]
            current_bbox = (min(c.x0 for c in line.chars), line.top, max(c.x1 for c in line.chars), line.bottom)
            current_lines = [idx]
            continue

        if current_label is not None:
            current_text.append(raw_text)
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

    deduped: dict[str, FootnoteBody] = {}
    for body in bodies:
        if body.label not in deduped:
            deduped[body.label] = body
    ordered = sorted(deduped.values(), key=lambda b: int(b.label))
    return ordered, consumed_lines


def _parse_label_from_line(line: LineInfo) -> tuple[str, str] | None:
    chars = [c for c in line.chars if c.text]
    if not chars:
        return None

    first_non_space = next((i for i, c in enumerate(chars) if c.text.strip()), None)
    if first_non_space is None:
        return None
    i = first_non_space

    # first token cluster
    first_cluster: list[CharInfo] = []
    while i < len(chars) and chars[i].text.strip():
        first_cluster.append(chars[i])
        i += 1
    first_token = "".join(c.text for c in first_cluster).strip()

    # optional second numeric cluster for split digits: "1 0 http"
    second_token = ""
    j = i
    while j < len(chars) and not chars[j].text.strip():
        j += 1
    k = j
    second_cluster: list[CharInfo] = []
    while k < len(chars) and chars[k].text.strip():
        second_cluster.append(chars[k])
        k += 1
    if second_cluster:
        second_token = "".join(c.text for c in second_cluster).strip()

    label_candidate = first_token
    consume_until = i
    if re.fullmatch(r"\d", first_token) and re.fullmatch(r"\d", second_token):
        gap = second_cluster[0].x0 - first_cluster[-1].x1
        if gap <= max(8.0, line.body_size * 0.9):
            label_candidate = first_token + second_token
            consume_until = k

    stripped_label = label_candidate.rstrip(').:-')
    if not re.fullmatch(r"\d{1,3}", stripped_label):
        return None

    remainder = "".join(c.text for c in chars[consume_until:]).strip()
    if not remainder:
        return None
    return str(int(stripped_label)), remainder


def _is_footer_artifact(text: str, line: LineInfo, *, page_height: float) -> bool:
    normalized = " ".join(text.split())
    if not normalized:
        return True
    if normalized in {"| Page", "Page", "|"}:
        return True
    if re.fullmatch(r"\d+\s*\|\s*Page", normalized, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"Page\s*\d+", normalized, flags=re.IGNORECASE):
        return True
    if line.top >= page_height * 0.95 and len(normalized.split()) <= 3:
        return True
    return False


def _has_vertical_gap(lines_by_top: list[tuple[int, LineInfo]], pos: int) -> bool:
    if pos == 0:
        return True
    prev_line = lines_by_top[pos - 1][1]
    line = lines_by_top[pos][1]
    gap = max(0.0, line.top - prev_line.bottom)
    return gap >= max(4.0, line.body_size * 1.05)


def _looks_like_footnote_lexical(text: str) -> bool:
    lowered = text.lower()
    return (
        "http://" in lowered
        or "https://" in lowered
        or "www." in lowered
        or "code references" in lowered
        or "found at:" in lowered
    )


def _page_body_size(lines: list[LineInfo]) -> float:
    sizes = sorted(line.body_size for line in lines if line.body_size > 0)
    if not sizes:
        return 0.0
    return sizes[len(sizes) // 2]
