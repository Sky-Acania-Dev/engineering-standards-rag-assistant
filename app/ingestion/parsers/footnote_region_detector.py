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


def detect_footnote_bodies(
    lines: list[LineInfo],
    page_height: float,
    *,
    excluded_line_indexes: set[int] | None = None,
) -> tuple[list[FootnoteBody], set[int]]:
    if not lines:
        return [], set()

    lines_by_top = sorted(enumerate(lines), key=lambda item: item[1].top)
    median_size = _page_body_size(lines)
    min_top = page_height * 0.4

    first_candidate_pos: int | None = None
    excluded = excluded_line_indexes or set()

    for pos, (line_idx, line) in enumerate(lines_by_top):
        if line_idx in excluded:
            continue
        parsed = _parse_label_from_line(line)
        text = "".join(c.text for c in line.chars).strip()
        if line.top < min_top or parsed is None:
            continue
        if _is_footer_artifact(text, line, page_height=page_height) or _looks_like_table_row(text):
            continue
        size_ok = line.body_size <= median_size * 1.2 if median_size > 0 else True
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
        if idx in excluded:
            continue
        raw_text = "".join(c.text for c in line.chars).strip()
        if not raw_text:
            continue
        if _is_footer_artifact(raw_text, line, page_height=page_height) or _looks_like_table_row(raw_text):
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

    if not _is_valid_footnote_block(ordered, lines_by_top, first_candidate_pos, page_height, median_size):
        return [], set()
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


def _looks_like_table_row(text: str) -> bool:
    normalized = " ".join(text.split())
    if "|" in normalized:
        return True
    return bool(re.search(r"\S+\s{2,}\S+\s{2,}\S+", text))


def _is_valid_footnote_block(
    bodies: list[FootnoteBody],
    lines_by_top: list[tuple[int, LineInfo]],
    first_candidate_pos: int,
    page_height: float,
    median_size: float,
) -> bool:
    if not bodies:
        return False

    # Hard negative: long, prose-like numbered list entries are usually normal body content.
    if len(bodies) >= 3:
        long_entries = sum(1 for body in bodies if len(body.content.split()) >= 6 and not _looks_like_footnote_lexical(body.content))
        if long_entries >= max(2, len(bodies) // 2):
            return False

    # Positive signals: require multiple strong indicators together (prefer omission over false positives).
    lexical_count = sum(1 for body in bodies if _looks_like_footnote_lexical(body.content))
    if lexical_count == 0:
        return False
    short_entry_count = sum(1 for body in bodies if len(body.content.split()) <= 8)
    first_top = min(body.bbox[1] for body in bodies)
    block_bottom = max(body.bbox[3] for body in bodies)
    compact_bottom_block = first_top >= page_height * 0.62 and (block_bottom - first_top) <= page_height * 0.33

    smaller_than_body = False
    if median_size > 0:
        body_sizes = [
            lines_by_top[idx][1].body_size
            for idx in range(first_candidate_pos)
            if lines_by_top[idx][1].top < first_top
        ]
        if body_sizes:
            main_body_size = sorted(body_sizes)[len(body_sizes) // 2]
            candidate_sizes = [line.body_size for _, line in lines_by_top[first_candidate_pos:]]
            candidate_size = sorted(candidate_sizes)[len(candidate_sizes) // 2] if candidate_sizes else 0.0
            smaller_than_body = candidate_size > 0 and candidate_size <= main_body_size * 0.95

    vertical_gap = _has_vertical_gap(lines_by_top, first_candidate_pos)

    positive_signals = 0
    if lexical_count >= max(1, len(bodies) // 3):
        positive_signals += 1
    if short_entry_count >= max(1, len(bodies) // 2):
        positive_signals += 1
    if compact_bottom_block:
        positive_signals += 1
    if smaller_than_body:
        positive_signals += 1
    if vertical_gap:
        positive_signals += 1

    if positive_signals < 2:
        return False

    if len(bodies) >= 6 and lexical_count <= 1 and not smaller_than_body:
        return False
    return True


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
