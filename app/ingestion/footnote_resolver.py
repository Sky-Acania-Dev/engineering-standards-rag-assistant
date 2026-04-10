from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_PAGE_RE = re.compile(r"^##\s+Page\s+(\d+)\b", flags=re.IGNORECASE)
_FOOTNOTE_START_RE = re.compile(r"^(\d{1,3})[\.)]?\s+(.+)$")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_FN_TAG_RE = re.compile(r"\[fn:(\d{1,3})\]")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


@dataclass(frozen=True)
class FootnoteResolvedDocument:
    text: str
    page_footnotes: dict[int, tuple[dict[str, Any], ...]]


def _parse_pages(text: str) -> list[tuple[int | None, list[str]]]:
    pages: list[tuple[int | None, list[str]]] = []
    current_page: int | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        page_match = _PAGE_RE.match(line.strip())
        if page_match:
            if current_lines or current_page is not None:
                pages.append((current_page, current_lines))
            current_page = int(page_match.group(1))
            current_lines = [line.strip()]
            continue
        current_lines.append(line)

    if current_lines or current_page is not None:
        pages.append((current_page, current_lines))
    return pages


def _extract_page_footnotes(lines: list[str]) -> tuple[list[str], dict[int, str]]:
    if len(lines) < 2:
        return lines, {}
    candidate_start = None
    scan_start = 1 if lines and _PAGE_RE.match(lines[0]) else 0
    for idx in range(len(lines) - 1, scan_start - 1, -1):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if _FOOTNOTE_START_RE.match(stripped):
            candidate_start = idx
        else:
            if candidate_start is not None:
                break

    if candidate_start is None:
        return lines, {}

    entries: dict[int, str] = {}
    footnote_line_indexes: set[int] = set()
    for offset, raw_line in enumerate(lines[candidate_start:], start=candidate_start):
        stripped = raw_line.strip()
        if not stripped:
            continue
        start_match = _FOOTNOTE_START_RE.match(stripped)
        if start_match:
            entries[int(start_match.group(1))] = start_match.group(2).strip()
            footnote_line_indexes.add(offset)

    if not entries:
        return lines, {}

    # Require likely footnote content (URL/prose), not numeric table rows.
    if any(re.fullmatch(r"[\d.\-–\s/%]+", value) for value in entries.values()):
        return lines, {}

    prefix_text = "\n".join(lines[:candidate_start])
    referenced = 0
    for note_id in entries:
        marker = str(note_id)
        if (
            re.search(rf"\[{marker}\]|\({marker}\)", prefix_text)
            or re.search(rf"\b[A-Za-z]{{3,}}{marker}(?=\b|[.,;:])", prefix_text)
        ):
            referenced += 1
            continue
        range_hits = re.findall(
            r"\b((?:\d{1,3}\s*-\s*\d{1,3}|\d{1,3})(?:\s*,\s*(?:\d{1,3}\s*-\s*\d{1,3}|\d{1,3}))+)",
            prefix_text,
        )
        for raw in range_hits:
            if note_id in _expand_number_list(raw):
                referenced += 1
                break
    url_heavy = any(_URL_RE.search(value) for value in entries.values())
    non_footnote_body = [line for line in lines[:candidate_start] if line.strip() and not _PAGE_RE.match(line.strip())]
    if referenced == 0 and not url_heavy and non_footnote_body:
        return lines, {}

    kept_lines = [line for idx, line in enumerate(lines) if idx not in footnote_line_indexes]
    return kept_lines, entries


def _expand_number_list(raw: str) -> list[int]:
    expanded: list[int] = []
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if "-" in part:
            left, right = [s.strip() for s in part.split("-", maxsplit=1)]
            if left.isdigit() and right.isdigit():
                low, high = int(left), int(right)
                if low <= high and high - low <= 5:
                    expanded.extend(range(low, high + 1))
                    continue
        if part.isdigit():
            expanded.append(int(part))
    return expanded


def _resolve_page_body(
    lines: list[str],
    *,
    known_ids: set[int],
    footnote_content: dict[int, str],
) -> tuple[list[str], tuple[dict[str, Any], ...]]:
    if not known_ids:
        return lines, ()

    attached: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, str]] = set()

    def _record(id_value: int, anchor: str) -> str:
        normalized_anchor = anchor.strip().strip(",.;:")
        pair = (id_value, normalized_anchor)
        if pair not in seen_pairs and id_value in footnote_content:
            entry: dict[str, Any] = {
                "id": id_value,
                "content": footnote_content[id_value],
                "anchor_text": normalized_anchor,
            }
            url_match = _URL_RE.search(footnote_content[id_value])
            if url_match:
                entry["url"] = url_match.group(0)
            attached.append(entry)
            seen_pairs.add(pair)
        return f"[fn:{id_value}]"

    resolved_lines: list[str] = []
    for line in lines:
        working = line.translate(_SUPERSCRIPT_MAP)

        # Split merged tokens like Rule3 / R80219 / 2123 (where suffix is known id).
        for note_id in sorted(known_ids, reverse=True):
            suffix = str(note_id)
            pattern = re.compile(rf"\b([A-Za-z][A-Za-z0-9]{{2,}}|\d{{2,}}){suffix}\b")

            def merged_repl(match: re.Match[str], *, _note_id: int = note_id) -> str:
                anchor = match.group(1)
                return f"{anchor} {_record(_note_id, anchor)}"

            working = pattern.sub(merged_repl, working)

        # Repair "Chapter 21 23" style corruption.
        def chapter_repl(match: re.Match[str]) -> str:
            chapter_token, marker = match.group(1), int(match.group(2))
            if marker not in known_ids:
                return match.group(0)
            return f"{chapter_token} {_record(marker, chapter_token)}"

        working = re.sub(r"\b(Chapter\s+\d{1,3})\s+(\d{1,3})\b", chapter_repl, working, flags=re.IGNORECASE)

        # Expand explicit lists/ranges like 8-9,18.
        def list_repl(match: re.Match[str]) -> str:
            raw = match.group(1)
            ids = [num for num in _expand_number_list(raw) if num in known_ids]
            if not ids:
                return match.group(0)
            tags = "".join(_record(num, "reference") for num in ids)
            return tags

        working = re.sub(r"\b((?:\d{1,3}\s*-\s*\d{1,3}|\d{1,3})(?:\s*,\s*(?:\d{1,3}\s*-\s*\d{1,3}|\d{1,3}))+)", list_repl, working)

        # Handle punctuation-attached markers like website.6 or Code,8
        def punct_repl(match: re.Match[str]) -> str:
            punct, raw_id = match.group(1), int(match.group(2))
            if raw_id not in known_ids:
                return match.group(0)
            return f"{punct} {_record(raw_id, 'punctuation')}"

        working = re.sub(r"(?<!\[fn)(?<!\d)([.,;:])\s*(\d{1,3})\b", punct_repl, working)

        resolved_lines.append(working)

    return resolved_lines, tuple(attached)


def resolve_footnotes_by_page(document_text: str) -> FootnoteResolvedDocument:
    pages = _parse_pages(document_text)

    body_pages: list[tuple[int | None, list[str]]] = []
    all_footnotes: dict[int, str] = {}
    for page_num, page_lines in pages:
        body_lines, extracted = _extract_page_footnotes(page_lines)
        body_pages.append((page_num, body_lines))
        for key, value in extracted.items():
            all_footnotes.setdefault(key, value)

    known_ids = set(all_footnotes)
    page_footnotes: dict[int, tuple[dict[str, Any], ...]] = {}
    rendered_pages: list[str] = []

    for page_num, body_lines in body_pages:
        resolved_lines, attached = _resolve_page_body(
            body_lines,
            known_ids=known_ids,
            footnote_content=all_footnotes,
        )
        if page_num is not None and attached:
            page_footnotes[page_num] = attached
        rendered_pages.append("\n".join(resolved_lines))

    return FootnoteResolvedDocument(
        text="\n\n".join(block for block in rendered_pages if block.strip()),
        page_footnotes=page_footnotes,
    )
