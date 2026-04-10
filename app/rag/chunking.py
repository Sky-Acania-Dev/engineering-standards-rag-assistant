from __future__ import annotations

from dataclasses import dataclass, replace
import re
from difflib import SequenceMatcher
from typing import Iterable


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    section: str
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    content_type: str = "body_text"
    section_path: tuple[str, ...] = ()
    table_id: str | None = None
    figure_id: str | None = None
    figure_ref: str | None = None
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None


@dataclass(frozen=True)
class _Block:
    text: str
    content_type: str
    page_start: int | None
    page_end: int | None
    section: str
    section_path: tuple[str, ...]
    table_id: str | None = None
    figure_id: str | None = None
    figure_ref: str | None = None
    heading_level: int | None = None
    protected: bool = False


def _tokenize(text: str) -> list[str]:
    return text.split()


def _normalize_line_noise(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"\b(\d+)\s*\|\s*P\s*a\s*g\s*e\b", r"Page \1", cleaned, flags=re.IGNORECASE)
    return cleaned


def _looks_like_toc_line(line: str) -> bool:
    return bool(re.search(r"\.{2,}\s*\d+$", line)) or bool(re.match(r"^(Contents|Table of Contents)\b", line, flags=re.IGNORECASE))


def _is_note_block(block: str) -> bool:
    first = block.splitlines()[0].strip() if block.splitlines() else ""
    return bool(re.match(r"^(note|notes|exception|warning|caution)\b[:\-]?", first, flags=re.IGNORECASE))


def _is_list_block(lines: list[str]) -> bool:
    if not lines:
        return False
    return all(
        line.startswith("-")
        or line.startswith("*")
        or bool(re.match(r"^\d+[\.)]\s", line))
        for line in lines
    )


def _parse_image_ref(block: str) -> str | None:
    match = re.search(r"^\[IMAGE\]\s*(.+)$", block.strip(), flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_numeric_id(prefix: str, text: str) -> str | None:
    pattern = rf"\b{re.escape(prefix)}\s*([A-Za-z0-9_.\-]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).rstrip(".:")
        return f"{prefix} {value}"
    return None


def _serialize_table_block(block: str) -> tuple[str, str | None]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    caption: str | None = None
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("[TABLE_CAPTION]"):
            caption = line.replace("[TABLE_CAPTION]", "", 1).strip()
            continue
        if line.startswith("[TABLE]") or line.startswith("[/TABLE]"):
            continue
        if "|" in line:
            parts = [part.strip() for part in line.split("|") if part.strip()]
            if parts:
                rows.append(parts)

    if not rows:
        return block, caption

    header = rows[0]
    data_rows = rows[1:]
    rendered: list[str] = []
    if caption:
        rendered.append(f"Table caption: {caption}")
    rendered.append("Table columns: " + "; ".join(header))

    for idx, row in enumerate(data_rows, start=1):
        if set(row) <= {"---", "--", "-"}:
            continue
        pairs = [f"{header[col_idx] if col_idx < len(header) else f'col_{col_idx + 1}'}={value}" for col_idx, value in enumerate(row)]
        rendered.append(f"Row {idx}: " + "; ".join(pairs))

    return "\n".join(rendered), caption


def _update_heading_stack(
    heading_stack: list[tuple[int, str]],
    *,
    level: int,
    heading_text: str,
) -> tuple[str, tuple[str, ...]]:
    while heading_stack and heading_stack[-1][0] >= level:
        heading_stack.pop()
    heading_stack.append((level, heading_text))
    return heading_text, tuple(title for _, title in heading_stack)


def _normalize_heading_text(text: str) -> str:
    cleaned = " ".join(text.split())
    cleaned = cleaned.replace("###", " ")
    cleaned = re.sub(r"\bIC-\s+P-", "IC-P-", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # OCR/extraction artifacts often append page digits directly to the heading.
    cleaned = re.sub(r"(?<=[A-Za-z])\d{1,4}$", "", cleaned).strip()
    cleaned = re.sub(r"\b(Section\s+[A-Za-z]\d{3,})\d{1,2}$", r"\1", cleaned, flags=re.IGNORECASE)
    return cleaned


def _detect_structural_heading(line: str) -> tuple[int, str] | None:
    markdown_heading = re.match(r"^(#{1,6})\s+(.+)$", line)
    if markdown_heading:
        # Markdown markers are parser hints. Only treat them as structural
        # sections if the embedded text matches known heading patterns.
        inner = markdown_heading.group(2).strip()
        detected = _detect_structural_heading(inner)
        if detected is not None:
            return detected
        normalized = _normalize_heading_text(inner)
        if (
            len(normalized.split()) >= 2
            and not re.fullmatch(r"[A-Z]{2,}\.?", normalized)
            and "http://" not in normalized.lower()
            and "https://" not in normalized.lower()
        ):
            return len(markdown_heading.group(1)), normalized
        return None

    chapter_heading = re.match(
        r"^(chapter)\s+(\d+[A-Za-z]?)\s*[:.\-–]?\s*(.+)?$",
        line,
        flags=re.IGNORECASE,
    )
    if chapter_heading:
        chapter_label = chapter_heading.group(1).capitalize()
        chapter_number = chapter_heading.group(2)
        chapter_title = (chapter_heading.group(3) or "").strip()
        heading = f"{chapter_label} {chapter_number}"
        if chapter_title:
            heading = f"{heading}: {chapter_title}"
        return 2, _normalize_heading_text(heading)

    section_heading = re.match(r"^(section)\s+([\w.\-]+)\s*[:.\-–]?\s*(.+)?$", line, flags=re.IGNORECASE)
    if section_heading:
        section_label = section_heading.group(1).capitalize()
        section_number = section_heading.group(2)
        section_title = (section_heading.group(3) or "").strip()
        heading = f"{section_label} {section_number}"
        if section_title:
            heading = f"{heading}: {section_title}"
        return 2, _normalize_heading_text(heading)

    decimal_heading = re.match(r"^(\d{1,2}(?:\.\d+){1,3})\s+(.+)$", line)
    if decimal_heading:
        depth = decimal_heading.group(1).count(".") + 3
        heading = f"{decimal_heading.group(1)} {decimal_heading.group(2)}"
        return min(depth, 6), _normalize_heading_text(heading)

    return None


def _is_image_artifact_block(text: str) -> bool:
    compact = " ".join(text.split())
    marker = r"\[IMAGES?\]\s+\d+\s+embedded\s+image\(s\)"
    if not re.search(marker, compact, flags=re.IGNORECASE):
        return False
    # Cover/image placeholder artifacts are usually very short title + marker text.
    # Avoid classifying regular running text that merely mentions an image marker.
    return len(compact.split()) <= 24 and bool(
        re.fullmatch(rf"(?:.+\s+)?{marker}", compact, flags=re.IGNORECASE)
    )


def _is_chapter_heading(heading: str) -> bool:
    return bool(re.match(r"^Chapter\s+\d+[A-Za-z]?(?:\s*:.*)?$", heading, flags=re.IGNORECASE))


def _is_section_heading(heading: str) -> bool:
    return bool(re.match(r"^\d+(?:\.\d+){1,3}\s+.+$", heading)) or bool(
        re.match(r"^Section\s+[\w.\-]+(?:\s*:.*)?$", heading, flags=re.IGNORECASE)
    )


def _is_standalone_heading_line(line: str) -> bool:
    compact = " ".join(line.split())
    if not compact:
        return False
    if "http://" in compact.lower() or "https://" in compact.lower():
        return False
    if re.search(r"\bwww\.", compact, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"[A-Z]{2,}\.?", compact):
        return False
    if re.fullmatch(r"[A-Z]-\d{4}-\d{3,}", compact):
        return False
    token_count = len(compact.split())
    if token_count > 14:
        return False
    if token_count <= 2 and compact.endswith("."):
        return False
    if token_count > 7 and re.search(r"\b(shall|must|comply|provided|required)\b", compact, flags=re.IGNORECASE):
        return False
    if re.search(r"[!?;]", compact):
        return False
    return True


def _table_row_cells(line: str) -> list[str] | None:
    if "|" in line:
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        return cells or None
    cells = [cell.strip() for cell in re.split(r"\s{2,}", line) if cell.strip()]
    if len(cells) >= 2:
        return cells
    return None


def _serialize_unmarked_table(lines: list[str], *, page: int | None) -> tuple[str, str | None] | None:
    header = _table_row_cells(lines[0]) if lines else None
    if header is None:
        return None
    column_count = len(header)
    if column_count < 2:
        return None
    rows: list[list[str]] = [header]

    for line in lines[1:]:
        cells = _table_row_cells(line)
        if cells is None:
            return None
        # Common PDF artifact: first column is separated by wide spaces, while
        # trailing numeric columns collapse to single spaces.
        if len(cells) < column_count and cells:
            prefix = cells[:-1]
            needed_tail_columns = column_count - len(prefix)
            expanded_tail = cells[-1].split(maxsplit=max(0, needed_tail_columns - 1))
            if len(prefix) + len(expanded_tail) == column_count:
                # Preserve multi-word values in the last column (e.g. "Not Required").
                cells = cells[:-1] + expanded_tail
        rows.append(cells)

    if len(rows) < 3:
        return None
    if any(len(row) != column_count for row in rows):
        return None

    separator = "| " + " | ".join("---" for _ in header) + " |"
    rendered = ["| " + " | ".join(header) + " |", separator]
    rendered.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(rendered), f"table:p{page}" if page is not None else "table:unknown"


def _segment_inline_table_runs(lines: list[str], *, page: int | None) -> list[tuple[str, list[str]]] | None:
    if len(lines) < 4:
        return None

    segments: list[tuple[str, list[str]]] = []
    idx = 0
    while idx < len(lines):
        if _table_row_cells(lines[idx]) is None:
            start = idx
            while idx < len(lines) and _table_row_cells(lines[idx]) is None:
                idx += 1
            segments.append(("body", lines[start:idx]))
            continue

        start = idx
        while idx < len(lines) and _table_row_cells(lines[idx]) is not None:
            idx += 1
        run = lines[start:idx]
        if len(run) >= 3 and _serialize_unmarked_table(run, page=page) is not None:
            segments.append(("table", run))
        else:
            segments.append(("body", run))

    merged: list[tuple[str, list[str]]] = []
    for kind, segment_lines in segments:
        if not segment_lines:
            continue
        if merged and merged[-1][0] == kind:
            merged[-1][1].extend(segment_lines)
        else:
            merged.append((kind, list(segment_lines)))

    if not any(kind == "table" for kind, _ in merged):
        return None
    if all(kind == "table" for kind, _ in merged):
        return None
    return merged


def _iter_structured_blocks(document_text: str) -> list[_Block]:
    raw_blocks = [b.strip() for b in document_text.split("\n\n") if b.strip()]
    blocks: list[_Block] = []
    page: int | None = None
    heading_stack: list[tuple[int, str]] = []
    section = "Section 1"
    section_path: tuple[str, ...] = ()
    pending_image_ref: str | None = None

    def _append_content_block(lines: list[str], *, force_content_type: str | None = None) -> None:
        nonlocal pending_image_ref
        if not lines:
            return
        if (
            force_content_type is None
            and not lines[0].startswith("[TABLE]")
            and not all(line.startswith("|") for line in lines)
        ):
            segmented = _segment_inline_table_runs(lines, page=page)
            if segmented is not None:
                for _, segment_lines in segmented:
                    _append_content_block(segment_lines)
                return

        full_block = "\n".join(lines)
        content_type = force_content_type or "body_text"
        protected = force_content_type == "toc"
        table_id: str | None = None
        figure_id: str | None = None
        figure_ref: str | None = None

        if force_content_type is None:
            if _looks_like_toc_line(lines[0]) or any(_looks_like_toc_line(line) for line in lines):
                content_type = "toc"
                protected = True
            elif lines[0].startswith("[IMAGE]"):
                pending_image_ref = _parse_image_ref(full_block)
                blocks.append(
                    _Block(
                        text=full_block,
                        content_type="header_footer_noise",
                        page_start=page,
                        page_end=page,
                        section=section,
                        section_path=section_path,
                        figure_ref=pending_image_ref,
                        protected=True,
                    )
                )
                return
            elif lines[0].startswith("[IMAGE_CAPTION]"):
                caption = lines[0].replace("[IMAGE_CAPTION]", "", 1).strip()
                content_type = "figure_caption"
                protected = True
                figure_id = _extract_numeric_id("Figure", caption)
                figure_ref = pending_image_ref
            elif lines[0].startswith("[TABLE]"):
                content_type = "table"
                protected = True
                full_block, table_caption = _serialize_table_block(full_block)
                table_id = _extract_numeric_id("Table", table_caption) if table_caption else None
                if not table_id:
                    table_id = f"table:p{page}" if page is not None else "table:unknown"
            elif all(line.startswith("|") for line in lines):
                content_type = "table"
                protected = True
                table_id = f"table:p{page}" if page is not None else "table:unknown"
            elif (serialized := _serialize_unmarked_table(lines, page=page)) is not None:
                content_type = "table"
                protected = True
                full_block, table_id = serialized
            elif _is_note_block(full_block):
                content_type = "note"
                protected = True
            elif _is_list_block(lines):
                content_type = "list"
                protected = True
            elif _is_image_artifact_block(full_block):
                content_type = "image_artifact"
                protected = True

        blocks.append(
            _Block(
                text=full_block,
                content_type=content_type,
                page_start=page,
                page_end=page,
                section=section,
                section_path=section_path,
                table_id=table_id,
                figure_id=figure_id,
                figure_ref=figure_ref,
                protected=protected,
            )
        )

    for raw_block in raw_blocks:
        lines = [_normalize_line_noise(line) for line in raw_block.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            continue

        body_buffer: list[str] = []
        cursor = 0
        while cursor < len(lines):
            current = lines[cursor]
            page_match = re.match(r"^##\s+Page\s+(\d+)\b", current, flags=re.IGNORECASE)
            if page_match:
                _append_content_block(body_buffer)
                body_buffer = []
                page = int(page_match.group(1))
                cursor += 1
                continue

            if _looks_like_toc_line(current):
                _append_content_block(body_buffer)
                body_buffer = []
                _append_content_block(lines[cursor:], force_content_type="toc")
                break

            heading_detected = _detect_structural_heading(current)
            if heading_detected is not None and _is_standalone_heading_line(current):
                _append_content_block(body_buffer)
                body_buffer = []
                level, heading_text = heading_detected
                section, section_path = _update_heading_stack(heading_stack, level=level, heading_text=heading_text)
                blocks.append(
                    _Block(
                        text=heading_text,
                        content_type="section_header",
                        page_start=page,
                        page_end=page,
                        section=section,
                        section_path=section_path,
                        heading_level=level,
                        protected=True,
                    )
                )
                cursor += 1
                continue

            body_buffer.append(current)
            cursor += 1

        _append_content_block(body_buffer)

    return blocks


def _normalized_chunk_text(text: str) -> str:
    collapsed = " ".join(text.lower().split())
    collapsed = re.sub(r"\bpage\s+\d+\b", "", collapsed, flags=re.IGNORECASE)
    return collapsed.strip()


def _is_near_duplicate(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.96


def _deduplicate_chunks(chunks: list[TextChunk]) -> list[TextChunk]:
    kept: list[TextChunk] = []
    normalized_seen: list[str] = []

    for chunk in chunks:
        normalized = _normalized_chunk_text(chunk.text)
        should_skip = any(_is_near_duplicate(normalized, seen) for seen in normalized_seen)
        if should_skip and chunk.content_type in {"body_text", "image_artifact", "toc"}:
            continue
        kept.append(chunk)
        normalized_seen.append(normalized)

    reindexed: list[TextChunk] = []
    for new_id, chunk in enumerate(kept):
        reindexed.append(replace(chunk, chunk_id=new_id))

    linked: list[TextChunk] = []
    for idx, chunk in enumerate(reindexed):
        prev_chunk_id = reindexed[idx - 1].chunk_id if idx > 0 else None
        next_chunk_id = reindexed[idx + 1].chunk_id if idx + 1 < len(reindexed) else None
        linked.append(replace(chunk, prev_chunk_id=prev_chunk_id, next_chunk_id=next_chunk_id))

    return linked


def chunk_document_by_section(
    document_text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 150,
    soft_split_ratio: float = 0.7,
    hard_split_ratio: float = 1.25,
) -> list[TextChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    if not 0 < soft_split_ratio <= 1:
        raise ValueError("soft_split_ratio must be in (0, 1]")
    if hard_split_ratio < 1:
        raise ValueError("hard_split_ratio must be >= 1")

    blocks = _iter_structured_blocks(document_text)
    chunks: list[TextChunk] = []
    chunk_id = 0

    active_chapter: str | None = None
    active_section_heading: str | None = None
    body_units: list[list[tuple[str, int | None]]] = []
    body_section = "Section 1"
    body_section_path: tuple[str, ...] = ()
    toc_buffer: list[_Block] = []

    def _current_section_metadata() -> tuple[str, tuple[str, ...]]:
        if active_section_heading is not None:
            if active_chapter and active_section_heading != active_chapter:
                return active_section_heading, (active_chapter, active_section_heading)
            return active_section_heading, (active_section_heading,)
        if active_chapter is not None:
            return active_chapter, (active_chapter,)
        return "Section 1", ()

    def _body_heading_prefix(page: int | None) -> list[tuple[str, int | None]]:
        section, path = _current_section_metadata()
        lines: list[str] = []
        if path:
            lines.extend(path)
        elif section:
            lines.append(section)
        return [(line, page) for line in lines if line]

    def _emit_toc_buffer() -> None:
        nonlocal chunk_id, toc_buffer
        if not toc_buffer:
            return
        # Intentionally emit a single contiguous TOC chunk even if it exceeds
        # body chunk_size, so TOC ordering stays complete and reviewable.
        text = "\n".join(block.text for block in toc_buffer if block.text.strip())
        page_start = next((block.page_start for block in toc_buffer if block.page_start is not None), None)
        page_end = next((block.page_end for block in reversed(toc_buffer) if block.page_end is not None), None)
        toc_section = active_section_heading or active_chapter or "Table of Contents"
        toc_path = tuple(p for p in (active_chapter, active_section_heading) if p) or ("Table of Contents",)
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                section=toc_section,
                text=text,
                token_count=len(_tokenize(text)),
                page_start=page_start,
                page_end=page_end,
                content_type="toc",
                section_path=toc_path,
            )
        )
        chunk_id += 1
        toc_buffer = []

    def _flush_body_chunks() -> None:
        nonlocal body_units, chunk_id
        if not body_units:
            return
        soft_limit = max(1, int(chunk_size * soft_split_ratio))
        hard_limit = max(chunk_size + 1, int(chunk_size * hard_split_ratio))

        def emit_tokens(tokens: list[tuple[str, int | None]]) -> None:
            nonlocal chunk_id
            chunk_pages = [page for _, page in tokens if page is not None]
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=body_section,
                    text=" ".join(token for token, _ in tokens),
                    token_count=len(tokens),
                    page_start=chunk_pages[0] if chunk_pages else None,
                    page_end=chunk_pages[-1] if chunk_pages else None,
                    content_type="body_text",
                    section_path=body_section_path,
                )
            )
            chunk_id += 1

        current: list[tuple[str, int | None]] = []
        for unit in body_units:
            unit_tokens = unit
            if len(unit_tokens) > chunk_size:
                if current:
                    # Avoid emitting tiny heading-only chunks; attach heading prefix
                    # to the first large paragraph and then split.
                    if len(current) <= max(8, overlap // 2):
                        unit_tokens = current + unit_tokens
                        current = []
                    else:
                        emit_tokens(current)
                        current = current[-overlap:] if overlap else []
                while len(unit_tokens) > chunk_size:
                    emit_tokens(unit_tokens[:chunk_size])
                    unit_tokens = unit_tokens[chunk_size - overlap :] if overlap else unit_tokens[chunk_size:]
                if unit_tokens:
                    current = unit_tokens
                continue

            projected = len(current) + len(unit_tokens)
            if current and projected > chunk_size and len(current) >= soft_limit:
                # Soft policy: prefer splitting at this paragraph boundary.
                emit_tokens(current)
                current = current[-overlap:] if overlap else []
            elif current and projected > hard_limit:
                # No section boundary was found by hard limit; fall back to
                # normal target-sized split with overlap behavior.
                merged = current + unit_tokens
                emit_tokens(merged[:chunk_size])
                current = merged[chunk_size - overlap :] if overlap else merged[chunk_size:]
                continue
            current.extend(unit_tokens)

        if current:
            emit_tokens(current)

        body_units = []

    def _start_new_section(page: int | None) -> None:
        nonlocal body_section, body_section_path, body_units
        body_section, body_section_path = _current_section_metadata()
        prefix = _body_heading_prefix(page)
        body_units = [prefix] if prefix else []

    for block in blocks:
        if block.content_type == "toc":
            if body_units:
                _flush_body_chunks()
            toc_buffer.append(block)
            continue
        _emit_toc_buffer()

        if block.content_type == "section_header":
            _flush_body_chunks()
            if _is_chapter_heading(block.text):
                active_chapter = block.text
                active_section_heading = None
            elif _is_section_heading(block.text):
                active_section_heading = block.text
                if active_chapter is None:
                    active_chapter = block.text if block.text.lower().startswith("chapter ") else None
            else:
                active_section_heading = block.text
            continue

        if block.content_type == "header_footer_noise":
            continue

        if block.content_type == "image_artifact":
            _flush_body_chunks()
            artifact_tokens = _tokenize(block.text)
            section_name, section_path = _current_section_metadata()
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=section_name,
                    text=block.text,
                    token_count=len(artifact_tokens),
                    page_start=block.page_start,
                    page_end=block.page_end,
                    content_type="image_artifact",
                    section_path=section_path,
                )
            )
            chunk_id += 1
            continue

        block_tokens = _tokenize(block.text)
        if not block_tokens:
            continue

        if block.protected and block.content_type != "body_text":
            _flush_body_chunks()
            section_name, section_path = _current_section_metadata()
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=section_name,
                    text=block.text,
                    token_count=len(block_tokens),
                    page_start=block.page_start,
                    page_end=block.page_end,
                    content_type=block.content_type,
                    section_path=section_path,
                    table_id=block.table_id,
                    figure_id=block.figure_id,
                    figure_ref=block.figure_ref,
                )
            )
            chunk_id += 1
            continue

        if not body_units:
            _start_new_section(block.page_start)
        body_units.append([(token, block.page_end) for token in block_tokens])

    _emit_toc_buffer()
    _flush_body_chunks()

    return _deduplicate_chunks(chunks)


def chunks_to_text(chunks: Iterable[TextChunk]) -> list[str]:
    return [chunk.text for chunk in chunks]
