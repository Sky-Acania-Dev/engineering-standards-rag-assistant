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


@dataclass(frozen=True)
class _BufferToken:
    token: str
    page: int | None
    section: str
    section_path: tuple[str, ...]


def _tokenize(text: str) -> list[str]:
    return text.split()


def _normalize_line_noise(line: str) -> str:
    cleaned = " ".join(line.split())
    cleaned = re.sub(r"\b(\d+)\s*\|\s*P\s*a\s*g\s*e\b", r"Page \1", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


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
    # OCR/extraction artifacts often append page digits directly to the heading.
    cleaned = re.sub(r"(?<=[A-Za-z])\d{1,4}$", "", cleaned).strip()
    return cleaned


def _detect_structural_heading(line: str) -> tuple[int, str] | None:
    markdown_heading = re.match(r"^(#{1,6})\s+(.+)$", line)
    if markdown_heading:
        return len(markdown_heading.group(1)), _normalize_heading_text(markdown_heading.group(2))

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

    decimal_heading = re.match(r"^(\d+(?:\.\d+){1,3})\s+(.+)$", line)
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


def _iter_structured_blocks(document_text: str) -> list[_Block]:
    raw_blocks = [b.strip() for b in document_text.split("\n\n") if b.strip()]
    blocks: list[_Block] = []
    page: int | None = None
    heading_stack: list[tuple[int, str]] = []
    section = "Section 1"
    section_path: tuple[str, ...] = ()
    pending_image_ref: str | None = None

    for raw_block in raw_blocks:
        lines = [_normalize_line_noise(line) for line in raw_block.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            continue

        cursor = 0
        while cursor < len(lines):
            current = lines[cursor]
            page_match = re.match(r"^##\s+Page\s+(\d+)\b", current, flags=re.IGNORECASE)
            if page_match:
                page = int(page_match.group(1))
                cursor += 1
                continue

            heading_detected = _detect_structural_heading(current)
            if heading_detected is None:
                break

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

        lines = lines[cursor:]
        if not lines:
            continue

        full_block = "\n".join(lines)
        content_type = "body_text"
        protected = False
        table_id: str | None = None
        figure_id: str | None = None
        figure_ref: str | None = None

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
            continue
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
            table_id = _extract_numeric_id("Table", table_caption or full_block)
            if table_caption and not table_id:
                table_id = f"table:p{page}" if page is not None else "table:unknown"
        elif all(line.startswith("|") for line in lines):
            content_type = "table"
            protected = True
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
) -> list[TextChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    blocks = _iter_structured_blocks(document_text)
    chunks: list[TextChunk] = []
    chunk_id = 0
    buffer_tokens: list[_BufferToken] = []
    active_section = "Section 1"
    active_section_path: tuple[str, ...] = ()

    def _emit_chunk_from_tokens(tokens: list[_BufferToken]) -> None:
        nonlocal chunk_id
        if not tokens:
            return
        first = tokens[0]
        last = tokens[-1]
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                section=first.section,
                text=" ".join(token.token for token in tokens),
                token_count=len(tokens),
                page_start=first.page,
                page_end=last.page,
                content_type="body_text",
                section_path=first.section_path,
            )
        )
        chunk_id += 1

    def flush_buffer(*, keep_overlap: bool) -> None:
        nonlocal buffer_tokens
        if not buffer_tokens:
            return
        _emit_chunk_from_tokens(buffer_tokens)
        buffer_tokens = buffer_tokens[-overlap:] if keep_overlap and overlap else []

    def _append_block_tokens(block: _Block) -> None:
        nonlocal buffer_tokens
        block_tokens = _tokenize(block.text)
        buffer_tokens.extend(
            _BufferToken(
                token=token,
                page=block.page_end,
                section=block.section,
                section_path=block.section_path,
            )
            for token in block_tokens
        )

    for block in blocks:
        if block.content_type == "section_header":
            # Structural anchor: hard boundary and deterministic metadata reset.
            flush_buffer(keep_overlap=False)
            active_section = block.section
            active_section_path = block.section_path
            continue

        if block.content_type in {"toc", "image_artifact"}:
            flush_buffer(keep_overlap=False)
            block_tokens = _tokenize(block.text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=block.section,
                    text=block.text,
                    token_count=len(block_tokens),
                    page_start=block.page_start,
                    page_end=block.page_end,
                    content_type=block.content_type,
                    section_path=block.section_path,
                )
            )
            chunk_id += 1
            continue

        if block.content_type == "header_footer_noise":
            continue

        block_tokens = _tokenize(block.text)
        if not block_tokens:
            continue

        block_section = block.section if block.section != "Section 1" else active_section
        block_section_path = block.section_path if block.section_path else active_section_path

        if block.protected and block.content_type != "body_text":
            flush_buffer(keep_overlap=False)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=block_section,
                    text=block.text,
                    token_count=len(block_tokens),
                    page_start=block.page_start,
                    page_end=block.page_end,
                    content_type=block.content_type,
                    section_path=block_section_path,
                    table_id=block.table_id,
                    figure_id=block.figure_id,
                    figure_ref=block.figure_ref,
                )
            )
            chunk_id += 1
            continue

        if buffer_tokens and block_section != buffer_tokens[0].section:
            # Semantic section boundary: do not leak overlap into the new section.
            flush_buffer(keep_overlap=False)

        _append_block_tokens(
            replace(
                block,
                section=block_section,
                section_path=block_section_path,
            )
        )
        while len(buffer_tokens) > chunk_size:
            _emit_chunk_from_tokens(buffer_tokens[:chunk_size])
            buffer_tokens = buffer_tokens[chunk_size - overlap :] if overlap else []

    if buffer_tokens:
        _emit_chunk_from_tokens(buffer_tokens)

    return _deduplicate_chunks(chunks)


def chunks_to_text(chunks: Iterable[TextChunk]) -> list[str]:
    return [chunk.text for chunk in chunks]
