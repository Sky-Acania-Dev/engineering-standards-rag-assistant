from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    section: str
    text: str
    token_count: int
    page: int | None = None
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
    page: int | None
    section: str
    section_path: tuple[str, ...]
    table_id: str | None = None
    figure_id: str | None = None
    figure_ref: str | None = None
    protected: bool = False


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


def _iter_structured_blocks(document_text: str) -> list[_Block]:
    raw_blocks = [b.strip() for b in document_text.split("\n\n") if b.strip()]
    blocks: list[_Block] = []
    page: int | None = None
    heading_stack: list[tuple[int, str]] = []
    section = "Section 1"
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

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", current)
            chapter_match = re.match(r"^(Chapter\s+\d+.*|Section\s+[\w.\-]+.*)$", current, flags=re.IGNORECASE)
            if heading_match or chapter_match:
                if heading_match:
                    level = len(heading_match.group(1))
                    heading_text = heading_match.group(2).strip()
                else:
                    level = 2
                    heading_text = current.strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))
                section = heading_text
                blocks.append(
                    _Block(
                        text=heading_text,
                        content_type="section_header",
                        page=page,
                        section=section,
                        section_path=tuple(title for _, title in heading_stack),
                        protected=True,
                    )
                )
                cursor += 1
                continue
            break

        lines = lines[cursor:]
        if not lines:
            continue

        full_block = "\n".join(lines)
        section_path = tuple(title for _, title in heading_stack)
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
                    page=page,
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

        blocks.append(
            _Block(
                text=full_block,
                content_type=content_type,
                page=page,
                section=section,
                section_path=section_path,
                table_id=table_id,
                figure_id=figure_id,
                figure_ref=figure_ref,
                protected=protected,
            )
        )

    return blocks


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
    buffer_tokens: list[str] = []
    buffer_page: int | None = None
    buffer_section = "Section 1"
    buffer_section_path: tuple[str, ...] = ()

    def flush_buffer() -> None:
        nonlocal chunk_id, buffer_tokens
        if not buffer_tokens:
            return
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                section=buffer_section,
                text=" ".join(buffer_tokens),
                token_count=len(buffer_tokens),
                page=buffer_page,
                content_type="body_text",
                section_path=buffer_section_path,
            )
        )
        chunk_id += 1
        buffer_tokens = buffer_tokens[-overlap:] if overlap else []

    for block in blocks:
        if block.content_type in {"section_header", "header_footer_noise"}:
            continue

        if block.content_type == "toc":
            # Keep TOC isolated from body-text chunks.
            flush_buffer()
            block_tokens = _tokenize(block.text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=block.section,
                    text=block.text,
                    token_count=len(block_tokens),
                    page=block.page,
                    content_type="toc",
                    section_path=block.section_path,
                )
            )
            chunk_id += 1
            continue

        block_tokens = _tokenize(block.text)
        if not block_tokens:
            continue

        if block.protected and block.content_type != "body_text":
            flush_buffer()
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=block.section,
                    text=block.text,
                    token_count=len(block_tokens),
                    page=block.page,
                    content_type=block.content_type,
                    section_path=block.section_path,
                    table_id=block.table_id,
                    figure_id=block.figure_id,
                    figure_ref=block.figure_ref,
                )
            )
            chunk_id += 1
            continue

        if buffer_tokens and block.section != buffer_section:
            flush_buffer()

        if not buffer_tokens:
            buffer_page = block.page
            buffer_section = block.section
            buffer_section_path = block.section_path

        combined = buffer_tokens + block_tokens
        while len(combined) > chunk_size:
            current = combined[:chunk_size]
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    section=buffer_section,
                    text=" ".join(current),
                    token_count=len(current),
                    page=buffer_page,
                    content_type="body_text",
                    section_path=buffer_section_path,
                )
            )
            chunk_id += 1
            combined = combined[chunk_size - overlap :]
        buffer_tokens = combined

    if buffer_tokens:
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                section=buffer_section,
                text=" ".join(buffer_tokens),
                token_count=len(buffer_tokens),
                page=buffer_page,
                content_type="body_text",
                section_path=buffer_section_path,
            )
        )

    linked: list[TextChunk] = []
    for idx, chunk in enumerate(chunks):
        prev_chunk_id = chunks[idx - 1].chunk_id if idx > 0 else None
        next_chunk_id = chunks[idx + 1].chunk_id if idx + 1 < len(chunks) else None
        linked.append(
            TextChunk(
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                text=chunk.text,
                token_count=chunk.token_count,
                page=chunk.page,
                content_type=chunk.content_type,
                section_path=chunk.section_path,
                table_id=chunk.table_id,
                figure_id=chunk.figure_id,
                figure_ref=chunk.figure_ref,
                prev_chunk_id=prev_chunk_id,
                next_chunk_id=next_chunk_id,
            )
        )

    return linked


def chunks_to_text(chunks: Iterable[TextChunk]) -> list[str]:
    return [chunk.text for chunk in chunks]
