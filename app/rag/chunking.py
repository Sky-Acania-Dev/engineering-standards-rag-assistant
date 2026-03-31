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


def _tokenize(text: str) -> list[str]:
    return text.split()


def _is_protected_block(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return False

    def is_table_or_list(line: str) -> bool:
        return (
            line.startswith("|")
            or line.startswith("-")
            or line.startswith("*")
            or bool(re.match(r"^\d+\.\s", line))
        )

    return all(is_table_or_list(line) for line in lines)


def _split_sections(document_text: str) -> list[tuple[str, str]]:
    section_break = re.compile(r"(?m)^(#{1,6}\s+.*|Chapter\s+\d+.*|Section\s+[\w.\-]+.*)$")
    matches = list(section_break.finditer(document_text))
    if not matches:
        return [("Section 1", document_text.strip())]

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
        body = document_text[start:end].strip()
        sections.append((title, body))
    return sections


def _to_blocks(section_text: str) -> list[tuple[str, bool]]:
    blocks: list[tuple[str, bool]] = []
    for block in [b.strip() for b in section_text.split("\n\n") if b.strip()]:
        blocks.append((block, _is_protected_block(block)))
    return blocks


def chunk_document_by_section(
    document_text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[TextChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[TextChunk] = []
    chunk_id = 0

    for section_name, section_text in _split_sections(document_text):
        if not section_text:
            continue

        blocks = _to_blocks(section_text)
        buffer_tokens: list[str] = []

        for block_text, protected in blocks:
            block_tokens = _tokenize(block_text)
            if not block_tokens:
                continue

            if protected:
                if buffer_tokens:
                    text = " ".join(buffer_tokens)
                    chunks.append(TextChunk(chunk_id=chunk_id, section=section_name, text=text, token_count=len(buffer_tokens)))
                    chunk_id += 1
                    buffer_tokens = buffer_tokens[-overlap:] if overlap else []
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        section=section_name,
                        text=block_text,
                        token_count=len(block_tokens),
                    )
                )
                chunk_id += 1
                continue

            combined = buffer_tokens + block_tokens
            while len(combined) > chunk_size:
                current = combined[:chunk_size]
                text = " ".join(current)
                chunks.append(TextChunk(chunk_id=chunk_id, section=section_name, text=text, token_count=len(current)))
                chunk_id += 1
                combined = combined[chunk_size - overlap :]
            buffer_tokens = combined

        if buffer_tokens:
            text = " ".join(buffer_tokens)
            chunks.append(TextChunk(chunk_id=chunk_id, section=section_name, text=text, token_count=len(buffer_tokens)))
            chunk_id += 1

    return chunks


def chunks_to_text(chunks: Iterable[TextChunk]) -> list[str]:
    return [chunk.text for chunk in chunks]
