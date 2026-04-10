from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.ingestion.footnote_linker import LinkedFootnote
from app.ingestion.pdf_layout_extractor import LayoutToken


def render_page_text_with_footnotes(
    tokens: list[LayoutToken],
    links: list[LinkedFootnote],
) -> tuple[list[str], tuple[dict[str, Any], ...]]:
    if not tokens:
        return [], ()

    link_by_token: dict[int, list[LinkedFootnote]] = defaultdict(list)
    for link in links:
        link_by_token[link.token_index].append(link)

    lines: dict[int, list[str]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    seen_meta: set[tuple[int, str]] = set()

    for idx, token in enumerate(tokens):
        lines[token.line_id].append(token.text)
        for link in link_by_token.get(idx, []):
            lines[token.line_id].append(f"[fn:{link.id}]")
            key = (link.id, link.anchor_text)
            if key not in seen_meta:
                entry: dict[str, Any] = {
                    "id": link.id,
                    "content": link.content,
                    "anchor_text": link.anchor_text,
                    "confidence": link.confidence,
                    "page": token.page,
                }
                metadata.append(entry)
                seen_meta.add(key)

    ordered = [" ".join(lines[line_id]).strip() for line_id in sorted(lines)]
    return ordered, tuple(metadata)
