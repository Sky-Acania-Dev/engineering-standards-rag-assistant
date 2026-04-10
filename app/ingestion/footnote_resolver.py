from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_PAGE_RE = re.compile(r"^##\s+Page\s+(\d+)\b", flags=re.IGNORECASE)
_DEF_RE = re.compile(r"^\[FOOTNOTE_DEF\]\s*(\d{1,3})\|([^|]*)\|(.*)$")


@dataclass(frozen=True)
class FootnoteResolvedDocument:
    text: str
    page_footnotes: dict[int, tuple[dict[str, Any], ...]]


def resolve_footnotes_by_page(document_text: str) -> FootnoteResolvedDocument:
    """Resolve parser-emitted layout footnote definitions conservatively.

    This function intentionally does *not* infer footnote markers from plain digits.
    It only consumes explicit [FOOTNOTE_DEF] lines emitted by the layout-aware PDF
    pipeline and keeps existing [fn:N] anchors in body text unchanged.
    """

    page_footnotes: dict[int, list[dict[str, Any]]] = {}
    current_page: int | None = None
    cleaned_lines: list[str] = []

    for raw_line in document_text.splitlines():
        stripped = raw_line.rstrip()
        page_match = _PAGE_RE.match(stripped.strip())
        if page_match:
            current_page = int(page_match.group(1))
            cleaned_lines.append(stripped)
            continue

        def_match = _DEF_RE.match(stripped.strip())
        if def_match and current_page is not None:
            note_id = int(def_match.group(1))
            anchor_text = def_match.group(2).strip()
            content = def_match.group(3).strip()
            if not content:
                continue
            entry: dict[str, Any] = {
                "id": note_id,
                "anchor_text": anchor_text,
                "content": content,
                "page": current_page,
            }
            url_match = re.search(r"https?://\S+|www\.\S+", content, flags=re.IGNORECASE)
            if url_match:
                entry["url"] = url_match.group(0)
            page_footnotes.setdefault(current_page, []).append(entry)
            continue

        cleaned_lines.append(stripped)

    deduped: dict[int, tuple[dict[str, Any], ...]] = {}
    for page, entries in page_footnotes.items():
        seen: set[tuple[int, str]] = set()
        unique: list[dict[str, Any]] = []
        for entry in entries:
            key = (int(entry["id"]), str(entry.get("anchor_text", "")))
            if key in seen:
                continue
            unique.append(entry)
            seen.add(key)
        deduped[page] = tuple(unique)

    return FootnoteResolvedDocument(text="\n".join(cleaned_lines), page_footnotes=deduped)
