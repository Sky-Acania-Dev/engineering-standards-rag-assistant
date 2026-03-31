from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class _StructuredHTMLParser(HTMLParser):
    """Extract structured text from HTML with heading, table, and image cues."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._lines: list[str] = []
        self._tag_stack: list[str] = []
        self._table_depth = 0
        self._table_row: list[str] = []
        self._table_rows: list[list[str]] = []
        self._in_caption = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {k: v for k, v in attrs}
        self._tag_stack.append(tag)

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._lines.append("")

        if tag == "table":
            self._table_depth += 1
            self._table_rows = []
            self._lines.append("\n[TABLE]")

        if tag == "tr" and self._table_depth > 0:
            self._table_row = []

        if tag in {"th", "td"} and self._table_depth > 0:
            self._table_row.append("")

        if tag == "img":
            alt = (attrs_map.get("alt") or "").strip()
            src = (attrs_map.get("src") or "").strip()
            details = alt if alt else (src if src else "unlabeled image")
            self._lines.append(f"[IMAGE] {details}")

        if tag == "figcaption":
            self._in_caption = True

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack:
            self._tag_stack.pop()

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "section", "article"}:
            self._lines.append("")

        if tag in {"th", "td"} and self._table_depth > 0 and self._table_row:
            self._table_row[-1] = self._table_row[-1].strip()

        if tag == "tr" and self._table_depth > 0 and self._table_row:
            self._table_rows.append([cell for cell in self._table_row if cell])
            self._table_row = []

        if tag == "table" and self._table_depth > 0:
            self._table_depth -= 1
            for row in self._table_rows:
                self._lines.append(" | ".join(row))
            self._lines.append("[/TABLE]\n")
            self._table_rows = []

        if tag == "figcaption":
            self._in_caption = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return

        if self._table_depth > 0 and self._table_row:
            self._table_row[-1] = f"{self._table_row[-1]} {text}".strip()
            return

        if self._in_caption:
            self._lines.append(f"[IMAGE_CAPTION] {text}")
            return

        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(current_tag[-1])
            self._lines.append(f"{'#' * level} {text}")
        else:
            self._lines.append(text)

    def get_text(self) -> str:
        normalized = [line.strip() for line in self._lines]
        result: list[str] = []
        for line in normalized:
            if line == "" and result and result[-1] == "":
                continue
            result.append(line)
        return "\n".join(result).strip()


def parse_html_content(content: str) -> str:
    parser = _StructuredHTMLParser()
    parser.feed(content)
    parser.close()
    return parser.get_text()


def ingest_html_folder(folder_path: str, encoding: str = "utf-8") -> list[dict[str, Any]]:
    """Read all .html/.htm files recursively and return parsed documents."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    documents: list[dict[str, Any]] = []

    for file_path in sorted(folder.rglob("*.htm*")):
        if file_path.suffix.lower() not in {".html", ".htm"}:
            continue
        try:
            raw = file_path.read_text(encoding=encoding)
            documents.append(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "content": parse_html_content(raw),
                    "content_type": "text/html",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive continuation in batch mode
            print(f"Skipping {file_path}: {exc}")

    return documents
