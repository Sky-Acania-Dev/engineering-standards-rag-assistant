from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import os
import sys

from app.ingestion.parsers.char_superscript_detector import detect_superscript_anchors
from app.ingestion.parsers.footnote_linker import link_anchors_to_bodies
from app.ingestion.parsers.footnote_region_detector import detect_footnote_bodies
from app.ingestion.parsers.pdfplumber_layout import build_visual_lines, extract_page_chars
from app.ingestion.parsers.text_reconstructor import reconstruct_page_text


logger = logging.getLogger(__name__)


def _require_pdfplumber() -> Any:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment extras
        raise ImportError(
            "PDF ingestion prefers 'pdfplumber'. Install it with `pip install pdfplumber`."
        ) from exc
    return pdfplumber


def _require_pypdf() -> Any:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment extras
        raise ImportError(
            "PDF ingestion requires 'pypdf'. Install it with `pip install pypdf`."
        ) from exc
    return PdfReader


def _table_rows(table: list[list[str | None]]) -> list[list[str]]:
    rows = [[(cell or "").strip() for cell in row] for row in table if row]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _table_to_markdown_lines(table: list[list[str | None]]) -> list[str]:
    normalized_rows = _table_rows(table)
    if not normalized_rows:
        return []
    return [f"| {' | '.join(row)} |" for row in normalized_rows]


def _normalize_match_line(line: str) -> str:
    return " ".join(line.split()).strip().lower()


def _table_plain_rows(tables: list[list[list[str | None]]]) -> list[str]:
    plain_rows: list[str] = []
    for table in tables:
        rows = _table_rows(table)
        plain_rows.extend(" ".join(cell for cell in row if cell).strip() for row in rows)
    return [row for row in plain_rows if row]


def _dedupe_table_lines_from_body(page_text_lines: list[str], tables: list[list[list[str | None]]]) -> list[str]:
    if not tables:
        return page_text_lines
    row_norms = {_normalize_match_line(row) for row in _table_plain_rows(tables)}
    if not row_norms:
        return page_text_lines
    return [line for line in page_text_lines if _normalize_match_line(line) not in row_norms]


def _table_line_indexes_in_visual_lines(
    visual_lines: list[Any], tables: list[list[list[str | None]]]
) -> set[int]:
    if not tables:
        return set()
    row_norms = {_normalize_match_line(row) for row in _table_plain_rows(tables)}
    if not row_norms:
        return set()
    matched: set[int] = set()
    for idx, line in enumerate(visual_lines):
        text = "".join(c.text for c in line.chars).strip()
        if _normalize_match_line(text) in row_norms:
            matched.add(idx)
    return matched


def _inject_tables_into_text_lines(page_text_lines: list[str], tables: list[list[list[str | None]]]) -> list[str]:
    if not tables:
        return page_text_lines
    lines = list(page_text_lines)
    for table in tables:
        rows = _table_rows(table)
        if not rows:
            continue
        plain_rows = [" ".join(cell for cell in row if cell).strip() for row in rows]
        markdown_rows = [f"| {' | '.join(row)} |" for row in rows]
        inserted = False

        for idx in range(0, max(0, len(lines) - len(plain_rows) + 1)):
            if _normalize_match_line(lines[idx]) != _normalize_match_line(plain_rows[0]):
                continue
            if all(
                _normalize_match_line(lines[idx + offset]) == _normalize_match_line(plain_rows[offset])
                for offset in range(len(plain_rows))
            ):
                lines[idx : idx + len(plain_rows)] = markdown_rows
                inserted = True
                break

        # Avoid duplicating table content if similar table-like lines are already present;
        # otherwise append markdown rows so table data is not lost.
        if not inserted:
            normalized_first = _normalize_match_line(plain_rows[0])
            already_table_like = any(
                _normalize_match_line(line).startswith(normalized_first[: max(8, len(normalized_first) // 2)])
                for line in lines
            )
            if not already_table_like:
                if lines and lines[-1]:
                    lines.append("")
                lines.extend(markdown_rows)
    return lines


def _extract_structured_page_text_pdfplumber(page: Any, page_number: int) -> str:
    lines = [f"## Page {page_number}"]

    chars = extract_page_chars(page, page_number)
    page_text_lines: list[str] = []
    resolved_for_page = []
    unlinked_bodies = []
    table_data = page.extract_tables() or []
    if chars:
        visual_lines = build_visual_lines(chars)
        anchors = detect_superscript_anchors(visual_lines)
        page_height = float(getattr(page, "height", 0.0) or 0.0)
        table_line_indexes = _table_line_indexes_in_visual_lines(visual_lines, table_data)
        footnote_bodies, footnote_line_indexes = detect_footnote_bodies(
            visual_lines,
            page_height=page_height,
            excluded_line_indexes=table_line_indexes,
        )
        resolved_for_page, unresolved, unlinked_bodies = link_anchors_to_bodies(anchors, footnote_bodies)
        reconstructed = reconstruct_page_text(
            visual_lines,
            resolved=resolved_for_page,
            unresolved=unresolved,
            footnote_line_indexes=footnote_line_indexes,
        )
        page_text_lines = reconstructed.lines
        page_text_lines = _dedupe_table_lines_from_body(page_text_lines, table_data)

        if os.getenv("PDF_FOOTNOTE_DEBUG", "").lower() in {"1", "true", "yes"}:
            logger.debug(
                "pdf footnote debug page=%s line_count=%s anchors=%s bodies=%s resolved=%s dropped_unresolved=%s",
                page_number,
                len(visual_lines),
                len(anchors),
                len(footnote_bodies),
                len(resolved_for_page),
                len(unresolved),
            )
            for anchor in anchors:
                logger.debug(
                    "anchor page=%s id=%s line=%s line_final=%s punct_adjacent=%s heading_like=%s bbox=%s anchor_text=%s",
                    anchor.page_number,
                    anchor.label,
                    anchor.line_index,
                    anchor.anchor_insert_x >= max(c.x1 for c in visual_lines[anchor.line_index].chars) - 0.5,
                    bool(anchor.anchor_text and anchor.anchor_text[-1:] in {'.', ',', ')', '"'}),
                    len(''.join(c.text for c in visual_lines[anchor.line_index].chars).split()) <= 12,
                    anchor.bbox,
                    anchor.anchor_text,
                )
            for body in footnote_bodies:
                logger.debug("footer_body page=%s id=%s lines=%s content=%s", body.page_number, body.label, body.line_indexes, body.content[:120])
            for dropped in unresolved:
                logger.debug("dropped_anchor page=%s id=%s line=%s anchor_text=%s", dropped.page_number, dropped.label, dropped.line_index, dropped.anchor_text)
            debug_summary = (
                f"[PDF_FOOTNOTE_DEBUG] page={page_number} anchors={len(anchors)} bodies={len(footnote_bodies)} "
                f"resolved={len(resolved_for_page)} dropped={len(unresolved)}"
            )
            print(debug_summary, file=sys.stderr)
            for anchor in anchors:
                print(
                    f"[PDF_FOOTNOTE_DEBUG] anchor id={anchor.label} line={anchor.line_index} text={anchor.anchor_text}",
                    file=sys.stderr,
                )
    else:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            for raw_line in page_text.splitlines():
                line = raw_line.strip()
                if line:
                    page_text_lines.append(line)

    lines.extend(_inject_tables_into_text_lines(page_text_lines, table_data))

    for footnote in resolved_for_page:
        serialized = f"[FNDEF page={page_number} id={footnote.label} anchor={footnote.anchor_text}] {footnote.content}"
        lines.append(serialized)
    for footnote in unlinked_bodies:
        serialized = f"[FNUNLINK page={page_number} id={footnote.label} reason={footnote.debug_reason}] {footnote.content}"
        lines.append(serialized)

    image_count = len(getattr(page, "images", []) or [])
    if image_count:
        lines.append(f"[IMAGES] {image_count} embedded image(s)")

    return "\n".join(lines)


def _extract_structured_page_text_pypdf(page: Any, page_number: int) -> str:
    text = (page.extract_text() or "").strip()
    lines = [f"## Page {page_number}"]

    if text:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.isupper() and len(line.split()) <= 14:
                lines.append(f"### {line}")
            else:
                lines.append(line)

    image_count = len(getattr(page, "images", []) or [])
    if image_count:
        lines.append(f"[IMAGES] {image_count} embedded image(s)")

    return "\n".join(lines)


def _extract_with_pdfplumber(pdf_path: str | Path) -> str:
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(pdf_path)) as doc:
        page_chunks = [
            _extract_structured_page_text_pdfplumber(page, i)
            for i, page in enumerate(doc.pages, start=1)
        ]
    return "\n\n".join(chunk for chunk in page_chunks if chunk.strip()).strip()


def _extract_with_pypdf(pdf_path: str | Path) -> str:
    PdfReader = _require_pypdf()
    reader = PdfReader(str(pdf_path))

    page_chunks: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        page_chunks.append(_extract_structured_page_text_pypdf(page, i))

    return "\n\n".join(chunk for chunk in page_chunks if chunk.strip()).strip()


def _is_usable_extraction(text: str) -> bool:
    if not text.strip():
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## Page "):
            continue
        if stripped in {"[TABLE]", "[/TABLE]"}:
            return True
        if len(stripped) >= 3:
            return True
    return False


def parse_pdf_file(pdf_path: str | Path) -> str:
    """Extract text from a PDF while preserving structure and tables."""
    pdfplumber_error: Exception | None = None
    try:
        extracted = _extract_with_pdfplumber(pdf_path)
        if _is_usable_extraction(extracted):
            return extracted
    except Exception as exc:  # pragma: no cover - fallback behavior
        pdfplumber_error = exc

    try:
        extracted = _extract_with_pypdf(pdf_path)
        if _is_usable_extraction(extracted):
            return extracted
    except Exception as exc:  # pragma: no cover - fallback behavior
        if pdfplumber_error is not None:
            raise RuntimeError(
                f"PDF extraction failed with pdfplumber ({pdfplumber_error}) and pypdf ({exc})."
            ) from exc
        raise

    if pdfplumber_error is not None:
        raise RuntimeError(f"PDF extraction failed: unusable pdfplumber output ({pdfplumber_error}).")
    raise RuntimeError("PDF extraction failed: both pdfplumber and pypdf produced unusable output.")


def ingest_pdf_folder(folder_path: str) -> list[dict[str, Any]]:
    """Read all .pdf files recursively and return parsed documents."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    documents: list[dict[str, Any]] = []

    for file_path in sorted(folder.rglob("*.pdf")):
        try:
            documents.append(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "content": parse_pdf_file(file_path),
                    "content_type": "application/pdf",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive continuation in batch mode
            print(f"Skipping {file_path}: {exc}")

    return documents
