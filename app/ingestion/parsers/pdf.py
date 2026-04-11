from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import os

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

        if not inserted:
            if lines and lines[-1]:
                lines.append("")
            lines.extend(markdown_rows)
    return lines


def _extract_structured_page_text_pdfplumber(page: Any, page_number: int) -> str:
    lines = [f"## Page {page_number}"]

    chars = extract_page_chars(page, page_number)
    page_text_lines: list[str] = []
    resolved_for_page = []
    if chars:
        visual_lines = build_visual_lines(chars)
        anchors = detect_superscript_anchors(visual_lines)
        page_height = float(getattr(page, "height", 0.0) or 0.0)
        footnote_bodies, footnote_line_indexes = detect_footnote_bodies(visual_lines, page_height=page_height)
        resolved_for_page, unresolved = link_anchors_to_bodies(anchors, footnote_bodies)
        reconstructed = reconstruct_page_text(
            visual_lines,
            resolved=resolved_for_page,
            unresolved=unresolved,
            footnote_line_indexes=footnote_line_indexes,
        )
        page_text_lines = reconstructed.lines

        if os.getenv("PDF_FOOTNOTE_DEBUG", "").lower() in {"1", "true", "yes"}:
            logger.debug(
                "pdf footnote debug page=%s line_count=%s anchors=%s resolved=%s dropped_unresolved=%s",
                page_number,
                len(visual_lines),
                len(anchors),
                len(resolved_for_page),
                len(unresolved),
            )
    else:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            for raw_line in page_text.splitlines():
                line = raw_line.strip()
                if line:
                    page_text_lines.append(line)

    table_data = page.extract_tables() or []
    lines.extend(_inject_tables_into_text_lines(page_text_lines, table_data))

    for footnote in resolved_for_page:
        serialized = f"[FNDEF page={page_number} id={footnote.label} anchor={footnote.anchor_text}] {footnote.content}"
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
