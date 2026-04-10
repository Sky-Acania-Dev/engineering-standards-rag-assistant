from __future__ import annotations

from pathlib import Path
from typing import Any


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


def _table_to_markdown_lines(table: list[list[str | None]]) -> list[str]:
    rows = [[(cell or "").strip() for cell in row] for row in table if row]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    markdown_rows = [f"| {' | '.join(row)} |" for row in normalized_rows]
    return ["[TABLE]", *markdown_rows, "[/TABLE]"]


def _extract_structured_page_text_pdfplumber(page: Any, page_number: int) -> str:
    lines = [f"## Page {page_number}"]

    page_text = (page.extract_text() or "").strip()
    if page_text:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if line:
                lines.append(line)

    for table in page.extract_tables() or []:
        table_lines = _table_to_markdown_lines(table)
        if table_lines:
            lines.extend(table_lines)
            lines.append("")

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
