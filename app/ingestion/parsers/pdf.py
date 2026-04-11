from __future__ import annotations

from pathlib import Path
import re
from statistics import median
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

    page_text = (page.extract_text() or "").strip()
    page_text_lines: list[str] = []
    if page_text:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if line:
                page_text_lines.append(line)

    table_data = page.extract_tables() or []
    lines.extend(_inject_tables_into_text_lines(page_text_lines, table_data))

    image_count = len(getattr(page, "images", []) or [])
    if image_count:
        lines.append(f"[IMAGES] {image_count} embedded image(s)")

    return "\n".join(lines)


def _group_page_chars_into_lines(chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not chars:
        return []
    sorted_chars = sorted(chars, key=lambda c: (float(c.get("doctop", c.get("top", 0.0))), float(c.get("x0", 0.0))))
    lines: list[list[dict[str, Any]]] = []
    y_tolerance = 8.0
    for ch in sorted_chars:
        ch_top = float(ch.get("doctop", ch.get("top", 0.0)))
        if not lines:
            lines.append([ch])
            continue
        last_line = lines[-1]
        last_top = median(float(item.get("doctop", item.get("top", 0.0))) for item in last_line)
        if abs(ch_top - last_top) <= y_tolerance:
            last_line.append(ch)
        else:
            lines.append([ch])
    return [sorted(line, key=lambda c: float(c.get("x0", 0.0))) for line in lines]


def _line_text(line_chars: list[dict[str, Any]]) -> str:
    return "".join(str(ch.get("text", "")) for ch in line_chars)


def _superscript_geometry(
    *,
    char_size: float,
    char_top: float,
    line_size_median: float,
    line_top_median: float,
) -> tuple[bool, bool]:
    is_smaller = char_size <= (line_size_median * 0.84)
    top_delta = line_top_median - char_top
    # Some PDFs encode superscripts with smaller size but near-equal (or slightly
    # lower) `top` values due to glyph metrics. Allow a tight near-baseline window
    # when the size reduction is strong.
    is_raised = top_delta >= 0.35 or (is_smaller and abs(top_delta) <= 0.55)
    return is_smaller, is_raised


def _build_anchor_debug_for_page(page: Any, page_number: int) -> dict[str, Any]:
    chars = list(getattr(page, "chars", []) or [])
    lines = _group_page_chars_into_lines(chars)
    detected_anchors: list[dict[str, Any]] = []
    heading_tokens = ("chapter ", "section ", "appendix ", "article ")

    for line_index, line_chars in enumerate(lines):
        if not line_chars:
            continue
        line_sizes = [float(ch.get("size", 0.0) or 0.0) for ch in line_chars if float(ch.get("size", 0.0) or 0.0) > 0]
        if not line_sizes:
            continue
        line_size_median = median(line_sizes)
        line_top_median = median(float(ch.get("doctop", ch.get("top", 0.0))) for ch in line_chars)
        line_text = _line_text(line_chars)
        heading_like = line_text.strip().lower().startswith(heading_tokens)

        idx = 0
        while idx < len(line_chars):
            ch = line_chars[idx]
            text = str(ch.get("text", ""))
            if not text.isdigit():
                idx += 1
                continue

            run_chars = [ch]
            j = idx + 1
            while j < len(line_chars):
                next_text = str(line_chars[j].get("text", ""))
                if next_text.isdigit():
                    run_chars.append(line_chars[j])
                    j += 1
                    continue
                break

            anchor_chars = list(run_chars)
            if len(run_chars) > 2:
                superscript_like = [
                    (
                        lambda geom: geom[0] and geom[1]
                    )(
                        _superscript_geometry(
                            char_size=float(dch.get("size", line_size_median) or line_size_median),
                            char_top=float(dch.get("doctop", dch.get("top", line_top_median))),
                            line_size_median=line_size_median,
                            line_top_median=line_top_median,
                        )
                    )
                    for dch in run_chars
                ]
                suffix_start = len(run_chars)
                for pos in range(len(run_chars) - 1, -1, -1):
                    if superscript_like[pos]:
                        suffix_start = pos
                    else:
                        break
                suffix_len = len(run_chars) - suffix_start
                if 1 <= suffix_len <= 2 and suffix_start > 0:
                    anchor_chars = run_chars[suffix_start:]

            anchor_id = "".join(str(ac.get("text", "")) for ac in anchor_chars)
            if len(anchor_id) > 2 or len(anchor_id) == 0:
                idx = j
                continue

            anchor_size_median = median(float(ac.get("size", line_size_median) or line_size_median) for ac in anchor_chars)
            anchor_top_median = median(float(ac.get("doctop", ac.get("top", line_top_median))) for ac in anchor_chars)
            is_smaller, is_raised = _superscript_geometry(
                char_size=anchor_size_median,
                char_top=anchor_top_median,
                line_size_median=line_size_median,
                line_top_median=line_top_median,
            )
            if not (is_smaller and is_raised):
                idx = j
                continue

            prev_char = line_chars[idx - 1] if idx > 0 else None
            anchor_start_in_line = idx + (len(run_chars) - len(anchor_chars))
            prev_char = line_chars[anchor_start_in_line - 1] if anchor_start_in_line > 0 else None
            next_char = line_chars[j] if j < len(line_chars) else None
            prev_text = str(prev_char.get("text", "")) if prev_char else ""
            next_text = str(next_char.get("text", "")) if next_char else ""
            if not prev_char or prev_text.isspace():
                idx = j
                continue
            punctuation_adjacent = prev_text in {".", ",", ";", ":", ")", "]"} or next_text in {".", ",", ";", ":"}
            line_final = next_char is None or next_text.strip() == ""

            confidence = 0.55
            if is_smaller:
                confidence += 0.2
            if is_raised:
                confidence += 0.2
            if punctuation_adjacent:
                confidence += 0.05
            confidence = max(0.0, min(1.0, round(confidence, 2)))

            anchor_text_window = line_chars[max(0, idx - 12) : min(len(line_chars), j + 12)]
            nearby_text = _line_text(anchor_text_window).strip()
            detected_anchors.append(
                {
                    "anchor_id": anchor_id,
                    "bbox": {
                        "x0": float(anchor_chars[0].get("x0", 0.0)),
                        "top": float(min(float(ac.get("top", 0.0)) for ac in anchor_chars)),
                        "x1": float(anchor_chars[-1].get("x1", anchor_chars[-1].get("x0", 0.0))),
                        "bottom": float(max(float(ac.get("bottom", 0.0)) for ac in anchor_chars)),
                    },
                    "line_index": line_index,
                    "nearby_anchor_text": nearby_text,
                    "confidence": confidence,
                    "flags": {
                        "line_final": line_final,
                        "punctuation_adjacent": punctuation_adjacent,
                        "heading_like": heading_like,
                    },
                }
            )
            idx = j
    return {"page": page_number, "detected_anchors": detected_anchors}


def extract_phase1_superscript_anchor_debug(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Phase 1 debug helper: char-level superscript anchor detection only."""
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(pdf_path)) as doc:
        return [
            _build_anchor_debug_for_page(page, i)
            for i, page in enumerate(doc.pages, start=1)
        ]


def _line_bbox(line_chars: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "x0": float(min(float(ch.get("x0", 0.0)) for ch in line_chars)),
        "top": float(min(float(ch.get("top", 0.0)) for ch in line_chars)),
        "x1": float(max(float(ch.get("x1", ch.get("x0", 0.0))) for ch in line_chars)),
        "bottom": float(max(float(ch.get("bottom", 0.0)) for ch in line_chars)),
    }


def _collect_bottom_region_lines(page: Any) -> list[dict[str, Any]]:
    chars = list(getattr(page, "chars", []) or [])
    if not chars:
        return []
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    if page_height <= 0:
        page_height = float(max(float(ch.get("bottom", 0.0)) for ch in chars))
    cutoff_top = page_height * 0.72
    bottom_chars = [ch for ch in chars if float(ch.get("top", 0.0)) >= cutoff_top]
    line_chars = _group_page_chars_into_lines(bottom_chars)
    lines: list[dict[str, Any]] = []
    for chars_in_line in line_chars:
        line_text = _line_text(chars_in_line).strip()
        if not line_text:
            continue
        line_sizes = [float(ch.get("size", 0.0) or 0.0) for ch in chars_in_line if float(ch.get("size", 0.0) or 0.0) > 0]
        lines.append(
            {
                "text": line_text,
                "bbox": _line_bbox(chars_in_line),
                "median_size": float(median(line_sizes)) if line_sizes else 0.0,
            }
        )
    return lines


def _parse_footnote_bodies_from_lines(lines: list[dict[str, Any]]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    current_label: str | None = None
    label_pattern = re.compile(r"^(\d{1,2})\s+(\S.*)$")

    for line in lines:
        text = str(line["text"]).strip()
        match = label_pattern.match(text)
        if match:
            current_label = match.group(1)
            parsed[current_label] = match.group(2).strip()
            continue
        if current_label is not None:
            parsed[current_label] = f"{parsed[current_label]} {text}".strip()
    return parsed


def _classify_bottom_region(page: Any, lines: list[dict[str, Any]]) -> dict[str, Any]:
    if not lines:
        return {
            "classification": "unknown",
            "reasons": ["no_bottom_lines_detected"],
            "parsed_body_labels": [],
            "parsed_bodies": {},
        }

    page_chars = list(getattr(page, "chars", []) or [])
    region_top = min(float(line["bbox"]["top"]) for line in lines)
    non_bottom_sizes = [
        float(ch.get("size", 0.0) or 0.0)
        for ch in page_chars
        if float(ch.get("size", 0.0) or 0.0) > 0 and float(ch.get("top", 0.0)) < region_top
    ]
    page_sizes = [float(ch.get("size", 0.0) or 0.0) for ch in page_chars if float(ch.get("size", 0.0) or 0.0) > 0]
    page_size_median = float(median(non_bottom_sizes)) if non_bottom_sizes else (float(median(page_sizes)) if page_sizes else 0.0)
    line_size_median = float(median(line["median_size"] for line in lines if line["median_size"] > 0)) if lines else 0.0

    numbered_prefix = re.compile(r"^\s*(\d{1,2})([.)])\s+")
    footnote_prefix = re.compile(r"^\s*(\d{1,2})\s+\S")
    numbered_list_lines = [line for line in lines if numbered_prefix.match(str(line["text"]))]
    footnote_candidate_lines = [line for line in lines if footnote_prefix.match(str(line["text"]))]

    reasons: list[str] = []

    extracted_tables = page.extract_tables() or []
    if extracted_tables:
        table_token_count = 0
        for table in extracted_tables:
            for row in _table_rows(table):
                for cell in row:
                    if len(cell.split()) <= 6 and cell.strip():
                        if cell in " ".join(str(line["text"]) for line in lines):
                            table_token_count += 1
        if table_token_count >= 2:
            reasons.append("bottom_text_matches_extracted_table_cells")
            return {
                "classification": "table_region",
                "reasons": reasons,
                "parsed_body_labels": [],
                "parsed_bodies": {},
            }

    if len(numbered_list_lines) >= 3 and len(footnote_candidate_lines) <= 1:
        reasons.append("multi_line_numbered_list_prefix_pattern")
        return {
            "classification": "ordinary_numbered_list",
            "reasons": reasons,
            "parsed_body_labels": [],
            "parsed_bodies": {},
        }

    if len(numbered_list_lines) >= 3 and line_size_median >= page_size_median * 0.95:
        reasons.append("numbered_lines_are_body_sized")
        return {
            "classification": "ordinary_numbered_list",
            "reasons": reasons,
            "parsed_body_labels": [],
            "parsed_bodies": {},
        }

    parsed_bodies = _parse_footnote_bodies_from_lines(lines)
    parsed_labels = list(parsed_bodies.keys())
    parsed_ids = [int(label) for label in parsed_labels if label.isdigit()]

    if len(parsed_ids) >= 2:
        sorted_ids = sorted(parsed_ids)
        small_text = line_size_median > 0 and page_size_median > 0 and line_size_median <= page_size_median * 0.9
        has_local_sequence = any((b - a) == 1 for a, b in zip(sorted_ids, sorted_ids[1:]))
        if has_local_sequence and small_text:
            reasons.extend(["sequential_numeric_labels", "smaller_bottom_text"])
            return {
                "classification": "true_footnote_block",
                "reasons": reasons,
                "parsed_body_labels": parsed_labels,
                "parsed_bodies": parsed_bodies,
            }

    reasons.append("insufficient_footnote_signals")
    return {
        "classification": "unknown",
        "reasons": reasons,
        "parsed_body_labels": [],
        "parsed_bodies": {},
    }


def _build_phase2_bottom_region_debug_for_page(page: Any, page_number: int) -> dict[str, Any]:
    lines = _collect_bottom_region_lines(page)
    if lines:
        region_bbox = {
            "x0": float(min(float(line["bbox"]["x0"]) for line in lines)),
            "top": float(min(float(line["bbox"]["top"]) for line in lines)),
            "x1": float(max(float(line["bbox"]["x1"]) for line in lines)),
            "bottom": float(max(float(line["bbox"]["bottom"]) for line in lines)),
        }
    else:
        region_bbox = None
    classified = _classify_bottom_region(page, lines)
    return {
        "page": page_number,
        "region_bbox": region_bbox,
        "classification": classified["classification"],
        "reasons_for_classification": classified["reasons"],
        "parsed_body_labels": classified["parsed_body_labels"],
        "parsed_bodies": classified["parsed_bodies"],
    }


def extract_phase2_page_bottom_debug(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Phase 2 debug helper: classify page-bottom regions and parse local body labels."""
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(pdf_path)) as doc:
        return [
            _build_phase2_bottom_region_debug_for_page(page, i)
            for i, page in enumerate(doc.pages, start=1)
        ]


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
