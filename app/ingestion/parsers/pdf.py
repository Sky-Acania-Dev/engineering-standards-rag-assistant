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
    # Keep Phase 2 conservative but avoid clipping footnote starts.
    cutoff_top = page_height * 0.75
    bottom_chars = [ch for ch in chars if float(ch.get("top", 0.0)) >= cutoff_top]
    if len(bottom_chars) < 10:
        cutoff_top = page_height * 0.60
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
                "chars": chars_in_line,
            }
        )
    return lines


def _parse_footnote_bodies_from_lines(lines: list[dict[str, Any]]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    current_label: str | None = None
    label_pattern = re.compile(r"^(\d{1,3})(?:[.)])?(?:\s+(.*))?$")

    for line in lines:
        text = str(line["text"]).strip()
        match = label_pattern.match(text)
        if match:
            current_label = match.group(1)
            initial_content = (match.group(2) or "").strip()
            parsed[current_label] = initial_content
            continue
        if current_label is not None:
            parsed[current_label] = f"{parsed[current_label]} {text}".strip()
    return parsed


def _line_starts_with_superscript_numeric_label(line: dict[str, Any]) -> bool:
    chars = list(line.get("chars", []) or [])
    if not chars:
        return False
    idx = 0
    while idx < len(chars) and str(chars[idx].get("text", "")).isspace():
        idx += 1
    run: list[dict[str, Any]] = []
    while idx < len(chars) and str(chars[idx].get("text", "")).isdigit() and len(run) < 3:
        run.append(chars[idx])
        idx += 1
    if not run or idx >= len(chars):
        return False
    next_text = str(chars[idx].get("text", ""))
    if not next_text.isspace() and next_text not in {".", ")"}:
        return False
    # Require superscript-like geometry relative to the rest of the line.
    line_chars = [
        ch
        for ch in chars
        if ch not in run and str(ch.get("text", "")).strip() and str(ch.get("text", "")) not in {".", ")"}
    ]
    if not line_chars:
        return False
    line_size = median(float(ch.get("size", 0.0) or 0.0) for ch in line_chars)
    line_top = median(float(ch.get("doctop", ch.get("top", 0.0))) for ch in line_chars)
    run_size = median(float(ch.get("size", line_size) or line_size) for ch in run)
    run_top = median(float(ch.get("doctop", ch.get("top", line_top))) for ch in run)
    return run_size <= line_size * 0.9 and (line_top - run_top) >= 0.2


def _classify_bottom_region(page: Any, lines: list[dict[str, Any]], *, page_number: int) -> dict[str, Any]:
    def _result(
        *,
        classification: str,
        reasons: list[str],
        parsed_body_labels: list[str],
        parsed_bodies: dict[str, str],
        checks: dict[str, Any],
        starting_label_candidates: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "classification": classification,
            "reasons": reasons,
            "parsed_body_labels": parsed_body_labels,
            "parsed_bodies": parsed_bodies,
            "starting_label_candidates": starting_label_candidates or [],
            "detected_content": [
                {"label": label, "content": parsed_bodies.get(label, "")}
                for label in parsed_body_labels
            ],
            "checks": checks,
        }

    if not lines:
        return _result(
            classification="unknown",
            reasons=["no_bottom_lines_detected"],
            parsed_body_labels=[],
            parsed_bodies={},
            checks={"passes_true_footnote_check": False},
        )

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

    numbered_prefix = re.compile(r"^\s*(\d{1,3})([.)])\s+")
    bullet_prefix = re.compile(r"^\s*[•◦▪‣●○·*\-]\s+\S")
    footnote_prefix = re.compile(r"^\s*(\d{1,3})(?:[.)])?(?:\s+\S.*)?\s*$")
    numbered_list_lines = [line for line in lines if numbered_prefix.match(str(line["text"]))]
    bullet_list_lines = [line for line in lines if bullet_prefix.match(str(line["text"]))]
    footnote_candidate_lines = [line for line in lines if footnote_prefix.match(str(line["text"]))]

    reasons: list[str] = []
    checks: dict[str, Any] = {
        "table_negative": False,
        "numbered_list_negative": False,
        "body_sized_numbered_negative": False,
        "label_only_small_candidate": False,
        "parsed_ids_count": 0,
        "superscript_label_present": False,
        "small_text": False,
        "passes_true_footnote_check": False,
    }

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
            checks["table_negative"] = True
            return _result(
                classification="table_region",
                reasons=reasons,
                parsed_body_labels=[],
                parsed_bodies={},
                checks=checks,
            )

    parsed_bodies_raw = _parse_footnote_bodies_from_lines(lines)
    starting_label_candidates = list(parsed_bodies_raw.keys())

    # Numbered-list-first, footnote-later handling:
    # if there is a long list run (numbered or bulleted), re-parse only the
    # tail lines after the last list item so trailing footnotes are not masked.
    list_indices = [
        idx
        for idx, line in enumerate(lines)
        if numbered_prefix.match(str(line["text"])) or bullet_prefix.match(str(line["text"]))
    ]
    if len(list_indices) >= 2:
        tail_start = max(list_indices) + 1
        tail_lines = lines[tail_start:]
        if tail_lines:
            tail_parsed = _parse_footnote_bodies_from_lines(tail_lines)
            if tail_parsed:
                parsed_bodies_raw = tail_parsed
        elif len(list_indices) >= 3:
            parsed_bodies_raw = {}

    parsed_bodies = {
        label: content
        for label, content in parsed_bodies_raw.items()
        if not (content.strip() in {"| Page", "Page", "|Page"})
        and not (label.isdigit() and int(label) == page_number and content.strip().startswith("| Page"))
        and not (label.isdigit() and int(label) >= 100)
    }
    parsed_labels = list(parsed_bodies.keys())
    parsed_ids = [int(label) for label in parsed_labels if label.isdigit()]
    checks["parsed_ids_count"] = len(parsed_ids)
    label_only_pattern = re.compile(r"^\s*(\d{1,3})(?:[.)])?\s*$")
    label_only_small_candidate = any(
        label_only_pattern.match(str(line["text"]))
        and float(line.get("median_size", 0.0) or 0.0) > 0
        and page_size_median > 0
        and float(line.get("median_size", 0.0) or 0.0) <= page_size_median * 0.9
        for line in footnote_candidate_lines
    )
    checks["label_only_small_candidate"] = label_only_small_candidate

    if (
        (numbered_list_lines or bullet_list_lines)
        and line_size_median >= page_size_median * 0.95
        and len(parsed_ids) == 0
        and not label_only_small_candidate
    ):
        reasons.append("numbered_lines_are_body_sized")
        checks["body_sized_numbered_negative"] = True
        return _result(
            classification="ordinary_numbered_list",
            reasons=reasons,
            parsed_body_labels=[],
            parsed_bodies={},
            checks=checks,
            starting_label_candidates=starting_label_candidates,
        )

    # Ignore footer-only regions (e.g., isolated page numbers at far-right).
    if len(lines) <= 2 and len(parsed_ids) == 0:
        reasons.append("footer_or_page_number_only")
        return _result(
            classification="unknown",
            reasons=reasons,
            parsed_body_labels=[],
            parsed_bodies={},
            checks=checks,
            starting_label_candidates=starting_label_candidates,
        )

    if len(parsed_ids) >= 1:
        small_text = line_size_median > 0 and page_size_median > 0 and line_size_median <= page_size_median * 0.9
        superscript_labels: set[str] = set()
        for line in footnote_candidate_lines:
            label_match = re.match(r"^\s*(\d{1,3})(?:[.)])?(?:\s+\S.*)?\s*$", str(line["text"]))
            if not label_match:
                continue
            if _line_starts_with_superscript_numeric_label(line):
                superscript_labels.add(label_match.group(1))
                continue
            if (
                label_only_pattern.match(str(line["text"]))
                and float(line.get("median_size", 0.0) or 0.0) > 0
                and page_size_median > 0
                and float(line.get("median_size", 0.0) or 0.0) <= page_size_median * 0.9
            ):
                superscript_labels.add(label_match.group(1))
        superscript_label_present = bool(superscript_labels)
        if superscript_labels:
            parsed_labels = [label for label in parsed_labels if label in superscript_labels]
            parsed_bodies = {label: parsed_bodies[label] for label in parsed_labels}
        checks["small_text"] = small_text
        checks["superscript_label_present"] = superscript_label_present
        if superscript_label_present:
            reasons.extend(["superscript_numeric_label_prefix"])
            if small_text:
                reasons.append("smaller_bottom_text")
            checks["passes_true_footnote_check"] = True
            return _result(
                classification="true_footnote_block",
                reasons=reasons,
                parsed_body_labels=parsed_labels,
                parsed_bodies=parsed_bodies,
                checks=checks,
                starting_label_candidates=starting_label_candidates,
            )

    reasons.append("insufficient_footnote_signals")
    return _result(
        classification="unknown",
        reasons=reasons,
        parsed_body_labels=parsed_labels,
        parsed_bodies=parsed_bodies,
        checks=checks,
        starting_label_candidates=starting_label_candidates,
    )


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
    classified = _classify_bottom_region(page, lines, page_number=page_number)
    detected_footnotes: list[dict[str, Any]] = []
    if classified["classification"] == "true_footnote_block":
        for label in classified["parsed_body_labels"]:
            detected_footnotes.append(
                {
                    "anchor_number": int(label),
                    "footnote_content_page": page_number,
                    "footnote_content_detected": classified["parsed_bodies"].get(label, ""),
                }
            )
    return {
        "page": page_number,
        "region_bbox": region_bbox,
        "classification": classified["classification"],
        "reasons_for_classification": classified["reasons"],
        "parsed_body_labels": classified["parsed_body_labels"],
        "parsed_bodies": classified["parsed_bodies"],
        "starting_label_candidates": classified.get("starting_label_candidates", []),
        "detected_content": classified["detected_content"],
        "checks": classified["checks"],
        "detected_footnotes": detected_footnotes,
    }


def extract_phase2_page_bottom_debug(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Phase 2 debug helper: classify page-bottom regions and parse local body labels."""
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(pdf_path)) as doc:
        return [
            _build_phase2_bottom_region_debug_for_page(page, i)
            for i, page in enumerate(doc.pages, start=1)
        ]


def _build_phase3_linking_debug(
    phase1_anchor_debug: list[dict[str, Any]],
    phase2_bottom_debug: list[dict[str, Any]],
) -> dict[str, Any]:
    """Phase 3 linker: page-local first, then cross-page fallback, no text mutation."""
    page_marker_map: dict[int, list[dict[str, Any]]] = {}
    for page in phase1_anchor_debug:
        page_number = int(page.get("page", 0) or 0)
        markers = list(page.get("detected_anchors", []) or [])
        page_marker_map[page_number] = markers

    content_records: list[dict[str, Any]] = []
    page_content_map: dict[int, dict[str, dict[str, Any]]] = {}
    for page in phase2_bottom_debug:
        page_number = int(page.get("page", 0) or 0)
        if str(page.get("classification", "")) != "true_footnote_block":
            page_content_map.setdefault(page_number, {})
            continue
        parsed_bodies = dict(page.get("parsed_bodies", {}) or {})
        page_map = page_content_map.setdefault(page_number, {})
        for label, content in parsed_bodies.items():
            normalized_id = str(label)
            record = {
                "content_id": normalized_id,
                "footnote_content_page": page_number,
                "footnote_content_detected": str(content),
                "linked_markers": [],
            }
            # Keep first content occurrence per id per page to avoid accidental id shifting.
            page_map.setdefault(normalized_id, record)
            content_records.append(record)

    # Explicit orphan pools for deferred reconciliation.
    orphan_marker_pool: list[dict[str, Any]] = []
    for page_number in sorted(page_marker_map):
        for marker in page_marker_map[page_number]:
            orphan_marker_pool.append(
                {
                    "page": page_number,
                    "anchor_id": str(marker.get("anchor_id", "")),
                    "marker": marker,
                }
            )
    orphan_content_pool: list[dict[str, Any]] = [
        {
            "content_id": str(record["content_id"]),
            "footnote_content_page": int(record["footnote_content_page"]),
            "footnote_content_detected": str(record["footnote_content_detected"]),
        }
        for record in content_records
    ]

    links: list[dict[str, Any]] = []

    def _remove_marker_from_orphan_pool(page_number: int, anchor_id: str, marker_ref: dict[str, Any]) -> None:
        for idx, orphan in enumerate(orphan_marker_pool):
            if (
                int(orphan.get("page", 0) or 0) == page_number
                and str(orphan.get("anchor_id", "")) == anchor_id
                and orphan.get("marker") is marker_ref
            ):
                orphan_marker_pool.pop(idx)
                return

    def _remove_content_from_orphan_pool(content_page: int, content_id: str) -> None:
        for idx, orphan in enumerate(orphan_content_pool):
            if (
                int(orphan.get("footnote_content_page", 0) or 0) == content_page
                and str(orphan.get("content_id", "")) == content_id
            ):
                orphan_content_pool.pop(idx)
                return

    # Pass 1: page-local ID linking.
    unresolved_markers: list[dict[str, Any]] = []
    for marker_ref in list(orphan_marker_pool):
        page_number = int(marker_ref["page"])
        anchor_id = str(marker_ref["anchor_id"])
        marker = marker_ref["marker"]
        local_content = page_content_map.get(page_number, {}).get(anchor_id)
        if local_content is None:
            unresolved_markers.append(marker_ref)
            continue
        link = {
            "anchor_page": page_number,
            "anchor_id": anchor_id,
            "content_page": int(local_content["footnote_content_page"]),
            "link_mode": "page_local_id_match",
            "marker": marker,
            "content": {
                "content_id": str(local_content["content_id"]),
                "footnote_content_page": int(local_content["footnote_content_page"]),
                "footnote_content_detected": str(local_content["footnote_content_detected"]),
            },
        }
        local_content["linked_markers"].append(
            {
                "anchor_id": anchor_id,
                "anchor_page": page_number,
                "line_index": marker.get("line_index"),
                "bbox": marker.get("bbox"),
                "nearby_anchor_text": marker.get("nearby_anchor_text"),
            }
        )
        links.append(link)
        _remove_marker_from_orphan_pool(page_number, anchor_id, marker)
        _remove_content_from_orphan_pool(int(local_content["footnote_content_page"]), anchor_id)

    # Pass 2: fallback cross-page ID linking for unresolved orphan markers.
    content_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in content_records:
        content_by_id.setdefault(str(record["content_id"]), []).append(record)
    for records in content_by_id.values():
        records.sort(key=lambda item: int(item["footnote_content_page"]))

    dropped_anchors: list[dict[str, Any]] = []
    for marker_ref in unresolved_markers:
        page_number = int(marker_ref["page"])
        anchor_id = str(marker_ref["anchor_id"])
        marker = marker_ref["marker"]
        candidates = content_by_id.get(anchor_id, [])
        if not candidates:
            dropped_anchors.append(
                {
                    "anchor_page": page_number,
                    "anchor_id": anchor_id,
                    "reason": "no_matching_content_id",
                }
            )
            continue
        selected = sorted(
            candidates,
            key=lambda item: (abs(int(item["footnote_content_page"]) - page_number), int(item["footnote_content_page"])),
        )[0]
        selected["linked_markers"].append(
            {
                "anchor_id": anchor_id,
                "anchor_page": page_number,
                "line_index": marker.get("line_index"),
                "bbox": marker.get("bbox"),
                "nearby_anchor_text": marker.get("nearby_anchor_text"),
            }
        )
        links.append(
            {
                "anchor_page": page_number,
                "anchor_id": anchor_id,
                "content_page": int(selected["footnote_content_page"]),
                "link_mode": "cross_page_id_fallback",
                "marker": marker,
                "content": {
                    "content_id": str(selected["content_id"]),
                    "footnote_content_page": int(selected["footnote_content_page"]),
                    "footnote_content_detected": str(selected["footnote_content_detected"]),
                },
            }
        )
        _remove_marker_from_orphan_pool(page_number, anchor_id, marker)
        _remove_content_from_orphan_pool(int(selected["footnote_content_page"]), anchor_id)

    pages = sorted(set(page_marker_map.keys()) | set(page_content_map.keys()))
    per_page_debug: list[dict[str, Any]] = []
    for page_number in pages:
        marker_ids = [str(item.get("anchor_id", "")) for item in page_marker_map.get(page_number, [])]
        body_ids = sorted(page_content_map.get(page_number, {}).keys(), key=lambda raw: int(raw) if raw.isdigit() else raw)
        page_links = [
            {
                "anchor_id": str(link["anchor_id"]),
                "anchor_page": int(link["anchor_page"]),
                "content_page": int(link["content_page"]),
                "link_mode": str(link["link_mode"]),
            }
            for link in links
            if int(link["anchor_page"]) == page_number
        ]
        page_dropped = [entry for entry in dropped_anchors if int(entry["anchor_page"]) == page_number]
        per_page_debug.append(
            {
                "page": page_number,
                "anchor_ids": marker_ids,
                "body_ids": body_ids,
                "resolved_links": page_links,
                "dropped_anchors": page_dropped,
            }
        )

    linked_footnote_contents = sorted(
        content_records,
        key=lambda item: (int(item["footnote_content_page"]), int(item["content_id"]) if str(item["content_id"]).isdigit() else str(item["content_id"])),
    )
    orphan_markers = [
        {
            "anchor_page": int(marker["page"]),
            "anchor_id": str(marker["anchor_id"]),
        }
        for marker in orphan_marker_pool
    ]

    return {
        "pages": per_page_debug,
        "resolved_links": [
            {
                "anchor_page": int(link["anchor_page"]),
                "anchor_id": str(link["anchor_id"]),
                "content_page": int(link["content_page"]),
                "link_mode": str(link["link_mode"]),
            }
            for link in links
        ],
        "linked_footnote_contents": [
            {
                "content_id": str(item["content_id"]),
                "footnote_content_page": int(item["footnote_content_page"]),
                "footnote_content_detected": str(item["footnote_content_detected"]),
                "linked_markers": list(item["linked_markers"]),
            }
            for item in linked_footnote_contents
        ],
        "orphan_markers": orphan_markers,
        "orphan_content_pool": orphan_content_pool,
    }


def extract_phase3_linking_debug(pdf_path: str | Path) -> dict[str, Any]:
    """Phase 3 debug helper: link Phase 1 anchors to Phase 2 bodies."""
    phase1 = extract_phase1_superscript_anchor_debug(pdf_path)
    phase2 = extract_phase2_page_bottom_debug(pdf_path)
    return _build_phase3_linking_debug(phase1, phase2)


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
