from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingestion.parsers.pdf import (
    _collect_bottom_region_lines,
    _line_starts_with_superscript_numeric_label,
    _parse_footnote_bodies_from_lines,
    _require_pdfplumber,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 per-page bottom-line diagnostics.")
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument("--page", type=int, required=True, help="1-indexed page number")
    args = parser.parse_args()

    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(args.pdf)) as doc:
        page = doc.pages[args.page - 1]
        lines = _collect_bottom_region_lines(page)
        parsed = _parse_footnote_bodies_from_lines(lines)

    output = {
        "page": args.page,
        "line_count": len(lines),
        "lines": [
            {
                "text": line["text"],
                "bbox": line["bbox"],
                "median_size": line["median_size"],
                "starts_superscript_numeric_label": _line_starts_with_superscript_numeric_label(line),
            }
            for line in lines
        ],
        "parsed_labels_raw": parsed,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
