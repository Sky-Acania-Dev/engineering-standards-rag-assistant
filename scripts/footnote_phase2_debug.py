from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingestion.parsers.pdf import extract_phase2_page_bottom_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 page-bottom region classifier debug extractor.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF document.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <pdf>.phase2_bottom_regions.json.",
    )
    args = parser.parse_args()

    output_path = args.out or args.pdf.with_suffix(".phase2_bottom_regions.json")
    debug = extract_phase2_page_bottom_debug(args.pdf)
    output_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
