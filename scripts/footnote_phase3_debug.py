from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ingestion.parsers.pdf import extract_phase3_linking_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 footnote linker debug extractor.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF document.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <pdf>.phase3_linking.json.",
    )
    args = parser.parse_args()

    output_path = args.out or args.pdf.with_suffix(".phase3_linking.json")
    phase3_debug = extract_phase3_linking_debug(args.pdf)
    result = {
        "linked_footnote_objects": phase3_debug.get("linked_footnote_contents", []),
        "orphan_markers": phase3_debug.get("orphan_markers", []),
        "orphan_content_pool": phase3_debug.get("orphan_content_pool", []),
        "resolved_links": phase3_debug.get("resolved_links", []),
        "pages": phase3_debug.get("pages", []),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    linked_count = len(result["linked_footnote_objects"])
    orphan_marker_count = len(result["orphan_markers"])
    orphan_content_count = len(result["orphan_content_pool"])
    print(f"Wrote {output_path}")
    print(f"Linked footnote objects: {linked_count}")
    print(f"Orphan markers: {orphan_marker_count}")
    print(f"Orphan content objects: {orphan_content_count}")


if __name__ == "__main__":
    main()
