from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ingestion.parsers.pdf import extract_phase4_page_integration_artifact_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 page-integration debug extractor.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF document.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <pdf>.phase4_page_integration.json.",
    )
    args = parser.parse_args()

    output_path = args.out or args.pdf.with_suffix(".phase4_page_integration.json")
    phase4_debug = extract_phase4_page_integration_artifact_debug(args.pdf)
    output_path.write_text(json.dumps(phase4_debug, indent=2), encoding="utf-8")

    page_count = len(phase4_debug.get("pages", []))
    resolved_anchor_count = int(phase4_debug.get("total_resolved_anchor_count", 0) or 0)
    print(f"Wrote {output_path}")
    print(f"Pages with resolved anchors: {page_count}")
    print(f"Resolved anchors: {resolved_anchor_count}")


if __name__ == "__main__":
    main()
