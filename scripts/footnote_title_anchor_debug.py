from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from typing import Any

from app.ingestion.parsers.pdf import _group_page_chars_into_lines, _line_text, _require_pdfplumber


TARGET_TITLES = (
    "chapter 1: administration and general requirements",
    "chapter 3: foundations",
)


def _normalized(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _print_candidate(
    *,
    page_num: int,
    line_index: int,
    line_chars: list[dict[str, Any]],
    char_index: int,
) -> None:
    line_text = _line_text(line_chars)
    line_size_median = median(
        float(ch.get("size", 0.0) or 0.0)
        for ch in line_chars
        if float(ch.get("size", 0.0) or 0.0) > 0
    )
    line_top_median = median(float(ch.get("doctop", ch.get("top", 0.0))) for ch in line_chars)

    ch = line_chars[char_index]
    char_text = str(ch.get("text", ""))
    char_size = float(ch.get("size", line_size_median) or line_size_median)
    char_top = float(ch.get("doctop", ch.get("top", line_top_median)))
    is_smaller = char_size <= (line_size_median * 0.84)
    is_raised = char_top < (line_top_median - 1.2)

    prefix_chars = line_chars[max(0, char_index - 25) : char_index]
    prefix_text = _line_text(prefix_chars)
    prefix_detail = [
        {
            "text": str(p.get("text", "")),
            "size": float(p.get("size", 0.0) or 0.0),
            "top": float(p.get("doctop", p.get("top", 0.0))),
        }
        for p in prefix_chars
    ]

    print("=" * 80)
    print(f"page={page_num} line_index={line_index} digit={char_text!r} char_index={char_index}")
    print(f"line_text={line_text!r}")
    print(f"line_size_median={line_size_median:.3f} line_top_median={line_top_median:.3f}")
    print(
        f"digit_size={char_size:.3f} digit_top={char_top:.3f} "
        f"is_smaller={is_smaller} is_raised={is_raised}"
    )
    print(f"prefix_text={prefix_text!r}")
    print("prefix_chars:")
    for item in prefix_detail:
        print(
            f"  text={item['text']!r} size={item['size']:.3f} top={item['top']:.3f}"
        )


def run(pdf_path: Path) -> None:
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc.pages, start=1):
            chars = list(getattr(page, "chars", []) or [])
            for line_index, line_chars in enumerate(_group_page_chars_into_lines(chars)):
                line_text = _line_text(line_chars)
                normalized = _normalized(line_text)
                if not any(title in normalized for title in TARGET_TITLES):
                    continue
                print("-" * 80)
                print(f"Found target title on page={page_num} line_index={line_index}: {line_text!r}")
                for idx, ch in enumerate(line_chars):
                    text = str(ch.get("text", ""))
                    if text not in {"1", "9"}:
                        continue
                    # Prioritize trailing title markers, not chapter numbers at the front.
                    if idx < max(3, len(line_chars) // 2):
                        continue
                    _print_candidate(
                        page_num=page_num,
                        line_index=line_index,
                        line_chars=line_chars,
                        char_index=idx,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug chapter-title anchor candidates (1 and 9) for Phase 1 superscript detection."
    )
    parser.add_argument("pdf", type=Path, help="Path to source PDF.")
    args = parser.parse_args()
    run(args.pdf)


if __name__ == "__main__":
    main()
