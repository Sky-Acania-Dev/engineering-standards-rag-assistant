from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayoutToken:
    page: int
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    fontname: str
    line_id: int
    reading_order: int


def tokens_from_pdfplumber_words(page: Any, page_number: int) -> list[LayoutToken]:
    if not hasattr(page, "extract_words"):
        return []
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=True,
        extra_attrs=["size", "fontname", "top", "bottom"],
    )
    tokens: list[LayoutToken] = []
    for idx, word in enumerate(words or []):
        line_id = int(round(float(word.get("top", 0.0))))
        tokens.append(
            LayoutToken(
                page=page_number,
                text=str(word.get("text", "")),
                x0=float(word.get("x0", 0.0)),
                x1=float(word.get("x1", 0.0)),
                top=float(word.get("top", 0.0)),
                bottom=float(word.get("bottom", 0.0)),
                size=float(word.get("size", 0.0) or 0.0),
                fontname=str(word.get("fontname", "")),
                line_id=line_id,
                reading_order=idx,
            )
        )
    return tokens
