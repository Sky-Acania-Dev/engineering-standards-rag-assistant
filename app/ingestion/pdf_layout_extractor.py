from __future__ import annotations

from dataclasses import dataclass
import re
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
    chars = list(getattr(page, "chars", []) or [])
    tokens: list[LayoutToken] = []
    emitted_idx = 0

    def _split_word_with_superscript(word: dict[str, Any]) -> list[tuple[str, float, float, float, float, float]]:
        text = str(word.get("text", ""))
        if not text or not chars:
            return [(text, float(word.get("x0", 0.0)), float(word.get("x1", 0.0)), float(word.get("top", 0.0)), float(word.get("bottom", 0.0)), float(word.get("size", 0.0) or 0.0))]
        if not re.search(r"\d$", text):
            return [(text, float(word.get("x0", 0.0)), float(word.get("x1", 0.0)), float(word.get("top", 0.0)), float(word.get("bottom", 0.0)), float(word.get("size", 0.0) or 0.0))]

        w_x0 = float(word.get("x0", 0.0))
        w_x1 = float(word.get("x1", 0.0))
        w_top = float(word.get("top", 0.0))
        w_bottom = float(word.get("bottom", 0.0))
        in_word_chars = [
            ch for ch in chars
            if float(ch.get("x0", -1e9)) >= w_x0 - 0.5
            and float(ch.get("x1", 1e9)) <= w_x1 + 0.5
            and float(ch.get("top", 0.0)) <= w_bottom + 1.0
            and float(ch.get("bottom", 0.0)) >= w_top - 1.0
        ]
        if not in_word_chars:
            return [(text, w_x0, w_x1, w_top, w_bottom, float(word.get("size", 0.0) or 0.0))]
        in_word_chars = sorted(in_word_chars, key=lambda c: float(c.get("x0", 0.0)))
        char_text = "".join(str(ch.get("text", "")) for ch in in_word_chars)
        if not char_text.endswith(tuple("0123456789")):
            return [(text, w_x0, w_x1, w_top, w_bottom, float(word.get("size", 0.0) or 0.0))]

        trailing: list[dict[str, Any]] = []
        for ch in reversed(in_word_chars):
            if str(ch.get("text", "")).isdigit():
                trailing.append(ch)
            else:
                break
        trailing.reverse()
        if not trailing:
            return [(text, w_x0, w_x1, w_top, w_bottom, float(word.get("size", 0.0) or 0.0))]

        base_chars = in_word_chars[: len(in_word_chars) - len(trailing)]
        if not base_chars:
            return [(text, w_x0, w_x1, w_top, w_bottom, float(word.get("size", 0.0) or 0.0))]
        base_size = sum(float(ch.get("size", 0.0) or 0.0) for ch in base_chars) / len(base_chars)
        base_top = sum(float(ch.get("top", 0.0)) for ch in base_chars) / len(base_chars)
        trailing_size = sum(float(ch.get("size", 0.0) or 0.0) for ch in trailing) / len(trailing)
        trailing_top = sum(float(ch.get("top", 0.0)) for ch in trailing) / len(trailing)
        if trailing_size > base_size * 0.9 or trailing_top >= base_top - 0.3:
            return [(text, w_x0, w_x1, w_top, w_bottom, float(word.get("size", 0.0) or 0.0))]

        base_text = "".join(str(ch.get("text", "")) for ch in base_chars).strip()
        sup_text = "".join(str(ch.get("text", "")) for ch in trailing).strip()
        if not base_text or not sup_text:
            return [(text, w_x0, w_x1, w_top, w_bottom, float(word.get("size", 0.0) or 0.0))]
        return [
            (
                base_text,
                float(base_chars[0].get("x0", w_x0)),
                float(base_chars[-1].get("x1", w_x1)),
                min(float(ch.get("top", w_top)) for ch in base_chars),
                max(float(ch.get("bottom", w_bottom)) for ch in base_chars),
                base_size,
            ),
            (
                sup_text,
                float(trailing[0].get("x0", w_x0)),
                float(trailing[-1].get("x1", w_x1)),
                min(float(ch.get("top", w_top)) for ch in trailing),
                max(float(ch.get("bottom", w_bottom)) for ch in trailing),
                trailing_size,
            ),
        ]

    for word in words or []:
        split_parts = _split_word_with_superscript(word)
        for text, x0, x1, top, bottom, size in split_parts:
            line_id = int(round(top))
            tokens.append(
                LayoutToken(
                    page=page_number,
                    text=text,
                    x0=x0,
                    x1=x1,
                    top=top,
                    bottom=bottom,
                    size=size,
                    fontname=str(word.get("fontname", "")),
                    line_id=line_id,
                    reading_order=emitted_idx,
                )
            )
            emitted_idx += 1
    return tokens
