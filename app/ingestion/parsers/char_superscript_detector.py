from __future__ import annotations

from dataclasses import dataclass
import re

from app.ingestion.parsers.pdfplumber_layout import CharInfo, LineInfo


@dataclass(frozen=True)
class AnchorCandidate:
    page_number: int
    line_index: int
    label: str
    char_orders: tuple[int, ...]
    bbox: tuple[float, float, float, float]
    anchor_text: str
    anchor_insert_x: float


_ALLOWED = re.compile(r"^\d{1,3}$")
_PUNCT = set(").,;:'\"”’]}»")


def detect_superscript_anchors(lines: list[LineInfo]) -> list[AnchorCandidate]:
    anchors: list[AnchorCandidate] = []
    for line_index, line in enumerate(lines):
        chars = line.chars
        idx = 0
        while idx < len(chars):
            ch = chars[idx]
            if not ch.text.strip() or not ch.text.isdigit():
                idx += 1
                continue

            run: list[CharInfo] = [ch]
            j = idx + 1
            while j < len(chars):
                nxt = chars[j]
                gap = nxt.x0 - run[-1].x1
                if nxt.text.isspace() and gap <= max(3.5, line.body_size * 0.5):
                    j += 1
                    continue
                if (
                    nxt.text.isdigit()
                    and gap <= max(2.6, line.body_size * 0.4)
                    and abs(nxt.bottom - run[-1].bottom) <= max(1.4, line.body_size * 0.3)
                ):
                    run.append(nxt)
                    j += 1
                    continue
                break

            label = "".join(c.text for c in run)
            if not _ALLOWED.match(label):
                idx = j
                continue

            run_size = sum(c.size for c in run) / max(1, len(run))
            run_bottom = sum(c.bottom for c in run) / max(1, len(run))
            prev_char = chars[idx - 1] if idx > 0 else None
            next_char = chars[j] if j < len(chars) else None
            prev_gap = ch.x0 - prev_char.x1 if prev_char else 99.0
            next_gap = next_char.x0 - run[-1].x1 if next_char else 99.0

            is_small = run_size <= line.body_size * 0.97
            is_raised = run_bottom <= line.baseline - max(0.15, line.body_size * 0.03)
            attached_left = prev_char is not None and prev_gap <= max(2.8, line.body_size * 0.35)
            line_final = next_char is None
            punctuation_adjacent = prev_char is not None and prev_char.text in _PUNCT
            separated_right = line_final or next_gap >= -max(0.8, line.body_size * 0.15)

            heading_like = len("".join(c.text for c in chars).split()) <= 12
            if is_small and (is_raised or heading_like) and attached_left and separated_right:
                left = min(c.x0 for c in run)
                top = min(c.top for c in run)
                right = max(c.x1 for c in run)
                bottom = max(c.bottom for c in run)
                anchor_text = _neighbor_text(chars, idx, punctuation_adjacent=punctuation_adjacent)
                anchors.append(
                    AnchorCandidate(
                        page_number=line.page_number,
                        line_index=line_index,
                        label=label,
                        char_orders=tuple(c.order for c in run),
                        bbox=(left, top, right, bottom),
                        anchor_text=anchor_text,
                        anchor_insert_x=right,
                    )
                )
            idx = j
    return anchors


def _neighbor_text(chars: tuple[CharInfo, ...], anchor_start: int, *, punctuation_adjacent: bool) -> str:
    lookback = max(0, anchor_start - 24)
    prefix = "".join(c.text for c in chars[lookback:anchor_start]).strip()
    if not prefix:
        return ""
    if punctuation_adjacent:
        prefix = prefix.rstrip(".,;:'\"”’)]}")
    tokens = prefix.split()
    window = tokens[-6:]
    if len(window) >= 3 and len(window[0]) == 1:
        window = window[1:]
    cleaned = " ".join(window).strip()
    cleaned = re.sub(r"^(?:and|or|for)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned
