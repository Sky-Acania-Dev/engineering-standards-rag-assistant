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


_ALLOWED = re.compile(r"^(\d{1,3}|[*†‡])$")


def detect_superscript_anchors(lines: list[LineInfo]) -> list[AnchorCandidate]:
    anchors: list[AnchorCandidate] = []
    for line_index, line in enumerate(lines):
        chars = line.chars
        idx = 0
        while idx < len(chars):
            ch = chars[idx]
            if not ch.text.strip():
                idx += 1
                continue

            run: list[CharInfo] = [ch]
            j = idx + 1
            while j < len(chars):
                nxt = chars[j]
                if (
                    nxt.text.isdigit()
                    and run[-1].text.isdigit()
                    and nxt.x0 - run[-1].x1 <= max(1.5, line.body_size * 0.12)
                    and abs(nxt.bottom - run[-1].bottom) <= max(0.8, line.body_size * 0.18)
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

            is_small = run_size <= line.body_size * 0.9
            is_raised = run_bottom <= line.baseline - max(0.5, line.body_size * 0.12)
            tight_left = prev_gap <= max(2.0, line.body_size * 0.22)
            separated_right = next_gap >= max(1.0, line.body_size * 0.15)
            if is_small and is_raised and tight_left and separated_right:
                left = min(c.x0 for c in run)
                top = min(c.top for c in run)
                right = max(c.x1 for c in run)
                bottom = max(c.bottom for c in run)
                anchor_text = _neighbor_text(chars, idx)
                anchors.append(
                    AnchorCandidate(
                        page_number=line.page_number,
                        line_index=line_index,
                        label=label,
                        char_orders=tuple(c.order for c in run),
                        bbox=(left, top, right, bottom),
                        anchor_text=anchor_text,
                        anchor_insert_x=left,
                    )
                )
            idx = j
    return anchors


def _neighbor_text(chars: tuple[CharInfo, ...], anchor_start: int) -> str:
    start = max(0, anchor_start - 4)
    prefix = "".join(c.text for c in chars[start:anchor_start]).strip()
    return prefix[-80:] if prefix else ""
