from __future__ import annotations

import re
from collections import Counter


def _normalize_page_artifacts(line: str) -> str:
    collapsed = " ".join(line.split())
    collapsed = re.sub(r"\b(\d+)\s*\|\s*P\s*a\s*g\s*e\b", r"Page \1", collapsed, flags=re.IGNORECASE)
    return collapsed.strip()


def _is_probable_noise(line: str) -> bool:
    if not line:
        return True
    if re.match(r"^Page\s+\d+$", line, flags=re.IGNORECASE):
        return True
    if line.startswith("[IMAGES]"):
        return True
    return False


def normalize_ingested_text(text: str) -> str:
    """Normalize extracted parser text and drop obvious repeated header/footer noise."""
    if not text.strip():
        return ""

    lines = [_normalize_page_artifacts(line) for line in text.splitlines()]

    # Keep structural blanks but compute repeat noise on non-empty text lines.
    non_empty = [line for line in lines if line]
    frequencies = Counter(non_empty)
    repeated = {
        line
        for line, count in frequencies.items()
        if count >= 3 and not line.startswith("#") and len(line.split()) <= 10
    }

    normalized: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if normalized and normalized[-1] == "":
                continue
            normalized.append("")
            continue
        if line in repeated or _is_probable_noise(line):
            continue
        normalized.append(line)

    while normalized and normalized[0] == "":
        normalized.pop(0)
    while normalized and normalized[-1] == "":
        normalized.pop()

    return "\n".join(normalized)
