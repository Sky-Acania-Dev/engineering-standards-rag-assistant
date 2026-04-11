# Phase 1 Footnote Anchor Debug — Quick Test Guide

This guide explains how to run the Phase 1 superscript-anchor debug extractor and inspect output.

## Script

- `scripts/footnote_phase1_debug.py`
- `scripts/footnote_title_anchor_debug.py` (targeted title-marker diagnostics for `...Requirements1` and `...Foundations9`)

The script reads a PDF and writes page-level JSON debug output using:

- `page`
- `detected_anchors[]`
  - `anchor_id`
  - `bbox`
  - `line_index`
  - `nearby_anchor_text`
  - `confidence`
  - `flags` (`line_final`, `punctuation_adjacent`, `heading_like`)

## PowerShell (VS Code) run command

From the repository root:

```powershell
Set-Location "C:\path\to\engineering-standards-rag-assistant"
$env:PYTHONPATH = "."

$pdfPath = "C:\path\to\input.pdf"
$outPath = "C:\path\to\phase1_anchors.json"

python .\scripts\footnote_phase1_debug.py $pdfPath --out $outPath
```

If you omit `--out`, output defaults to:

- `<input_pdf>.phase1_anchors.json`

## Targeted Chapter Title Diagnostics

To inspect why chapter-title markers (`1` and `9`) are accepted/rejected:

```powershell
python .\scripts\footnote_title_anchor_debug.py $pdfPath
```

This script prints, for the title-line digit candidates:

- digit size/top vs line medians,
- `is_smaller` and `is_raised` checks,
- the title text,
- and preceding character details.

## Common error: `No module named app`

Cause: running from outside repo root or missing `PYTHONPATH`.

Fix:

1. `Set-Location` to repo root.
2. `Set-Item Env:PYTHONPATH "."` (or `$env:PYTHONPATH = "."`).
3. Re-run the script.

## What to validate in Phase 1

1. Char-level anchors are detected in running prose/headings.
2. Punctuation-adjacent anchors are detected (e.g., after period/comma/paren).
3. Two-digit anchors are captured as a single id (`10`, `11`, etc.).
4. Superscript-like digits with whitespace/newline before them are excluded from anchor detection.
5. Trailing superscripts attached to numeric tokens (for example patterns like `R80219` and `21.624`) are captured as suffix anchors when visually raised/smaller.

## Investigation notes (for next phases)

- Current Phase 1 anchor detection runs directly on `pdfplumber` page chars; it does **not** depend on chunking/structured-block heading removal.
- Chapter/section title exclusion happens later in chunking, so headings are still visible to this debug step.
- Keep cross-page cases in mind for linking (example: anchor on page N, body on page N+1 such as footnote 12 in the current test PDF).

## Scope boundary (important)

Phase 1 does **not**:

- link anchors to footnote bodies,
- classify page-bottom regions,
- strip or rewrite body text,
- emit chunk-level footnote metadata.
