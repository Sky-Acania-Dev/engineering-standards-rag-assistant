# Phase 1 Footnote Anchor Debug — Quick Test Guide

This guide explains how to run the Phase 1 superscript-anchor debug extractor and inspect output.

## Script

- `scripts/footnote_phase1_debug.py`

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
4. URL-style reference rows like `2 http://...` are **ignored** as anchors in this phase.

## Scope boundary (important)

Phase 1 does **not**:

- link anchors to footnote bodies,
- classify page-bottom regions,
- strip or rewrite body text,
- emit chunk-level footnote metadata.

