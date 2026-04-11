# Footnote Debug — Phase 1 + Phase 2 Quick Test Guide

This guide explains how to run Phase 1 superscript-anchor debug extraction and Phase 2 page-bottom region classification.

## Script

- `scripts/footnote_phase1_debug.py`
- `scripts/footnote_phase2_debug.py`
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
Set-Location "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant"
$env:PYTHONPATH = "."

$pdfPath = "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant\temp\my_pdfs\14-TMCS.pdf"
$outPath = "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant\temp\footnote_tests\phase1_anchors.json"

py .\scripts\footnote_phase1_debug.py $pdfPath --out $outPath
```

If you omit `--out`, output defaults to:

- `<input_pdf>.phase1_anchors.json`

## Targeted Chapter Title Diagnostics

To inspect why chapter-title markers (`1` and `9`) are accepted/rejected:

```powershell
py .\scripts\footnote_title_anchor_debug.py $pdfPath
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

## Phase 2 run command

```powershell
Set-Location "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant"
$env:PYTHONPATH = "."

$pdfPath = "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant\temp\my_pdfs\14-TMCS.pdf"
$outPath = "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant\temp\footnote_tests\phase2_bottom_regions.json"

py .\scripts\footnote_phase2_debug.py $pdfPath --out $outPath
```

Phase 2 output is page-level JSON with:

- `page`
- `region_bbox`
- `classification` (`true_footnote_block`, `ordinary_numbered_list`, `table_region`, `unknown`)
- `reasons_for_classification`
- `parsed_body_labels` (only when `true_footnote_block`)
- `parsed_bodies` (only when `true_footnote_block`)
- `detected_content` (label/content pairs parsed from the region)
- `checks` (pass/fail booleans for internal gating checks)
- `detected_footnotes` with:
  - `anchor_number`
  - `footnote_content_page`
  - `footnote_content_detected`

## Phase 2 scope boundary

Phase 2 does **not**:

- link anchors to bodies,
- rewrite or strip source body text,
- emit chunk footnote metadata.

Current positive criteria for `true_footnote_block` are conservative and include:

- lower-page region scan beginning at ~65% page height (with fallback lower if sparse),
- table and ordinary-numbered-list negatives checked first,
- at least one parsed numeric label (`1`-`3` digits),
- and a line-start superscript-like numeric label prefix before content text.
