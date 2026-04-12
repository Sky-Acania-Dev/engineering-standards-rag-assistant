# Footnote Handling System (Phases 1–2 Complete, Phase 3 Planned)

## Scope and current status

This repository currently implements **Phase 1** and **Phase 2** of a staged PDF footnote pipeline:

- **Phase 1:** detect inline superscript anchors from `pdfplumber` character geometry.
- **Phase 2:** classify bottom-of-page regions and parse candidate footnote bodies only when the region is validated as a true footnote block.
- **Phase 3:** not yet implemented in code (linking anchors ↔ bodies only).
- **Phase 4:** not yet implemented in code (integration into extracted text/chunks).

The phased strategy is explicit in `temp/Codex Prompt Temp.txt`, including guardrails about avoiding false positives and avoiding id-shifting when labels are missing.

---

## High-level architecture in code

All implemented logic is in:

- `app/ingestion/parsers/pdf.py`

Core phase-specific entry points:

- `extract_phase1_superscript_anchor_debug(pdf_path)`
- `extract_phase2_page_bottom_debug(pdf_path)`

Both return **debug-first** artifacts, intentionally separate from full PDF text extraction (`parse_pdf_file`), so phases can be verified without mutating content.

---

## Phase 1 in detail: char-level superscript anchor detection

### Main function path

1. `extract_phase1_superscript_anchor_debug` iterates PDF pages via `pdfplumber`.
2. For each page, `_build_anchor_debug_for_page`:
   - groups chars into lines (`_group_page_chars_into_lines`),
   - scans digit runs on each line,
   - evaluates superscript geometry (`_superscript_geometry`),
   - emits debug records for accepted anchors.

### Geometry and acceptance rules

A candidate anchor must satisfy:

- **small relative size** (`char_size <= line_median * 0.84`), and
- **raised positioning** (`top_delta >= 0.35`) or a near-baseline exception for strongly size-reduced glyphs.

Additional guardrails:

- ignore runs that produce empty ids or ids longer than 2 digits,
- reject when the preceding char is missing/whitespace (prevents list-leading numeric noise),
- allow extraction of a superscript suffix from long digit runs (e.g., body text ending in 2 superscript digits).

### Debug output schema (Phase 1)

Per page:

- `page`
- `detected_anchors[]` items with:
  - `anchor_id`
  - `bbox` (`x0`, `top`, `x1`, `bottom`)
  - `line_index`
  - `nearby_anchor_text`
  - `confidence`
  - `flags`:
    - `line_final`
    - `punctuation_adjacent`
    - `heading_like`

This matches the requested “detect-only, no linking, no text mutation” contract.

---

## Phase 2 in detail: page-bottom region classification + local body parsing

### Main function path

1. `extract_phase2_page_bottom_debug` iterates pages.
2. `_build_phase2_bottom_region_debug_for_page`:
   - collects bottom-region lines (`_collect_bottom_region_lines`),
   - classifies region (`_classify_bottom_region`),
   - if class is `true_footnote_block`, emits parsed footnote detections.

### Bottom-region collection

`_collect_bottom_region_lines` starts with the lower 25% of page height (top cutoff 75%); if too sparse, it expands to lower 40% (top cutoff 60%).

Each line carries:

- text
- line bbox
- median font size
- raw chars for geometry checks

### Classification labels used

- `true_footnote_block`
- `ordinary_numbered_list`
- `table_region`
- `unknown`

### Negative protections (false-positive suppression)

`_classify_bottom_region` includes hard negatives for known regressions:

1. **Table region detection:** if extracted table cell tokens significantly overlap bottom text, classify as `table_region`.
2. **Body-sized list detection:** when numbered/bulleted lines look like normal body font and no strong footnote evidence exists, classify as `ordinary_numbered_list`.
3. **Footer-only guard:** very small/empty footer-like regions become `unknown`.

### Body parsing behavior

- `_parse_footnote_bodies_from_lines` parses `1`, `1.`, `1)` style labels and line-wrapped continuations.
- Parsed labels are further filtered by:
  - plausibility exclusions (e.g., page-number-like artifacts, very large ids),
  - superscript-leading-label evidence (`_line_starts_with_superscript_numeric_label`) or small label-only lines.
- Region is promoted to `true_footnote_block` only when superscript-style label evidence survives.

### Debug output schema (Phase 2)

Per page:

- `page`
- `region_bbox`
- `classification`
- `reasons_for_classification`
- `parsed_body_labels`
- `parsed_bodies`
- `starting_label_candidates`
- `detected_content`
- `checks`
- `detected_footnotes` (only for true blocks)

---

## Relationship to text extraction/chunking

Current phase implementation is intentionally **non-destructive**:

- Phase 1 and 2 functions are debug helpers only.
- The extraction path used by `parse_pdf_file` still focuses on page text + table insertion and does **not** yet link or inject footnote markers.

This separation is important because it prevents section/chapter drift and text corruption while Phase 3/4 validation is pending.

---

## Current footnote marker/content schema status

There are currently **no dedicated dataclasses** for footnote markers or footnote contents in the parser/chunking model.

Current structures are dictionary-based debug payloads:

- Phase 1 marker items (`detected_anchors[]`):
  - `anchor_id`
  - `bbox`
  - `line_index`
  - `nearby_anchor_text`
  - `confidence`
  - `flags` (`line_final`, `punctuation_adjacent`, `heading_like`)

- Phase 2 content items (`detected_footnotes[]`):
  - `anchor_number`
  - `footnote_content_page`
  - `footnote_content_detected`

If Phase 3 needs stronger typing, introducing dedicated marker/content dataclasses (or `TypedDict`s) is recommended before linking and integration.

---

## Test coverage currently present (Phase 1–2)

`tests/unit/test_parsers_html_pdf.py` includes focused coverage for:

- heading/punctuation-adjacent superscript anchors,
- ignoring leading-space superscript-like numbers,
- two-digit trailing superscript extraction,
- near-baseline title superscript acceptance,
- true footnote block classification,
- dotted labels and wrapped continuation parsing,
- trailing footnotes after list blocks,
- avoiding ordinary numbered list/table misclassification.

Fixtures include representative debug samples:

- `tests/fixtures/phase1_anchor_debug_sample.json`
- `tests/fixtures/phase2_bottom_region_debug_sample.json`

---

## Phase 3 plan (linking only, no text mutation)

The requested Phase 3 should be implemented as a pure linkage layer using only Phase 1 + Phase 2 artifacts.

### Planned implementation steps

1. **Create id-indexed maps from existing debug outputs (no text/tag mutation)**
   - markers by `anchor_id` from Phase 1 (`detected_anchors`), retaining full marker records and pages.
   - footnote contents by label id from Phase 2 where classification is `true_footnote_block`.
   - **Do not inject** inline tags such as `[fn:N]` in this step.

2. **Link by id equality only (allow many markers → one content)**
   - link markers and footnote contents whenever `anchor_id == content_label_id`.
   - if multiple markers share the same id, all may link to one footnote content record.
   - no relative-order shifting and no nearest-neighbor reassignment.

3. **Emit grouped linkage + explicit orphan debugging outputs**
   - return a **linked footnote content list**, where each footnote content item includes:
     - footnote id
     - footnote text/content
     - footnote content page/source
     - list of linked marker records
   - return **orphan marker list** (markers with no content id match).
   - do not return a separate orphan-content list; entries in the full footnote content list with empty linked marker arrays are implicit orphan contents.

4. **Add phase-3 debug API**
   - e.g., `extract_phase3_linking_debug(pdf_path)` returning per-page:
     - marker ids
     - content ids
     - linked footnote contents (with linked marker lists)
     - orphan markers
     - (no separate orphan-content list; infer from empty linked-marker arrays)

5. **Keep extraction/chunk code untouched**
   - no marker insertion, no stripping, no metadata emission yet.
   - Phase 3 should emit a link artifact for Phase 4 consumption (rather than mutating `document_text` directly).

### Planned tests for Phase 3

- page 5 anchor `1` links to body `1`.
- page 7 mapping for `2/3/4/5` remains stable (no id shifts).
- repeated markers with the same id can all link to one content record.
- no accidental `6`→`7` merge.
- id `24` link does not alter numeric prose/citations (e.g., `21.62`).

### Out-of-scope in Phase 3

- no chunk text edits,
- no footnote metadata in final chunks,
- no inline marker insertion (e.g., no `[fn:N]` yet).

---

## Operational guidance for future Phase 4

When Phase 3 is passing, Phase 4 should integrate conservatively:

- strip only validated `true_footnote_block` lines,
- inject `[footnote: N]` only for resolved links,
- preserve ordering/section metadata/table behavior,
- never duplicate table text inline + table chunk,
- never reinterpret ordinary numbered lists or table rows as footnotes.

This keeps the current “prefer omission over false positives” strategy intact.

### Integration point in pipeline

Phase 4 should run as an **ingestion-time pre-chunking integration step**:

1. parse/extract page text and table/image structure,
2. run Phase 1/2 detection + Phase 3 linking,
3. apply Phase 4 text integration from resolved links,
4. then pass integrated `document_text` into chunking.

This means `chunk_document_by_section(...)` receives already-integrated text and should not be responsible for rerunning char-geometry marker detection.

### How Phase 4 uses Phase 1 outputs

Phase 4 should use Phase 1 marker detections **indirectly** through Phase 3 resolved link artifacts.

- Phase 1 provides marker candidates (`detected_anchors[]`).
- Phase 2 provides validated footnote contents.
- Phase 3 links marker ids to content ids and emits linked-content groups + orphan markers.
- Phase 4 consumes that linked result to inject markers and metadata safely.

If a workflow executes Phase 4 without persisted intermediate artifacts, Phase 1/2/3 must be recomputed in the same ingestion run before integration.
