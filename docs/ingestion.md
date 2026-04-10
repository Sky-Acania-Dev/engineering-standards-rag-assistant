# Ingestion Notes

## PDF layout-aware footnote pipeline

The PDF ingestion path now prefers layout evidence over flattened-text digit heuristics.

Pipeline stages:

1. **Layout token extraction** (`app/ingestion/pdf_layout_extractor.py`)
   - Reads ordered word spans from PDF pages with geometry and typography metadata (`x0/x1/top/bottom`, font size, font name, reading order).
2. **Superscript anchor detection** (`app/ingestion/footnote_detector.py`)
   - Scores anchors using multi-signal rules: short numeric token, smaller font ratio, raised baseline proxy, and tight adjacency.
3. **Footnote region detection** (`app/ingestion/footnote_detector.py`)
   - Detects bottom-page footnote bodies and preserves full content (including multi-line accumulation by ID).
4. **Anchor linking** (`app/ingestion/footnote_linker.py`)
   - Links anchor IDs to same-page footnote body IDs.
5. **Text normalization** (`app/ingestion/text_normalizer.py`)
   - Emits inline `[fn:N]` markers and explicit `[FOOTNOTE_DEF]` records with `id|anchor_text|content`.
6. **Chunk metadata attachment** (`app/ingestion/footnote_resolver.py`, `app/rag/chunking.py`)
   - Consumes explicit footnote definitions and attaches deduplicated structured metadata to chunks.

### Fallback behavior

If layout signals are unavailable or weak, ingestion does **not** infer footnotes from arbitrary inline digits.
This conservative strategy intentionally prefers false negatives over false positives to protect legal citations,
standards, years, and URLs from corruption.

### Tradeoffs

- **Safer text integrity:** avoids corrupting values like `2306`, `2009`, `R802`, `ASSE 1051-2009`, and long URLs.
- **Potential misses:** true footnotes may be skipped when PDFs do not expose reliable span geometry.
- **Maintains chunking stability:** section/table/TOC behavior remains unchanged because footnote metadata is injected as a dedicated subsystem.
