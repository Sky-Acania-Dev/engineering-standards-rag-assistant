# Footnote Phased Testing — PowerShell Commands

Use these commands from the repository root in **VS Code PowerShell**.

---

## 0) Session setup (required)

```powershell
Set-Location "C:\path\to\engineering-standards-rag-assistant"
$env:PYTHONPATH = "."
```

This fixes the common `No module named app` error.

---

## 1) Phase 1 — run the debug extractor

```powershell
$pdfPath = "C:\path\to\input.pdf"
$outPath = "C:\path\to\phase1_anchors.json"

python .\scripts\footnote_phase1_debug.py $pdfPath --out $outPath
```

If you omit `--out`, output defaults to `<input_pdf>.phase1_anchors.json`.

---

## 2) Phase 1 — run focused parser tests

```powershell
python -m pytest -q tests\unit\test_parsers_html_pdf.py
```

---

## 3) Phase 1 — run related ingestion regression checks

```powershell
python -m pytest -q tests\unit\test_day2_ingestion_flow.py -k "footnote_contaminated_heading_is_normalized or url_fragment_and_footnote_spillover_do_not_create_section"
```

---

## 4) Optional quick checks against JSON output

### Pretty-print JSON

```powershell
Get-Content $outPath | python -m json.tool
```

### Count detected anchors per page

```powershell
python -c "import json; d=json.load(open(r'$outPath', encoding='utf-8')); print([(p['page'], len(p.get('detected_anchors', []))) for p in d])"
```

### Show pages where anchors were found

```powershell
python -c "import json; d=json.load(open(r'$outPath', encoding='utf-8')); print([p['page'] for p in d if p.get('detected_anchors')])"
```

---

## Notes for future phases

- Keep this file as the single place for phase-by-phase PowerShell commands.
- Add sections for Phase 2/3/4 scripts and tests as they are introduced.
