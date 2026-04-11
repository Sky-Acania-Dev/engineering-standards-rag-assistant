
## 1) Identify Footnote Marks

``` VS Code Powershell
Set-Location "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant"
$env:PYTHONPATH = "."
$pdfPath = "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant\temp\my_pdfs\14-TMCS.pdf"
$outJson = "C:\Personal Folder\Work\WorkRepo\engineering-standards-rag-assistant\temp\tests\sample.phase1_anchors.json"

py scripts/footnote_phase1_debug.py $pdfPath --out $outJson 

```