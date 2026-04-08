# E2E Vertical Slice Procedure (`14-TMCS.pdf`)

## 0) Install semantic embedding dependency (required for sentence-transformer embedder)

### Bash

```bash
python -m pip install sentence-transformers
```

### VS Code PowerShell

```powershell
py -m pip install sentence-transformers
```

## 1) Build index with semantic embeddings

### Bash

```bash
mkdir -p ./temp/e2e_slice_index
python scripts/build_index.py \
  --input ./temp/my_pdfs \
  --output ./temp/e2e_slice_index \
  --chunk-size 800 \
  --overlap 150 \
  --embedder sentence_transformer \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

### VS Code PowerShell

```powershell
New-Item -ItemType Directory -Force -Path .\temp\e2e_slice_index | Out-Null
py scripts/build_index.py `
  --input .\temp\my_pdfs `
  --output .\temp\e2e_slice_index `
  --chunk-size 800 `
  --overlap 150 `
  --embedder sentence_transformer `
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Confirm `chunk_index.json`, `chunk_store.jsonl`, and `index_manifest.json` are present.

## 2) Start API (extractive baseline)

Run this in **Terminal A** and keep it running for steps 3 and 4.

### Bash

```bash
export INDEX_DIR=./temp/e2e_slice_index
export EMBEDDER_PROVIDER=sentence_transformer
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export QUERY_DEFAULT_MODE=extractive
uvicorn app.main:app --reload --port 8000
```

### VS Code PowerShell

```powershell
$env:INDEX_DIR = ".\temp\e2e_slice_index"
$env:EMBEDDER_PROVIDER = "sentence_transformer"
$env:EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
$env:QUERY_DEFAULT_MODE = "extractive"
py -m uvicorn app.main:app --reload --port 8000
```

If embedder settings do not match manifest, startup should fail immediately with a clear mismatch error.

## 3) Positive extractive query

Run this in **Terminal B** while Terminal A keeps the API server running.

### Bash

```bash
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key requirements described in this document?","top_k":5,"mode":"extractive","debug":true}' | jq
```

### VS Code PowerShell

```powershell
$body = @{
  question = "What are the key requirements described in this document?"
  top_k = 5
  mode = "extractive"
  debug = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

Expected: non-empty citations, no refusal, debug retrieval scores.

## 4) Negative query refusal

Run this in **Terminal B** while Terminal A keeps the API server running.

### Bash

```bash
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the population of Mars City in 2040?","top_k":5,"mode":"extractive","debug":true}' | jq
```

### VS Code PowerShell

```powershell
$body = @{
  question = "What is the population of Mars City in 2040?"
  top_k = 5
  mode = "extractive"
  debug = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

Expected: refusal reason set (`low_retrieval_score` or `insufficient_chunks`), citations empty, confidence 0.0.
With `sentence_transformer`, the default refusal floor is stricter (`min_score_threshold` shows `0.4` in debug).

## Ollama local LLM setup (prerequisite for generative mode)

If you want to run Step 5 (`mode=generative`), ensure a local Ollama model is installed and running.
You can run Ollama in **Terminal C** (or as a background service/app), while Terminal A continues running the API server.

### Bash

```bash
# 1) Install Ollama from https://ollama.com/download (one-time)
# 2) Start Ollama service/app (if not already running)
ollama serve

# 3) Pull a model (example)
ollama pull llama3.1

# 4) Quick sanity check
ollama run llama3.1 "Respond with: Ollama ready"
```

### VS Code PowerShell

```powershell
# 1) Install Ollama from https://ollama.com/download (one-time)
# 2) Start Ollama service/app (if not already running)
ollama serve

# 3) Pull a model (example)
ollama pull llama3.1

# 4) Quick sanity check
ollama run llama3.1 "Respond with: Ollama ready"
```

If `ollama serve` is already managed by the desktop app/service on your machine, you can skip launching it manually and just verify the model is available with `ollama list`.

## 5) Generative mode with Ollama

Start Ollama locally with your model (example `llama3.1`), then update env vars in **Terminal A** and restart the API there:

### Bash

```bash
export QUERY_DEFAULT_MODE=generative
export QUERY_GENERATION_ENABLED=true
export QUERY_GENERATION_PROVIDER=ollama
export QUERY_GENERATION_MODEL=llama3.1
export QUERY_GENERATION_ENDPOINT=http://localhost:11434/api/generate
```

### VS Code PowerShell

```powershell
$env:QUERY_DEFAULT_MODE = "generative"
$env:QUERY_GENERATION_ENABLED = "true"
$env:QUERY_GENERATION_PROVIDER = "ollama"
$env:QUERY_GENERATION_MODEL = "llama3.1"
$env:QUERY_GENERATION_ENDPOINT = "http://localhost:11434/api/generate"
```

Then query:

Run these query commands in **Terminal B** while Terminal A serves the API.

### Bash

```bash
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize this document in plain language.","mode":"generative","debug":true}' | jq
```

### VS Code PowerShell

```powershell
$body = @{
  question = "Summarize this document in plain language."
  mode = "generative"
  debug = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

Expected: retrieval-first behavior; debug shows whether generator was used or explicit fallback triggered.
