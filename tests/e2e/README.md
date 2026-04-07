# E2E Vertical Slice Procedure (`14-TMCS.pdf`)

## Executive Summary

Use this procedure to validate a full vertical slice of the RAG flow against a single known PDF (`14-TMCS.pdf`) located in `/temp/my_pdfs`.

At a high level, you will:

1. Build retrieval artifacts from `/temp/my_pdfs`.
2. Start the API using those artifacts via `INDEX_DIR`.
3. Run a health check.
4. Run a positive query and verify grounded output with citations.
5. Run a negative query and verify safe refusal behavior.
6. Optionally run a generative-mode check to validate fallback behavior.

This is intended as a manual end-to-end smoke test you can run after unit tests pass.

---

## Detailed Step-by-Step Instructions

### Preconditions

- Python 3.11+ environment is ready.
- Dependencies are installed.
- `/temp/my_pdfs` exists and contains only `14-TMCS.pdf`.

---

### 1) Build index artifacts from `/temp/my_pdfs`

Create a project-local output directory and build index artifacts (`chunk_index.json`, `chunk_store.jsonl`, `index_manifest.json`).
Using `./temp/...` avoids accidentally writing to an OS-level temp directory (for example, `C:\temp` on Windows).

```bash
mkdir -p ./temp/e2e_slice_index

python scripts/build_index.py \
  --input ./temp/my_pdfs \
  --output ./temp/e2e_slice_index \
  --chunk-size 800 \
  --overlap 150 \
  --embedding-dim 256 \
  --embedder hash
```

or

```bash
py scripts/build_index.py --input ./temp/my_pdfs --output ./temp/e2e_slice_index --chunk-size 800 --overlap 150 --embedding-dim 256 --embedder hash
```


**Success indicators**

- Command exits with code `0`.
- `./temp/e2e_slice_index/chunk_index.json` exists.
- `./temp/e2e_slice_index/chunk_store.jsonl` exists.
- `./temp/e2e_slice_index/index_manifest.json` exists.

---

### 2) Start the API against this index

Point the service to the fresh artifacts and launch FastAPI.

```bash
export INDEX_DIR=./temp/e2e_slice_index
uvicorn app.main:app --reload --port 8000
```

**Success indicators**

- Uvicorn starts cleanly.
- The app is reachable at `http://127.0.0.1:8000`.

---

### 3) Health check

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/health
```

**Expected output**

```json
{"status":"ok"}
```

---

### 4) Positive query (extractive + debug)

Run a relevant query and inspect retrieval/citation/debug fields.

```bash
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key requirements described in this document?",
    "top_k": 5,
    "mode": "extractive",
    "debug": true
  }' | jq
```

**Success indicators**

- `retrieved_chunks > 0`
- `citations` is non-empty
- `refusal_reason` is `null`
- `debug_info` is present

---

### 5) Negative query (refusal-path validation)

Run an intentionally unrelated question.

```bash
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the population of Mars City in 2040?",
    "top_k": 5,
    "mode": "extractive",
    "debug": true
  }' | jq
```

**Expected refusal indicators**

- `citations` is empty
- `confidence` is `0.0`
- `refusal_reason` is set (for example `insufficient_chunks` or `low_retrieval_score`)

---

### 6) Optional: Generative-mode fallback check

If generation is not configured, service should fall back safely to extractive behavior.

```bash
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize this document in plain language.",
    "mode": "generative",
    "debug": true
  }' | jq
```

**What to confirm**

- Response returns successfully.
- If generation is unavailable, answer indicates fallback behavior.
- Citations still map to retrieved evidence.

---

## Completion Criteria

The vertical slice is considered validated when all of the following are true:

1. Index artifacts are built successfully from `/temp/my_pdfs` into `./temp/e2e_slice_index`.
2. API starts and `GET /health` returns `{"status":"ok"}`.
3. Positive query returns grounded answer with citations and no refusal.
4. Negative query returns safe refusal with explicit reason.

---

## Notes

- Keep this as a deterministic smoke test baseline.
- Re-run this procedure after major changes to ingestion, retrieval, or query-response wiring.
