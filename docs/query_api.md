# Query API Guide

This document explains how to call the `POST /query` endpoint, what each request option does, and how response/refusal/debug behavior works.

## Endpoint

- **Method:** `POST`
- **Path:** `/query`
- **Content-Type:** `application/json`

## Request Body

```json
{
  "question": "What are lockout/tagout training requirements?",
  "top_k": 5,
  "mode": "extractive",
  "debug": false
}
```

## `mode` Behavior

### `extractive`

- Uses retrieved chunks directly to build a deterministic evidence summary.
- Works as the non-LLM baseline.

### `generative`

- Always performs retrieval + evidence sufficiency checks first.
- Uses configured generator backend (for example, local Ollama).
- If generator is unavailable/fails, the service falls back explicitly to extractive output without crashing.
- Citations are always from retrieved chunks, never from free-form generated claims.

## `debug` Behavior

When `debug: true`, `debug_info` includes:

- selected `mode`, `top_k`
- `timings_ms`
- retrieval scores and selected chunks
- `evidence_sufficiency` result
- generation details (`provider`, `used`, `fallback_used`, `fallback_reason`)
- refusal reason/trigger when applicable
- embedder backend/model/dimension used at query time

## Response Schema

```json
{
  "answer": "...",
  "citations": [],
  "retrieved_chunks": 5,
  "confidence": 0.74,
  "refusal_reason": null,
  "debug_info": {}
}
```

## Refusal Behavior

The service refuses when retrieval evidence is insufficient. Refusals return:

- empty `citations`
- `confidence: 0.0`
- `refusal_reason` (`insufficient_chunks`, `low_retrieval_score`, `no_query_embedding`, etc.)

## Runtime Configuration

- Embedding: `EMBEDDER_PROVIDER`, `EMBEDDING_MODEL`
- Query policy: `QUERY_DEFAULT_MODE`, `QUERY_MIN_SCORE`, `QUERY_MIN_CHUNKS`, `QUERY_TOP_K`
- Generation provider: `QUERY_GENERATION_PROVIDER`, `QUERY_GENERATION_ENABLED`, `QUERY_GENERATION_MODEL`, `QUERY_GENERATION_ENDPOINT`, `QUERY_GENERATION_TIMEOUT_SECONDS`, `QUERY_GENERATION_TEMPERATURE`

On startup, the API validates query embedding settings against `index_manifest.json` and fails fast on mismatch.
