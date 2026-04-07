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

### Fields

| Field | Type | Required | Default | Rules | Description |
|---|---|---:|---|---|---|
| `question` | string | Yes | — | Minimum length: 1 character | User question to answer. |
| `top_k` | integer \| null | No | `null` | If provided: `1 <= top_k <= 20` | Overrides service retrieval chunk count for this request. If omitted/null, service uses configured default. |
| `mode` | `"extractive" \| "generative"` | No | `"extractive"` | Must be one of the two literal values | Controls answer strategy. |
| `debug` | boolean | No | `false` | — | Enables debug diagnostics in response. |

## `mode` Behavior

### `extractive`

- Uses retrieved chunks directly to build an evidence-based summary.
- Keeps behavior closest to conservative RAG preview style.
- Citations are ordered from chunks actually used in the answer.

### `generative`

- Builds a constrained prompt from retrieved evidence and asks the configured generation endpoint/model to synthesize an answer.
- If generation is not enabled/configured, or generation fails, the service falls back to extractive summary text.
- Citations still map to retrieved chunks selected for generation.

## `debug` Behavior

### When `debug: false`

- `debug_info` is omitted from the response.
- You still receive:
  - `answer`
  - `citations`
  - `retrieved_chunks`
  - `confidence`
  - `refusal_reason`

### When `debug: true`

The response includes `debug_info` with structured diagnostics, including:

- `mode`, `top_k`
- `timings_ms` (`embedding_ms`, `retrieval_ms`, `answer_ms`, `total_ms`)
- `retrieval` block:
  - hit counts and thresholds
  - scores
  - selected/used chunk ids
  - `retrieved_documents` (metadata + score + snippet)
- `evidence_sufficiency`
- `refusal_handling` (`triggered`, `reason`)

## Response Schema

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_uid": "docA:12",
      "doc_id": "docA",
      "title": "OSHA 1910",
      "section": "1910.147",
      "chunk_id": 12,
      "page": 88,
      "score": 0.81
    }
  ],
  "retrieved_chunks": 5,
  "confidence": 0.74,
  "refusal_reason": null,
  "debug_info": {
    "mode": "extractive",
    "top_k": 5,
    "timings_ms": {"embedding_ms": 1.23, "retrieval_ms": 2.7, "answer_ms": 0.4, "total_ms": 4.5},
    "retrieval": {
      "hit_count": 5,
      "min_score_threshold": 0.2,
      "min_chunks_threshold": 1,
      "scores": [0.81, 0.76, 0.71],
      "selected_chunk_uids": ["docA:12", "docB:5"],
      "used_chunk_uids": ["docA:12"],
      "retrieved_documents": []
    },
    "evidence_sufficiency": {"hit_count": 5, "best_score": 0.81, "refusal_reason": null},
    "refusal_handling": {"triggered": false, "reason": null}
  }
}
```

## Refusal Behavior

The service may refuse when evidence is insufficient, returning a safe refusal answer with:

- empty citations
- `confidence: 0.0`
- `refusal_reason` set (for example: `insufficient_chunks`, `low_retrieval_score`, `no_query_embedding`)

If `debug=true`, refusal responses also include diagnostics and refusal metadata in `debug_info`.

## Configuration Notes

Runtime behavior is tunable via environment variables (configured in dependency wiring), including:

- `QUERY_DEFAULT_MODE`
- `QUERY_MIN_SCORE`
- `QUERY_MIN_CHUNKS`
- `QUERY_TOP_K`
- generation options (`QUERY_GENERATION_ENABLED`, `QUERY_GENERATION_MODEL`, `QUERY_GENERATION_ENDPOINT`, `QUERY_GENERATION_TIMEOUT_SECONDS`, `QUERY_GENERATION_TEMPERATURE`)

These defaults are used when request-level overrides are absent.
