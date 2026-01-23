# Engineering Standards RAG Assistant

Production-style GenAI assistant for querying engineering and regulatory standards using advanced RAG and bounded agentic orchestration. Returns citation-grounded answers, runs automated evaluation, and exposes a FastAPI service with cost and safety controls.

## Scope and Non-Goals
**In scope**
- Grounded Q&A over standards/regulatory documents with citations
- Deterministic, bounded agent workflow (no open-ended autonomy)
- Evaluation harness (faithfulness, context relevance, latency, cost)
- Production API packaging (FastAPI + Docker)

**Out of scope**
- Legal/engineering advice or authoritative compliance decisions
- Document redistribution of copyrighted standards

## Key Features
- Document ingestion: PDF + HTML (extensible)
- Retrieval: hybrid search + reranking (configurable)
- Orchestration: multi-step agent graph (query → retrieve → synthesize → validate)
- Citations: chunk-level source references returned with answers
- Evaluation: automated runs with stored metrics and regression tracking
- Ops: structured logging, request tracing, and basic cost controls

## Architecture (High-Level)
1. **Ingest** documents → parse → normalize → chunk
2. **Index** chunks → embeddings + metadata → vector store
3. **Query** → agent graph selects retrieval strategy
4. **Retrieve** → hybrid search + rerank → context pack
5. **Answer** → grounded synthesis with citations
6. **Validate** → critic checks grounding and refusal rules
7. **Measure** → evaluation records quality, latency, and cost

See `docs/architecture.md` for details.

## Quickstart
### Prerequisites
- Python 3.11+
- (Optional) Docker

### Local run
1. Copy env template:
   - `cp .env.example .env`
2. Create and activate venv, install deps:
   - `pip install -r requirements.txt`
3. Ingest sample docs (or your own):
   - `python scripts/ingest.py --input app/data/sample_docs`
4. Start API:
   - `uvicorn app.main:app --reload`
5. Query:
   - `POST /query`

### Docker run
- `docker compose up --build`

## Configuration
- Environment variables are documented in `.env.example`
- Model, retrieval, and evaluation settings live in `app/core/config.py`

## Evaluation
- Run:
  - `python scripts/eval.py --suite baseline`
- Outputs:
  - `app/evaluation/results/` (gitignored by default)

## Safety and Disclaimer
This tool provides **retrieval and summarization with citations** and is not a substitute for professional engineering judgment, code compliance review, or legal advice. See `docs/risk_and_disclaimer.md`.

## Roadmap
- Add version-aware document metadata and filtering
- Add structured “standard section” parsing for hierarchical citations
- Add caching + rate limiting
- Add offline regression set and CI evaluation gate

## License
MIT (see `LICENSE`).
