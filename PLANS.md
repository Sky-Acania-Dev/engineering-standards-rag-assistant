# Engineering Standards RAG Assistant — Development Plan

## Project Objective

Build a production-style RAG assistant that:

- Answers questions over engineering/regulatory documents
- Provides grounded citations
- Uses bounded agentic orchestration (LangGraph)
- Runs locally (Ollama + FAISS)
- Exposes a FastAPI service
- Includes evaluation (faithfulness > context relevance)
- Includes basic observability (logging, tracing)

---

# Development Phases

## Phase 0 — Scope & Guardrails

**Goal:** Define constraints before building.

### Tasks
- Define supported query types:
  - lookup, summarize, compare, extract requirements
- Define non-goals:
  - no legal/engineering advice
  - no compliance guarantees
  - no copyrighted redistribution
- Define success criteria:
  - grounded answers with citations
  - correct refusal on insufficient evidence
- Freeze tech stack:
  - Python, FAISS, Ollama, LangGraph, FastAPI, Docker

---

## Phase 1 — Foundation MVP

**Goal:** End-to-end working pipeline.

### Tasks
- Repository structure setup
- Small controlled corpus (5–20 documents)
- Ingestion pipeline:
  - load → normalize → chunk → embed → store
- Metadata schema:
  - doc_id, title, page, section, chunk_id
- Chunking (basic):
  - 500–1000 tokens, overlap 100–150
- Retrieval:
  - FAISS top-k
- Generation:
  - grounded answer with citations
- API:
  - `/query`, `/ingest`, `/health`

### Deliverable
- Working RAG pipeline with citation output

---

## Phase 2 — RAG Quality Pass

**Goal:** Improve correctness and evaluation.

### Tasks
- Build evaluation dataset (30–100 Q/A pairs)
- Add retrieval diagnostics:
  - retrieved chunks, scores, latency
- Improve chunking:
  - heading-aware
  - section-aware
- Add reranking:
  - retrieve (top 20–40) → rerank → top 5–10
- Improve citation formatting:
  - title, page, section
- Add refusal handling:
  - insufficient / conflicting evidence

### Deliverable
- Measurable retrieval quality
- Reduced hallucination
- Improved citations

---

## Phase 3 — Bounded Agentic Orchestration

**Goal:** Add structured reasoning without chaos.

### Graph Design
- Query analysis
- Retrieval
- Evidence sufficiency check
- Answer synthesis
- Citation validation
- Final response

### Tasks
- Query classification:
  - lookup, compare, summarize, unclear
- Evidence sufficiency node:
  - sufficient / weak / conflicting / none
- Citation validation node:
  - verify claims match sources
- Add retry logic:
  - max retrieval attempts: 2–3
  - bounded loops
- Add fallback behavior:
  - refusal or cautious answer

### Deliverable
- Deterministic LangGraph workflow
- Controlled reasoning and retries

---

## Phase 4 — Productionization & Polish

**Goal:** Make it look like real engineering.

### Tasks

#### Logging
- Structured JSON logs:
  - request_id, latency, retrieved chunks, model

#### Tracing
- Step-level tracking:
  - retrieval, generation, validation

#### Config Management
- Centralized configs:
  - chunk size, top-k, model, timeout

#### Dockerization
- Reproducible setup
- Local deployment instructions

#### Testing
- Unit tests:
  - ingestion, chunking, retrieval
- Integration test:
  - end-to-end query

#### Benchmarking
- Eval runner:
  - faithfulness proxy
  - retrieval hit rate
  - latency

#### Documentation
- Architecture
- Design decisions
- Evaluation methodology
- Known limitations

### Deliverable
- Production-style repo with reproducibility

---

# Milestone Timeline

## Milestone 1 — Week 1
- Repo setup
- Environment + configs
- Ingestion skeleton

## Milestone 2 — Week 2
- Chunking
- Embeddings
- FAISS indexing

## Milestone 3 — Week 3
- Query pipeline
- Answer generation
- Citation formatting

## Milestone 4 — Week 4
- FastAPI endpoints
- End-to-end demo
- Initial README

## Milestone 5 — Week 5
- Evaluation dataset
- Retrieval diagnostics
- Chunking improvements

## Milestone 6 — Week 6
- Reranking
- Refusal handling
- Evidence checks

## Milestone 7 — Week 7
- LangGraph workflow
- Validation node
- Retry constraints

## Milestone 8 — Week 8
- Logging & tracing
- Testing
- Docker
- Benchmark report
- Final polish

---

# Priority Order (Strict)

1. Ingestion
2. Metadata
3. Chunking
4. Embeddings + FAISS
5. Query endpoint
6. Grounded generation
7. Citations
8. Evaluation dataset
9. Reranking
10. Refusal handling
11. LangGraph
12. Logging/tracing/testing
13. Docker/docs

---

# Core Risks

## 1. Bad PDF Parsing
- Fix: store raw extracted text, modular parser

## 2. Broken Chunking
- Fix: preserve section structure

## 3. Weak Local Models
- Fix: rely more on retrieval + validation

## 4. Fake Citations
- Fix: add validation node

## 5. Scope Creep
- Fix:
  - no UI early
  - no cloud infra early
  - no multi-agent complexity

---

# Success Criteria

A strong version of this project:

- Produces grounded answers with traceable citations
- Refuses when evidence is insufficient
- Has measurable evaluation results
- Uses bounded orchestration (not open-ended agents)
- Is reproducible via Docker
- Has clean, inspectable architecture

---

# Stretch Goals (Optional)

## Good
- Hybrid retrieval
- Better reranker
- Table-aware parsing
- Document versioning
- Compare-documents mode

## Avoid Early
- Multi-agent systems
- Autonomous browsing
- Complex frontend
- Cloud microservices

---

# Philosophy

- Clarity > Cleverness  
- Evaluation > Hype  
- Constraints > Autonomy  
- Correctness > Fluency  
