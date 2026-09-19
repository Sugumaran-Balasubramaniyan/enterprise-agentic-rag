# Research & Design — Enterprise Agentic RAG

> **Status:** living research notebook · **Last updated:** 2026-09-19
> **Scope:** architecture rationale, key papers and references, and the forward roadmap.
> Companion doc: [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — the state-of-the-art gap analysis behind this repo.

This document is written as an engineer's research notebook: it captures *why* the architecture is
shaped the way it is, which papers and references inform each layer, and where the project intends to
go next. Design decisions here are meant to be challenged — open an issue if a claim is wrong.

---

## 1. System Overview

`enterprise-agentic-rag` is an enterprise-grade, **agentic** Retrieval-Augmented Generation platform.
It couples a PostgreSQL 16 + PGVector (HNSW) vector store with an autonomous multi-step agent
orchestrator and a two-stage **deterministic** guardrail stack.

The design stance is deliberately *defense-in-depth with determinism at the security boundary*:
retrieval and generation can be fast and flexible, but anything touching safety (prompt injection,
PII, RBAC) is enforced by deterministic code, never by an LLM's judgment.

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer 1  Client & Consumption   (Streamlit UI :8501, FastAPI :8000) │
├────────────────────────────────────────────────────────────────────┤
│ Layer 2  Deterministic Safety Firewall  (Pre-execution guardrail)    │
├────────────────────────────────────────────────────────────────────┤
│ Layer 3  Agent Orchestrator     (intent planning → tool dispatch)    │
├────────────────────────────────────────────────────────────────────┤
│ Layer 4  Retrieval & Storage    (PGVector HNSW + tsvector hybrid,    │
│                                   in-memory fallback)                │
├────────────────────────────────────────────────────────────────────┤
│ Layer 5  Post-Execution Safety  (PII scrub + claim-level grounding)  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Layers & Rationale

### 2.1 Client & Consumption Layer
FastAPI `/api/v1` exposes a typed JSON API plus an SSE streaming endpoint; a Streamlit "Mission
Control" dashboard renders the same pipeline. Streaming (SSE) surfaces reasoning steps, tool
dispatches, and tokens in real time — important for agentic systems where the final answer alone does
not convey *how* the system arrived there, and for building user trust in the retrieval steps.

### 2.2 Deterministic Safety Firewall (Pre-Execution)
Every query passes a deterministic pre-execution gate before any tool runs:

- Adversarial / jailbreak detection (`DAN`-style modes, developer-prompt exfiltration, system-prompt
  overrides).
- Obfuscation decoding: Base64, hex, ROT13 are decoded *before* evaluation so encoded payloads cannot
  bypass filters.
- Injection filters for SQL patterns (`UNION SELECT`, `DROP`, `1=1`) and shell injection (`$(...)`,
  backticks).
- **RBAC** screening against the caller's `user_role`.

**Why deterministic?** In the OWASP LLM Top 10 threat model ([see references](#7-references)), prompt
injection and data leakage are top risks; a guardrail that "reasons" about a prompt is itself
injectable. A fixed-pattern, auditable filter is stronger for the cases that matter most, at the cost
of some false positives. Determinism also makes the guardrail unit-testable and measurable — see the
100% block-rate results in the README benchmark section.

### 2.3 Agent Orchestrator (Multi-Step Tool Dispatcher)
A state machine classifies intent and plans a sequence of tool invocations (retrieval, deterministic
calculation, citation verification), capped at `max_steps`. Each step records a `ToolExecutionTrace`
(latency, arguments, output) for auditability. This mirrors the *agentic* pattern from the Agentic RAG
survey lineage — the system decides **whether / what / how** to retrieve, iterating on tool results
rather than emitting a single fixed retrieval call. The reference architecture for this iterative
loop is the **ReAct** paradigm ([Yao et al., 2022](#7-references)).

### 2.4 Retrieval & Storage
- **PGVector HNSW** graph indexing (`m=16, ef_construction=64, ef_search=32`) over 1536-dim,
  normalized embeddings for sub-20 ms median latency.
- **Hybrid retrieval**: dense cosine similarity fused with PostgreSQL `tsvector` lexical search via
  **Reciprocal Rank Fusion** (`k=60`). Hybrid sparse+dense retrieval is widely shown to beat dense-only
  on out-of-domain and keyword-heavy queries.
- **Dual-engine fallback**: an in-memory SIMD/NumPy store keeps the system usable when PostgreSQL is
  offline — same interface, no downtime.

### 2.5 Post-Execution Safety & Grounding
- Deterministic **PII scrubbing** (API keys, SSNs, card numbers, emails, phones, IPs).
- **Claim-level factual grounding**: token-overlap consistency between generated claims and retrieved
  source chunks, plus a citation verifier reporting precision/coverage. This is the "faithfulness"
  problem RAG is known for ([Self-RAG, Asai et al., 2023](#7-references)) — here enforced with
  deterministic, measurable checks rather than relying on the generator to be honest.

---

## 3. Key References

The following papers and resources inform the design and the roadmap. Citations are given so reviewers
can verify claims directly.

| Reference | Topic | Why it matters here |
|-----------|-------|---------------------|
| [Self-RAG — arXiv:2310.11511](https://arxiv.org/abs/2310.11511) — Asai et al., 2023 | Self-reflective RAG: retrieval on demand, then reflection to assess whether the output is grounded | Grounds the post-execution grounding/reflection layer and the planned "reflect & correct" loop |
| [CRAG — arXiv:2401.15884](https://arxiv.org/abs/2401.15884) — Yan et al., 2024 | Corrective RAG: evaluate retrieval quality and trigger re-retrieval when it is poor | Informs the roadmap's retrieval-correction path for low-confidence recalls |
| [Adaptive-RAG — arXiv:2403.14403](https://arxiv.org/abs/2403.14403) — Jeong et al., 2024 | Route queries to no-retrieval / single-step / multi-step agents based on complexity | Routing strategy for the agent planner to avoid over- or under-retrieving |
| [RAG-Fusion — arXiv:2402.03367](https://arxiv.org/abs/2402.03367) — Lau et al., 2024 | Multi-query generation + RRF to broaden recall | Basis for multi-query query-transformation in the roadmap |
| [ReAct — arXiv:2210.03629](https://arxiv.org/abs/2210.03629) — Yao et al., 2022 | Reasoning + acting interleaved in language models | The conceptual backbone of the tool-dispatching agent loop |
| [GraphRAG — arXiv:2404.16130](https://arxiv.org/abs/2404.16130) — Edge et al., 2024 | Global text-graph communities to answer *global* questions over a corpus | Motivates the roadmap's graph-backed indexing for corpus-wide queries |
| [LightRAG — arXiv:2410.05779](https://arxiv.org/abs/2410.05779) — Guo et al., 2024 | Lightweight, incrementally-updatable GraphRAG with dual-level retrieval | Chosen implementation path for GraphRAG (cheap to run, incremental updates) |
| [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 2024 | Prepending a short LLM "context chunk" to each embedding to disambiguate | Roadmap: contextual chunking to lift retrieval precision on terse enterprise docs |
| [OWASP LLM Top 10 — genai.owasp.org](https://genai.owasp.org/llm-top-10/) | Top risks: prompt injection, sensitive-information disclosure, etc. | Threat model that justifies the deterministic guardrail stack |
| [RAGAS — docs.ragas.io](https://docs.ragas.io) | Metrics: faithfulness, answer relevancy, context precision/recall | Evaluation harness for grounding + retrieval quality (see roadmap) |

---

## 4. Design Tension & Trade-offs (explicit decisions)

- **Deterministic guards over learned guards for security-critical paths.** Accepted cost: possible
  false positives on legitimate-but-adversarial-looking text. This is a deliberate bet — a deterministic
  filter is provable and testable; a learned guard is not.
- **Hybrid retrieval over dense-only.** Adds a dependency on `tsvector` but materially improves recall
  on technical / keyword-dense enterprise queries, consistent with the hybrid-search literature.
- **Dual-engine storage.** The in-memory fallback trades full SQL functionality for availability.
  It exists so the demo and tests never hard-crash on a missing database.
- **Explicit agent loop over a single-shot pipeline.** More moving parts, but the capacity for
  multi-tool, multi-step questions (e.g. "retrieve, then size, then verify citations") is the entire
  point of *agentic* RAG.

---

## 5. Roadmap

Ranked by impact-to-effort, following the gap analysis in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

### 5.1 GraphRAG via LightRAG
Provide graph-based indexing over the corpus to answer *global* / corpus-summarising questions that
flat top-k retrieval answers poorly. Use LightRAG ([arXiv:2410.05779](#7-references)) for its low cost
and incremental updates rather than a heavyweight GraphRAG pipeline; expose it as an additional
retrieval tool the orchestrator can dispatch to when the query is corpus-level.

### 5.2 MCP (Model Context Protocol) Server
Expose the retrieval, calculation, and citation tools as an MCP server so external agents/clients can
discover and call them with a standard, typed interface. This turns the platform from a single app into
a *service* other AI systems can compose.

### 5.3 Deep-Research Mode
A higher `max_steps`, parallel multi-query expansion (RAG-Fusion style), retrieve→reflect→re-retrieve
(CRAG / Self-RAG style), and a structured long-form composer. Aimed at "write me a report on X" queries
rather than single-answer lookups.

### 5.4 Query Transformation
Explicit rewrite / multi-query expansion / **HyDE** (hypothetical document embeddings) transforms before
retrieval to lift recall on ambiguous or conversational queries. Complements the existing RRF hybrid.

### 5.5 Reranking
A cross-encoder reranker between retrieval and generation to tighten precision on the fused top-k.

### 5.6 Evaluation Harness
A RAGAS-style suite (faithfulness, answer relevancy, context precision/recall) wired into `make eval`,
so every retrieval or generation change is measured, not eyeballed.

### 5.7 Observed-life hardening
Semantic caching, more observability (per-step latency tracebacks are already present), auth/rate-limit
wiring, and CI hooks for the eval suite.

---

## 6. Reproducibility Notes

- Latency and grounding evaluations are standalone scripts under `benchmarks/` — see the README
  benchmark section and run `make benchmark` / `make eval`.
- `LLM_PROVIDER=mock` ships as the offline-safe default; real provider keys are opt-in.
- The seeded enterprise documents that back interactive demos live in `data/documents/` and mirror the
  departments the guardrails' RBAC and the dashboard's tenant scoping refer to
  (Platform Engineering, Data Engineering, Security & Compliance).

---

## 7. References (full list, link-ready)

- Self-RAG: <https://arxiv.org/abs/2310.11511>
- CRAG: <https://arxiv.org/abs/2401.15884>
- Adaptive-RAG: <https://arxiv.org/abs/2403.14403>
- RAG-Fusion: <https://arxiv.org/abs/2402.03367>
- ReAct: <https://arxiv.org/abs/2210.03629>
- GraphRAG: <https://arxiv.org/abs/2404.16130>
- LightRAG: <https://arxiv.org/abs/2410.05779>
- Anthropic Contextual Retrieval: <https://www.anthropic.com/news/contextual-retrieval>
- OWASP LLM Top 10: <https://genai.owasp.org/llm-top-10/>
- RAGAS: <https://docs.ragas.io>