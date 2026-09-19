# Research Gap Analysis — Enterprise Agentic RAG

> **Date:** 2026-09-19
> **Purpose:** State-of-the-art (SOTA) gap analysis for this repository, grounded in a 5-track research sweep
> (agentic architectures, retrieval, evaluation, enterprise production, GraphRAG). Every gap is cited so
> reviewers can verify the claims. This document is the "why" behind the implementation plan and the
> roadmap in this repo.

---

## 1. Executive Summary

This repository is a solid *deterministic prototype*: FastAPI + PGVector + Streamlit with regex
guardrails, an in-memory RRF hybrid, and a scripted "agent" loop. The test suite is green and the
benchmarks are honest. However, measured against the 2025–2026 literature it has three
**credibility-critical gaps** and a long tail of missed SOTA techniques:

1. **There is no real LLM call in the codebase.** `_synthesize_response` pastes retrieved chunks
   together; `LLM_PROVIDER=mock` is the only provider. Interviewers and PhD reviewers will notice
   immediately. A showcase repo must have a real, pluggable LLM client with structured output,
   streaming, and tool calling.
2. **Embeddings are hash-based random projections, not a real model.** The 1536-dim vectors are
   deterministic pseudo-random projections of tokens. Retrieval quality therefore cannot be what the
   README claims, and it cannot be improved by known techniques (reranking, contextual retrieval, …).
   This is the root-cause blocker for everything else.
3. **The "agent" is a scripted regex pipeline, not an agent.** Intent classification is keyword
   matching; the tool sequence is fixed. The literature consensus (Agentic RAG Survey,
   arXiv:2501.09136) is that a real agent decides *whether/what/how* to retrieve iteratively.

The 2025–2026 consensus architecture (see §2) is a **layered, configurable system**: real embeddings →
structure-aware + contextual chunking → hybrid dense+sparse retrieval (RRF) → cross-encoder reranking →
query transformation → an LLM-driven ReAct/Plan-and-Execute loop with an Adaptive-RAG router, CRAG-style
retrieval correction, and Self-RAG-style reflection → grounded generation with citations → evaluation
(RAGAS-style) and observability (OpenTelemetry/Langfuse) → enterprise hardening (auth, rate limits,
semantic cache, audit). This plan adds those layers *without* destroying what exists.

---

## 2. Research Tracks & Findings (with citations)

### 2.1 Agentic RAG architectures (track 1)

| Finding | Source | Why it matters here |
|---|---|---|
| The field has converged on **agentic RAG**: an agent decides whether/what/how to retrieve, iterating with tools. | Agentic RAG Survey — arXiv:2501.09136 | Current repo is the textbook *fixed pipeline* this survey describes as the old generation. |
| **Self-RAG**: LM emits reflection tokens (Retrieve/ISREL/ISSUP/ISUSE) to critique its own retrieval and generation. | arXiv:2310.11511 | Cheap to approximate: LLM critique of chunk relevance/support *before* generation, gates hallucination. |
| **CRAG**: a retrieval evaluator grades retrieved docs {correct, incorrect, ambiguous} and triggers correction (keep / fall back to web search / blend). | arXiv:2401.15884 | Directly upgrades the fixed `vector_search → citation_verifier` sequence into a corrective loop. |
| **Adaptive-RAG**: a classifier routes each query to none / single-hop / multi-hop retrieval by complexity. | arXiv:2403.14403 | The natural successor to this repo's regex intent classifier; also cuts cost on simple queries. |
| **RAG-Fusion / multi-query**: generate N query variants, retrieve each, fuse with RRF. | arXiv:2402.03367 | Cheap recall win; the repo already has an RRF scaffold to reuse. |
| Agent loop standards: **ReAct**, **Plan-and-Execute**, **Reflexion**, using *native tool calling* (not prompt-parsed JSON). | arXiv:2210.03629, arXiv:2305.04091, arXiv:2303.11366 | Replaces the scripted loop; MCP (modelcontextprotocol.io) is the industry-standard tool/transport layer. |
| **Memory**: short-term conversation + long-term episodic/semantic memory (MemGPT-style paging). | arXiv:2310.08560, mem0.ai | The repo has no conversation memory at all; sessions are stateless. |
| Deep-research agents work via plan → iterate → synthesize-with-citations in a large context. | OpenAI deep research (Feb 2025), Gemini Deep Research | A "research mode" endpoint is a strong differentiator, but it is a *stretch* goal. |

### 2.2 Retrieval techniques (track 2)

| Finding | Source | Why it matters here |
|---|---|---|
| **Embeddings are the highest-leverage upgrade.** Modern defaults: Qwen3-Embedding (0.6B/4B/8B, Apache-2.0, MTEB 70.58 top multilingual, MRL), BGE-M3 (dense+sparse+multi-vector in one model), or API options OpenAI text-embedding-3-large / Cohere Embed v3 / Voyage. | qwenlm.github.io/blog/qwen3-embedding/; arXiv:2402.03216; openai.com/blog/new-embedding-models; cohere.com/blog/introducing-embed-v3 | Replaces the fake hash embeddings; unblocks every downstream technique. |
| **Matryoshka (MRL)** lets one model serve multiple dims (OpenAI `dimensions` param, Qwen3 variable dims). | arXiv:2205.13147 | Keep the pgvector column at 512–1024 dims to shrink HNSW size while keeping quality. |
| **Hybrid retrieval (dense + BM25) fused with RRF (k≈60) beats either alone** (~4–5% MAP) and is robust across domains. | Cormack et al., SIGIR'09 (RRF); Anthropic contextual retrieval | The repo's hybrid is in-memory-only and its sparse leg is a naive scorer — needs a real PG tsvector/BM25 leg and RRF in SQL. |
| **Reranking** (cross-encoder BGE-reranker-v2-m3, Qwen3-Reranker, or Cohere Rerank 3.5): Anthropic measured top-20 retrieval-failure drop 5.7% → 1.9% (**−67%**) when reranking was added. | huggingface.co/BAAI/bge-reranker-v2-m3; anthropic.com/news/contextual-retrieval | Second-highest-impact/low-effort win after real embeddings. |
| **Contextual retrieval** (prepend LLM chunk context before embedding *and* BM25): −35% (dense) / −49% (dense+BM25) top-20 retrieval failures, ≈ $1.02/M tokens with prompt caching. | anthropic.com/news/contextual-retrieval | The #1 chunking-level technique; pairs with hybrid. |
| **Query transformation**: rewrite–retrieve–read, HyDE (arXiv:2212.10496), multi-query expansion, step-back prompting (arXiv:2310.06117, +27% TimeQA). | arXiv:2305.14283, arXiv:2212.10496, arXiv:2310.06117 | The repo has zero query processing — high-value, low-effort adds. |
| **Late chunking** (embed long context, pool per chunk) — elegant but needs long-context embedders. | arXiv:2409.04701 | Documented stretch; not blockable. |
| **pgvector best practice**: HNSW (not IVFFlat) with tunable ef_search, `halfvec` + binary quantization for memory, `maintenance_work_mem` for fast builds; dedicated engines (Qdrant/Milvus) only past ~10M vectors or for ColBERT/multi-vector. | github.com/pgvector/pgvector; arXiv:1603.09320 (HNSW) | Config-level wins, zero quality cost. |
| MMR for diversity: only as a post-process for redundant result sets; off for factoid QA. | Carbonell & Goldstein (1998) | Small, config-gated post-processor. |

### 2.3 Evaluation & observability (track 3)

| Finding | Source | Why it matters here |
|---|---|---|
| **RAGAS** is the de-facto standard metric set: *faithfulness, answer relevancy, context precision, context recall, answer correctness*, plus noise sensitivity; LLM-as-judge based, with a synthetic testset generator. | docs.ragas.io | The repo's grounding eval is token-overlap only — a quality gate, not a real eval. Both should coexist. |
| **LLM-as-judge** best practice: pairwise or pointwise scoring with a strong judge, bias mitigation (position/verbosity), self-consistency. | arXiv:2306.05685 (MT-Bench / Chatbot Arena) | Needed for faithfulness/relevancy metrics without human labels. |
| Retrieval metrics: **nDCG@k, MRR, recall@k, hit rate** — computable without labels via LLM-based relevance. | BEIR: arXiv:2004.12832; MTEB: arXiv:2305.10435 | Add a labeled golden set + retrieval metric harness so every change is measured. |
| **Observability**: OpenTelemetry **GenAI semantic conventions** (`gen_ai.*` span attributes) with Langfuse / Phoenix / LangSmith for tracing RAG retrieval→generation, token usage, cost. | opentelemetry.io/docs/specs/semconv/gen-ai/; langfuse.com | The repo has in-memory metrics only; no per-request tracing, no cost/token tracking. |
| Hallucination detection beyond overlap: NLI-style (TrueTeacher/AlignScore) or LLM-judge; token-overlap ≠ grounding. | arXiv:2305.12000 (AlignScore family) | Upgrade `verify_factual_grounding` semantics (keep deterministic path for CI). |
| Agent evaluation (tool-call correctness, trajectory) is an open research area most repos skip. | Agentic RAG Survey — arXiv:2501.09136 | Document trajectory eval as an extension point. |

### 2.4 Enterprise production (track 4)

| Finding | Source | Why it matters here |
|---|---|---|
| **Prompt injection is OWASP LLM01 (top risk)**, and RAG does not itself mitigate it; need defense-in-depth: constrain behavior, validate output, filter I/O, least privilege. **Indirect injection via retrieved documents is a demonstrated attack** (arXiv:2302.12173) — retrieved content must be treated as untrusted data. | genai.owasp.org/llmrisk/llm01-prompt-injection/; arXiv:2302.12173 | Current guardrails are regex-only; add retrieval-time injection screening + system-prompt hardening + output validation. |
| Guardrail frameworks (NeMo Guardrails, Guardrails AI, Llama Guard) **complement** deterministic rules; both layers are additive. | github.com/NVIDIA/NeMo-Guardrails; github.com/guardrails-ai/guardrails; meta-llama/PurpleLlama | Keep deterministic layer (fast, precise); add optional model-based layer. |
| **PII at ingestion AND output** (Presidio) so PII never reaches embeddings/vector store. | github.com/microsoft/presidio | Currently only output-side regex scrubbing; ingestion-side redaction is missing. |
| **Auth**: FastAPI OAuth2/JWT is the standard; the API currently has none. **Rate limiting** is both cost and security control (OWASP LLM10). **Audit logging** with correlation IDs is required for compliance (GDPR Art. 5 accountability). | fastapi.tiangolo.com/tutorial/security/oauth2-jwt/; github.com/slowapi/slowapi | Three concrete production gaps, all easy to close behind config. |
| **Semantic caching** (GPTCache/Redis embedding cache) plus **provider prompt caching** (Anthropic: up to −90% cost, −85% latency on long prompts; OpenAI −90% cached tokens) are the biggest cost/latency levers. | github.com/zilliztech/GPTCache; anthropic.com/news/prompt-caching | Add a query-embedding semantic cache; structure prompts so the static prefix is cacheable. |
| **Streaming**: SSE (FastAPI StreamingResponse) is correct for 1-way token streaming; **structured output**: provider JSON-schema + Pydantic validation with bounded retry. | fastapi.tiangolo.com/advanced/custom-response/; platform.openai.com/docs/guides/structured-outputs | The repo "streams" a pre-built string; needs real LLM token streaming + typed outputs. |
| **Reliability**: tenacity retries with backoff, circuit breakers, timeouts, health checks, graceful degradation. | github.com/jd/tenacity | None present; add to LLM/embedding calls. |
| **Multi-tenancy**: `tenant_id` column + Postgres Row-Level Security + metadata pre-filtering; not just app-level filtering. | postgresql.org/docs/current/ddl-rowsecurity.html | Currently RBAC is keyword-screening only; DB-enforced isolation is the enterprise standard. |

### 2.5 GraphRAG (track 5)

| Finding | Source | Why it matters here |
|---|---|---|
| **GraphRAG** (MS): LLM-built entity KG + Leiden community detection + community summaries; global search answers "sensemaking" questions flat vector RAG cannot. Cost is heavy (~600K tokens global search). | arXiv:2404.16130 | Big differentiator but token-hungry; schedule after core pipeline. |
| **LightRAG** (HKU): graph+vector dual representation, dual-level retrieval, incremental updates, single call / <100 tokens per query — *practical* alternative; has a Postgres/pgvector backend. | arXiv:2410.05779 (EMNLP 2025) | The recommended way to add graphs to this stack. |
| **Graphs win on multi-hop + whole-corpus aggregation; vectors win on cheap fact lookup** — so route between them (query router). | arXiv:2604.09666 (RAGSearch); neo4j.com/blog/graphrag-manifesto/ | A "graph-enhanced local search" (entity-link → 1–2 hop traversal → inject context) is the highest-impact/effort feature. |
| **Entity extraction without LLM** (GLiNER, spaCy) keeps indexing offline/cheap; **KG construction quality** (dedup, coreference, typed relation schema) is the #1 quality lever. | arXiv:2311.08526 (GLiNER); arXiv:2408.08921 (GraphRAG survey) | Use GLiNER/spaCy for the indexing path; enforce a small relation vocabulary. |
| Storage for a self-contained Docker showcase: keep Postgres; add **Apache AGE** (Cypher-as-extension) or use in-memory `networkx` — avoid a second DB service. | age.apache.org; github.com/apache/age | No new infra needed if we go the networkx route. |

---

## 3. Gap Register (prioritized)

Legend: **C** = credibility-critical (a reviewer will question the repo without it), **H** = high impact, **M** = medium.
`CFG` = behind config flag; `MOCK` = has a deterministic fallback so CI/tests run offline (no API keys).

| # | Gap | Severity | Fix summary |
|---|---|---|---|
| G1 | No real LLM integration (client, structured output, streaming, tool calling) | **C** | Pluggable `LLMClient` (OpenAI-compatible; mock fallback) + Pydantic-validated outputs + real SSE streaming |
| G2 | Embeddings are fake hash projections | **C** | `EmbeddingProvider` interface: OpenAI / local sentence-transformers / deterministic mock |
| G3 | "Agent" is a scripted regex pipeline | **C** | LLM-driven ReAct loop over tools; Adaptive-RAG router; CRAG evaluator; Self-RAG-lite reflection (mock-compatible) |
| G4 | Hybrid RRF search is in-memory-only; PG path is pure cosine | **H** | Real PG hybrid: cosine + tsvector/BM25 fused with RRF in SQL; HNSW tuning; in-memory fallback preserved |
| G5 | No query transformation | **H** | Query rewrite + multi-query expansion (config-gated; HyDE/step-back optional) |
| G6 | No reranking | **H** | Reranker interface: cross-encoder / Cohere API / linear-scores fallback (`CFG`; the -67% failure reduction) |
| G7 | No evaluation framework (RAGAS-style) | **H** | `benchmarks/rag_eval.py`: faithfulness/relevancy/context metrics (deterministic fallback + optional LLM-judge) + nDCG/MRR/recall@k on a golden set; CI gate |
| G8 | No observability/tracing | **H** | Lightweight OTel-gen_ai-style tracing module + optional Langfuse export + per-request token/cost telemetry |
| G9 | No auth, rate limiting, or audit log | **H** | Optional JWT/API-key auth, slowapi rate limits, structured audit log with correlation IDs (`CFG`) |
| G10 | No semantic cache / prompt-cache structure | M | Query-embedding semantic cache (Redis or in-proc fallback) + static-prefix prompt ordering |
| G11 | Chunker is basic (no structure awareness, no parent-child) | M | Structure-aware + token-budget chunker; parent_id links for small-to-big |
| G12 | PII only at output; none at ingestion | M | Presidio-style ingestion redaction (optional dep) + existing output scrubbing |
| G13 | No conversation memory | M | Session store + context-window management (mem0-style optional, mock-safe) |
| G14 | No GraphRAG layer | M→**H** | Graph-enhanced local search via LightRAG/networkx + entity extraction; **roadmap** (after core pipeline) |
| G15 | MCP tool exposure | M | Expose tools via an MCP server (roadmap; the 2026 standard transport) |
| G16 | Deep-research mode | M | Plan → iterate → cited-synthesis endpoint (roadmap) |
| G17 | Repo hygiene: no LICENSE, CONTRIBUTING, SECURITY, sample data | M | Add Apache-2.0 LICENSE, CONTRIBUTING.md, SECURITY.md, `data/` sample corpus |
| G18 | README overclaims vs code | **C** | README v2: claims backed by real providers/benchmarks; document architecture + research links |

---

## 4. Implementation Plan (waves)

- **Wave A — Foundations (parallel, disjoint files):** G2 embeddings+config · LLM client (new `app/llm/`) · retrieval upgrades (PG hybrid + reranker + query transforms + chunker) · repo hygiene/docs.
- **Wave B — Agentic + Enterprise:** G3 agent loop (depends on LLM client/retrieval) · G9/G10/G12/G13 hardening (auth, cache, ingestion PII, memory).
- **Wave C — Evaluation & integration:** G7 eval harness + golden dataset · CI · full-suite verification · benchmark refresh · README v2.
- **Roadmap (documented, not built here):** G14 GraphRAG (LightRAG + networkx), G15 MCP server, G16 deep-research mode, late chunking, ColBERT.

**Guardrails (non-negotiables):**
1. `LLM_PROVIDER=mock` and `EMBEDDING_PROVIDER=mock` remain the CI/test defaults → **no API keys needed, ever**.
2. All 65 existing tests keep passing; new features add new tests.
3. Real providers are config-swapped, never hard-required.
4. Dependencies: optional heavy deps (sentence-transformers, redis, presidio, tenacity, slowapi) go to a separate extras file.

---

## 5. Sources

*Agentic RAG Survey* arXiv:2501.09136 · *Self-RAG* arXiv:2310.11511 · *CRAG* arXiv:2401.15884 · *Adaptive-RAG* arXiv:2403.14403 · *RAG-Fusion* arXiv:2402.03367 · *ReAct* arXiv:2210.03629 · *Reflexion* arXiv:2303.11366 · *MemGPT* arXiv:2310.08560 · *GraphRAG* arXiv:2404.16130 · *LightRAG* arXiv:2410.05779 · *GLiNER* arXiv:2311.08526 · *GraphRAG survey* arXiv:2408.08921 · *BGE-M3* arXiv:2402.03216 · *MRL* arXiv:2205.13147 · *HyDE* arXiv:2212.10496 · *Rewrite-Retrieve-Read* arXiv:2305.14283 · *Step-Back* arXiv:2310.06117 · *Late chunking* arXiv:2409.04701 · *ColBERT* arXiv:2004.12832 · *HNSW* arXiv:1603.09320 · *RRF* plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf · *Indirect prompt injection* arXiv:2302.12173 · *LLM-as-judge* arXiv:2306.05685 ·
Anthropic contextual retrieval: https://www.anthropic.com/news/contextual-retrieval · Anthropic prompt caching: https://www.anthropic.com/news/prompt-caching ·
Qwen3 embedding: https://qwenlm.github.io/blog/qwen3-embedding/ · OWASP LLM Top 10: https://genai.owasp.org/llm-top-10/ · OTel GenAI conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/ · Langfuse: https://langfuse.com · RAGAS: https://docs.ragas.io · MCP: https://modelcontextprotocol.io ·
OpenAI deep research: https://openai.com/index/introducing-deep-research/ · Presidio: https://github.com/microsoft/presidio ·
NeMo Guardrails: https://github.com/NVIDIA/NeMo-Guardrails · Guardrails AI: https://github.com/guardrails-ai/guardrails ·
GPTCache: https://github.com/zilliztech/GPTCache · slowapi: https://github.com/slowapi/slowapi · tenacity: https://github.com/jd/tenacity ·
Apache AGE: https://age.apache.org/ · pgvector: https://github.com/pgvector/pgvector · navbar nano-graphrag: https://github.com/gusye1234/nano-graphrag