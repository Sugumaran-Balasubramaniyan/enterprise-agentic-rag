# PGVector HNSW Tuning Guide
<!--
  source: database_optimization_guide_v1
  department: Data Engineering
  version: 1.8
  author: Data Platform Team
  category: Database Performance
-->
## 1. Scope

This guide covers indexing, distance-metric selection, and runtime tuning for **PGVector** vector
indexes on PostgreSQL 16, for data engineers operating the embedding store that backs the enterprise
RAG retrieval layer.

## 2. Index Architecture Overview

PGVector offers HNSW and IVFFlat index types. For online datasets where inserts and queries
interleave, **HNSW is the recommended default** — no offline rebuild, and it degrades gracefully under
concurrent writes.

Key HNSW build parameters:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `m` | Max connections per node (fan-out) | `16` |
| `ef_construction` | Build-time search breadth (build cost ↑) | `64` |
| `ef_search` | Query-time search breadth (recall ↔ latency) | `32` |

Higher `m`/`ef_construction` improve recall at build cost; `ef_search` is the primary runtime dial.

## 3. Distance Metric Selection

Metrics must match how embeddings are normalized and exposed.

- For **normalized embeddings** (L2-normalized, as here), **cosine distance** equals inner product and
  is the correct choice — scale-invariant and stable across model versions.
- Use L2 (Euclidean) only for raw, non-normalized vectors where magnitude matters.
- Never mix metrics within an index; mismatches silently degrade retrieval quality.

With cosine distance, application-facing "similarity" is `1.0 - cosine_distance`.

## 4. Runtime Tuning Recommendations

### ef_search
Start at `32`; raise to `64`–`128` when recall-at-k regresses or for high-precision queries (latency
rises roughly linearly); drop to `16`–`24` for latency-critical, high-QPS paths.

### ANN Open-Search Trade-off
HNSW is approximate. For exact-equality or guaranteed recall, raise `ef_search` well beyond
neighborhood size or run an exact scan as verification — do not assume an exact top-k by default.

### Concurrent Writes
HNSW tolerates concurrent inserts, but transactional batch inserts avoid write stalls. Schedule bulk
loads off-peak; batch hundreds to thousands of rows per transaction depending on index size.

## 5. Hybrid Search with RRF
Dense-only retrieval underperforms on technical, keyword-dense queries. Combine cosine similarity with
PostgreSQL `tsvector` lexical search via **Reciprocal Rank Fusion** (RRF):

```
RRF(d) = alpha / (k + rank_vector(d)) + (1 - alpha) / (k + rank_lexical(d))
```

- Default `k = 60` smooths fusion; `alpha = 0.5` weights dense and lexical equally; use the same text
  for lexical and dense so terms associate. RRF fuses *ranks*, not raw scores, so it tolerates the
  incompatible similarity/`tsvector` scales.

## 6. Memory & Sizing
HNSW works in memory. Baseline RAM per vector set:

```
bytes ≈ N × D × 4 (float32)  +  HNSW adjacency overhead ~ m × 8 bytes per node
```

10M vectors × 1536 dims ≈ 61.4 GB before adjacency/index overhead. Budget RAM accordingly; enlarge
`shared_buffers`/`work_mem` for index build.

## 7. Monitoring & Verification
- Track median/P95 latency and recall-at-k per workload; alert on baseline deviation.
- Confirm via `EXPLAIN` that the HNSW index (not a sequential scan) is used.
- After any parameter change, re-run `make benchmark` and compare against the documented baseline.