# Cloud Cost Optimization

<!--
  source: cloud_cost_optimization_v1
  department: Platform Engineering
  version: 1.2
  author: Platform Cost Efficiency Team
  category: Cost & FinOps
-->

## 1. Purpose

This guide defines the enterprise approach to cloud cost management: how we size capacity, monitor
spend, and optimize compute and storage in a defensible, repeatable way. Cost optimization must never
compromise availability, security, or data governance requirements.

## 2. Cost Model Fundamentals

Understand the unit economics of each workload before optimizing. For every service, maintain:

- **Base sizing dimensions:** requests-per-second, tokens-per-request (for AI workloads), storage
  growth per month, and peak-to-average ratio.
- **Capacity planning formula:** provision to the measured P99, not the average, to absorb burst while
  staying below hard latency SLAs.
- A **cost per unit** (e.g. per 1M tokens, per GiB-month, per instance-hour) so optimizations can be
  compared on a common basis.

## 3. Sizing Instances & GPUs for AI Workloads

For embedding and inference workloads:

- Size GPU/CPU instances to the **memory footprint** of the model plus working set, not nominal card
  count. For an embedding index, budget ≈ `N × D × 4 bytes` (float32) plus HNSW adjacency overhead.
- Prefer **fewer, correctly-sized instances** over over-provisioned padding; idle GPU capacity is pure
  cost.
- Use autoscaling with warm standby tiers: an always-on minimum plus burst capacity activated on queue
  depth — never pay for peak headroom that is idle 95% of the time.

## 4. Storage Tiering

- **Hot tier:** frequently accessed vectors and documents on high-throughput storage; this is where
  the PGVector working set lives.
- **Warm tier:** recent-but-cold data on standard storage with transparent retrieval.
- **Cold/archive tier:** compliant, immutable archives at lowest cost; only ever read for audits or
  recovery.
- Classify every object to a tier; do not let hot-tier pricing apply to cold data by default.

## 5. Batch vs. Real-Time Cost Curves

- Real-time embedding and inference is more expensive per unit than **batched** processing.
- Where latency permits, queue ingestion and bulk embedding in nightly/off-peak batches to exploit
  lower per-unit and power pricing.
- Distinguish SLA-bound real-time paths (which must stay at peak provisioning) from idempotent batch
  jobs (which should run at off-peak, at reduced parallelism, optionally on spot/ecologically-priced
  capacity with checkpointing).

## 6. Monitoring & Guardrails

- Track **cost per successful request** and **cost per 1M tokens** as first-class metrics.
- Set anomaly alerts on spend deltas (`budget_consumed`, `forecasted_spend`) so cost drift is caught in
  days, not months.
- Run a **FinOps review** each billing cycle that ties spend to delivered capacity and SLA; reclaim
  drift and report unit-cost trends.

## 7. Cost-Saving Checklist (by leverage)

1. **Right-size instances** first — usually the largest unclaimed win.
2. **Move cold data to archive tier** and delete nothing governed by retention before its date.
3. **Batch what can tolerate latency**; keep real-time hot paths lean.
4. **Eliminate idle capacity** via evidence-based autoscaling.
5. **Renegotiate/baseline** only after measurement — never optimize unit price before unit demand.

## 8. Compliance & Exceptions

Automated cost controls (budget caps, autoscale thresholds) are the default. Deviating from them
requires platform approval and an exception record. All cost decisions remain subordinate to the
security, availability, and data-retention obligations in the platform and compliance policies.