# Data Governance Framework
<!--
  source: data_governance_framework_v1
  department: Security & Compliance
  version: 2.0
  author: Data Governance Office
  category: Governance & Compliance
-->
## 1. Purpose

This framework defines how data is owned, classified, protected, retained, and disposed across the
enterprise. It applies to all data ingested into, stored in, or produced by the platform — including
the documents and embeddings in the retrieval system.

## 2. Principles

1. **Ownership over custody:** every dataset has a named data owner accountable for its accuracy,
   classification, and lifecycle.
2. **Least privilege:** access is the exception, not the default; granted per role and re-verified.
3. **Classify at creation:** every asset receives a data classification (Public, Internal, Restricted,
   Highly Restricted) at the moment of creation.
4. **Traceability:** every derived asset (chunk, embedding, synthesized answer) must be linkable back to
   its source and provenance.
5. **Retention by policy, not habit:** nothing is kept because deletion is inconvenient.

## 3. Data Ownership & Accountability

- Each dataset has a registered **data owner**, a **steward** (day-to-day quality/lifecycle), and a
  **technical custodian** (system of record). Owners approve classification changes and access grants;
  custodians implement controls.
- Ownership is recorded in the data catalog; a dataset without a registered owner is frozen from
  production ingestion until assigned.

## 4. Lineage & Provenance

- Document chunks carry metadata recording their **source**, **department**, **version**, and
  **author**, so any answer traces to its originating document.
- Derived artifacts (embeddings, retrieved results, generated outputs) inherit input lineage; synthesized
  answers must cite the source chunks they are grounded in.
- Any transformation that loses lineage (re-embedding, copying to a new store) is itself a change
  requiring governance review.

## 5. Retention & Disposal

- Retention periods are set per classification and regulatory domain at dataset creation and are
   enforced automatically where possible.
- **Public / Internal:** retain for the business need window, then dispose.
- **Restricted:** retain per minimum regulatory retention (e.g. audit-required periods), then dispose.
- **Highly Restricted:** retain only for the shortest lawful period; disposal requires a documented,
   attested deletion.
- Physical disposal must also cover **copies**: backups, archives, and vectors derived from deleted
  source content — deleting a source without purging its derived embeddings is incomplete disposal.

## 6. Access & Approvals

- Access grants are role- and purpose-bound, time-boxed, and reviewed quarterly.
- The RBAC model partitions data by department; cross-department access requires an exception approved
   by the owning data owner and InfoSec.
- Every access decision affecting Restricted data is auditable end-to-end.

## 7. Quality & Integrity

- Data owners define quality gates (completeness, freshness, schema conformance) for their datasets.
- Datasets that fail quality gates are quarantined from retrieval rather than served with degraded
   confidence.
- Quality is reported per dataset in the data catalog; retrieval confidence should reflect the source
   dataset's quality state.

## 8. Governance of AI & Derived Content

- AI-generated outputs that reference governed data inherit that data's access and retention obligations;
  model outputs are **not** "new data" free of governance — they are derived assets with provenance.
- Training or fine-tuning on governed data requires classification review and InfoSec approval per the
  security & compliance policy.

## 9. Audit & Compliance

- All access to Restricted and Highly Restricted data is immutable-audited.
- Compliance reviews validate ownership, classification, retention, and disposal against this framework;
   findings are tracked to closure with owners and due dates.