# Incident Response Runbook

<!--
  source: incident_response_runbook_v1
  department: Platform Engineering
  version: 1.4
  author: Platform On-Call Guild
  category: Operations & Runbook
-->

## 1. When to Use This Runbook

This runbook drives response for service outages, degraded availability, security incidents, and
data-quality incidents affecting the retrieval platform. It is a step-by-step operational reference:
when in doubt, **follow the steps in order** and record every action in the incident channel.

## 2. Severity Definitions

| Severity | Impact                                                   | Response window |
|----------|----------------------------------------------------------|-----------------|
| SEV-1    | Total service outage; core query path unusable           | Immediate, 24/7  |
| SEV-2    | Major degradation; partial users affected                | Within 15 min    |
| SEV-3    | Minor degradation; single feature / non-critical path    | Within 1 hour    |
| SEV-4    | Cosmetic / monitoring-only; no user impact                | Next business day|

A security incident (suspected compromise or data exposure) is always treated at **SEV-2 or above**
regardless of apparent scope, up to and including SEV-1 for confirmed Restricted-data exposure.

## 3. Triage (First 15 Minutes)

1. **Acknowledge** the alert and open a dedicated incident channel or thread. Assign a single incident
   commander.
2. **Classify severity** using the table above; adjust as new information arrives.
3. **Identify blast radius:** which services, tenants, and data classes are affected. For security
   incidents, declare whether Restricted or Highly Restricted data is involved.
4. **Decide go/no-go early:** do not fix in production unless the fix is a documented rollback or a
   rehearsed hotfix. Default to **rollback** over novel repair.

## 4. Containment

- **For availability incidents:** fail over to the secondary availability zone store if the primary is
  unhealthy; engage circuit breakers to protect downstream dependencies from cascading failure.
- **For the vector store:** if PostgreSQL is unavailable, the retrieval layer transparently falls back
  to the in-memory store. Confirm the active backend (`GET /api/v1/metrics`, field `active_backend`)
  and state the degraded mode in the incident log.
- **For security incidents:** immediately rotate any exposed credentials, revoke suspect sessions, and
  restrict access before investigating. Preserve evidence for root cause analysis.

## 5. Diagnosis

- Correlate the distributed trace (request ID) with logs and metrics to isolate the failing component.
- Check, in order: dependency health → index status → query latency percentiles (P50/P95/P99) → guardrail
  block-rate → recent deployment revision.
- For retrieval-quality regressions, re-run the benchmark (`make benchmark` / `make eval`) against the
  deployed revision to distinguish an infrastructure fault from a retrieval behavior change.

## 6. Recovery

- Apply the rehearsed rollback or hotfix; verify the change at the API layer with a synthetic
  healthcheck query before declaring recovery.
- Confirm the correct backend and index are back in service.
- Hold the incident open until monitoring shows latency and error rate back to baseline for at least
  one full alert window.

## 7. Post-Incident Review

Within 5 business days, file a post-incident review containing:

1. Timeline (all events, with owners and timestamps).
2. Root cause and contributing factors.
3. The corrective actions with owners and due dates.
4. Any runbook or monitoring gaps discovered — and the follow-up tickets to close them.

No SEV-1/SEV-2 closes without an approved post-incident review on record.

## 8. Runbook Maintenance

Treat this runbook as living documentation. Every post-incident review that reveals a missing or
incorrect step MUST result in a runbook update in the same review. An outdated runbook is worse than no
runbook.