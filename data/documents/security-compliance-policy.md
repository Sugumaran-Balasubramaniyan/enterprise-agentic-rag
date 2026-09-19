# Security & Compliance Policy

<!--
  source: security_compliance_policy_v1
  department: Security & Compliance
  version: 3.1
  author: InfoSec Compliance Team
  category: Security & Privacy
-->

## 1. Purpose & Authority

This policy establishes the mandatory security and compliance controls for all systems, data, and AI
services within the enterprise. It is authoritative during design, operation, and incident response.
Violations of this policy are managed under the corporate incident and disciplinary processes.

## 2. Guardrail & Safety Requirements for AI Services

Any service that consumes or emits LLM output must comply with the deterministic guardrail contract:

- **Pre-execution guardrails** inspect prompts for prompt injection, adversarial jailbreak attempts,
  and SQL/command injection patterns *before* any tool executes.
- **Post-execution filters** sanitize PII — including email addresses, phone numbers, credit card
  numbers, SSNs, API keys, and tokens — and verify claim-level grounding before output is emitted.
- Guardrails must be **deterministic** for security-critical decisions. LLM judgment may not be
  substituted for these checks.

## 3. Data Classification & Handling

Data is classified as **Public**, **Internal**, **Restricted**, and **Highly Restricted**.

| Classification | Handling summary                                             |
|----------------|--------------------------------------------------------------|
| Public         | No special handling; may be served openly                    |
| Internal       | Requires authentication; no transmission without authorization |
| Restricted     | Must remain in home region; encrypted at rest and in transit |
| Highly Restricted | Additional need-to-know access control and full audit trail  |

Models and embeddings must never be trained or fine-tuned on Restricted or Highly Restricted data
without explicit InfoSec approval of the downstream storage and retention.

## 4. Access Control (RBAC)

Access boundaries must be strictly partitioned by department and role.

- Data access is granted on a least-privilege basis per **role**: `standard_user`,
  `enterprise_analyst`, `compliance_officer`, and `system_admin`.
- **Unauthenticated access or cross-department permission escalation must be rejected immediately.**
- Role assignments are reviewed quarterly; revocations take effect on the next request after the
  change is committed.
- The agent orchestrator and API layer enforce these boundaries on every query, screening resource
  access against the caller's role before executing retrieval.

## 5. Encryption & Key Management

- All data at rest classified above **Public** must use AES-256 (or equivalent).
- All data in transit must use TLS 1.2+; service-to-service traffic uses mTLS under the service mesh.
- Secrets (API keys, tokens, database credentials) live in the enterprise secret manager — **never** in
  source code, environment files committed to version control, logs, or container images.
- Keys are rotated at least every 90 days; compromised keys are revoked immediately and reported.

## 6. Logging, Monitoring & Auditing

- All access to Restricted and Highly Restricted data is fully audited with an immutable audit trail.
- Security-relevant events (guardrail blocks, access denials, privilege changes) are logged centrally.
- Logs must not contain raw secrets or PII; redaction occurs at the source.

## 7. Incident Response

Security incidents are handled per the corporate incident response runbook. Any confirmed compromise
of Restricted data triggers:

1. Immediate containment and credential rotation.
2. Disclosure to the Data Protection Officer within 72 hours where regulated.
3. Root-cause analysis and a remediating change record.

## 8. Compliance & Exceptions

Compliance is verified continuously; known gaps are logged as findings with owners and due dates.
Exceptions require InfoSec sign-off, are time-boxed, and are re-reviewed at each compliance cycle.