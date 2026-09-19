# Security Policy

This project takes the security of its runtime, its guardrail stack, and your data seriously. If you
believe you have found a security vulnerability, please read this document and follow the disclosure
process below.

---

## Supported Versions

Security fixes are applied to the latest release and, where feasible, backported to the two most
recent minor releases.

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | ✅ Active support  |
| 0.1.x   | ⚠️ Security patches only |
| < 0.1.0 | ❌ Not supported   |

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.** Public disclosure of an
unfixed issue can expose users before a patch is available.

Instead, report privately:

- **Preferred:** Open a **private security advisory** on GitHub:
  [https://github.com/Sugumaran-Balasubramaniyan/enterprise-agentic-rag/security/advisories/new](https://github.com/Sugumaran-Balasubramaniyan/enterprise-agentic-rag/security/advisories/new)
- **Alternative:** Email the maintainer at `bsugumaran@hotmail.com` (PGP-encrypted if you have the key;
  otherwise clearly mark the subject `[SECURITY]`).

Please include in your report:

1. A description of the vulnerability and its **security impact** (what an attacker can do).
2. Steps to reproduce, including the exact environment (`LLM_PROVIDER`, database mode, version).
3. If known, a suggested fix or hardening step.

### What happens next

- We aim to acknowledge your report **within 72 hours**.
- We will investigate and confirm the issue, then coordinate a fix and a private pre-disclosure window.
- We will credit you in the advisory unless you prefer to stay anonymous.

---

## Security Posture

This repository is an **enterprise RAG platform**, so its threat model spans prompt/jailbreak attacks,
data exfiltration, prompt-injection through retrieved content, and accidental PII leakage. The system's
guidance and defaults reflect that model:

### Deterministic Guardrails (defense-in-depth)

- **Pre-execution:** adversarial / jailbreak detection, SQL & shell injection filters, obfuscation
  decoding (Base64 / hex / ROT13), and **role-based access control (RBAC)**. These checks are
  *deterministic* by design — security-critical decisions must never be delegated to an LLM.
- **Post-execution:** PII & secret sanitization (API keys, tokens, SSNs, card numbers, emails, phones,
  IPs) and **claim-level factual grounding** before any output is emitted to a client.

### PII Sanitization

All model output passes through a deterministic PII scrubber before it is returned or persisted.
Emitting raw PII to client endpoints is treated as a defect. Tests in `tests/test_adversarial_guardrails.py`
cover masking coverage — do not regress these.

### Authentication & Authorization Flags

- Access to sensitive organizational resources is gated by the `user_role` field
  (`standard_user`, `enterprise_analyst`, `compliance_officer`, `system_admin`) and screened by the
  pre-execution RBAC layer.
- When a real LLM provider is enabled (see below), API access is expected to sit behind your
  organisation's own authentication/authorization and network perimeter — the service itself does not
  ship a standalone user registry.

### No Secrets in Environment Files

- `.env.example` contains **placeholders only** — never real keys.
- **Never commit a real `.env` file or hard-code API keys.** `.env` is git-ignored; keep it out of
  version control and out of CI logs.
- `OPENAI_API_KEY` / `MISTRAL_API_KEY` and similar provider secrets must be injected at deploy time via
  your secret manager or CI secrets, never baked into the image or the repository.
- If you ever suspect a secret was committed, rotate it immediately and file a private report.

### LLM / Embedding Provider Keys

```dotenv
# .env.example (sanitized — these are placeholders, not real values)
LLM_PROVIDER=mock
OPENAI_API_KEY=
MISTRAL_API_KEY=
```

`LLM_PROVIDER=mock` is the offline-safe default (no network, no keys). When enabling a real provider,
leave the unused keys blank and store real values out-of-band.

---

## Reporting Non-Security Bugs

For general bugs, performance concerns, or feature requests, open a regular GitHub issue or pull
request — see [CONTRIBUTING.md](CONTRIBUTING.md).