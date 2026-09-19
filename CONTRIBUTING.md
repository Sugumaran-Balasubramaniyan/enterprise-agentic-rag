# Contributing to Enterprise Agentic RAG

Thanks for your interest in contributing. This project is maintained as a professional, production-oriented open-source repository, and we appreciate every contribution — from a typo fix to a full retrieval architecture improvement.

Please read this guide and our [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) before getting started.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Project Layout](#project-layout)
- [Running Tests & Linting](#running-tests--linting)
- [Branching & PR Process](#branching--pr-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Code Style & Review Guidelines](#code-style--review-guidelines)
- [Reporting Bugs](#reporting-bugs)

---

## Development Setup

The project targets **Python 3.11 / 3.12**. All dependencies are managed through `requirements.txt` with `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/Sugumaran-Balasubramaniyan/enterprise-agentic-rag.git
cd enterprise-agentic-rag

# 2. Create and activate a virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy the environment template and review it
cp .env.example .env
# Defaults are offline-safe (LLM_PROVIDER=mock), so you can start without any keys.
```

> **Note:** You do **not** need a running PostgreSQL instance to develop against the core retrieval
> logic — the vector store automatically falls back to an in-memory store when the database is
> unavailable. To exercise the PostgreSQL + PGVector path, `docker compose up -d` first.

---

## Project Layout

```
app/
  agent/            # Agent orchestrator, tool dispatcher
  api/              # FastAPI routes & request/response schemas
  guardrails/       # Pre/Post-execution deterministic guardrails
  rag/              # Parser, chunker, embeddings, vector store
  config.py         # Settings & environment configuration
benchmarks/         # Latency + grounding/safety reproducibility harnesses
data/documents/     # Sample enterprise documents for ingestion demos
docs/               # Research, gap analysis, architecture decision records
tests/              # pytest suite
streamlit_app.py    # Mission Control dashboard
```

---

## Running Tests & Linting

Always run the full suite and the linter before opening a PR.

```bash
# Run the entire test suite
pytest -q

# Verbose output with a short traceback (handy when debugging)
pytest -v --tb=short

# Run a single test file
pytest tests/test_vector_store.py -q

# Lint with ruff (configured in pyproject.toml)
make lint          # or: ruff check .

# Optionally run the benchmarking / evaluation harnesses
make benchmark     # vector retrieval latency benchmark
make eval          # grounding, citation & safety evaluation
```

A PR will not be merged if tests or linting fail on CI, so please confirm both are green locally.

---

## Branching & PR Process

We use a lightweight GitHub Flow style workflow.

1. **Create a branch** off `main` with a descriptive, hyphenated name:
   ```bash
   git checkout -b feat/pg-native-hybrid-search
   ```

2. **Make focused changes.** Keep each commit and each PR small and self-contained. If you are
   touching Python source, make sure new code is covered by a test in `tests/`.

3. **Run tests, lint, and benchmarks** as described above.

4. **Push and open a pull request.**
   ```bash
   git push -u origin feat/pg-native-hybrid-search
   ```
   In the PR description, summarize *what* changed, *why*, and any benchmark/eval numbers that moved.

5. **Address review feedback** by pushing follow-up commits to the same branch. Keep the conversation
   civil and constructive — see the [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

6. **Merge.** Maintainers will squash-merge once CI is green and at least one review has approved.

---

## Commit Message Conventions

We follow **[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)**. A commit message
has the form:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Common types:

| Type      | Purpose                                                        |
|-----------|----------------------------------------------------------------|
| `feat`    | A new capability (e.g. a new rag provider or tool)             |
| `fix`     | A bug fix                                                      |
| `docs`    | Documentation-only changes                                     |
| `refactor`| Code change that neither fixes a bug nor adds a feature        |
| `test`    | Adding or updating tests                                        |
| `chore`   | Build, tooling, and maintenance changes                        |
| `perf`    | A performance improvement                                      |

Examples:

```bash
feat(rag): add pluggable OpenAI embedding provider
fix(guardrails): strip trailing whitespace in PII regex before matching
docs: add RAGAS-style evaluation section to README
```

---

## Developer Certificate of Origin (DCO)

This project requires that every commit is **signed off** to certify that you have the right to submit
the work under the project's license, per the
[Developer Certificate of Origin](https://developercertificate.org/).

Sign off each commit by adding `Signed-off-by: Your Name <you@example.com>`:

```bash
git commit -s -m "feat(rag): add pluggable OpenAI embedding provider"
```

The `-s` flag appends the sign-off trailer automatically. A `Signed-off-by` trailer is required on every
commit; missing trailers may cause the PR to be flagged by CI.

---

## Code Style & Review Guidelines

- **Formatting & linting:** `ruff` is the canonical linter and formatter; run `make lint` before pushing.
- **Type hints:** use them on new public functions and classes.
- **Async correctness:** `app/` is async-first. Never call blocking I/O in a coroutine without
  offloading it; new retriever/storage code should be `async def` and use the existing session pattern.
- **Config over magic numbers:** new knobs (thresholds, index params, provider settings) belong in
  `app/config.py` behind an environment variable with a sane default in `.env.example`.
- **Determinism:** the guardrail stack is intentionally deterministic — do not replace deterministic
  checks with LLM judgment for security-critical paths.
- **Benchmarks:** when you change retrieval or generation behavior, re-run `make benchmark` / `make eval`
  and report the deltas in the PR.

---

## Reporting Bugs

For **security vulnerabilities**, please **do not** open a public issue. See
[SECURITY.md](SECURITY.md) for private disclosure instructions.

For anything else, open a GitHub issue with:

- A clear, reproducible title.
- Steps to reproduce (including OS, Python version, and any relevant `.env` settings).
- Expected vs. actual behavior.
- If possible, a minimal reproducer or the failing test.