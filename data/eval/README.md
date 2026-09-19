# Golden QA Evaluation Dataset (`golden_qa.jsonl`)

This file is the golden question/answer set used by the RAG evaluation harness
(`benchmarks/rag_eval.py`). It is a small, hand-curated, offline, deterministic
dataset: every row is grounded in the actual seeded corpus stored in
`app/rag/vector_store.py` (`DEFAULT_ENTERPRISE_DOCUMENTS`), so the harness can
compute retrieval, faithfulness, and generation metrics **with zero API keys**
in mock mode (the CI path).

## Row shape

The file is JSONL — one JSON object per line, no trailing commas. Each row has
exactly these keys:

| Key                  | Type        | Meaning                                                                 |
|----------------------|-------------|-------------------------------------------------------------------------|
| `id`                 | `str`       | Unique row id (g001, g002, ...).                                        |
| `question`           | `str`       | The user-facing question fed to retrieval and the orchestrator.         |
| `reference_answer`   | `str`       | An ideal, grounded answer (used by the optional LLM judge in `openai` mode). |
| `relevant_chunk_ids` | `list[str]` | Doc-chunk ids (from `DEFAULT_ENTERPRISE_DOCUMENTS`) that must be retrieved. |
| `question_type`      | `str`       | One of `factoid`, `technical`, `multi-hop`, `calculation`.              |

## Question-type mix

The set contains exactly 20 rows:

- **8 `factoid`** — single-chunk lookup questions (one relevant chunk each).
- **4 `technical`** — identifier-heavy questions (mTLS, RRF, ef_construction,
  cosine, eu-west-3, P99) that exercise the hybrid (RRF) retrieval + rerank path.
- **4 `multi-hop`** — questions that connect **two** chunks; `relevant_chunk_ids`
  lists both.
- **4 `calculation`** — sizing questions whose expected numbers must match the
  deterministic math in `app/agent/tools/calculator.py`:
  - `vector_ram`: `num_vectors * dimension * 4` (bytes)
  - `concurrency` (Little's law): `qps * (latency_ms / 1000)`
  - `monthly_storage_cost`: `storage_gb * price_per_gb`
  - `replicas_needed`: `ceil(total_qps / qps_per_instance)`

## Grounding rules for new rows

1. **Every** `relevant_chunk_id` must exist in `DEFAULT_ENTERPRISE_DOCUMENTS`
   (see `tests/test_rag_eval.py::test_relevant_chunk_ids_exist`).
2. `reference_answer` should paraphrase/quote the actual chunk text — do not
   invent facts that are not present in the corpus.
3. Write questions with strong keyword overlap to their target chunk(s): the
   offline harness uses hybrid (RRF) retrieval, and the deterministic mock
   embedding is token/semantic based, so matching vocabulary drives recall.
4. For `calculation` rows, keep the numbers literal and verify them against the
   `CalculatorTool` formulas above.

## How to add a row

1. Append one JSON object per line to `golden_qa.jsonl` (keep `id` unique).
2. Re-run the validators and the harness:

```bash
python3 -m pytest tests/test_rag_eval.py -q
python3 benchmarks/rag_eval.py --provider mock
```

The harness prints per-row and aggregate metrics and enforces the CI gate
(`hit@k >= 0.60` and mean grounding `>= 0.50`). If you add questions and the
measured aggregate drops below the gate, either improve the question's keyword
overlap to its chunks or confirm the pipeline genuinely regressed before
adjusting any threshold.

## Files

- `golden_qa.jsonl` — the 20 golden rows.
- `README.md` — this file.