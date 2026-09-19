"""
Enterprise Agentic RAG - RAG Evaluation Harness (RAGAS-style).

Evaluates the retrieval + generation stack of the agentic RAG pipeline over a
hand-labeled golden QA set (``data/eval/golden_qa.jsonl``) grounded in the
seeded corpus of ``app/rag/vector_store.py``:

Retrieval metrics (RAGAS-style, label-based):
    - hit@k     : did any relevant chunk appear in the top-k results?
    - MRR       : reciprocal rank of the first relevant chunk
    - recall@k  : fraction of relevant chunks that were retrieved in top-k

Generation metrics (deterministic, offline):
    - faithfulness proxy: mean PostExecutionGuardrail factual-grounding score
      over the deterministic orchestrator synthesis (mock mode is fully
      deterministic and requires NO API keys, so it is safe to gate CI).
    - citation coverage / precision from the CitationVerifierTool output.

Optional ``--provider openai`` mode additionally scores reference-answer
relevancy and answer faithfulness with an LLM judge (0-5 scaled), using the
app's own ``get_llm_client()``. It is skipped with a warning if the configured
client is a mock/offline provider or no ``OPENAI_API_KEY`` is set.

The CLI enforces a CI gate: exit code 0 iff aggregate ``hit@k >= 0.60`` and
aggregate grounding score ``>= 0.50``; otherwise a nonzero exit code with a
clear failure summary of the real measured values. The harness is
deterministic on repeated runs: the seeded mock embeddings derive features
from Python's ``hash()`` which is randomized per interpreter unless
``PYTHONHASHSEED`` is pinned, so the module re-executes itself with
``PYTHONHASHSEED=0`` before importing any app code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Determinism guard: the app's mock embedding provider derives features from
# Python's randomized ``hash()``, whose per-process seed changes embeddings
# across interpreter runs. Pin the hash seed by re-executing ourselves with
# PYTHONHASHSEED=0 so mock mode yields byte-identical numbers on every run.
# Guard runs before any app import so seeded vectors are stable.
# ---------------------------------------------------------------------------
if os.environ.get("PYTHONHASHSEED") != "0":
    _env = os.environ.copy()
    _env["PYTHONHASHSEED"] = "0"
    raise SystemExit(subprocess.call([sys.executable, __file__, *sys.argv[1:]], env=_env))

# Ensure project root is importable when invoked as a bare script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.orchestrator import AgentOrchestrator
from app.agent.tools.vector_search import VectorSearchTool
from app.guardrails.post_execution import PostExecutionGuardrail
from app.rag.reranker import get_reranker
from app.rag.vector_store import PGVectorStore

# ---------------------------------------------------------------------------
# CI gate thresholds (measured honestly against the deterministic pipeline)
# ---------------------------------------------------------------------------
HIT_AT_K_THRESHOLD = 0.60
GROUNDING_THRESHOLD = 0.50
TOP_K = 4

REQUIRED_ROW_KEYS = {"id", "question", "reference_answer", "relevant_chunk_ids", "question_type"}
VALID_TYPES = {"factoid", "multi-hop", "calculation", "technical"}

# ---------------------------------------------------------------------------
# Golden set loading & validation
# ---------------------------------------------------------------------------


def load_golden(path: str) -> List[Dict[str, Any]]:
    """Load the golden QA set from a JSONL file, validating each row's shape.

    Raises ``ValueError`` when a row is malformed so a bad dataset fails loudly
    in CI rather than silently producing skewed metrics.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: row is not a JSON object")
            missing = REQUIRED_ROW_KEYS - set(obj.keys())
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {sorted(missing)}")
            if not isinstance(obj["relevant_chunk_ids"], list) or not obj["relevant_chunk_ids"]:
                raise ValueError(f"{path}:{line_no}: 'relevant_chunk_ids' must be a non-empty list")
            if obj.get("question_type") not in VALID_TYPES:
                raise ValueError(
                    f"{path}:{line_no}: 'question_type' must be one of {sorted(VALID_TYPES)}"
                )
            rows.append(obj)

    if not rows:
        raise ValueError(f"{path}: empty golden dataset")
    return rows


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def retrieval_metrics(question: Dict[str, Any], retrieved: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-question retrieval metrics against the golden labels.

    ``retrieved`` is the ordered top-k list of result dicts (each with an
    ``id`` field) as returned by the real pipeline (hybrid search + rerank).
    """
    relevant = set(question["relevant_chunk_ids"])
    ranked_ids = [str(r.get("id", "")) for r in retrieved]

    hit = int(any(cid in ranked_ids for cid in relevant))
    mrr = 0.0
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            mrr = 1.0 / rank
            break
    if relevant:
        recall = sum(1 for cid in relevant if cid in ranked_ids) / len(relevant)
    else:
        recall = 0.0

    return {
        "hit_at_k": hit,
        "mrr": round(mrr, 4),
        "recall_at_k": round(recall, 4),
        "retrieved_ids": ranked_ids,
    }


async def run_retrieval(
    vector_tool: VectorSearchTool,
    reranker: Any,
    question: str,
    top_k: int = TOP_K,
) -> Tuple[List[Dict[str, Any]], str]:
    """Run the REAL retrieval path: hybrid (RRF) search, then rerank.

    Mirrors the orchestrator's retrieval strategy: technical/enterprise queries
    go through hybrid (RRF) search and the configured reranker is applied when
    available. ``top_k`` defaults to 4 per the golden-set contract.
    """
    search_res = await vector_tool.execute(
        query=question,
        limit=top_k,
        use_hybrid=True,  # technical & identifier-heavy questions -> hybrid RRF
    )
    results = search_res.get("results", [])
    search_type = search_res.get("search_type", "hybrid_rrf")

    if len(results) > 1 and reranker is not None:
        try:
            reranked = await asyncio.wait_for(reranker.rerank(question, results), timeout=15.0)
            if reranked:
                results = reranked
        except Exception:  # noqa: BLE001 - rerank must never fail an eval run
            pass

    # The reranker preserves the full list; keep only the top-k in final order.
    return results[:top_k], search_type


# ---------------------------------------------------------------------------
# Generation metrics (deterministic offline path)
# ---------------------------------------------------------------------------


async def generation_metrics(
    orchestrator: AgentOrchestrator, question: str
) -> Dict[str, Any]:
    """Run the deterministic orchestrator synthesis and derive faithfulness.

    Uses the app's real mock-mode synthesis path (``AgentOrchestrator.execute``
    is byte-for-byte deterministic when no real LLM client is configured).
    Faithfulness proxy = mean PostExecutionGuardrail factual-grounding score of
    the synthesized answer against the chunks it actually retrieved.
    """
    resp = await orchestrator.execute(query=question)
    answer = resp.answer or ""
    sources = resp.sources or []

    _, grounding = PostExecutionGuardrail.verify_factual_grounding(answer, sources)

    g = resp.guardrail_metrics or {}
    return {
        "grounding_score": round(float(grounding), 4),
        "citation_coverage": float(g.get("citation_coverage", 1.0)),
        "citation_precision": float(g.get("citation_precision", 1.0)),
        "has_sources": bool(sources),
    }


# ---------------------------------------------------------------------------
# Optional LLM-judge metrics (openai provider mode)
# ---------------------------------------------------------------------------


def _is_offline_provider(client: Any) -> bool:
    """True when the client is missing or backed by a mock/offline provider."""
    if client is None:
        return True
    provider = str(getattr(client, "provider", "") or "").strip().lower()
    return provider in {"mock", "offline", "none", ""}


async def llm_judge_score(llm_client: Any, judge_prompt: str) -> Optional[float]:
    """Return a normalized 0.0-1.0 score from an LLM judge for ``judge_prompt``."""
    if _is_offline_provider(llm_client):
        return None
    try:
        from pydantic import BaseModel, Field

        class JudgeScore(BaseModel):
            score: float = Field(..., ge=0.0, le=5.0, description="Relevance/faithfulness score 0-5.")
            rationale: str = Field("", description="One-sentence justification.")

        result, _meta = await asyncio.wait_for(
            llm_client.structured(
                messages=[
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": "Score the response with a single 0-5 value and a one-sentence rationale."},
                ],
                response_model=JudgeScore,
            ),
            timeout=30.0,
        )
        return max(0.0, min(1.0, float(result.score) / 5.0))
    except Exception:  # noqa: BLE001 - judge failures are non-fatal
        return None


_RELEVANCY_PROMPT = (
    "You are an answer-relevancy judge. Score how well the given answer actually "
    "addresses the user's question on a scale of 0 (completely off-topic) to 5 "
    "(fully relevant and complete). Reply only with the requested JSON."
)

_FAITHFULNESS_PROMPT = (
    "You are a faithfulness judge. Given a question, a generated answer, and the "
    "retrieved context it was grounded in, score on a scale of 0 (answer is fully "
    "hallucinated / unsupported) to 5 (every claim is supported by the context) "
    "how faithfully the answer stays grounded in the context. Reply only with the "
    "requested JSON."
)


async def llm_judge_metrics(
    llm_client: Any, rows: Sequence[Dict[str, Any]], orchestrator: AgentOrchestrator
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Score answer_relevancy and faithfulness with the LLM judge (openai mode).

    Returns ``({aggregate}, {per-question})``. Skips gracefully with a warning
    when the client is offline/mock. ``orchestrator`` only ensures the same
    retrieval grounding used for the deterministic metrics is available.
    """
    if _is_offline_provider(llm_client):
        return {"llm_answer_relevancy": None, "llm_faithfulness": None}, {}

    relevancies: List[float] = []
    faithfulnesses: List[float] = []
    per_q: Dict[str, Any] = {}

    for row in rows:
        qid = row["id"]
        resp = await orchestrator.execute(query=row["question"])
        context = " ".join(s.get("content", "") for s in (resp.sources or []))
        relevancy_prompt = (
            f"{_RELEVANCY_PROMPT}\n\nQuestion: {row['question']}\nAnswer: {resp.answer}"
        )
        faithfulness_prompt = (
            f"{_FAITHFULNESS_PROMPT}\n\nQuestion: {row['question']}\n"
            f"Answer: {resp.answer}\nContext: {context}"
        )
        rel = await llm_judge_score(llm_client, relevancy_prompt)
        faith = await llm_judge_score(llm_client, faithfulness_prompt)
        per_q[qid] = {"llm_answer_relevancy": rel, "llm_faithfulness": faith}
        if rel is not None:
            relevancies.append(rel)
        if faith is not None:
            faithfulnesses.append(faith)

    def _mean(vals: List[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "llm_answer_relevancy": _mean(relevancies),
        "llm_faithfulness": _mean(faithfulnesses),
    }, per_q


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------


def _mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def aggregate(per_question: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question retrieval/generation dicts into dataset means."""
    hit = [q["hit_at_k"] for q in per_question]
    mrr = [q["mrr"] for q in per_question]
    recall = [q["recall_at_k"] for q in per_question]
    grounding = [q["grounding_score"] for q in per_question]
    coverage = [q["citation_coverage"] for q in per_question]
    precision = [q["citation_precision"] for q in per_question]
    return {
        "n": len(per_question),
        "hit_at_k": _mean(hit),
        "mrr": _mean(mrr),
        "recall_at_k": _mean(recall),
        "grounding": _mean(grounding),
        "citation_coverage": _mean(coverage),
        "citation_precision": _mean(precision),
    }


def print_markdown_table(header: List[str], rows_cells: List[List[str]]) -> None:
    """Print a clean markdown table to stdout."""
    widths = [max(len(str(h)), *(len(str(c)) for c in col)) for h, col in
              ((h, [r[i] for r in rows_cells]) for i, h in enumerate(header))]
    print("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows_cells:
        print("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")


# ---------------------------------------------------------------------------
# Main evaluation run
# ---------------------------------------------------------------------------


async def run_eval(rows: List[Dict[str, Any]], provider: str, verbose: bool) -> Dict[str, Any]:
    """Run retrieval + generation metrics for every golden row."""
    vector_store = PGVectorStore(mode="in_memory", seed_defaults=True)
    vector_tool = VectorSearchTool(vector_store=vector_store)
    orchestrator = AgentOrchestrator(vector_store=vector_store, max_steps=3)
    try:
        reranker = get_reranker()
    except Exception:  # noqa: BLE001 - reranker is optional
        reranker = None

    per_question: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        retrieved, search_type = await run_retrieval(vector_tool, reranker, row["question"])
        r_metrics = retrieval_metrics(row, retrieved)
        g_metrics = await generation_metrics(orchestrator, row["question"])
        per_question.append({"id": row["id"], **r_metrics, **g_metrics})
        if verbose:
            found = [c for c in row["relevant_chunk_ids"] if c in r_metrics["retrieved_ids"]]
            print(
                f"  [{idx}/{len(rows)}] {row['id']:<6} {row['question_type']:<11} "
                f"hit@k={r_metrics['hit_at_k']} mrr={r_metrics['mrr']:.3f} "
                f"recall@k={r_metrics['recall_at_k']:.3f} "
                f"grounding={g_metrics['grounding_score']:.3f} "
                f"({search_type}, found={found})"
            )

    summary = aggregate(per_question)

    llm: Dict[str, Any] = {}
    if provider == "openai":
        from app.llm.client import get_llm_client

        client = get_llm_client()
        if _is_offline_provider(client) or not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  LLM-judge metrics skipped: provider=openai requested but the "
                  "client is mock/offline or OPENAI_API_KEY is unset.")
        else:
            judge_agg, per_q_llm = await llm_judge_metrics(client, rows, orchestrator)
            llm = judge_agg
            for qid, scores in per_q_llm.items():
                for pq in per_question:
                    if pq.get("id") == qid:
                        pq.update(scores)
    summary["llm"] = llm
    summary["per_question"] = per_question
    return summary


def render_report(summary: Dict[str, Any]) -> str:
    """Render the aggregate metrics as a markdown report."""
    lines = []
    lines.append("## RAG Evaluation — Aggregate Metrics")
    lines.append("")
    rows_cells = [
        ["Rows", str(summary["n"])],
        ["hit@k (threshold 0.60)", f"{summary['hit_at_k']:.4f}"],
        ["MRR", f"{summary['mrr']:.4f}"],
        ["recall@k", f"{summary['recall_at_k']:.4f}"],
        ["grounding / faithfulness proxy (threshold 0.50)", f"{summary['grounding']:.4f}"],
        ["citation coverage", f"{summary['citation_coverage']:.4f}"],
        ["citation precision", f"{summary['citation_precision']:.4f}"],
    ]
    llm = summary.get("llm") or {}
    if llm.get("llm_answer_relevancy") is not None:
        rows_cells.append(["LLM-judge answer_relevancy (0-1)", f"{llm['llm_answer_relevancy']:.4f}"])
    if llm.get("llm_faithfulness") is not None:
        rows_cells.append(["LLM-judge faithfulness (0-1)", f"{llm['llm_faithfulness']:.4f}"])
    print_markdown_table(["Metric", "Value"], rows_cells)
    lines.append("")
    return "\n".join(lines)


def gate_pass(summary: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Evaluate the CI gate. Returns (passed, list of failure messages)."""
    failures = []
    if summary["hit_at_k"] < HIT_AT_K_THRESHOLD:
        failures.append(
            f"hit@k {summary['hit_at_k']:.3f} < threshold {HIT_AT_K_THRESHOLD:.2f}"
        )
    if summary["grounding"] < GROUNDING_THRESHOLD:
        failures.append(
            f"grounding {summary['grounding']:.3f} < threshold {GROUNDING_THRESHOLD:.2f}"
        )
    return (not failures), failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG evaluation harness: retrieval + generation metrics on a golden QA set."
    )
    parser.add_argument(
        "--golden",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "eval", "golden_qa.jsonl"
        ),
        help="Path to the golden QA JSONL dataset.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openai"],
        default="mock",
        help="'mock' (deterministic, offline, no keys) or 'openai' (adds LLM-judge metrics).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-question metrics.")
    args = parser.parse_args(argv)

    golden_path = os.path.abspath(args.golden)
    print("=" * 72)
    print("🧪 ENTERPRISE AGENTIC RAG — RAG EVALUATION HARNESS (RAGAS-style)")
    print("=" * 72)
    print(f"ℹ️  Golden set: {golden_path}")
    print(f"ℹ️  Provider:   {args.provider}  (top_k={TOP_K}, hybrid RRF + rerank)")
    print("-" * 72)

    rows = load_golden(golden_path)
    print(f"📊 Loaded {len(rows)} golden rows.\n")

    if args.verbose:
        print("Retrieval & generation detail:")
    summary = asyncio.run(run_eval(rows, args.provider, args.verbose))

    render_report(summary)
    passed, failures = gate_pass(summary)

    # ----- optional LLM-judge per-question table ---------------------------
    per_q_llm_enabled = (args.provider == "openai") and summary.get("llm", {}).get("llm_faithfulness") is not None
    if per_q_llm_enabled:
        print("\n## LLM-judge per-question scores")
        print("")
        cells = [
            [q["id"], f"{q['hit_at_k']}", f"{q.get('llm_answer_relevancy', float('nan')):.3f}",
             f"{q.get('llm_faithfulness', float('nan')):.3f}"]
            for q in summary["per_question"]
        ]
        print_markdown_table(["id", "hit@k", "answer_relevancy", "faithfulness"], cells)

    print("-" * 72)
    if passed:
        print("✅ CI GATE PASSED (hit@k and grounding thresholds met).")
        print(f"   hit@k={summary['hit_at_k']:.4f} (>= {HIT_AT_K_THRESHOLD:.2f}), "
              f"grounding={summary['grounding']:.4f} (>= {GROUNDING_THRESHOLD:.2f})")
        return 0

    print("❌ CI GATE FAILED:")
    for msg in failures:
        print(f"   - {msg}")
    print("   The retrieval/generation metrics fell below thresholds; see the "
          "table above for real measured values.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())