"""Retrieval + (optional) faithfulness benchmark runner.

Run via::

    python -m attune_rag.benchmark
    python -m attune_rag.benchmark --queries path/to/queries.yaml
    python -m attune_rag.benchmark --min-precision 0.70
    python -m attune_rag.benchmark --with-faithfulness \\
        --min-faithfulness 0.85
    python -m attune_rag.benchmark --with-faithfulness \\
        --compare-thinking --json out.json

Prints precision@1, recall@3, and mean latency. Exits 1 if
precision@1 falls below ``--min-precision`` so CI can gate.

With ``--with-faithfulness`` (opt-in — spends API tokens),
also runs the default prompt variant through
``run_and_generate`` + :class:`FaithfulnessJudge` and
gates on the resulting mean faithfulness score. Requires
``ANTHROPIC_API_KEY`` and the ``[claude]`` extra.

With ``--compare-thinking``, runs the judge twice (thinking
off, thinking on) and prints a side-by-side comparison so the
extended-thinking knob can be calibrated against real data.
Use with ``--json PATH`` to also dump the full per-query
verdict records (reasoning + claim text) for offline analysis.

Queries file format matches tests/golden/queries.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any


def _default_queries_path() -> Path:
    """Resolve the bundled golden-query set's path.

    Walks up from this file (``src/attune_rag/benchmark.py``)
    to the repo root and returns ``tests/golden/queries.yaml``.
    Used as the ``--queries`` default; callers can override.
    """
    return Path(__file__).resolve().parent.parent.parent / "tests" / "golden" / "queries.yaml"


def _default_negatives_path() -> Path:
    """Resolve the bundled out-of-corpus (negative) query set's path."""
    return (
        Path(__file__).resolve().parent.parent.parent / "tests" / "golden" / "queries_negative.yaml"
    )


def _default_extended_path() -> Path:
    """Resolve the bundled extended (advisory hard) query set's path."""
    return (
        Path(__file__).resolve().parent.parent.parent / "tests" / "golden" / "queries_extended.yaml"
    )


def _default_corpus_b_path() -> Path:
    """Resolve the bundled unseen second-corpus directory (generalization)."""
    return Path(__file__).resolve().parent.parent.parent / "tests" / "golden" / "corpus_b"


def _default_corpus_b_queries_path() -> Path:
    """Resolve the golden queries for the unseen second corpus."""
    return (
        Path(__file__).resolve().parent.parent.parent / "tests" / "golden" / "queries_corpus_b.yaml"
    )


def _env_bool(name: str) -> bool:
    """Parse an env var as a boolean ("1"/"true"/"yes" → True)."""
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    """Parse an env var as int or return None when unset/malformed."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _load_queries(path: Path) -> list[dict[str, Any]]:
    """Read the ``queries:`` list from a YAML file at ``path``.

    Expected schema: ``{queries: [{id: str, query: str,
    expected_in_top_3: list[str], difficulty?: str}, ...]}``.
    Raises ``ValueError`` if the file parses but the queries
    list is empty or missing.
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def _run_benchmark(
    queries: list[dict[str, Any]],
    k: int,
    corpus: Any = None,
) -> dict[str, Any]:
    """Run retrieval-only benchmark across ``queries`` and aggregate metrics.

    ``corpus`` (optional): a CorpusProtocol to retrieve against. When
    ``None`` the pipeline uses its default (bundled attune-help). Pass a
    ``DirectoryCorpus`` here to measure generalization to an unseen
    corpus (Phase 2).

    Returns a report dict with this shape (consumed by
    :func:`_print_summary`, the dashboard, and ``main``):

    - ``retriever`` (str) — class name of the active retriever
    - ``corpus`` (str) — corpus.name from the active pipeline
    - ``total_queries`` (int)
    - ``precision_at_1`` (float) — fraction where the top-1 hit
      is in ``expected_in_top_3``
    - ``recall_at_k`` (float) — fraction where any hit in the
      top-k intersects ``expected_in_top_3``
    - ``k`` (int) — the top-k cap used
    - ``mean_latency_ms`` (float)
    - ``max_latency_ms`` (float)
    - ``per_query`` (list[dict]) — one record per query with
      ``id``, ``difficulty``, ``query``, ``expected``, ``actual``,
      ``top1_match``, ``topk_match``

    Pure: no LLM calls, no disk writes. Spends only the cost of
    retrieval (CPU-bound for ``KeywordRetriever``).
    """
    from . import RagPipeline

    pipeline = RagPipeline(corpus=corpus) if corpus is not None else RagPipeline()
    retriever_name = type(pipeline.retriever).__name__
    corpus_name = pipeline.corpus.name

    precision_hits = 0  # top-1 matches an expected path
    recall_hits = 0  # any expected path is in top-k hits
    latencies_ms: list[float] = []
    per_query_results: list[dict[str, Any]] = []

    for entry in queries:
        expected = set(entry.get("expected_in_top_3", []))
        start = time.perf_counter()
        result = pipeline.run(entry["query"], k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        hit_paths = [h.template_path for h in result.citation.hits]
        top1_hit = bool(hit_paths and hit_paths[0] in expected)
        topk_hit = bool(set(hit_paths) & expected)

        if top1_hit:
            precision_hits += 1
        if topk_hit:
            recall_hits += 1

        per_query_results.append(
            {
                "id": entry["id"],
                "difficulty": entry.get("difficulty", ""),
                "query": entry["query"],
                "expected": sorted(expected),
                "actual": hit_paths,
                "top1_match": top1_hit,
                "topk_match": topk_hit,
            }
        )

    total = len(queries)
    return {
        "retriever": retriever_name,
        "corpus": corpus_name,
        "total_queries": total,
        "precision_at_1": precision_hits / total if total else 0.0,
        "recall_at_k": recall_hits / total if total else 0.0,
        "k": k,
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "max_latency_ms": max(latencies_ms) if latencies_ms else 0.0,
        "by_difficulty": _aggregate_by_difficulty(per_query_results),
        "per_query": per_query_results,
    }


def _aggregate_by_difficulty(per_query: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group precision@1 / recall@k counts by difficulty tier.

    Promoted into the report dict (not just the printed summary) so the
    JSON artifact carries per-difficulty signal — a regression on the few
    ``hard`` queries is invisible in the corpus-wide average when easy
    queries sit at ceiling.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for q in per_query:
        d = q.get("difficulty") or "unknown"
        b = buckets.setdefault(d, {"total": 0, "top1": 0, "topk": 0})
        b["total"] += 1
        if q["top1_match"]:
            b["top1"] += 1
        if q["topk_match"]:
            b["topk"] += 1
    for b in buckets.values():
        # A bucket only exists once a query landed in it, so total >= 1.
        b["precision_at_1"] = b["top1"] / b["total"]
        b["recall_at_k"] = b["topk"] / b["total"]
    return buckets


def _run_negative_benchmark(neg_queries: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Measure abstention on out-of-corpus queries.

    A negative query has no correct answer in the corpus, so the right
    behavior is to **abstain** — return no hit above the retriever's
    ``MIN_SCORE``. ``false_answer_rate`` is the fraction that returned at
    least one hit anyway (lower is better); ``abstention_rate`` is its
    complement. This is measurement only: the retriever is unchanged, so
    the number establishes a baseline for the false-positive blind spot.
    """
    from . import RagPipeline

    pipeline = RagPipeline()
    false_answers = 0
    per_query_results: list[dict[str, Any]] = []
    for entry in neg_queries:
        result = pipeline.run(entry["query"], k=k)
        hit_paths = [h.template_path for h in result.citation.hits]
        top_score = result.citation.hits[0].score if result.citation.hits else 0.0
        answered = bool(hit_paths)
        if answered:
            false_answers += 1
        per_query_results.append(
            {
                "id": entry["id"],
                "query": entry["query"],
                "abstained": not answered,
                "leaked_hits": hit_paths[:k],
                "top_score": top_score,
            }
        )
    total = len(neg_queries)
    return {
        "total_negatives": total,
        "false_answers": false_answers,
        "false_answer_rate": false_answers / total if total else 0.0,
        "abstention_rate": (total - false_answers) / total if total else 0.0,
        "per_query": per_query_results,
    }


def _print_summary(report: dict[str, Any], verbose: bool) -> None:
    total = report["total_queries"]
    p1 = report["precision_at_1"]
    rk = report["recall_at_k"]
    k = report["k"]
    print(f"Retriever:  {report['retriever']}")
    print(f"Corpus:     {report['corpus']}")
    print(f"Queries:    {total}")
    print(f"Precision@1: {p1:.2%} ({int(p1 * total)}/{total})")
    print(f"Recall@{k}:    {rk:.2%} ({int(rk * total)}/{total})")
    print(f"Mean latency: {report['mean_latency_ms']:.2f}ms")
    print(f"Max latency:  {report['max_latency_ms']:.2f}ms")

    by_difficulty = report.get("by_difficulty") or _aggregate_by_difficulty(report["per_query"])
    if by_difficulty:
        print("\nBreakdown by difficulty:")
        for d in ("easy", "medium", "hard"):
            b = by_difficulty.get(d)
            if not b:
                continue
            print(f"  {d:7} P@1={b['top1']}/{b['total']}  R@{k}={b['topk']}/{b['total']}")

    if verbose:
        print("\nPer-query detail:")
        for q in report["per_query"]:
            mark = "OK" if q["topk_match"] else "MISS"
            top1 = "*" if q["top1_match"] else " "
            print(f"  [{mark}{top1}] {q['id']} ({q['difficulty']}): {q['query']!r}")
            if not q["topk_match"]:
                print(f"        expected: {q['expected']}")
                print(f"        actual:   {q['actual']}")


def _print_negatives(neg: dict[str, Any], verbose: bool) -> None:
    """Print the out-of-corpus abstention summary."""
    total = neg["total_negatives"]
    print("\nOut-of-corpus (negative) set:")
    print(f"  Negatives:        {total}")
    print(
        f"  Abstention rate:  {neg['abstention_rate']:.2%} "
        f"({total - neg['false_answers']}/{total} correctly returned nothing)"
    )
    print(
        f"  False-answer rate:{neg['false_answer_rate']:.2%} "
        f"({neg['false_answers']}/{total} confidently answered an unanswerable query)"
    )
    if verbose:
        for q in neg["per_query"]:
            if not q["abstained"]:
                print(
                    f"    LEAK {q['id']}: {q['query']!r} -> "
                    f"{q['leaked_hits']} (score {q['top_score']:.1f})"
                )


async def _score_faithfulness(
    queries: list[dict[str, Any]],
    k: int,
    use_native_citations: bool = False,
    *,
    use_thinking: bool = False,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    """Run the default variant through run_and_generate + judge for each query.

    Returns a dict with mean faithfulness, claim-citations stats,
    and p95 latency plus per-query detail. Spends API tokens (2
    LLM calls per query). Respect the caller's budget.

    ``use_native_citations`` toggles the Anthropic Citations API
    path. When True, ``citation_emit_rate`` is the fraction of
    queries the model returned at least one ``ClaimCitation`` for;
    when False it is always 0.0.

    ``use_thinking`` opts the judge into Anthropic extended
    thinking. ``thinking_budget_tokens`` overrides the judge's
    default ceiling when set.
    """
    from . import RagPipeline
    from .eval import FaithfulnessJudge

    pipeline = RagPipeline()
    judge = FaithfulnessJudge()

    score_kwargs: dict[str, Any] = {"use_thinking": use_thinking}
    if thinking_budget_tokens is not None:
        score_kwargs["thinking_budget_tokens"] = thinking_budget_tokens

    scores: list[float] = []
    latencies: list[float] = []
    hallu = 0
    refuse = 0
    cited = 0
    per_query: list[dict[str, Any]] = []

    for entry in queries:
        query = entry["query"]
        print(f"  judging: {entry['id']} — {query!r}", file=sys.stderr)
        t0 = time.perf_counter()
        answer, rag_result = await pipeline.run_and_generate(
            query,
            provider="claude",
            k=k,
            use_native_citations=use_native_citations,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        verdict = await judge.score(
            query=query,
            answer=answer,
            passages=rag_result.context,
            **score_kwargs,
        )
        scores.append(verdict.score)
        if verdict.total_claims == 0:
            refuse += 1
        if len(verdict.unsupported_claims) > 0:
            hallu += 1
        if rag_result.used_native_citations and rag_result.claim_citations:
            cited += 1
        per_query.append(
            {
                "id": entry["id"],
                "query": query,
                "answer": answer,
                "context": rag_result.context,
                "score": verdict.score,
                "supported": len(verdict.supported_claims),
                "unsupported": len(verdict.unsupported_claims),
                "supported_claims": list(verdict.supported_claims),
                "unsupported_claims": list(verdict.unsupported_claims),
                "reasoning": verdict.reasoning,
                "latency_ms": latencies[-1],
                "claim_citation_count": len(rag_result.claim_citations),
                "used_native_citations": rag_result.used_native_citations,
                "thinking_used": verdict.thinking_used,
            }
        )

    n = len(scores)
    return {
        "mean_faithfulness": sum(scores) / n if n else 0.0,
        "refusal_rate": refuse / n if n else 0.0,
        "hallucination_rate": hallu / n if n else 0.0,
        "citation_emit_rate": cited / n if n else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_latency_ms": sum(latencies) / n if n else 0.0,
        "per_query": per_query,
    }


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile of ``values`` (e.g. ``0.95``).

    Uses nearest-rank, not interpolation — sufficient for the
    benchmark's latency-summary use case. Returns 0.0 for an
    empty input rather than raising; ``pct`` must be in
    ``[0.0, 1.0]``.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return ordered[idx]


def _print_faithfulness(
    fr: dict[str, Any],
    label: str = "",
) -> None:
    """Print one faithfulness report (``mean``, refusal, hallucination, latency).

    Reads the dict shape returned by :func:`_score_faithfulness`.
    ``label`` is appended in parens to each metric line so a
    caller can disambiguate when printing multiple passes.
    """
    suffix = f" ({label})" if label else ""
    print()
    print(f"Mean faithfulness{suffix}:   {fr['mean_faithfulness']:.3f}")
    print(f"Refusal rate{suffix}:        {fr['refusal_rate']:.1%}")
    print(f"Hallucination rate{suffix}:  {fr['hallucination_rate']:.1%}")
    print(f"Citation emit rate{suffix}:  {fr['citation_emit_rate']:.1%}")
    print(f"Mean latency{suffix}:        {fr['mean_latency_ms']:.0f} ms")
    print(f"p95 latency{suffix}:         {fr['p95_latency_ms']:.0f} ms")


def _print_side_by_side(
    a_report: dict[str, Any],
    b_report: dict[str, Any],
    *,
    a_label: str = "Legacy [P{n}]",
    b_label: str = "Native cites",
) -> None:
    """Side-by-side comparison table for two faithfulness passes."""
    print()
    print(f"{'Metric':<24}  {a_label:>14}  {b_label:>14}  {'Δ':>10}")
    print(f"{'-' * 24}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    def row(name: str, key: str, fmt: str = ".3f", *, percent: bool = False) -> None:
        a = a_report[key]
        b = b_report[key]
        d = b - a
        if percent:
            print(f"{name:<24}  {a:>14.1%}  {b:>14.1%}  {d:>+10.1%}")
        else:
            print(f"{name:<24}  {a:>14{fmt}}  {b:>14{fmt}}  {d:>+10{fmt}}")

    row("Mean faithfulness", "mean_faithfulness")
    row("Refusal rate", "refusal_rate", percent=True)
    row("Hallucination rate", "hallucination_rate", percent=True)
    row("Citation emit rate", "citation_emit_rate", percent=True)
    row("Mean latency (ms)", "mean_latency_ms", ".0f")
    row("p95 latency (ms)", "p95_latency_ms", ".0f")


def _print_per_query_compare(
    a_report: dict[str, Any],
    b_report: dict[str, Any],
    *,
    a_label: str,
    b_label: str,
) -> None:
    """Per-query verdict diff: where do A and B disagree?"""
    by_id_a = {q["id"]: q for q in a_report["per_query"]}
    by_id_b = {q["id"]: q for q in b_report["per_query"]}
    common = [qid for qid in by_id_a if qid in by_id_b]
    if not common:
        return

    print()
    print(f"Per-query verdict comparison ({a_label} vs {b_label}):")
    print(f"  {'id':<18}  {'score Δ':>8}  {'claims A→B':>14}  " f"{'verdict shift':<24}")
    n_changed = 0
    for qid in common:
        qa = by_id_a[qid]
        qb = by_id_b[qid]
        delta = qb["score"] - qa["score"]
        claims_a = qa["supported"] + qa["unsupported"]
        claims_b = qb["supported"] + qb["unsupported"]
        shift_bits: list[str] = []
        if abs(delta) >= 0.001:
            shift_bits.append("score")
        if claims_a != claims_b:
            shift_bits.append("count")
        if (qa["supported"], qa["unsupported"]) != (
            qb["supported"],
            qb["unsupported"],
        ):
            shift_bits.append("partition")
        shift = ",".join(shift_bits) or "—"
        if shift_bits:
            n_changed += 1
        print(f"  {qid:<18}  {delta:>+8.3f}  " f"{claims_a:>5} → {claims_b:<6}  {shift:<24}")
    pct = n_changed / len(common) if common else 0.0
    print(f"\nVerdict-shift rate: {n_changed}/{len(common)} = {pct:.1%}")


def _dump_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write ``payload`` as pretty-printed, sorted-key JSON to ``path``.

    Creates parent directories if missing. ``sort_keys=True``
    keeps run-to-run diffs deterministic so that two artifacts
    from the same data are byte-identical. Prints the written
    path to stderr so the user sees where the file landed even
    when stdout is being captured.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nWrote per-query JSON: {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attune-rag-benchmark",
        description=(
            "Measure retrieval precision@1 / recall@k — optionally also "
            "faithfulness — against a golden-query set."
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=_default_queries_path(),
        help=f"Path to queries.yaml (default: {_default_queries_path()})",
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        default=_default_negatives_path(),
        help=(
            "Path to the out-of-corpus negative query set "
            f"(default: {_default_negatives_path()}). Skipped if absent. "
            "Measures abstention / false-answer rate (advisory, not gated)."
        ),
    )
    parser.add_argument(
        "--extended",
        type=Path,
        default=_default_extended_path(),
        help=(
            "Path to the extended advisory hard-query set "
            f"(default: {_default_extended_path()}). Skipped if absent. "
            "Measured + reported separately; never gates (keeps queries.yaml "
            "SHA lock intact)."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_default_corpus_b_path(),
        help=(
            "Path to an UNSEEN second corpus directory for generalization "
            f"measurement (default: {_default_corpus_b_path()}). Loaded as a "
            "DirectoryCorpus and benchmarked with --corpus-queries. Skipped if "
            "either is absent. Advisory; never gates."
        ),
    )
    parser.add_argument(
        "--corpus-queries",
        type=Path,
        default=_default_corpus_b_queries_path(),
        help=(
            "Golden queries for the --corpus generalization pass "
            f"(default: {_default_corpus_b_queries_path()})."
        ),
    )
    parser.add_argument("-k", type=int, default=3, help="Top-k for recall (default 3)")
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.70,
        help="Fail (exit 1) if precision@1 falls below this (default 0.70)",
    )
    parser.add_argument(
        "--with-faithfulness",
        action="store_true",
        help=(
            "Additionally run the default prompt variant through an LLM "
            "judge. Requires ANTHROPIC_API_KEY. Spends API tokens — one "
            "generator call + one judge call per query."
        ),
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.85,
        help=(
            "With --with-faithfulness: fail (exit 1) if mean faithfulness "
            "falls below this (default 0.85)."
        ),
    )
    parser.add_argument(
        "--native-citations",
        action="store_true",
        help=(
            "With --with-faithfulness: ALSO run a side-by-side pass "
            "using the Anthropic Citations API and print a "
            "comparison table. Doubles API spend. Requires the "
            "Claude provider; assumes the same model on both paths."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=_env_bool("ATTUNE_RAG_FAITHFULNESS_THINKING"),
        help=(
            "With --with-faithfulness: enable Anthropic extended "
            'thinking on the judge. Forces tool_choice="auto" '
            "and adds a thinking block. Env default: "
            "ATTUNE_RAG_FAITHFULNESS_THINKING."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=_env_int("ATTUNE_RAG_FAITHFULNESS_THINKING_BUDGET"),
        help=(
            "Ceiling for thinking tokens (only when --thinking). "
            "Billing is per emitted token, not the budget. Env: "
            "ATTUNE_RAG_FAITHFULNESS_THINKING_BUDGET. Default: "
            "32768 (set in the judge)."
        ),
    )
    parser.add_argument(
        "--compare-thinking",
        action="store_true",
        help=(
            "With --with-faithfulness: run two passes (thinking off, "
            "thinking on at --thinking-budget) and print a side-by-side "
            "comparison. Doubles API spend on the faithfulness step "
            "(thinking pass burns extra tokens on top). Mutually "
            "exclusive with --thinking and --native-citations."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "With --with-faithfulness: dump the full structured "
            "faithfulness report (per-query verdicts incl. reasoning "
            "and claim text) to PATH as JSON. Useful for offline "
            "calibration analysis."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-query detail including misses",
    )
    args = parser.parse_args(argv)

    if args.compare_thinking and args.thinking:
        print(
            "error: --compare-thinking already runs both thinking "
            "modes; --thinking is redundant. Drop one.",
            file=sys.stderr,
        )
        return 2
    if args.compare_thinking and args.native_citations:
        print(
            "error: --compare-thinking and --native-citations cannot "
            "be combined (would require 4 faithfulness passes). Run "
            "them in separate invocations.",
            file=sys.stderr,
        )
        return 2
    if args.compare_thinking and not args.with_faithfulness:
        print(
            "error: --compare-thinking requires --with-faithfulness.",
            file=sys.stderr,
        )
        return 2
    if not args.queries.is_file():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        return 2

    queries = _load_queries(args.queries)
    report = _run_benchmark(queries, k=args.k)
    _print_summary(report, verbose=args.verbose)

    # Out-of-corpus abstention measurement (advisory; never gates).
    negatives_report: dict[str, Any] | None = None
    if args.negatives and args.negatives.is_file():
        negatives_report = _run_negative_benchmark(_load_queries(args.negatives), k=args.k)
        _print_negatives(negatives_report, verbose=args.verbose)

    # Extended advisory hard-query pass (advisory; never gates).
    extended_report: dict[str, Any] | None = None
    if args.extended and args.extended.is_file():
        extended_report = _run_benchmark(_load_queries(args.extended), k=args.k)
        print("\n=== Extended (advisory hard) set ===")
        _print_summary(extended_report, verbose=args.verbose)

    # Generalization pass: same retriever against an UNSEEN second corpus
    # (advisory; never gates). Quantifies how far keyword+alias retrieval
    # drops on a corpus it was not tuned on.
    generalization_report: dict[str, Any] | None = None
    if (
        args.corpus
        and args.corpus.is_dir()
        and args.corpus_queries
        and args.corpus_queries.is_file()
    ):
        from .corpus import DirectoryCorpus

        corpus_b = DirectoryCorpus(args.corpus)
        generalization_report = _run_benchmark(
            _load_queries(args.corpus_queries), k=args.k, corpus=corpus_b
        )
        print(f"\n=== Generalization: unseen corpus ({args.corpus.name}) ===")
        _print_summary(generalization_report, verbose=args.verbose)

    if report["precision_at_1"] < args.min_precision:
        print(
            f"\nFAIL: precision@1 {report['precision_at_1']:.2%} < gate {args.min_precision:.2%}",
            file=sys.stderr,
        )
        return 1

    # Retrieval-only `--json` path. Used by the CI quality gate
    # when faithfulness gating is off-budget for the PR. The dump
    # carries `retrieval` + `queries_path`; downstream consumers
    # treat the absence of `faithfulness_legacy` as
    # "retrieval-only run".
    if args.json is not None and not args.with_faithfulness:
        retrieval_payload: dict[str, Any] = {
            "retrieval": report,
            "queries_path": str(args.queries),
        }
        if negatives_report is not None:
            retrieval_payload["negatives"] = negatives_report
        if extended_report is not None:
            retrieval_payload["extended"] = extended_report
        if generalization_report is not None:
            retrieval_payload["generalization"] = generalization_report
        _dump_json(args.json, retrieval_payload)

    if args.with_faithfulness:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "error: --with-faithfulness requires ANTHROPIC_API_KEY.",
                file=sys.stderr,
            )
            return 2

        json_payload: dict[str, Any] = {
            "retrieval": report,
            "queries_path": str(args.queries),
        }

        if args.compare_thinking:
            print(
                "\nRunning faithfulness pass A (thinking OFF)...",
                file=sys.stderr,
            )
            pass_a = asyncio.run(
                _score_faithfulness(
                    queries,
                    k=args.k,
                    use_native_citations=False,
                    use_thinking=False,
                )
            )
            _print_faithfulness(pass_a, label="thinking off")
            print(
                "\nRunning faithfulness pass B (thinking ON)...",
                file=sys.stderr,
            )
            pass_b = asyncio.run(
                _score_faithfulness(
                    queries,
                    k=args.k,
                    use_native_citations=False,
                    use_thinking=True,
                    thinking_budget_tokens=args.thinking_budget,
                )
            )
            _print_faithfulness(pass_b, label="thinking on")
            _print_side_by_side(
                pass_a,
                pass_b,
                a_label="thinking off",
                b_label="thinking on",
            )
            _print_per_query_compare(
                pass_a,
                pass_b,
                a_label="off",
                b_label="on",
            )
            json_payload["faithfulness_thinking_off"] = pass_a
            json_payload["faithfulness_thinking_on"] = pass_b
            gate_report = pass_a
            gate_label = "thinking-off"
        else:
            print(
                "\nRunning faithfulness pass (legacy [P{n}] path)...",
                file=sys.stderr,
            )
            legacy = asyncio.run(
                _score_faithfulness(
                    queries,
                    k=args.k,
                    use_native_citations=False,
                    use_thinking=args.thinking,
                    thinking_budget_tokens=args.thinking_budget,
                )
            )
            _print_faithfulness(legacy, label="legacy")
            json_payload["faithfulness_legacy"] = legacy

            if args.native_citations:
                print(
                    "\nRunning faithfulness pass (native citations path)...",
                    file=sys.stderr,
                )
                native = asyncio.run(
                    _score_faithfulness(
                        queries,
                        k=args.k,
                        use_native_citations=True,
                        use_thinking=args.thinking,
                        thinking_budget_tokens=args.thinking_budget,
                    )
                )
                _print_faithfulness(native, label="native")
                _print_side_by_side(legacy, native)
                json_payload["faithfulness_native"] = native

            gate_report = legacy
            gate_label = "legacy"

        if args.json is not None:
            _dump_json(args.json, json_payload)

        if gate_report["mean_faithfulness"] < args.min_faithfulness:
            print(
                f"\nFAIL: {gate_label} mean_faithfulness "
                f"{gate_report['mean_faithfulness']:.3f} "
                f"< gate {args.min_faithfulness:.3f}",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nPASS: P@1 ≥ {args.min_precision:.2%} and "
            f"{gate_label} faithfulness ≥ {args.min_faithfulness:.3f}."
        )
        return 0

    print(f"\nPASS: precision@1 meets gate ({args.min_precision:.2%}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
