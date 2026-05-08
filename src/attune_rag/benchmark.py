"""Retrieval + (optional) faithfulness benchmark runner.

Run via::

    python -m attune_rag.benchmark
    python -m attune_rag.benchmark --queries path/to/queries.yaml
    python -m attune_rag.benchmark --min-precision 0.70
    python -m attune_rag.benchmark --with-faithfulness \\
        --min-faithfulness 0.85

Prints precision@1, recall@3, and mean latency. Exits 1 if
precision@1 falls below ``--min-precision`` so CI can gate.

With ``--with-faithfulness`` (opt-in — spends API tokens),
also runs the default prompt variant through
``run_and_generate`` + :class:`FaithfulnessJudge` and
gates on the resulting mean faithfulness score. Requires
``ANTHROPIC_API_KEY`` and the ``[claude]`` extra.

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
    return Path(__file__).resolve().parent.parent.parent / "tests" / "golden" / "queries.yaml"


def _load_queries(path: Path) -> list[dict[str, Any]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def _run_benchmark(
    queries: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    from . import RagPipeline

    pipeline = RagPipeline()
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

    by_difficulty: dict[str, dict[str, int]] = {}
    for q in report["per_query"]:
        d = q.get("difficulty", "unknown") or "unknown"
        bucket = by_difficulty.setdefault(d, {"total": 0, "top1": 0, "topk": 0})
        bucket["total"] += 1
        if q["top1_match"]:
            bucket["top1"] += 1
        if q["topk_match"]:
            bucket["topk"] += 1

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


async def _score_faithfulness(
    queries: list[dict[str, Any]],
    k: int,
    use_native_citations: bool = False,
) -> dict[str, Any]:
    """Run the default variant through run_and_generate + judge for each query.

    Returns a dict with mean faithfulness, claim-citations stats,
    and p95 latency plus per-query detail. Spends API tokens (2
    LLM calls per query). Respect the caller's budget.

    ``use_native_citations`` toggles the Anthropic Citations API
    path. When True, ``citation_emit_rate`` is the fraction of
    queries the model returned at least one ``ClaimCitation`` for;
    when False it is always 0.0.
    """
    from . import RagPipeline
    from .eval import FaithfulnessJudge

    pipeline = RagPipeline()
    judge = FaithfulnessJudge()

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
        verdict = await judge.score(query=query, answer=answer, passages=rag_result.context)
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
                "score": verdict.score,
                "supported": len(verdict.supported_claims),
                "unsupported": len(verdict.unsupported_claims),
                "claim_citation_count": len(rag_result.claim_citations),
                "used_native_citations": rag_result.used_native_citations,
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
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return ordered[idx]


def _print_faithfulness(
    fr: dict[str, Any],
    label: str = "",
) -> None:
    suffix = f" ({label})" if label else ""
    print()
    print(f"Mean faithfulness{suffix}:   {fr['mean_faithfulness']:.3f}")
    print(f"Refusal rate{suffix}:        {fr['refusal_rate']:.1%}")
    print(f"Hallucination rate{suffix}:  {fr['hallucination_rate']:.1%}")
    print(f"Citation emit rate{suffix}:  {fr['citation_emit_rate']:.1%}")
    print(f"Mean latency{suffix}:        {fr['mean_latency_ms']:.0f} ms")
    print(f"p95 latency{suffix}:         {fr['p95_latency_ms']:.0f} ms")


def _print_side_by_side(
    legacy: dict[str, Any],
    native: dict[str, Any],
) -> None:
    """Side-by-side comparison table for the two paths."""
    print()
    print(f"{'Metric':<24}  {'Legacy [P{n}]':>14}  {'Native cites':>14}  {'Δ':>10}")
    print(f"{'-' * 24}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    def row(name: str, key: str, fmt: str = ".3f", *, percent: bool = False) -> None:
        a = legacy[key]
        b = native[key]
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
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-query detail including misses",
    )
    args = parser.parse_args(argv)

    if not args.queries.is_file():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        return 2

    queries = _load_queries(args.queries)
    report = _run_benchmark(queries, k=args.k)
    _print_summary(report, verbose=args.verbose)

    if report["precision_at_1"] < args.min_precision:
        print(
            f"\nFAIL: precision@1 {report['precision_at_1']:.2%} < gate {args.min_precision:.2%}",
            file=sys.stderr,
        )
        return 1

    if args.with_faithfulness:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "error: --with-faithfulness requires ANTHROPIC_API_KEY.",
                file=sys.stderr,
            )
            return 2
        print("\nRunning faithfulness pass (legacy [P{n}] path)...", file=sys.stderr)
        legacy = asyncio.run(_score_faithfulness(queries, k=args.k, use_native_citations=False))
        _print_faithfulness(legacy, label="legacy")

        if args.native_citations:
            print(
                "\nRunning faithfulness pass (native citations path)...",
                file=sys.stderr,
            )
            native = asyncio.run(_score_faithfulness(queries, k=args.k, use_native_citations=True))
            _print_faithfulness(native, label="native")
            _print_side_by_side(legacy, native)

        if legacy["mean_faithfulness"] < args.min_faithfulness:
            print(
                f"\nFAIL: legacy mean_faithfulness "
                f"{legacy['mean_faithfulness']:.3f} "
                f"< gate {args.min_faithfulness:.3f}",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nPASS: P@1 ≥ {args.min_precision:.2%} and "
            f"legacy faithfulness ≥ {args.min_faithfulness:.3f}."
        )
        return 0

    print(f"\nPASS: precision@1 meets gate ({args.min_precision:.2%}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
