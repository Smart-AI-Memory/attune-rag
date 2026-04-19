"""Prompt-variant A/B benchmark.

For every prompt variant in
:data:`attune_rag.prompts.PROMPT_VARIANTS`, run every
golden query through the full pipeline (retrieve → augment
→ generate → judge). Print a comparison table.

Run via::

    python -m attune_rag.eval.bench_prompts
    python -m attune_rag.eval.bench_prompts --variants baseline,strict
    python -m attune_rag.eval.bench_prompts --queries path/to/queries.yaml
    python -m attune_rag.eval.bench_prompts --output report.json

Requires ``attune-rag[claude]`` and ``ANTHROPIC_API_KEY``.

The benchmark is slow and spends API tokens (2 LLM calls
per query per variant). The default golden set is 15
queries × 4 variants = 120 LLM calls (60 generator + 60
judge). Budget accordingly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _default_queries_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent.parent / "tests" / "golden" / "queries.yaml"
    )


def _load_queries(path: Path) -> list[dict[str, Any]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


@dataclass
class _QueryRun:
    id: str
    query: str
    difficulty: str
    variant: str
    answer: str
    retrieval_top1_match: bool
    retrieval_topk_match: bool
    faithfulness_score: float
    supported_claims: int
    unsupported_claims: int
    total_claims: int
    generate_ms: float
    judge_ms: float


@dataclass
class _VariantReport:
    variant: str
    total_queries: int
    precision_at_1: float
    recall_at_k: float
    mean_faithfulness: float
    refusal_rate: float  # fraction of answers with zero claims
    hallucination_rate: float  # fraction of answers with ≥1 unsupported
    mean_generate_ms: float
    mean_judge_ms: float
    runs: list[_QueryRun] = field(default_factory=list)


async def _score_one(
    pipeline: Any,
    judge: Any,
    entry: dict[str, Any],
    variant: str,
    k: int,
    model: str | None,
) -> _QueryRun:
    query = entry["query"]
    expected = set(entry.get("expected_in_top_3", []))
    difficulty = entry.get("difficulty", "") or ""

    gen_start = time.perf_counter()
    answer, rag_result = await pipeline.run_and_generate(
        query,
        provider="claude",
        k=k,
        model=model,
        prompt_variant=variant,
    )
    generate_ms = (time.perf_counter() - gen_start) * 1000.0

    hit_paths = [h.template_path for h in rag_result.citation.hits]
    top1 = bool(hit_paths and hit_paths[0] in expected)
    topk = bool(set(hit_paths) & expected)

    judge_start = time.perf_counter()
    verdict = await judge.score(query=query, answer=answer, passages=rag_result.context)
    judge_ms = (time.perf_counter() - judge_start) * 1000.0

    return _QueryRun(
        id=entry["id"],
        query=query,
        difficulty=difficulty,
        variant=variant,
        answer=answer,
        retrieval_top1_match=top1,
        retrieval_topk_match=topk,
        faithfulness_score=verdict.score,
        supported_claims=len(verdict.supported_claims),
        unsupported_claims=len(verdict.unsupported_claims),
        total_claims=verdict.total_claims,
        generate_ms=generate_ms,
        judge_ms=judge_ms,
    )


def _aggregate(variant: str, runs: list[_QueryRun]) -> _VariantReport:
    n = len(runs)
    if n == 0:
        return _VariantReport(
            variant=variant,
            total_queries=0,
            precision_at_1=0.0,
            recall_at_k=0.0,
            mean_faithfulness=0.0,
            refusal_rate=0.0,
            hallucination_rate=0.0,
            mean_generate_ms=0.0,
            mean_judge_ms=0.0,
            runs=[],
        )
    return _VariantReport(
        variant=variant,
        total_queries=n,
        precision_at_1=sum(r.retrieval_top1_match for r in runs) / n,
        recall_at_k=sum(r.retrieval_topk_match for r in runs) / n,
        mean_faithfulness=sum(r.faithfulness_score for r in runs) / n,
        refusal_rate=sum(1 for r in runs if r.total_claims == 0) / n,
        hallucination_rate=sum(1 for r in runs if r.unsupported_claims > 0) / n,
        mean_generate_ms=sum(r.generate_ms for r in runs) / n,
        mean_judge_ms=sum(r.judge_ms for r in runs) / n,
        runs=runs,
    )


async def _run(
    queries: list[dict[str, Any]],
    variants: list[str],
    k: int,
    model: str | None,
    judge_model: str,
) -> list[_VariantReport]:
    from .. import RagPipeline
    from .faithfulness import FaithfulnessJudge

    pipeline = RagPipeline()
    judge = FaithfulnessJudge(model=judge_model)

    reports: list[_VariantReport] = []
    for variant in variants:
        print(f"\n=== variant: {variant} ===", file=sys.stderr)
        runs: list[_QueryRun] = []
        for idx, entry in enumerate(queries, start=1):
            print(
                f"  [{idx:2d}/{len(queries)}] {entry['id']}: {entry['query']!r}",
                file=sys.stderr,
            )
            run = await _score_one(pipeline, judge, entry, variant, k=k, model=model)
            runs.append(run)
        reports.append(_aggregate(variant, runs))
    return reports


def _print_table(reports: list[_VariantReport]) -> None:
    print()
    print(
        f"{'variant':<12} {'P@1':>6} {'R@k':>6} {'faith':>6} "
        f"{'refuse':>7} {'hallu':>6} {'gen_ms':>7} {'jdg_ms':>7}"
    )
    print("-" * 64)
    for r in reports:
        print(
            f"{r.variant:<12} "
            f"{r.precision_at_1:>6.1%} "
            f"{r.recall_at_k:>6.1%} "
            f"{r.mean_faithfulness:>6.2f} "
            f"{r.refusal_rate:>7.1%} "
            f"{r.hallucination_rate:>6.1%} "
            f"{r.mean_generate_ms:>7.0f} "
            f"{r.mean_judge_ms:>7.0f}"
        )
    print()
    print(
        "faith = mean faithfulness score in [0, 1]; refuse = share of "
        "answers with zero extracted claims; hallu = share of answers "
        "with ≥1 unsupported claim."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attune-rag-bench-prompts",
        description=(
            "A/B benchmark prompt variants for retrieval quality and "
            "faithfulness. Requires ANTHROPIC_API_KEY."
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=_default_queries_path(),
        help=f"Path to queries.yaml (default: {_default_queries_path()})",
    )
    parser.add_argument(
        "--variants",
        default="baseline,strict,citation,anti_prior",
        help="Comma-separated variant list (default: all four)",
    )
    parser.add_argument("-k", type=int, default=3, help="Top-k for retrieval (default 3)")
    parser.add_argument(
        "--model",
        default=None,
        help="Generator model (default: ClaudeProvider.DEFAULT_MODEL)",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-6",
        help="Judge model (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path for the full run detail",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY not set; cannot run prompt A/B.",
            file=sys.stderr,
        )
        return 2

    if not args.queries.is_file():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        return 2

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    from ..prompts import PROMPT_VARIANTS

    unknown = [v for v in variants if v not in PROMPT_VARIANTS]
    if unknown:
        print(
            f"error: unknown variants {unknown}; valid: {sorted(PROMPT_VARIANTS)}",
            file=sys.stderr,
        )
        return 2

    queries = _load_queries(args.queries)
    reports = asyncio.run(
        _run(
            queries=queries,
            variants=variants,
            k=args.k,
            model=args.model,
            judge_model=args.judge_model,
        )
    )

    _print_table(reports)

    if args.output:
        args.output.write_text(
            json.dumps(
                [
                    {
                        **{k: v for k, v in asdict(r).items() if k != "runs"},
                        "runs": [asdict(run) for run in r.runs],
                    }
                    for r in reports
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote full report to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
