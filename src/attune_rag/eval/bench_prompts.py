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

_SYSTEM_DIRS: tuple[str, ...] = ("/etc", "/sys", "/proc", "/dev", "/bin", "/sbin", "/boot")


def _default_queries_path() -> Path:
    """Return the in-repo golden queries path.

    Resolves relative to the installed package file, which
    points inside the source tree for editable / dev
    installs and inside ``site-packages/`` for wheel
    installs. The ``tests/`` directory is NOT shipped in
    the wheel, so this path only resolves for the
    dev-install case — callers must check ``is_file()``
    and fall back to a clear error for published installs.
    """
    return (
        Path(__file__).resolve().parent.parent.parent.parent / "tests" / "golden" / "queries.yaml"
    )


def _validate_read_path(raw: str | Path, kind: str) -> Path:
    """Validate and resolve a user-supplied input path.

    Rejects null-byte injection and unresolvable paths.
    Does NOT require the path to exist — callers should
    emit a more informative error than ``FileNotFoundError``
    when the file is missing.
    """
    text = str(raw)
    if "\x00" in text:
        raise ValueError(f"{kind} path contains null bytes: {text!r}")
    try:
        return Path(text).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid {kind} path {text!r}: {e}") from e


def _validate_write_path(raw: str | Path) -> Path:
    """Validate a user-supplied output path for writes.

    On top of :func:`_validate_read_path`, refuses writes
    to well-known system directories. This is a
    developer-run CLI, not a hardened production server
    path — the goal is to catch obvious footguns (a
    typo'd ``--output /etc/passwd``), not to enforce a
    full jail.
    """
    resolved = _validate_read_path(raw, kind="output")
    # Check both the raw input and the resolved path so macOS
    # symlinks like /etc -> /private/etc don't let a typo'd
    # --output /etc/passwd slip past the resolved-only check.
    raw_abs = str(Path(str(raw)).absolute())
    for candidate in (str(resolved), raw_abs):
        for sysdir in _SYSTEM_DIRS:
            if candidate == sysdir or candidate.startswith(sysdir + "/"):
                raise ValueError(
                    f"Refusing to write report to system directory " f"{sysdir}: {raw_abs}"
                )
    return resolved


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
    error: str | None = None  # non-None when the query failed mid-run


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
    error_count: int = 0  # queries that raised during the run
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
    """Aggregate per-query runs into a variant-level report.

    Errored runs (``run.error`` is not None) contribute to
    ``error_count`` but are excluded from every mean /
    rate, so a single transient failure does not skew the
    reported metrics.
    """
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
            error_count=0,
            runs=[],
        )

    ok = [r for r in runs if r.error is None]
    ok_n = len(ok)
    err_n = n - ok_n

    if ok_n == 0:
        # Every query errored — emit zeros with the error count
        return _VariantReport(
            variant=variant,
            total_queries=n,
            precision_at_1=0.0,
            recall_at_k=0.0,
            mean_faithfulness=0.0,
            refusal_rate=0.0,
            hallucination_rate=0.0,
            mean_generate_ms=0.0,
            mean_judge_ms=0.0,
            error_count=err_n,
            runs=runs,
        )

    return _VariantReport(
        variant=variant,
        total_queries=n,
        precision_at_1=sum(r.retrieval_top1_match for r in ok) / ok_n,
        recall_at_k=sum(r.retrieval_topk_match for r in ok) / ok_n,
        mean_faithfulness=sum(r.faithfulness_score for r in ok) / ok_n,
        refusal_rate=sum(1 for r in ok if r.total_claims == 0) / ok_n,
        hallucination_rate=sum(1 for r in ok if r.unsupported_claims > 0) / ok_n,
        mean_generate_ms=sum(r.generate_ms for r in ok) / ok_n,
        mean_judge_ms=sum(r.judge_ms for r in ok) / ok_n,
        error_count=err_n,
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
            try:
                run = await _score_one(pipeline, judge, entry, variant, k=k, model=model)
            except Exception as exc:  # noqa: BLE001
                # INTENTIONAL: one transient failure (rate limit,
                # network blip, malformed fixture entry) should not
                # abort an N-query benchmark. Record the error and
                # keep going; _aggregate excludes errored runs from
                # the reported means.
                print(
                    f"    ERROR ({type(exc).__name__}): {exc}; continuing",
                    file=sys.stderr,
                )
                run = _QueryRun(
                    id=str(entry.get("id", f"entry-{idx}")),
                    query=str(entry.get("query", "")),
                    difficulty=str(entry.get("difficulty", "") or ""),
                    variant=variant,
                    answer="",
                    retrieval_top1_match=False,
                    retrieval_topk_match=False,
                    faithfulness_score=0.0,
                    supported_claims=0,
                    unsupported_claims=0,
                    total_claims=0,
                    generate_ms=0.0,
                    judge_ms=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            runs.append(run)
        reports.append(_aggregate(variant, runs))
    return reports


def _print_table(reports: list[_VariantReport]) -> None:
    print()
    print(
        f"{'variant':<12} {'P@1':>6} {'R@k':>6} {'faith':>6} "
        f"{'refuse':>7} {'hallu':>6} {'gen_ms':>7} {'jdg_ms':>7} {'err':>4}"
    )
    print("-" * 70)
    for r in reports:
        print(
            f"{r.variant:<12} "
            f"{r.precision_at_1:>6.1%} "
            f"{r.recall_at_k:>6.1%} "
            f"{r.mean_faithfulness:>6.2f} "
            f"{r.refusal_rate:>7.1%} "
            f"{r.hallucination_rate:>6.1%} "
            f"{r.mean_generate_ms:>7.0f} "
            f"{r.mean_judge_ms:>7.0f} "
            f"{r.error_count:>4d}"
        )
    print()
    print(
        "faith = mean faithfulness score in [0, 1]; refuse = share of "
        "answers with zero extracted claims; hallu = share of answers "
        "with ≥1 unsupported claim; err = queries that raised and were "
        "excluded from the means."
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

    try:
        queries_path = _validate_read_path(args.queries, kind="queries")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not queries_path.is_file():
        using_default = args.queries == _default_queries_path()
        if using_default:
            print(
                "error: default golden queries file not found "
                f"at {queries_path}.\n"
                "The tests/golden/ directory is not shipped in the "
                "attune-rag wheel. Pass --queries <path> to your own "
                "YAML fixture (same schema as "
                "https://github.com/Smart-AI-Memory/attune-rag/blob/main/tests/golden/queries.yaml).",
                file=sys.stderr,
            )
        else:
            print(f"error: queries file not found: {queries_path}", file=sys.stderr)
        return 2

    output_path: Path | None = None
    if args.output is not None:
        try:
            output_path = _validate_write_path(args.output)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
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

    queries = _load_queries(queries_path)
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

    if output_path is not None:
        output_path.write_text(
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
        print(f"\nWrote full report to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
