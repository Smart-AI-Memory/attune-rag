"""Diagnostic for the LLMReranker default — D5 of the v1.0 roadmap.

Phase 5 deliverable from ``docs/specs/reranker-evaluation/tasks.md`` M1.

Runs two pipeline configurations against the bundled
``AttuneHelpCorpus`` + the golden query sets:

- **Run A** — ``RagPipeline(reranker=None)``. Deterministic; N=1.
  Reproduces the bundled baseline byte-identically (R1 of the spec).
- **Run B** — ``RagPipeline(reranker=LLMReranker())``. Non-deterministic;
  N=5 by default (per scoping decision #3). Aggregates per-metric
  mean / p50 / p95 across the 5 runs; per-query stability annotation
  on the residual paraphrased misses.

Emits a markdown report with all reproducibility metadata (query
SHA-256, commit SHA, reranker model, Anthropic SDK version, ISO
timestamp). Same input → byte-identical metadata block modulo
timestamp.

This is M1 — the script. M2 (live run + verdict) and M3 (cross-link
to user-corpus-onboarding/risks.md) ship in follow-up PRs.

Usage::

    python scripts/measure_reranker.py \\
        --baseline-queries tests/golden/queries.yaml \\
        --paraphrased-queries tests/golden/queries_paraphrased.yaml \\
        --rerank-runs 5 \\
        --out docs/specs/reranker-evaluation/diagnostic-1.md
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from attune_rag import RagPipeline
from attune_rag._scoring import QueryScore, score_queries

# Reproducibility constants. Locked here so the report's metadata
# block is one place to inspect.
_DEFAULT_RERANK_RUNS = 5  # per scoping decision #3
_DEFAULT_CANDIDATE_MULTIPLIER = 3
_HARNESS_VERSION = "0.1.0"

# R1 strict-dominance reference (rerank-off, bundled corpus). Today's
# actual numbers per tests/golden/measure_corpus_bundled.golden.md.
# The spec originally cited rerank-*on* aspirationals; corrected to
# rerank-*off* reality at D5 scoping (see tasks.md M1.3 note 2026-05-22).
_R1_REFERENCE = {
    "baseline_p1": 1.0000,
    "baseline_r3": 1.0000,
    "paraphrased_p1": 0.9000,
    "paraphrased_r3": 0.9875,
}


@dataclass(frozen=True)
class _RunBResult:
    """One Run-B pass (rerank-on) result.

    Carries both the aggregate metrics (for the run-over-run table)
    and the per-query records (so M2.3's per-query stability
    analysis can count how many of N runs lifted each Run-A
    paraphrased P@1 miss).
    """

    baseline_p1: float
    baseline_r3: float
    paraphrased_p1: float
    paraphrased_r3: float
    baseline_per: tuple[QueryScore, ...]
    paraphrased_per: tuple[QueryScore, ...]


def _load_queries(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load queries.yaml + return SHA-256 of canonical (LF-normalized) bytes."""
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n")
    sha = sha256(canonical).hexdigest()[:12]
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError(f"{path}: top-level YAML must be a mapping with a 'queries' key")
    return data["queries"], sha


def _git_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _anthropic_sdk_version() -> str:
    try:
        import anthropic

        return getattr(anthropic, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def run_a(
    pipeline_off: RagPipeline,
    baseline_queries: Sequence[dict[str, Any]],
    paraphrased_queries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Run A — rerank-off, deterministic, N=1. Asserts R1 reproduction."""
    base_per, base_agg = score_queries(pipeline_off, baseline_queries, k=3)
    para_per, para_agg = score_queries(pipeline_off, paraphrased_queries, k=3)
    return {
        "baseline_per": base_per,
        "baseline_agg": base_agg,
        "paraphrased_per": para_per,
        "paraphrased_agg": para_agg,
    }


def check_r1(run_a_result: dict[str, Any]) -> tuple[bool, str]:
    """Assert Run A reproduces today's bundled baseline byte-identically.

    Returns (ok, message). ok=False means the diagnostic is wrong and
    M1 needs to be re-opened per spec M2.2.
    """
    bag = run_a_result["baseline_agg"]
    pag = run_a_result["paraphrased_agg"]
    actual = {
        "baseline_p1": round(bag.p1, 4),
        "baseline_r3": round(bag.r3, 4),
        "paraphrased_p1": round(pag.p1, 4),
        "paraphrased_r3": round(pag.r3, 4),
    }
    if actual == _R1_REFERENCE:
        return True, f"R1 reproduction OK: {actual}"
    return False, f"R1 FAIL: expected {_R1_REFERENCE}, got {actual}"


def run_b(
    pipeline_on: RagPipeline,
    baseline_queries: Sequence[dict[str, Any]],
    paraphrased_queries: Sequence[dict[str, Any]],
    n: int,
) -> list[_RunBResult]:
    """Run B — rerank-on, non-deterministic, N runs.

    Returns one ``_RunBResult`` per run. The caller aggregates
    mean / p50 / p95 across runs.
    """
    out: list[_RunBResult] = []
    for _ in range(n):
        base_per, base_agg = score_queries(pipeline_on, baseline_queries, k=3)
        para_per, para_agg = score_queries(pipeline_on, paraphrased_queries, k=3)
        out.append(
            _RunBResult(
                baseline_p1=base_agg.p1,
                baseline_r3=base_agg.r3,
                paraphrased_p1=para_agg.p1,
                paraphrased_r3=para_agg.r3,
                baseline_per=tuple(base_per),
                paraphrased_per=tuple(para_per),
            )
        )
    return out


def _residual_stability(
    run_a_result: dict[str, Any],
    run_b_results: list[_RunBResult],
) -> list[dict[str, Any]]:
    """Per-query stability analysis on Run A's paraphrased P@1 misses (M2.3).

    For each paraphrased query where Run A (``rerank=off``) had P@1 = ✗,
    counts how many of the N Run-B passes lifted P@1 to ✓ and how many
    held R@3. Stability is expressed as ``k/N``; ``k >= ⌈0.8N⌉`` is the
    "≥4/5" bar named in the M3.1 rubric.
    """
    residuals: list[dict[str, Any]] = []
    a_paraphrased_per = run_a_result["paraphrased_per"]
    n = len(run_b_results)
    for q in a_paraphrased_per:
        if q.p1:
            continue  # Run A already at P@1 — not a residual
        p1_lifts = 0
        r3_holds = 0
        for run in run_b_results:
            for rq in run.paraphrased_per:
                if rq.qid == q.qid:
                    if rq.p1:
                        p1_lifts += 1
                    if rq.r3:
                        r3_holds += 1
                    break
        residuals.append(
            {
                "qid": q.qid,
                "difficulty": q.difficulty,
                "expected": list(q.expected),
                "p1_lifts": p1_lifts,
                "n": n,
                "p1_stability": f"{p1_lifts}/{n}",
                "r3_stability": f"{r3_holds}/{n}",
                "stable_lift": p1_lifts >= max(1, (n * 4 + 4) // 5),  # ⌈0.8N⌉
            }
        )
    return residuals


def _aggregate_run_b(results: list[_RunBResult]) -> dict[str, dict[str, float]]:
    """Mean / p50 / p95 per metric across N runs."""

    def _stats(values: list[float]) -> dict[str, float]:
        s = sorted(values)
        n = len(s)
        return {
            "mean": statistics.fmean(s),
            "p50": s[n // 2],
            "p95": s[min(n - 1, int(n * 0.95))],
        }

    return {
        "baseline_p1": _stats([r.baseline_p1 for r in results]),
        "baseline_r3": _stats([r.baseline_r3 for r in results]),
        "paraphrased_p1": _stats([r.paraphrased_p1 for r in results]),
        "paraphrased_r3": _stats([r.paraphrased_r3 for r in results]),
    }


def render_report(
    *,
    run_a_result: dict[str, Any],
    run_b_aggregated: dict[str, dict[str, float]] | None,
    run_b_n: int,
    metadata: dict[str, str],
    residuals: list[dict[str, Any]] | None = None,
) -> str:
    """Markdown report mirroring design.md §3 shape."""
    lines: list[str] = []
    lines.append("# D5 diagnostic: LLMReranker evaluation")
    lines.append("")
    lines.append("## Reproducibility metadata")
    lines.append("")
    for key in sorted(metadata.keys()):
        lines.append(f"- {key}: `{metadata[key]}`")
    lines.append("")

    # Run A
    bag = run_a_result["baseline_agg"]
    pag = run_a_result["paraphrased_agg"]
    lines.append("## Run A — `rerank=off` (deterministic, N=1)")
    lines.append("")
    lines.append("| Set | P@1 | R@3 |")
    lines.append("|-----|-----|-----|")
    lines.append(f"| baseline | {bag.p1:.4f} | {bag.r3:.4f} |")
    lines.append(f"| paraphrased | {pag.p1:.4f} | {pag.r3:.4f} |")
    lines.append("")
    ok, msg = check_r1(run_a_result)
    lines.append(f"**R1 strict-dominance check:** {msg}")
    lines.append("")

    # Run B
    lines.append(f"## Run B — `rerank=on` (Haiku, N={run_b_n})")
    lines.append("")
    if run_b_aggregated is None:
        lines.append("> Skipped (M1 ships the script; M2 runs the live diagnostic).")
        lines.append("")
    else:
        lines.append("| Metric | mean | p50 | p95 |")
        lines.append("|--------|------|-----|-----|")
        for metric in ("baseline_p1", "baseline_r3", "paraphrased_p1", "paraphrased_r3"):
            s = run_b_aggregated[metric]
            lines.append(f"| {metric} | {s['mean']:.4f} | {s['p50']:.4f} | {s['p95']:.4f} |")
        lines.append("")

    # M2.3 — per-query residual stability on Run A's paraphrased P@1 misses.
    if residuals is not None:
        lines.append("## Per-query residuals (paraphrased P@1 misses in Run A)")
        lines.append("")
        if not residuals:
            lines.append("> No residuals — Run A's paraphrased P@1 = 1.00.")
            lines.append("")
        else:
            lines.append(
                "Stability column is `k/N` = number of Run-B passes where "
                "rerank lifted that query to ✓. The rubric's "
                '"stable lift" bar is `≥ ⌈0.8·N⌉` (e.g. ≥4/5).'
            )
            lines.append("")
            lines.append(
                "| qid | difficulty | rerank fixes P@1 (k/N) | R@3 stability | stable lift? |"
            )
            lines.append(
                "|-----|-----------|------------------------|---------------|--------------|"
            )
            for r in residuals:
                lines.append(
                    f"| `{r['qid']}` | {r['difficulty'] or '—'} | "
                    f"{r['p1_stability']} | {r['r3_stability']} | "
                    f"{'✓' if r['stable_lift'] else '✗'} |"
                )
            lines.append("")
            stable_fixes = sum(1 for r in residuals if r["stable_lift"])
            lines.append(
                f"**Stable-lift count:** {stable_fixes} of {len(residuals)} "
                f"residuals fixed at ≥⌈0.8·N⌉ stability."
            )
            lines.append("")

    # Verdict placeholder — filled in at M3 per locked rubric.
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "> **TBD at M3** — applies the rubric from `tasks.md` M3.1 to the "
        "Run B numbers. One of `rerank-default-on` / `rerank-default-off` / "
        "`corpus-shape-dependent-default`."
    )
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D5 reranker evaluation diagnostic.")
    parser.add_argument(
        "--baseline-queries",
        type=Path,
        default=Path("tests/golden/queries.yaml"),
        help="Path to baseline queries YAML.",
    )
    parser.add_argument(
        "--paraphrased-queries",
        type=Path,
        default=Path("tests/golden/queries_paraphrased.yaml"),
        help="Path to paraphrased queries YAML.",
    )
    parser.add_argument(
        "--rerank-runs",
        type=int,
        default=_DEFAULT_RERANK_RUNS,
        help=f"N runs of Run B (rerank=on). Default {_DEFAULT_RERANK_RUNS}.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=_DEFAULT_CANDIDATE_MULTIPLIER,
        help=f"Reranker candidate over-fetch factor. Default {_DEFAULT_CANDIDATE_MULTIPLIER}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Markdown report destination.",
    )
    parser.add_argument(
        "--skip-run-b",
        action="store_true",
        help="Run A only — emit the report with Run B placeholder. Useful "
        "for the M1 PR (no live API spend); M2 fills in Run B.",
    )
    parser.add_argument(
        "--frozen-timestamp",
        type=str,
        default=None,
        help="ISO timestamp to write into the report (for deterministic "
        "metadata blocks in tests). Default: current UTC.",
    )
    args = parser.parse_args(argv)

    # Load queries (sha256 included in metadata).
    baseline_queries, baseline_sha = _load_queries(args.baseline_queries)
    paraphrased_queries, paraphrased_sha = _load_queries(args.paraphrased_queries)

    # Build pipelines.
    pipeline_off = RagPipeline()
    pipeline_on: RagPipeline | None = None
    reranker_model = "n/a"
    if not args.skip_run_b:
        from attune_rag.reranker import LLMReranker

        reranker = LLMReranker(candidate_multiplier=args.candidate_multiplier)
        reranker_model = getattr(reranker, "_model", "unknown")
        pipeline_on = RagPipeline(reranker=reranker)

    # Run A.
    a_result = run_a(pipeline_off, baseline_queries, paraphrased_queries)
    ok, _ = check_r1(a_result)
    if not ok:
        # Per spec M2.2: R1 self-test failure means M1 is wrong.
        print("error: R1 strict-dominance check failed; see report.", file=sys.stderr)
        # We still emit the report so the user can see what diverged.

    # Run B (skipped when --skip-run-b).
    b_aggregated: dict[str, dict[str, float]] | None = None
    residuals: list[dict[str, Any]] | None = None
    if not args.skip_run_b:
        assert pipeline_on is not None
        b_results = run_b(pipeline_on, baseline_queries, paraphrased_queries, args.rerank_runs)
        b_aggregated = _aggregate_run_b(b_results)
        residuals = _residual_stability(a_result, b_results)

    # Metadata.
    repo_root = Path(__file__).resolve().parent.parent
    timestamp = args.frozen_timestamp or (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    metadata = {
        "harness_version": _HARNESS_VERSION,
        "timestamp": timestamp,
        "commit_sha": _git_sha(repo_root),
        "baseline_queries_sha": baseline_sha,
        "paraphrased_queries_sha": paraphrased_sha,
        "rerank_runs": str(args.rerank_runs),
        "reranker_model": reranker_model,
        "anthropic_sdk_version": _anthropic_sdk_version(),
        "candidate_multiplier": str(args.candidate_multiplier),
        "run_b_skipped": str(args.skip_run_b).lower(),
    }

    report = render_report(
        run_a_result=a_result,
        run_b_aggregated=b_aggregated,
        run_b_n=args.rerank_runs if not args.skip_run_b else 0,
        metadata=metadata,
        residuals=residuals,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(report.encode("utf-8"))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
