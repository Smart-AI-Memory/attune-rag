"""Measure judge-side variance of FaithfulnessJudge.

For each of K queries (drawn from a calibration artifact),
re-runs the **judge only** M times in each condition
(thinking off, thinking on). The generator is not re-run —
the captured answer + context from the artifact are fed
back through the judge. This isolates judge-side variance
from generator-side variance, and is cheap (judge-only).

Outputs a JSON file with per-query mean/stdev for each
condition, plus aggregate pooled stdevs and the margin
stdev (stdev of per-query (off_mean - on_mean)).

Used by Phase 2 of the v1.0 roadmap to anchor the
"escalation if margin_stdev > 0.10" branch of the rubric
in docs/specs/faithfulness-thinking-decision/design.md.

Usage::

    python scripts/measure_judge_variance.py \\
        --artifact artifacts/calibration/thinking-2026-05-16.json \\
        --query-ids gq-001,gq-002,gq-004,gq-006,gq-012,gq-014,gq-018,gq-019 \\
        --runs 5 \\
        --out artifacts/calibration/variance-2026-05-16.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Make the package importable when running from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def _score_once(
    judge: Any,
    *,
    query: str,
    answer: str,
    context: str,
    use_thinking: bool,
) -> float:
    """One judge.score() call → score float."""
    result = await judge.score(
        query=query,
        answer=answer,
        passages=context,
        use_thinking=use_thinking,
    )
    return float(result.score)


async def _run_query(
    judge: Any,
    *,
    qid: str,
    query: str,
    answer: str,
    context: str,
    runs: int,
) -> dict[str, Any]:
    """Run M judge calls per condition for one query."""
    off_scores: list[float] = []
    on_scores: list[float] = []
    for i in range(runs):
        off_scores.append(
            await _score_once(
                judge, query=query, answer=answer, context=context, use_thinking=False
            )
        )
        print(f"  {qid}  off run {i + 1}/{runs} → {off_scores[-1]:.3f}", flush=True)
    for i in range(runs):
        on_scores.append(
            await _score_once(judge, query=query, answer=answer, context=context, use_thinking=True)
        )
        print(f"  {qid}  on  run {i + 1}/{runs} → {on_scores[-1]:.3f}", flush=True)
    return {
        "off": {
            "mean": statistics.fmean(off_scores),
            "stdev": statistics.stdev(off_scores) if len(off_scores) > 1 else 0.0,
            "raw": off_scores,
        },
        "on": {
            "mean": statistics.fmean(on_scores),
            "stdev": statistics.stdev(on_scores) if len(on_scores) > 1 else 0.0,
            "raw": on_scores,
        },
    }


def _per_query_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {q["id"]: q for q in report.get("per_query", [])}


def _aggregate(query_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Compute pooled stdevs across all queries.

    - off_stdev_pooled: stdev across the flat list of all off runs
      from all queries (captures the noise floor seen on this
      condition).
    - on_stdev_pooled: same for on.
    - margin_stdev: stdev of per-query (off_mean - on_mean).
      Captures how much the "judge condition" effect itself
      varies query-to-query, separate from within-query noise.
    """
    flat_off: list[float] = []
    flat_on: list[float] = []
    margins: list[float] = []
    for q in query_results.values():
        flat_off.extend(q["off"]["raw"])
        flat_on.extend(q["on"]["raw"])
        margins.append(q["off"]["mean"] - q["on"]["mean"])
    return {
        "off_stdev_pooled": statistics.stdev(flat_off) if len(flat_off) > 1 else 0.0,
        "on_stdev_pooled": statistics.stdev(flat_on) if len(flat_on) > 1 else 0.0,
        "margin_stdev": statistics.stdev(margins) if len(margins) > 1 else 0.0,
    }


async def _run(args: argparse.Namespace) -> int:
    if args.runs < 2:
        print("--runs must be >= 2 (stdev needs at least 2 samples)", file=sys.stderr)
        return 2

    artifact_path = Path(args.artifact)
    if not artifact_path.is_file():
        print(f"Artifact not found: {artifact_path}", file=sys.stderr)
        return 2
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    off_records = _per_query_by_id(artifact["faithfulness_thinking_off"])
    query_ids = [qid.strip() for qid in args.query_ids.split(",") if qid.strip()]
    missing = [qid for qid in query_ids if qid not in off_records]
    if missing:
        print(f"Query IDs not in artifact: {missing}", file=sys.stderr)
        return 2

    # Need answer + context per query (added in PR #26+).
    for qid in query_ids:
        rec = off_records[qid]
        if rec.get("answer") is None or rec.get("context") is None:
            print(
                f"{qid} missing answer or context — artifact predates PR #26. "
                f"Re-run the calibration benchmark to capture them.",
                file=sys.stderr,
            )
            return 2

    from attune_rag.eval.faithfulness import FaithfulnessJudge

    judge = FaithfulnessJudge()
    print(
        f"Judge model: {judge.model}  |  runs per condition: {args.runs}  |  "
        f"queries: {len(query_ids)}",
        flush=True,
    )

    query_results: dict[str, dict[str, Any]] = {}
    started = time.time()
    for i, qid in enumerate(query_ids, start=1):
        print(f"\n[{i}/{len(query_ids)}] {qid}: {off_records[qid]['query']!r}", flush=True)
        rec = off_records[qid]
        query_results[qid] = await _run_query(
            judge,
            qid=qid,
            query=rec["query"],
            answer=rec["answer"],
            context=rec["context"],
            runs=args.runs,
        )
        elapsed = time.time() - started
        print(
            f"  → off mean={query_results[qid]['off']['mean']:.3f} "
            f"σ={query_results[qid]['off']['stdev']:.3f}  "
            f"on mean={query_results[qid]['on']['mean']:.3f} "
            f"σ={query_results[qid]['on']['stdev']:.3f}  "
            f"(elapsed {elapsed:.0f}s)",
            flush=True,
        )

    aggregate = _aggregate(query_results)
    out = {
        "judge_model": judge.model,
        "source_artifact": str(artifact_path),
        "runs": args.runs,
        "query_ids": query_ids,
        "queries": query_results,
        "aggregate": aggregate,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"\nWrote {out_path}\n"
        f"Aggregate:\n"
        f"  off_stdev_pooled = {aggregate['off_stdev_pooled']:.4f}\n"
        f"  on_stdev_pooled  = {aggregate['on_stdev_pooled']:.4f}\n"
        f"  margin_stdev     = {aggregate['margin_stdev']:.4f}\n",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure judge-side variance by re-running the judge M times "
            "per query in each condition (off, on)."
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--query-ids",
        type=str,
        required=True,
        help="Comma-separated query IDs (e.g. gq-001,gq-002,gq-004).",
    )
    parser.add_argument(
        "--runs", type=int, required=True, help="Re-runs per query per condition (>= 2)."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
