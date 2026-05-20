"""Measure perf baseline for the four hot paths and lock dual-metric thresholds.

Phase 4 deliverable from docs/specs/ROADMAP-v1.md / Phase 4
(Downstream Validation). The companion to Phase 1's
``scripts/measure_baseline_variance.py``: same pattern (N runs on an
unchanged HEAD → mean / stdev / threshold per metric), but for
latency rather than retrieval quality.

Two metric axes are captured per run via
``time.perf_counter()`` + ``time.process_time()``:

- **wall** — wall-clock elapsed seconds (what users feel; includes
  network variance for Anthropic-bound calls).
- **cpu** — process CPU time in seconds (excludes wait time on
  external services; gate-friendly).

Per the spec's design decision (Phase 4 spec review, 2026-05-19),
both axes are tracked. CPU-time becomes the gating axis in W3.1;
wall-clock stays advisory through W3 and is re-evaluated in W4.

Threshold strategy is **mean + sigma * stdev** (latencies are
higher-is-worse, so we gate the upper bound), matching the Phase 1
``mean − sigma * stdev`` shape but inverted sign per Decision 1 of
the v1.0 roadmap.

Output:

- ``--out`` — markdown report (locked-baseline narrative).
- ``--thresholds-out`` — machine-readable JSON. Two rows per
  benchmark: ``<bench>.wall`` and ``<bench>.cpu``.

Usage::

    python scripts/measure_perf_baseline.py \\
        --runs 30 \\
        --out docs/specs/downstream-validation/perf-baseline.md \\
        --thresholds-out \\
            docs/specs/downstream-validation/perf-thresholds.json

Cheap smoke run (retrieval + corpus only, no LLM spend)::

    python scripts/measure_perf_baseline.py \\
        --runs 10 \\
        --out /tmp/perf.md --thresholds-out /tmp/perf.json

Full baseline including reranker + pipeline (requires
``ANTHROPIC_API_KEY``)::

    python scripts/measure_perf_baseline.py \\
        --runs 30 --include-llm \\
        --out docs/specs/downstream-validation/perf-baseline.md \\
        --thresholds-out \\
            docs/specs/downstream-validation/perf-thresholds.json

Pure stdlib except for the optional LLM-backed benchmarks (which
import ``anthropic`` only when ``--include-llm`` is set).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Hot paths to measure. The names are stable identifiers used as
# keys in the JSON output (``<name>.<axis>``); changing them is a
# breaking change to anything consuming perf-thresholds.json.
#
# Three of the four are LLM-free (RagPipeline.run is retrieval-only;
# the LLM call lives in the async run_and_generate path). Only the
# reranker actually hits Anthropic and so is gated behind --include-llm.
LLM_FREE_BENCHMARKS: tuple[str, ...] = (
    "keyword_retriever_retrieve",
    "directory_corpus_load",
    "rag_pipeline_run",
)
LLM_BENCHMARKS: tuple[str, ...] = ("llm_reranker_rerank",)

DEFAULT_SIGMA = 2.0
MIN_RUNS = 10  # statistics.stdev needs ≥2, but small N is unreliable


# ---------------------------------------------------------------------------
# Pure timing primitives (testable in isolation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimedResult:
    wall: float  # seconds
    cpu: float  # seconds


def time_call(fn: Callable[[], object]) -> TimedResult:
    """Time a no-arg callable. Captures both perf_counter and process_time.

    ``perf_counter`` is a monotonic wall-clock timer with the
    highest available resolution. ``process_time`` excludes time
    spent sleeping or waiting on external resources, so it isolates
    CPU work in this process. Their delta on the reranker and
    pipeline benchmarks is the network-bound fraction — that
    asymmetry is exactly why both axes get tracked.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    fn()
    return TimedResult(
        wall=time.perf_counter() - wall_start,
        cpu=time.process_time() - cpu_start,
    )


def run_benchmark_n(fn: Callable[[], object], n: int) -> dict[str, list[float]]:
    """Time ``fn`` ``n`` times. Returns ``{"wall": [...], "cpu": [...]}``."""
    wall: list[float] = []
    cpu: list[float] = []
    for _ in range(n):
        r = time_call(fn)
        wall.append(r.wall)
        cpu.append(r.cpu)
    return {"wall": wall, "cpu": cpu}


def compute_stats(values: list[float], sigma: float) -> dict[str, float]:
    """Mean, stdev, ``mean + sigma * stdev`` per axis (upper threshold).

    Returns floats rounded to 6 decimal places — latencies are sub-
    second on most hot paths, so 4-decimal rounding (Phase 1's
    convention for retrieval metrics in [0, 1]) would lose
    resolution here.
    """
    if not values:
        raise ValueError("compute_stats requires at least one value")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 6),
        "stdev": round(stdev, 6),
        "threshold": round(mean + sigma * stdev, 6),
        "n": len(values),
    }


def aggregate_results(
    raw: dict[str, dict[str, list[float]]], sigma: float
) -> dict[str, dict[str, float]]:
    """Flatten {bench: {axis: [values]}} → {f"{bench}.{axis}": stats}.

    Either axis may be absent (or empty) for a given benchmark —
    e.g. a benchmark recorded with only wall-clock data — and is
    skipped silently. The resulting JSON only contains the axes
    that have measurements, so downstream consumers can rely on
    presence to mean "real data".
    """
    out: dict[str, dict[str, float]] = {}
    for bench, axes in raw.items():
        for axis, values in axes.items():
            if not values:
                continue
            out[f"{bench}.{axis}"] = compute_stats(values, sigma)
    return out


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------


def git_sha(repo_root: Path) -> str:
    """Return the long HEAD SHA, or ``"unknown"`` if git fails."""
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


def environment_fingerprint() -> dict[str, str]:
    """Capture the runner SKU + Python identity so cross-machine
    re-measures can flag hardware drift.

    Phase 4 W0.4 locks the initial baseline on GitHub Actions
    ``ubuntu-latest``; subsequent re-measures compare against the
    committed fingerprint and warn on mismatch. Re-measure triggers
    are documented in design.md §2.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "ci_runner": os.environ.get("RUNNER_OS", "local"),
    }


# ---------------------------------------------------------------------------
# Output formatters (pure functions)
# ---------------------------------------------------------------------------


def build_thresholds_payload(
    *,
    measured_at: str,
    commit: str,
    runs: int,
    sigma: float,
    env: dict[str, str],
    stats_by_metric: dict[str, dict[str, float]],
    include_llm: bool,
) -> dict[str, Any]:
    return {
        "measured_at": measured_at,
        "commit": commit,
        "runs": runs,
        "sigma": sigma,
        "include_llm": include_llm,
        "environment": env,
        "metrics": stats_by_metric,
    }


def write_thresholds_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def render_markdown(payload: dict[str, Any]) -> str:
    """Format the locked baseline report. Sorted-key tables for diff stability."""
    lines: list[str] = []
    lines.append("# Perf baseline (Phase 4)")
    lines.append("")
    lines.append("> Locked record of one perf-baseline measurement run.")
    lines.append("> Generated by `scripts/measure_perf_baseline.py`.")
    lines.append("> Threshold = `mean + sigma × stdev` per (benchmark, axis)")
    lines.append("> — latencies are higher-is-worse, so the upper bound is the gate.")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Measured at | `{payload['measured_at']}` |")
    lines.append(f"| Commit | `{payload['commit']}` |")
    lines.append(f"| Runs (N) | {payload['runs']} |")
    lines.append(f"| Sigma | {payload['sigma']} |")
    lines.append(f"| LLM benchmarks included | {payload['include_llm']} |")
    env = payload["environment"]
    lines.append(f"| CI runner | `{env.get('ci_runner', 'unknown')}` |")
    lines.append(f"| Platform | `{env.get('platform', 'unknown')}` |")
    lines.append(
        f"| Python | `{env.get('python_implementation', '?')} {env.get('python_version', '?')}` |"
    )
    lines.append("")
    lines.append("## Per-metric stats")
    lines.append("")
    lines.append("| Metric | Mean (s) | Stdev (s) | Threshold (s) | N |")
    lines.append("|---|---:|---:|---:|---:|")
    metrics = payload["metrics"]
    for key in sorted(metrics):
        s = metrics[key]
        lines.append(
            f"| `{key}` | {s['mean']:.6f} | {s['stdev']:.6f} | "
            f"{s['threshold']:.6f} | {s['n']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Hot-path benchmark factories (lazy imports — only loaded if invoked)
# ---------------------------------------------------------------------------


def _resolve_benchmark_corpus_dir() -> Path:
    """Locate a corpus directory with enough material to time meaningfully.

    The bundled ``.help/templates/`` tree is the canonical corpus
    attune-rag's own benchmark uses (see ``attune_rag.benchmark``);
    fall back to ``tests/golden/`` if it's not present (test envs
    that strip vendored templates).
    """
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (repo_root / ".help" / "templates", repo_root / "tests" / "golden"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "no benchmark corpus directory found " "(checked .help/templates and tests/golden)"
    )


def _build_llm_free_benchmarks() -> dict[str, Callable[[], object]]:
    """Construct the three LLM-free benchmark callables.

    Imports of ``attune_rag.*`` happen here so the test suite can
    monkey-patch the factory wholesale without ever loading the
    package. Each callable is a closure over its fixture; calling
    it executes the hot path once.

    ``RagPipeline.run`` is retrieval-only — the LLM call lives in
    the async ``run_and_generate`` path. So it stays in this set
    despite the name; only ``LLMReranker.rerank`` calls Anthropic.
    """
    from attune_rag.corpus.directory import DirectoryCorpus  # noqa: PLC0415
    from attune_rag.pipeline import RagPipeline  # noqa: PLC0415
    from attune_rag.retrieval import KeywordRetriever  # noqa: PLC0415

    bench_corpus_dir = _resolve_benchmark_corpus_dir()
    query = "attune rag pipeline citation"

    # Build once, time the hot call.
    corpus = DirectoryCorpus(bench_corpus_dir)
    retriever = KeywordRetriever()

    def kw_retrieve() -> None:
        retriever.retrieve(query, corpus, k=3)

    # Cold-start corpus load — recreates each call so we measure
    # the load path, not cached state.
    def corpus_load() -> None:
        DirectoryCorpus(bench_corpus_dir)

    # RagPipeline.run — retrieval + prompt assembly orchestration.
    # No LLM call (that's run_and_generate).
    pipeline = RagPipeline(corpus=corpus)

    def pipeline_run() -> None:
        pipeline.run(query)

    return {
        "keyword_retriever_retrieve": kw_retrieve,
        "directory_corpus_load": corpus_load,
        "rag_pipeline_run": pipeline_run,
    }


def _build_llm_benchmarks() -> dict[str, Callable[[], object]]:
    """Construct the LLM-backed benchmark callable(s).

    The reranker is the only hot path that actually hits Anthropic
    (``RagPipeline.run`` is retrieval-only). Tracked separately so
    the cheap baseline can be measured without API spend.

    Raises if ``ANTHROPIC_API_KEY`` is missing — the caller should
    have validated that before ``--include-llm`` took effect.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY required for --include-llm benchmarks")
    from attune_rag.corpus.directory import DirectoryCorpus  # noqa: PLC0415
    from attune_rag.reranker import LLMReranker  # noqa: PLC0415
    from attune_rag.retrieval import KeywordRetriever  # noqa: PLC0415

    bench_corpus_dir = _resolve_benchmark_corpus_dir()
    query = "attune rag pipeline citation"
    corpus = DirectoryCorpus(bench_corpus_dir)
    retriever = KeywordRetriever()
    # LLMReranker carries its own Anthropic client; no provider arg.
    reranker = LLMReranker()

    # Pre-fetch hits once so the reranker benchmark times only the
    # reranking step, not the upstream retrieval.
    initial_hits = retriever.retrieve(query, corpus, k=5)

    def rerank() -> None:
        reranker.rerank(query, initial_hits)

    return {"llm_reranker_rerank": rerank}


def collect_raw_timings(runs: int, *, include_llm: bool) -> dict[str, dict[str, list[float]]]:
    """Wire factories to the timing loop. Returns {bench: {axis: [values]}}."""
    raw: dict[str, dict[str, list[float]]] = {}
    for name, fn in _build_llm_free_benchmarks().items():
        raw[name] = run_benchmark_n(fn, runs)
    if include_llm:
        for name, fn in _build_llm_benchmarks().items():
            raw[name] = run_benchmark_n(fn, runs)
    return raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_perf_baseline",
        description=(
            "Measure hot-path latency N times and lock dual-metric "
            "(wall + CPU) perf thresholds. Phase 4 W0.3 deliverable."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=True,
        help=f"Iterations per benchmark (must be >= {MIN_RUNS}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Markdown report destination.",
    )
    parser.add_argument(
        "--thresholds-out",
        type=Path,
        required=True,
        help="perf-thresholds.json destination.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"Threshold = mean + sigma * stdev (default {DEFAULT_SIGMA}).",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help=(
            "Also measure LLMReranker.rerank and RagPipeline.run "
            "(full). Costs API tokens — requires ANTHROPIC_API_KEY. "
            "Omitted by default so smoke runs are free."
        ),
    )
    args = parser.parse_args(argv)

    if args.runs < MIN_RUNS:
        print(
            f"error: --runs must be >= {MIN_RUNS} (got {args.runs})",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    commit = git_sha(repo_root)
    env = environment_fingerprint()
    measured_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    try:
        raw = collect_raw_timings(args.runs, include_llm=args.include_llm)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    stats_by_metric = aggregate_results(raw, args.sigma)
    payload = build_thresholds_payload(
        measured_at=measured_at,
        commit=commit,
        runs=args.runs,
        sigma=args.sigma,
        env=env,
        stats_by_metric=stats_by_metric,
        include_llm=args.include_llm,
    )

    write_thresholds_json(args.thresholds_out, payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"wrote {args.thresholds_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
