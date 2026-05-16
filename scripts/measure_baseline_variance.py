"""Measure attune-rag-benchmark variance to set CI thresholds.

Runs the benchmark N times back-to-back on an unchanged HEAD,
parses the per-run aggregate metrics from stdout, and emits:

- A locked markdown report  (``--out``)
- A machine-readable thresholds file  (``--thresholds-out``)

Threshold = ``mean − sigma * stdev`` per metric. Default
``sigma=2.0`` per Decision 1 of the v1.0 roadmap (see
``docs/specs/ROADMAP-v1.md``).

Re-run whenever a judge-affecting change lands (Phase 2's
``--thinking`` default flip is the first expected user).

Usage::

    python scripts/measure_baseline_variance.py \\
        --runs 20 \\
        --out docs/specs/release-quality-baseline/baseline-1.md \\
        --thresholds-out \\
            docs/specs/release-quality-baseline/thresholds.json

Cheap dry-run that skips the LLM judge (retrieval metrics only,
no API spend)::

    python scripts/measure_baseline_variance.py \\
        --runs 10 --skip-faithfulness \\
        --out /tmp/dry.md --thresholds-out /tmp/dry.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# Stdout patterns the benchmark prints. Keep these in sync with
# ``_print_summary`` and ``_print_faithfulness`` in
# ``src/attune_rag/benchmark.py``.
_PERCENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "precision_at_1": re.compile(r"Precision@1:\s+([\d.]+)%"),
    "recall_at_3": re.compile(r"Recall@3:\s+([\d.]+)%"),
}
_FLOAT_PATTERNS: dict[str, re.Pattern[str]] = {
    # Tolerant of an optional label suffix: the benchmark prints
    # "Mean faithfulness (legacy):" when called with just
    # --with-faithfulness, and "Mean faithfulness (thinking off):"
    # etc. under --compare-thinking. We only want the single
    # canonical default pass — match the first occurrence.
    "mean_faithfulness": re.compile(r"Mean faithfulness(?:\s*\([^)]*\))?:\s+([\d.]+)"),
}

MIN_RUNS = 10


def parse_metrics(stdout: str) -> dict[str, float]:
    """Extract aggregate metrics from one benchmark invocation's stdout.

    Percent-formatted metrics are converted to fractions (e.g.
    ``"82.50%"`` → ``0.825``) so the thresholds file is uniform.
    Missing metrics are simply absent from the returned dict —
    the caller decides whether that's an error.
    """
    out: dict[str, float] = {}
    for key, pat in _PERCENT_PATTERNS.items():
        m = pat.search(stdout)
        if m is not None:
            out[key] = float(m.group(1)) / 100.0
    for key, pat in _FLOAT_PATTERNS.items():
        m = pat.search(stdout)
        if m is not None:
            out[key] = float(m.group(1))
    return out


def compute_stats(values: list[float], sigma: float) -> dict[str, float | list[float]]:
    """Mean, stdev, and ``mean - sigma*stdev`` threshold for one metric."""
    if not values:
        raise ValueError("compute_stats requires at least one value")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "stdev": round(stdev, 4),
        "threshold": round(mean - sigma * stdev, 4),
        "raw": [round(v, 4) for v in values],
    }


def git_sha(repo_root: Path) -> str:
    """Return the short HEAD SHA, or ``"unknown"`` if git fails."""
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


def queries_sha256(path: Path | None) -> str | None:
    """Hash the queries file so a later silent edit fails the gate."""
    if path is None or not path.exists():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path | None, repo_root: Path) -> str | None:
    """Render ``path`` repo-relative when possible, else absolute.

    Keeps the locked baseline portable across machines and
    worktrees. Returns ``None`` if ``path`` is ``None``.
    """
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _resolve_default_queries_path() -> Path | None:
    """Mirror the benchmark's bundled queries-path default.

    Imported lazily so the variance script keeps working even if
    the benchmark module fails to import (e.g. missing optional
    deps). Returns ``None`` on import failure — the locked baseline
    will then record ``queries_path: null`` and the user is on the
    hook for re-running with an explicit ``--queries``.
    """
    try:
        from attune_rag.benchmark import _default_queries_path

        return _default_queries_path()
    except Exception:
        return None


def run_benchmark_once(
    queries: Path | None,
    with_faithfulness: bool,
    *,
    runner=None,
) -> dict[str, float]:
    """Invoke ``python -m attune_rag.benchmark`` once and parse metrics.

    ``runner`` is resolved at call time (defaulting to
    ``subprocess.run``) so test monkey-patches on
    ``subprocess.run`` are respected without re-importing.

    Raises :class:`RuntimeError` on non-zero exit so the caller can
    abort the whole measurement (we don't want a partial baseline).
    """
    if runner is None:
        runner = subprocess.run
    cmd: list[str] = [sys.executable, "-m", "attune_rag.benchmark"]
    if queries is not None:
        cmd.extend(["--queries", str(queries)])
    if with_faithfulness:
        cmd.append("--with-faithfulness")

    proc = runner(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-500:]
        raise RuntimeError(f"benchmark failed (exit {proc.returncode}): {tail}")

    metrics = parse_metrics(proc.stdout)
    expected: Iterable[str] = (
        ("precision_at_1", "recall_at_3", "mean_faithfulness")
        if with_faithfulness
        else ("precision_at_1", "recall_at_3")
    )
    missing = [k for k in expected if k not in metrics]
    if missing:
        raise RuntimeError(
            f"benchmark stdout missing metrics: {missing}; "
            f"last 300 chars: {proc.stdout[-300:]!r}"
        )
    return metrics


def write_thresholds_json(
    path: Path,
    *,
    commit: str,
    queries_path: Path | None,
    queries_hash: str | None,
    runs: int,
    sigma: float,
    stats_by_metric: dict[str, dict[str, float | list[float]]],
) -> None:
    payload = {
        "measured_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "commit": commit,
        "queries_path": str(queries_path) if queries_path else None,
        "queries_sha256": queries_hash,
        "runs": runs,
        "sigma": sigma,
        "metrics": {
            k: {kk: v[kk] for kk in ("mean", "stdev", "threshold")}
            for k, v in stats_by_metric.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def render_markdown(
    *,
    commit: str,
    queries_path: Path | None,
    queries_hash: str | None,
    runs: int,
    sigma: float,
    stats_by_metric: dict[str, dict[str, float | list[float]]],
    measured_at: str,
) -> str:
    """Format the locked baseline report. Sorted-key tables for diff stability."""
    lines: list[str] = []
    lines.append("# Baseline measurement")
    lines.append("")
    lines.append("> Locked record of one noise-floor measurement run.")
    lines.append("> Generated by `scripts/measure_baseline_variance.py`.")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Measured at | `{measured_at}` |")
    lines.append(f"| Commit | `{commit}` |")
    lines.append(f"| Queries path | `{queries_path}` |")
    lines.append(f"| Queries SHA-256 | `{queries_hash}` |")
    lines.append(f"| Runs (N) | {runs} |")
    lines.append(f"| Sigma | {sigma} |")
    lines.append("")
    lines.append("## Per-metric stats")
    lines.append("")
    lines.append("| Metric | Mean | Stdev | Threshold (mean − σ·stdev) |")
    lines.append("|---|---:|---:|---:|")
    for k in sorted(stats_by_metric):
        s = stats_by_metric[k]
        lines.append(f"| `{k}` | {s['mean']} | {s['stdev']} | {s['threshold']} |")
    lines.append("")
    lines.append("## Raw runs")
    lines.append("")
    for k in sorted(stats_by_metric):
        raw = stats_by_metric[k]["raw"]
        lines.append(f"### `{k}`")
        lines.append("")
        lines.append("```")
        for i, v in enumerate(raw, start=1):
            lines.append(f"run {i:>2}: {v}")
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_baseline_variance",
        description=(
            "Run attune-rag-benchmark N times on an unchanged HEAD and "
            "compute per-metric mean / stdev / threshold "
            "(threshold = mean − sigma * stdev)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=True,
        help=f"Number of benchmark runs (must be >= {MIN_RUNS}).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries.yaml (default: benchmark's bundled default).",
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
        help="thresholds.json destination.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Threshold = mean − sigma * stdev (default 2.0).",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help=(
            "Skip the LLM judge (retrieval metrics only, no API "
            "spend). Useful for validating the script before paying "
            "for a real run."
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
    # When --queries is not supplied the benchmark falls back to
    # its bundled default. Resolve that path here so the locked
    # baseline records *which* query set produced the numbers,
    # enabling CI to fail loud on a silent edit later.
    queries_path = args.queries or _resolve_default_queries_path()
    queries_hash = queries_sha256(queries_path)
    queries_path_recorded = _portable_path(queries_path, repo_root)

    metric_values: dict[str, list[float]] = {}
    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs}...", file=sys.stderr, flush=True)
        try:
            results = run_benchmark_once(
                args.queries,
                with_faithfulness=not args.skip_faithfulness,
            )
        except RuntimeError as e:
            print(f"\nerror: {e}", file=sys.stderr)
            return 1
        for k, v in results.items():
            metric_values.setdefault(k, []).append(v)

    stats_by_metric = {k: compute_stats(v, args.sigma) for k, v in metric_values.items()}
    measured_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    write_thresholds_json(
        args.thresholds_out,
        commit=commit,
        queries_path=queries_path_recorded,
        queries_hash=queries_hash,
        runs=args.runs,
        sigma=args.sigma,
        stats_by_metric=stats_by_metric,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render_markdown(
            commit=commit,
            queries_path=queries_path_recorded,
            queries_hash=queries_hash,
            runs=args.runs,
            sigma=args.sigma,
            stats_by_metric=stats_by_metric,
            measured_at=measured_at,
        )
    )
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"wrote {args.thresholds_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
