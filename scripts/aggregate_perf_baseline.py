"""Combine K per-invocation perf JSONs into the v2 locked baseline.

Phase 5 deliverable from
``docs/specs/perf-baseline-multi-run/tasks.md`` M1.

Companion to ``scripts/measure_perf_baseline.py``. The measurement
script writes per-invocation JSONs (raw N-trial timings); this
aggregator reads K of them and emits the locked v2 ``perf-thresholds.json``
+ markdown report.

The K-invocation methodology separates two noise sources:

- **intra_run_stdev** — within-invocation noise (the N trials' stdev,
  averaged across K invocations). Reflects per-trial measurement
  jitter on a single runner.
- **inter_run_stdev** — between-invocation noise (the stdev of the
  K per-invocation means). Reflects runner SKU drift, sibling-job
  contention, etc. — the noise floor a single sequential run can't see.

The locked threshold becomes ``mean + sigma × inter_run_stdev``: the
inter-run dimension is the load-bearing noise floor for the per-PR
delta-check gate.

Schema is backward-compatible per requirements.md R2: ``mean``,
``stdev``, ``threshold`` keep their meaning (``stdev`` aliases
``inter_run_stdev``); new keys (``intra_run_stdev``,
``inter_run_stdev``, ``runs_per_invocation``, ``invocations``,
``methodology_version: 2``) are added beside.

Usage::

    python scripts/aggregate_perf_baseline.py \\
        --per-invocation inv0.json --per-invocation inv1.json ... \\
        --out docs/specs/downstream-validation/perf-baseline.md \\
        --thresholds-out \\
            docs/specs/downstream-validation/perf-thresholds.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Minimum K per requirements.md R3.
MIN_INVOCATIONS = 2

# Default sigma for the v2 lock. Per design.md §5, σ rolls back from
# 3.0 → 2.0 in the same PR that lands the first v2 lock; the new
# inter_run_stdev basis absorbs the noise the v1 σ=3.0 was padding.
DEFAULT_SIGMA = 2.0


def _load_per_invocation(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load + sanity-check K per-invocation JSON payloads."""
    if len(paths) < MIN_INVOCATIONS:
        raise ValueError(f"need at least {MIN_INVOCATIONS} per-invocation JSONs; got {len(paths)}")
    payloads: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"per-invocation JSON not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON in {p}: {exc}") from exc
        if data.get("methodology_version") != 2:
            raise ValueError(
                f"{p}: expected methodology_version == 2; got "
                f"{data.get('methodology_version')!r}"
            )
        for required in (
            "invocation_index",
            "invocations",
            "runs_per_invocation",
            "raw_timings",
        ):
            if required not in data:
                raise ValueError(f"{p}: missing required key {required!r}")
        payloads.append(data)
    return payloads


def _validate_consistent(payloads: Sequence[dict[str, Any]]) -> None:
    """Cross-check K invocations agree on commit / config."""
    first = payloads[0]
    pinned = ("commit", "runs_per_invocation", "include_llm")
    for p in payloads[1:]:
        for key in pinned:
            if p.get(key) != first.get(key):
                raise ValueError(
                    f"inconsistent {key!r} across invocations: "
                    f"{first.get(key)!r} vs {p.get(key)!r}"
                )
    indices = sorted(p["invocation_index"] for p in payloads)
    expected = list(range(len(payloads)))
    if indices != expected:
        raise ValueError(
            f"invocation_index set {indices} does not match contiguous "
            f"range {expected}; missing or duplicate invocations"
        )


def aggregate_metric(per_invocation_values: list[list[float]], sigma: float) -> dict[str, Any]:
    """Compute v2 stats for one metric across K invocations.

    Args:
        per_invocation_values: K lists, each holding one invocation's N
            raw timing samples for this metric.
        sigma: Threshold multiplier (default 2.0 per spec).

    Returns:
        Dict with: ``mean``, ``intra_run_stdev``, ``inter_run_stdev``,
        ``stdev`` (alias of inter for backward-compat), ``threshold``,
        ``invocations``, ``runs_per_invocation``.
    """
    per_inv_means: list[float] = []
    per_inv_stdevs: list[float] = []
    for values in per_invocation_values:
        if not values:
            continue
        per_inv_means.append(statistics.mean(values))
        per_inv_stdevs.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    if len(per_inv_means) < 2:
        raise ValueError(
            f"need at least 2 invocations with non-empty samples; " f"got {len(per_inv_means)}"
        )
    mean = statistics.mean(per_inv_means)
    inter = statistics.stdev(per_inv_means)
    intra = statistics.mean(per_inv_stdevs)
    threshold = mean + sigma * inter
    return {
        "mean": round(mean, 6),
        "intra_run_stdev": round(intra, 6),
        "inter_run_stdev": round(inter, 6),
        "stdev": round(inter, 6),  # backward-compat alias
        "threshold": round(threshold, 6),
        "invocations": len(per_invocation_values),
        "runs_per_invocation": (len(per_invocation_values[0]) if per_invocation_values else 0),
    }


def aggregate_all(payloads: Sequence[dict[str, Any]], sigma: float) -> dict[str, dict[str, Any]]:
    """Aggregate all metrics across K invocations."""
    # Collect all metric keys across invocations (union — a metric might
    # be missing in one invocation if e.g. include_llm differed, though
    # _validate_consistent guards against that).
    all_metrics: set[str] = set()
    for p in payloads:
        all_metrics.update(p.get("raw_timings", {}).keys())
    out: dict[str, dict[str, Any]] = {}
    for metric in sorted(all_metrics):
        per_inv_values = [list(p["raw_timings"].get(metric, [])) for p in payloads]
        # Skip a metric entirely if any invocation lacks it — we don't
        # silently aggregate over a partial set.
        if any(not v for v in per_inv_values):
            continue
        out[metric] = aggregate_metric(per_inv_values, sigma)
    return out


def build_payload(
    *,
    payloads: Sequence[dict[str, Any]],
    sigma: float,
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first = payloads[0]
    return {
        "methodology_version": 2,
        "measured_at": (
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        ),
        "commit": first.get("commit", "unknown"),
        "sigma": sigma,
        "invocations": len(payloads),
        "runs_per_invocation": first.get("runs_per_invocation"),
        "include_llm": first.get("include_llm", False),
        "environment": first.get("environment", {}),
        "metrics": metrics,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """v2 markdown report. Mirrors the v1 shape but with the dual-noise table."""
    lines: list[str] = []
    lines.append("# Perf baseline (Phase 5 — multi-run methodology v2)")
    lines.append("")
    lines.append(f"- Methodology version: `{payload['methodology_version']}`")
    lines.append(f"- Measured at: `{payload['measured_at']}`")
    lines.append(f"- Commit: `{payload['commit']}`")
    lines.append(f"- Invocations: `{payload['invocations']}`")
    lines.append(f"- Runs per invocation: `{payload['runs_per_invocation']}`")
    lines.append(f"- Sigma: `{payload['sigma']}` (threshold = mean + σ × inter_run_stdev)")
    lines.append(f"- include_llm: `{payload['include_llm']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | mean | intra_run_stdev | inter_run_stdev | threshold |")
    lines.append("|--------|------|-----------------|-----------------|-----------|")
    for metric, stats in sorted(payload["metrics"].items()):
        lines.append(
            f"| `{metric}` | {stats['mean']:.6f} | "
            f"{stats['intra_run_stdev']:.6f} | "
            f"{stats['inter_run_stdev']:.6f} | "
            f"{stats['threshold']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aggregate_perf_baseline",
        description=("Combine K per-invocation perf JSONs into the v2 locked baseline."),
    )
    parser.add_argument(
        "--per-invocation",
        type=Path,
        action="append",
        required=True,
        help="Path to a per-invocation JSON. Pass --per-invocation K times.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="v2 markdown report destination.",
    )
    parser.add_argument(
        "--thresholds-out",
        type=Path,
        required=True,
        help="v2 perf-thresholds.json destination.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"Threshold multiplier (default {DEFAULT_SIGMA}).",
    )
    args = parser.parse_args(argv)

    try:
        payloads = _load_per_invocation(args.per_invocation)
        _validate_consistent(payloads)
        metrics = aggregate_all(payloads, args.sigma)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not metrics:
        print("error: no metrics aggregated (all per-invocation files empty?)", file=sys.stderr)
        return 2

    payload = build_payload(payloads=payloads, sigma=args.sigma, metrics=metrics)

    args.thresholds_out.parent.mkdir(parents=True, exist_ok=True)
    args.thresholds_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"wrote {args.thresholds_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
