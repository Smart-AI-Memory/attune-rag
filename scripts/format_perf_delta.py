"""Format a markdown delta comment comparing two perf-baseline JSONs.

Companion to ``scripts/measure_perf_baseline.py``. Reads a current
(per-PR) measurement and the locked baseline, emits a markdown
comment showing each metric's delta and a status verdict per axis.

Exit codes
----------
0 — all measured metrics are within the locked threshold (or
    baseline file is missing → advisory pending verdict).
1 — at least one metric exceeded its baseline threshold.
2 — validation error (current file missing or malformed).

Verdict per metric is **current_mean vs baseline.threshold**
(baseline.threshold = mean + sigma·stdev, the upper-bound gate
from the perf-baseline locking script). Metrics with no baseline
counterpart are reported as "new" — not a regression, but worth
the maintainer's eye.

Stable HTML marker (``<!-- attune-rag-perf-gate -->``) lets the
workflow grep + PATCH the same comment instead of stacking new
ones on every push.

Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

COMMENT_MARKER = "<!-- attune-rag-perf-gate -->"


@dataclass(frozen=True)
class MetricComparison:
    metric: str  # e.g. "keyword_retriever_retrieve.cpu"
    baseline_mean: float | None
    baseline_threshold: float | None  # mean + sigma·stdev
    current_mean: float
    status: str  # "ok" | "regression" | "new"

    @property
    def delta_pct(self) -> float | None:
        """Percent delta vs baseline mean, or None if no baseline."""
        if self.baseline_mean is None or self.baseline_mean == 0.0:
            return None
        return (self.current_mean - self.baseline_mean) / self.baseline_mean * 100.0


def compare(
    baseline_metrics: dict[str, dict[str, float]],
    current_metrics: dict[str, dict[str, float]],
) -> list[MetricComparison]:
    """Pair-up metrics by key, classify each.

    Iteration order: alphabetical by metric key, so the rendered
    comment is deterministic.
    """
    out: list[MetricComparison] = []
    all_keys = sorted(set(baseline_metrics) | set(current_metrics))
    for key in all_keys:
        b = baseline_metrics.get(key)
        c = current_metrics.get(key)
        if c is None:
            # In baseline but not current — happens if a benchmark
            # was removed. Skip silently; it isn't a regression and
            # tracking "removed benchmark" is outside this script's
            # scope.
            continue
        current_mean = float(c["mean"])
        if b is None:
            out.append(
                MetricComparison(
                    metric=key,
                    baseline_mean=None,
                    baseline_threshold=None,
                    current_mean=current_mean,
                    status="new",
                )
            )
            continue
        baseline_mean = float(b["mean"])
        baseline_threshold = float(b["threshold"])
        status = "regression" if current_mean > baseline_threshold else "ok"
        out.append(
            MetricComparison(
                metric=key,
                baseline_mean=baseline_mean,
                baseline_threshold=baseline_threshold,
                current_mean=current_mean,
                status=status,
            )
        )
    return out


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def render_baseline_pending_comment() -> str:
    """Emitted when no baseline file is on disk yet (pre-W0.4)."""
    lines = [
        COMMENT_MARKER,
        "## Perf delta — advisory",
        "",
        "Baseline `perf-thresholds.json` not yet committed (Phase 4 W0.4 "
        "still pending). Per-PR perf-delta gating is **inactive** until "
        "the initial baseline is locked. Run the `perf` workflow's "
        "`lock-baseline` dispatch to produce it.",
        "",
        COMMENT_MARKER,
    ]
    return "\n".join(lines) + "\n"


def render_comparison_comment(
    comparisons: list[MetricComparison],
    *,
    advisory: bool,
    gated_metrics: frozenset[str] = frozenset(),
) -> str:
    """Render the per-PR delta comment.

    ``advisory=True`` softens the global phrasing (Phase 4 W1–W2).
    ``gated_metrics`` lists metric keys whose regressions are
    blocking from W3.1 onward — those rows get a ⛔ icon and the
    intro names them explicitly. Non-gated regressions stay ⚠️.
    """
    has_regression = any(c.status == "regression" for c in comparisons)
    has_gated_regression = any(
        c.status == "regression" and c.metric in gated_metrics for c in comparisons
    )
    has_new = any(c.status == "new" for c in comparisons)

    if has_gated_regression:
        title = "Perf delta — REGRESSION (gating)"
    elif has_regression:
        if gated_metrics or advisory:
            # gated-metrics set but no gated regression → advisory
            # regressions only, gate stays green. Match `--advisory`
            # phrasing for this case.
            title = "Perf delta — possible regression"
        else:
            title = "Perf delta — REGRESSION (gating)"
    elif has_new:
        title = "Perf delta — new metrics (no baseline)"
    else:
        title = "Perf delta — within baseline"

    lines: list[str] = [
        COMMENT_MARKER,
        f"## {title}",
        "",
    ]
    if gated_metrics:
        gated_list = ", ".join(f"`{m}`" for m in sorted(gated_metrics))
        lines.extend(
            [
                f"Blocking on regression in: {gated_list}. Other metrics "
                "are advisory and don't block merge.",
                "",
            ]
        )
    elif advisory:
        lines.extend(
            [
                "Advisory only (Phase 4 W1–W2). Regressions don't block "
                "merge yet; the gate promotes to blocking on CPU-time "
                "axis in W3.1.",
                "",
            ]
        )

    lines.extend(
        [
            "| Metric | Baseline mean (s) | Current mean (s) | Δ | Threshold (s) | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )

    # Stable order: regressions first, then new, then ok — within
    # each group alphabetical by metric.
    rank = {"regression": 0, "new": 1, "ok": 2}
    ordered = sorted(comparisons, key=lambda c: (rank[c.status], c.metric))
    for c in ordered:
        if c.status == "regression":
            status_icon = "⛔ blocking" if c.metric in gated_metrics else "⚠️ over threshold"
        else:
            status_icon = {"new": "🆕 new", "ok": "ok"}[c.status]
        lines.append(
            f"| `{c.metric}` | {_format_seconds(c.baseline_mean)} | "
            f"{_format_seconds(c.current_mean)} | {_format_pct(c.delta_pct)} | "
            f"{_format_seconds(c.baseline_threshold)} | {status_icon} |"
        )

    if has_gated_regression or (has_regression and not advisory):
        lines.extend(
            [
                "",
                "**What to do**",
                "",
                "- Profile the change. If the regression is real, fix it " "before merging.",
                "- If hardware drift is suspected (Python upgrade, runner "
                "SKU change), re-lock the baseline via the `perf` "
                "workflow's `lock-baseline` dispatch and commit the new "
                "files in this PR.",
            ]
        )
    elif has_new:
        lines.extend(
            [
                "",
                "_New metrics have no baseline yet. They'll be tracked "
                "after the next baseline re-lock._",
            ]
        )

    lines.append("")
    lines.append(COMMENT_MARKER)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="format_perf_delta",
        description=(
            "Compare a per-PR perf measurement against the locked "
            "baseline and emit a markdown PR comment. Exits 0 "
            "(clean / baseline pending), 1 (regression), or 2 "
            "(validation error)."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to the locked perf-thresholds.json.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to the per-PR measurement (also a perf-thresholds.json shape).",
    )
    parser.add_argument(
        "--comment-out",
        type=Path,
        required=True,
        help="Where to write the markdown comment body.",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help=(
            "Soften the regression phrasing — Phase 4 W1–W2 mode. "
            "Exit code is unaffected; only the comment text changes. "
            "Has no effect when --gate-metric is set."
        ),
    )
    parser.add_argument(
        "--gate-metric",
        action="append",
        default=[],
        metavar="METRIC",
        help=(
            "Repeatable. Restrict the exit-1 condition to regressions "
            "in the named metric(s) only (e.g. "
            "'--gate-metric keyword_retriever_retrieve.cpu'). "
            "Phase 4 W3.1 promotes the CPU-time axis of "
            "KeywordRetriever.retrieve and RagPipeline.run to "
            "blocking; everything else stays advisory."
        ),
    )
    args = parser.parse_args(argv)

    if not args.current.exists():
        print(f"error: current measurement file missing at {args.current}", file=sys.stderr)
        return 2

    try:
        current = json.loads(args.current.read_text())
    except json.JSONDecodeError as e:
        print(f"error: {args.current} is not valid JSON: {e}", file=sys.stderr)
        return 2

    args.comment_out.parent.mkdir(parents=True, exist_ok=True)

    if not args.baseline.exists():
        # Baseline file not yet locked (W0.4 pending). Emit an
        # advisory note and exit 0; the workflow stays green.
        args.comment_out.write_text(render_baseline_pending_comment(), encoding="utf-8")
        return 0

    try:
        baseline = json.loads(args.baseline.read_text())
    except json.JSONDecodeError as e:
        print(f"error: {args.baseline} is not valid JSON: {e}", file=sys.stderr)
        return 2

    comparisons = compare(
        baseline.get("metrics", {}) or {},
        current.get("metrics", {}) or {},
    )

    gated_metrics = frozenset(args.gate_metric)
    args.comment_out.write_text(
        render_comparison_comment(
            comparisons,
            advisory=args.advisory,
            gated_metrics=gated_metrics,
        ),
        encoding="utf-8",
    )

    if gated_metrics:
        has_gate_breach = any(
            c.status == "regression" and c.metric in gated_metrics for c in comparisons
        )
        return 1 if has_gate_breach else 0

    has_regression = any(c.status == "regression" for c in comparisons)
    return 1 if has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
