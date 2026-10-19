"""Compare a benchmark JSON dump against locked thresholds.

CI runs ``attune-rag-benchmark --with-faithfulness --json out.json``
then invokes this script to gate the PR:

    python scripts/check_thresholds.py \\
        --dump out.json \\
        --thresholds docs/specs/release-quality-baseline/thresholds.json

Exit codes
----------
0 — all metrics meet or exceed their threshold.
1 — at least one metric is below threshold (regression).
2 — validation error: missing metric, missing file, queries_sha256
    mismatch, or malformed input.

Stderr lists each failure as one line per metric:

    FAIL precision_at_1: measured=0.9250 threshold=0.9500 delta=-0.0250

Pure stdlib. No LLM dependency. Safe to run in any CI image.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

# Where each metric lives in the benchmark's --json dump and how
# that key maps to the locked thresholds.json. The dump uses the
# literal key ``recall_at_k`` for any k; we translate to
# ``recall_at_<k>`` so the locked baseline keeps the k explicit.
#
# When --compare-thinking is in play the dump has
# ``faithfulness_thinking_off`` / ``faithfulness_thinking_on``
# instead of ``faithfulness_legacy``. Phase 1 only gates the
# default single-pass run; Phase 2 will revisit if --thinking
# defaults flip.


@dataclass(frozen=True)
class MetricFailure:
    metric: str
    measured: float
    threshold: float

    @property
    def delta(self) -> float:
        return self.measured - self.threshold


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} at {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{label} at {path} must be a JSON object")
    return data


def _quality_value(value: Any, label: str) -> float:
    """Accept finite JSON numbers in the quality metric domain, without coercion."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    # Check the range first so an arbitrarily large JSON integer cannot overflow
    # when math.isfinite converts it to a float.
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    return float(value)


def extract_metrics(dump: dict[str, Any]) -> dict[str, float]:
    """Pull aggregate metrics out of a benchmark JSON dump.

    Returns a flat ``{metric_name: value}`` dict using the same
    metric names as ``thresholds.json``. Raises :class:`KeyError`
    if any expected metric is missing, or :class:`ValueError` for
    invalid types or values. The caller turns either into an exit-2
    validation error so a malformed dump can't silently pass the gate.
    """
    if not isinstance(dump, Mapping):
        raise ValueError("dump must be an object")
    retrieval = dump.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise KeyError("dump missing top-level 'retrieval' object")

    k = retrieval.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("dump.retrieval 'k' must be a positive integer")

    out: dict[str, float] = {}
    if "precision_at_1" not in retrieval:
        raise KeyError("dump.retrieval missing 'precision_at_1'")
    out["precision_at_1"] = _quality_value(
        retrieval["precision_at_1"], "dump.retrieval.precision_at_1"
    )

    if "recall_at_k" not in retrieval:
        raise KeyError("dump.retrieval missing 'recall_at_k'")
    out[f"recall_at_{k}"] = _quality_value(retrieval["recall_at_k"], "dump.retrieval.recall_at_k")

    if "faithfulness_legacy" in dump:
        faith = dump["faithfulness_legacy"]
        if not isinstance(faith, Mapping):
            raise ValueError("dump.faithfulness_legacy must be an object")
        if "mean_faithfulness" not in faith:
            raise KeyError("dump.faithfulness_legacy missing 'mean_faithfulness'")
        out["mean_faithfulness"] = _quality_value(
            faith["mean_faithfulness"], "dump.faithfulness_legacy.mean_faithfulness"
        )
    # When the dump is retrieval-only (no --with-faithfulness) we
    # simply omit mean_faithfulness. The caller decides whether
    # that's a problem for a given thresholds.json.

    return out


def check(
    dump: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    verify_queries_sha256: bool = True,
    skip_metrics: frozenset[str] = frozenset(),
) -> tuple[list[MetricFailure], list[str]]:
    """Return ``(failures, validation_errors)``.

    A non-empty ``failures`` list means exit code 1 (regression).
    A non-empty ``validation_errors`` list means exit code 2
    (something's off about the inputs themselves — missing metric,
    queries mismatch, etc.); on exit 2 the failures list is moot.
    """
    validation: list[str] = []
    failures: list[MetricFailure] = []

    if not isinstance(thresholds, Mapping):
        return failures, ["thresholds must be an object"]
    try:
        measured = extract_metrics(dump)
    except (KeyError, ValueError) as e:
        # KeyError.__str__ repr-wraps its arg, which double-quotes
        # the message when the arg contains single quotes. Pull the
        # original message string directly to keep stderr clean.
        validation.append(e.args[0] if e.args else str(e))
        return failures, validation

    threshold_block = thresholds.get("metrics")
    if not isinstance(threshold_block, Mapping) or not threshold_block:
        validation.append("thresholds.json 'metrics' must be a non-empty object")
        return failures, validation

    if verify_queries_sha256:
        expected_sha = thresholds.get("queries_sha256")
        if expected_sha is not None:
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in expected_sha)
            ):
                return failures, ["thresholds.queries_sha256 must be a 64-character hex digest"]
            actual_path = dump.get("queries_path")
            if not isinstance(actual_path, str) or not actual_path.strip():
                return failures, ["dump.queries_path must name the queries file to verify sha256"]
            try:
                actual_sha = sha256(Path(actual_path).read_bytes()).hexdigest()
            except (OSError, ValueError) as e:
                validation.append(
                    f"could not read queries file {actual_path!r} to verify sha256: {e}"
                )
                return failures, validation
            if actual_sha != expected_sha.lower():
                validation.append(
                    "queries.yaml SHA-256 mismatch: dump used "
                    f"{actual_sha[:16]}…, thresholds expect "
                    f"{expected_sha[:16]}…. Re-measure with "
                    "scripts/measure_baseline_variance.py before "
                    "merging."
                )
                return failures, validation

    # Every threshold listed in the locked baseline must have a
    # measured counterpart, unless explicitly skipped (e.g. the CI
    # workflow runs retrieval-only on PRs that don't touch
    # faithfulness-affecting paths — see M3.3). Missing →
    # validation error (not a quiet pass).
    for metric_name, spec in threshold_block.items():
        if not isinstance(metric_name, str) or not metric_name:
            validation.append("thresholds.metrics keys must be non-empty metric names")
            continue
        if not isinstance(spec, Mapping):
            validation.append(f"thresholds.metrics.{metric_name} must be an object")
            continue
        if "threshold" not in spec:
            validation.append(f"thresholds.metrics.{metric_name} missing 'threshold'")
            continue
        try:
            threshold_val = _quality_value(
                spec["threshold"], f"thresholds.metrics.{metric_name}.threshold"
            )
        except ValueError as e:
            validation.append(str(e))
            continue
        if metric_name in skip_metrics:
            continue
        if metric_name not in measured:
            validation.append(
                f"dump missing measured value for '{metric_name}' (thresholds expect it)"
            )
            continue
        if measured[metric_name] < threshold_val:
            failures.append(
                MetricFailure(
                    metric=metric_name,
                    measured=measured[metric_name],
                    threshold=threshold_val,
                )
            )

    return failures, validation


# A stable HTML-comment marker so the CI workflow can find and
# edit the same comment instead of appending a new one each push.
COMMENT_MARKER = "<!-- attune-rag-quality-gate -->"


def format_failure_comment(failures: list[MetricFailure]) -> str:
    """Render a markdown PR-comment body for a non-empty failure list.

    The body is deterministic — no timestamps, no hostnames — so a
    golden test can pin it. Failures are sorted by metric name so
    two equivalent runs produce byte-identical comments. The
    leading and trailing :data:`COMMENT_MARKER` lets the workflow
    grep for and update the existing comment instead of stacking
    new ones on every push.

    Raises :class:`ValueError` on an empty list — formatting a
    "0 failures" comment is a caller bug, not a green-PR signal.
    """
    if not failures:
        raise ValueError(
            "format_failure_comment called with no failures; "
            "callers should skip commenting on a green run"
        )

    ordered = sorted(failures, key=lambda f: f.metric)
    lines: list[str] = [
        COMMENT_MARKER,
        "## Quality gate failed",
        "",
        "This PR's benchmark run did not meet the locked "
        "thresholds at "
        "`docs/specs/release-quality-baseline/thresholds.json`.",
        "",
        "| Metric | Measured | Threshold | Delta |",
        "|---|---:|---:|---:|",
    ]
    for f in ordered:
        lines.append(f"| `{f.metric}` | {f.measured:.4f} | {f.threshold:.4f} | {f.delta:+.4f} |")
    lines.extend(
        [
            "",
            "### What to do",
            "",
            "- If the regression is real, fix it before merging.",
            "- If this PR intentionally changes the corpus, the "
            "judge, or the prompts, re-measure the baseline:",
            "  ```",
            "  python scripts/measure_baseline_variance.py --runs 20 \\",
            "      --out docs/specs/release-quality-baseline/baseline-N.md \\",
            "      --thresholds-out docs/specs/release-quality-baseline/thresholds.json",
            "  ```",
            "  and commit the updated baseline in this same PR "
            "with `[baseline-update]` in the title.",
            "",
            COMMENT_MARKER,
        ]
    )
    return "\n".join(lines) + "\n"


def _print_failures(failures: list[MetricFailure]) -> None:
    for f in failures:
        print(
            f"FAIL {f.metric}: measured={f.measured:.4f} "
            f"threshold={f.threshold:.4f} delta={f.delta:+.4f}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_thresholds",
        description=(
            "Compare an attune-rag-benchmark JSON dump against a "
            "locked thresholds.json. Exits 0 (pass), 1 (regression), "
            "or 2 (validation error)."
        ),
    )
    parser.add_argument(
        "--dump",
        type=Path,
        required=True,
        help="Path to the benchmark JSON dump (from "
        "`attune-rag-benchmark --with-faithfulness --json PATH`).",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        required=True,
        help="Path to the locked thresholds.json.",
    )
    parser.add_argument(
        "--skip-queries-sha-check",
        action="store_true",
        help=(
            "Don't compare the queries file's SHA-256. Use only "
            "when CI runs the benchmark against a non-default "
            "queries set on purpose."
        ),
    )
    parser.add_argument(
        "--comment-out",
        type=Path,
        default=None,
        help=(
            "On regression (exit 1), also write a markdown "
            "PR-comment body to this path. The workflow then "
            "invokes `gh pr comment --body-file ...`. Not written "
            "on green runs or validation errors."
        ),
    )
    parser.add_argument(
        "--skip-metric",
        action="append",
        default=[],
        metavar="METRIC",
        help=(
            "Skip gating this metric even if it's listed in "
            "thresholds.json. Repeatable. Used by the CI workflow "
            "to run retrieval-only when faithfulness gating is "
            "either off-budget for the PR or the API key is not "
            "configured."
        ),
    )
    args = parser.parse_args(argv)

    try:
        dump = _load_json(args.dump, "dump")
        thresholds = _load_json(args.thresholds, "thresholds")
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    failures, validation = check(
        dump,
        thresholds,
        verify_queries_sha256=not args.skip_queries_sha_check,
        skip_metrics=frozenset(args.skip_metric),
    )
    if validation:
        for msg in validation:
            print(f"error: {msg}", file=sys.stderr)
        return 2
    if failures:
        _print_failures(failures)
        if args.comment_out is not None:
            args.comment_out.parent.mkdir(parents=True, exist_ok=True)
            args.comment_out.write_text(format_failure_comment(failures), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
