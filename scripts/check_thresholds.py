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
import sys
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
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} at {path} is not valid JSON: {e}") from e


def extract_metrics(dump: dict[str, Any]) -> dict[str, float]:
    """Pull aggregate metrics out of a benchmark JSON dump.

    Returns a flat ``{metric_name: value}`` dict using the same
    metric names as ``thresholds.json``. Raises :class:`KeyError`
    if any expected metric is missing — the caller turns that into
    an exit-2 validation error so a malformed dump can't silently
    pass the gate.
    """
    retrieval = dump.get("retrieval")
    if not isinstance(retrieval, dict):
        raise KeyError("dump missing top-level 'retrieval' object")

    k = retrieval.get("k")
    if not isinstance(k, int):
        raise KeyError("dump.retrieval missing integer 'k'")

    out: dict[str, float] = {}
    if "precision_at_1" not in retrieval:
        raise KeyError("dump.retrieval missing 'precision_at_1'")
    out["precision_at_1"] = float(retrieval["precision_at_1"])

    if "recall_at_k" not in retrieval:
        raise KeyError("dump.retrieval missing 'recall_at_k'")
    out[f"recall_at_{k}"] = float(retrieval["recall_at_k"])

    faith = dump.get("faithfulness_legacy")
    if isinstance(faith, dict) and "mean_faithfulness" in faith:
        out["mean_faithfulness"] = float(faith["mean_faithfulness"])
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

    try:
        measured = extract_metrics(dump)
    except KeyError as e:
        # KeyError.__str__ repr-wraps its arg, which double-quotes
        # the message when the arg contains single quotes. Pull the
        # original message string directly to keep stderr clean.
        validation.append(e.args[0] if e.args else str(e))
        return failures, validation

    threshold_block = thresholds.get("metrics") or {}
    if not threshold_block:
        validation.append("thresholds.json has no 'metrics' block")
        return failures, validation

    if verify_queries_sha256:
        expected_sha = thresholds.get("queries_sha256")
        actual_path = dump.get("queries_path")
        if expected_sha and actual_path:
            try:
                actual_sha = sha256(Path(actual_path).read_bytes()).hexdigest()
            except OSError as e:
                validation.append(
                    f"could not read queries file {actual_path!r} " f"to verify sha256: {e}"
                )
                return failures, validation
            if actual_sha != expected_sha:
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
        if metric_name in skip_metrics:
            continue
        threshold_val = spec.get("threshold")
        if threshold_val is None:
            validation.append(f"thresholds.metrics.{metric_name} missing 'threshold'")
            continue
        if metric_name not in measured:
            validation.append(
                f"dump missing measured value for '{metric_name}' " f"(thresholds expect it)"
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
        lines.append(
            f"| `{f.metric}` | {f.measured:.4f} | " f"{f.threshold:.4f} | {f.delta:+.4f} |"
        )
    lines.extend(
        [
            "",
            "### What to do",
            "",
            "- If the regression is real, fix it before merging.",
            "- If this PR intentionally changes the corpus, the "
            "judge, or the prompts, re-measure the baseline:",
            "  ```",
            "  python scripts/measure_baseline_variance.py " "--runs 20 \\",
            "      --out docs/specs/release-quality-baseline/" "baseline-N.md \\",
            "      --thresholds-out docs/specs/" "release-quality-baseline/thresholds.json",
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
    except (FileNotFoundError, ValueError) as e:
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
            args.comment_out.write_text(format_failure_comment(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
