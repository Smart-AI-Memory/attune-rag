"""Measure retrieval quality on an attune-rag corpus.

Standalone harness that scores a corpus + query set against the
attune-rag ``RagPipeline``, emitting a deterministic markdown
report with aggregate P@1 / R@3 plus per-query results.

The script is the freeze-time v0 of what becomes
``attune_rag.measure_corpus`` at v1.0.0 (see
``docs/specs/user-corpus-onboarding/``). Ship as ``### Changed``
internal tooling — no public API surface.

Usage::

    # User corpus, deterministic keyword path
    python scripts/measure_corpus.py \\
        --corpus-path docs/my-corpus \\
        --queries docs/my-corpus/queries.yaml \\
        --output report.md

    # Add rerank comparison (~$0.05 of Anthropic spend per pass)
    python scripts/measure_corpus.py \\
        --corpus-path docs/my-corpus \\
        --queries docs/my-corpus/queries.yaml \\
        --with-rerank \\
        --output report.md

    # Reproduce bundled-corpus golden snapshot
    python scripts/measure_corpus.py \\
        --corpus-bundled \\
        --queries tests/golden/queries.yaml \\
        --paraphrased tests/golden/queries_paraphrased.yaml \\
        --frozen-timestamp 2026-05-22T00:00:00Z \\
        --output /tmp/bundled.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from attune_rag import RagPipeline
from attune_rag._scoring import AggregateScore, QueryScore, score_queries

_HARNESS_VERSION = "0.1.0"


@dataclass(frozen=True)
class _SetResult:
    """Aggregate + per-query result for one query set."""

    label: str
    queries_path: Path
    queries_sha: str
    per_query: list[QueryScore]
    aggregate: AggregateScore


def _load_queries(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    # Normalize CRLF → LF so the SHA-256 in the report is identical
    # across Linux / macOS / Windows checkouts regardless of git's
    # eol-attribute application. The hash describes the canonical
    # content of the queries file, not the local line-ending policy.
    canonical = raw.replace(b"\r\n", b"\n")
    sha = sha256(canonical).hexdigest()[:12]
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError(f"Queries file {path} must be a YAML mapping with a 'queries' key.")
    queries = data["queries"]
    if not isinstance(queries, list):
        raise ValueError(f"Queries file {path}: 'queries' must be a list.")
    return queries, sha


def _build_pipeline(
    *,
    corpus_path: Path | None,
    bundled: bool,
    with_rerank: bool,
    candidate_multiplier: int,
) -> RagPipeline:
    """Construct a RagPipeline for either a user corpus or the bundled corpus."""
    if bundled:
        corpus = None  # RagPipeline._default_corpus picks AttuneHelpCorpus
    else:
        from attune_rag.corpus import DirectoryCorpus

        assert corpus_path is not None
        corpus = DirectoryCorpus(corpus_path)
    reranker = None
    if with_rerank:
        from attune_rag.reranker import LLMReranker

        reranker = LLMReranker(candidate_multiplier=candidate_multiplier)
    if corpus is None:
        return RagPipeline(reranker=reranker)
    return RagPipeline(corpus=corpus, reranker=reranker)


def _score_set(
    pipeline: RagPipeline,
    label: str,
    path: Path,
) -> _SetResult:
    queries, sha = _load_queries(path)
    per_query, aggregate = score_queries(pipeline, queries, k=3)
    return _SetResult(
        label=label,
        queries_path=path,
        queries_sha=sha,
        per_query=per_query,
        aggregate=aggregate,
    )


def _format_pct(x: float) -> str:
    return f"{x:.4f}"


def _render_report(
    *,
    corpus_label: str,
    baseline_sets: list[_SetResult],
    rerank_sets: list[_SetResult] | None,
    timestamp: str,
    rerank_used: bool,
) -> str:
    """Render the deterministic markdown report.

    When ``rerank_sets`` is provided (i.e. ``--with-rerank``), the
    aggregate table grows a side-by-side baseline vs +rerank column.
    """
    lines: list[str] = []
    lines.append("# Corpus measurement report")
    lines.append("")
    lines.append(f"- Corpus: `{corpus_label}`")
    lines.append(f"- Harness version: `{_HARNESS_VERSION}`")
    lines.append(f"- Timestamp: `{timestamp}`")
    for s in baseline_sets:
        lines.append(
            f"- Query set `{s.label}`: `{s.queries_path.as_posix()}` (sha256: `{s.queries_sha}`)"
        )
    lines.append(
        "- Mode: `keyword-only`" if not rerank_used else "- Mode: `keyword + LLM rerank (opt-in)`"
    )
    lines.append("")

    # Aggregate
    lines.append("## Aggregate")
    lines.append("")
    if rerank_sets is not None:
        lines.append("| Set | n | Baseline P@1 | Baseline R@3 | +Rerank P@1 | +Rerank R@3 |")
        lines.append("|-----|---|--------------|--------------|-------------|-------------|")
        for base, rer in zip(baseline_sets, rerank_sets, strict=False):
            lines.append(
                f"| {base.label} | {base.aggregate.n} | "
                f"{_format_pct(base.aggregate.p1)} | {_format_pct(base.aggregate.r3)} | "
                f"{_format_pct(rer.aggregate.p1)} | {_format_pct(rer.aggregate.r3)} |"
            )
    else:
        lines.append("| Set | n | P@1 | R@3 |")
        lines.append("|-----|---|-----|-----|")
        for s in baseline_sets:
            lines.append(
                f"| {s.label} | {s.aggregate.n} | "
                f"{_format_pct(s.aggregate.p1)} | {_format_pct(s.aggregate.r3)} |"
            )
    lines.append("")

    # Per-query tables (one per query set)
    for idx, s in enumerate(baseline_sets):
        lines.append(f"## Per-query — {s.label}")
        lines.append("")
        rer_lookup: dict[str, QueryScore] = {}
        if rerank_sets is not None:
            rer_lookup = {q.qid: q for q in rerank_sets[idx].per_query}
        if rerank_sets is not None:
            lines.append(
                "| qid | difficulty | baseline P@1 | baseline R@3 | +rerank P@1 | +rerank R@3 |"
            )
            lines.append(
                "|-----|-----------|--------------|--------------|-------------|-------------|"
            )
            for q in s.per_query:
                r = rer_lookup.get(q.qid)
                rp1 = "—" if r is None else ("✓" if r.p1 else "✗")
                rr3 = "—" if r is None else ("✓" if r.r3 else "✗")
                lines.append(
                    f"| {q.qid} | {q.difficulty or '—'} | "
                    f"{'✓' if q.p1 else '✗'} | {'✓' if q.r3 else '✗'} | {rp1} | {rr3} |"
                )
        else:
            lines.append("| qid | difficulty | P@1 | R@3 |")
            lines.append("|-----|-----------|-----|-----|")
            for q in s.per_query:
                lines.append(
                    f"| {q.qid} | {q.difficulty or '—'} | "
                    f"{'✓' if q.p1 else '✗'} | {'✓' if q.r3 else '✗'} |"
                )
        lines.append("")

    if not rerank_used:
        lines.append(
            "> 💡 Run with `--with-rerank` to measure whether rerank earns "
            "its keep on your corpus — a lift on marginal queries means leave "
            "it on; a neutral result means the keyword path is already doing "
            "the work. Either outcome is informative. Typical cost: ~$0.05 "
            "for an 80-query set at Haiku pricing. "
            "See `docs/USER_CORPUS_GUIDE.md` §6.2."
        )
        lines.append("")

    return "\n".join(lines)


def _watermark_check(
    sets: Sequence[_SetResult],
    *,
    p1_floor: float | None,
    r3_floor: float | None,
) -> list[str]:
    """Return list of human-readable watermark failures (empty = pass)."""
    failures: list[str] = []
    for s in sets:
        if r3_floor is not None and s.aggregate.r3 < r3_floor:
            failures.append(f"[{s.label}] R@3 {s.aggregate.r3:.4f} < watermark {r3_floor:.4f}")
        if p1_floor is not None and s.aggregate.p1 < p1_floor:
            failures.append(f"[{s.label}] P@1 {s.aggregate.p1:.4f} < watermark {p1_floor:.4f}")
    return failures


def _now_iso(frozen: str | None) -> str:
    if frozen is not None:
        return frozen
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure retrieval quality on an attune-rag corpus."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--corpus-path",
        type=Path,
        help="Path to a directory of markdown files (DirectoryCorpus root).",
    )
    src.add_argument(
        "--corpus-bundled",
        action="store_true",
        help="Use the bundled AttuneHelpCorpus (requires the [attune-help] extra).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to a queries YAML (shape: {'queries': [{'id', 'query', "
        "'expected_in_top_3', 'difficulty'?}, ...]}).",
    )
    parser.add_argument(
        "--paraphrased",
        type=Path,
        default=None,
        help="Optional second query set (e.g. paraphrased variants). Same shape.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the markdown report. Default: stdout.",
    )
    parser.add_argument(
        "--watermark-r3",
        type=float,
        default=0.85,
        help="Minimum aggregate R@3 per set. Non-zero exit if below. Default: 0.85.",
    )
    parser.add_argument(
        "--watermark-p1",
        type=float,
        default=None,
        help="Minimum aggregate P@1 per set. Off by default.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=3,
        help="Reranker candidate multiplier (only used with --with-rerank). Default: 3.",
    )
    parser.add_argument(
        "--with-rerank",
        action="store_true",
        help="Run a second pass with the LLM reranker and emit a side-by-side "
        "comparison. Requires ANTHROPIC_API_KEY. Costs ~$0.05 per 80-query set.",
    )
    parser.add_argument(
        "--frozen-timestamp",
        type=str,
        default=None,
        help="ISO timestamp to write into the report (for deterministic golden "
        "snapshots). Default: current UTC time.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    bundled = bool(args.corpus_bundled)
    corpus_label = "bundled (AttuneHelpCorpus)" if bundled else args.corpus_path.as_posix()

    pipeline_base = _build_pipeline(
        corpus_path=args.corpus_path,
        bundled=bundled,
        with_rerank=False,
        candidate_multiplier=args.candidate_multiplier,
    )
    baseline_sets: list[_SetResult] = [_score_set(pipeline_base, "baseline", args.queries)]
    if args.paraphrased is not None:
        baseline_sets.append(_score_set(pipeline_base, "paraphrased", args.paraphrased))

    rerank_sets: list[_SetResult] | None = None
    if args.with_rerank:
        pipeline_rer = _build_pipeline(
            corpus_path=args.corpus_path,
            bundled=bundled,
            with_rerank=True,
            candidate_multiplier=args.candidate_multiplier,
        )
        rerank_sets = [_score_set(pipeline_rer, "baseline", args.queries)]
        if args.paraphrased is not None:
            rerank_sets.append(_score_set(pipeline_rer, "paraphrased", args.paraphrased))

    timestamp = _now_iso(args.frozen_timestamp)
    report = _render_report(
        corpus_label=corpus_label,
        baseline_sets=baseline_sets,
        rerank_sets=rerank_sets,
        timestamp=timestamp,
        rerank_used=args.with_rerank,
    )

    if args.output is None:
        sys.stdout.write(report)
        if not report.endswith("\n"):
            sys.stdout.write("\n")
    else:
        # Write bytes directly so output has LF line endings on every
        # platform (Path.write_text translates \n → \r\n on Windows).
        args.output.write_bytes(report.encode("utf-8"))

    failures = _watermark_check(
        baseline_sets,
        p1_floor=args.watermark_p1,
        r3_floor=args.watermark_r3,
    )
    if failures:
        for f in failures:
            print(f"WATERMARK FAIL: {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
