"""Measure retrieval quality on an attune-rag corpus.

Public Python API + CLI for scoring a corpus + query set against
the attune-rag ``RagPipeline``. Per
[`docs/specs/user-corpus-onboarding/`](https://github.com/Smart-AI-Memory/attune-rag/tree/main/docs/specs/user-corpus-onboarding)
M1 — promotes the freeze-time ``scripts/measure_corpus.py``
harness into a packaged module + console_script.

The package default mirrors ``RagPipeline(reranker=None)`` —
i.e. rerank is **off** by default per D5's verdict
([`reranker-evaluation/diagnostic-1.md`](https://github.com/Smart-AI-Memory/attune-rag/blob/main/docs/specs/reranker-evaluation/diagnostic-1.md)).
``--with-rerank`` is an opt-in flag for users who want to measure
whether the rerank lift earns its keep on their corpus.

Python API::

    from pathlib import Path
    from attune_rag.measure_corpus import measure

    result = measure(
        corpus_path=Path("./my-corpus"),
        queries_path=Path("./my-corpus/queries.yaml"),
        paraphrased_path=Path("./my-corpus/paraphrased.yaml"),
    )
    print(f"P@1 = {result.p1:.4f}  R@3 = {result.r3:.4f}")
    if result.watermark_failures(r3_floor=0.85):
        sys.exit(1)
    Path("report.md").write_text(result.report_markdown())

CLI::

    python -m attune_rag.measure_corpus \\
        --corpus-path ./my-corpus \\
        --queries ./my-corpus/queries.yaml \\
        --output report.md \\
        --watermark-r3 0.85

    # console_script alias
    attune-rag-measure --corpus-path ./my-corpus --queries ... --output ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from . import RagPipeline
from ._scoring import QueryScore, score_queries

__all__ = ["measure", "MeasureResult"]
# ``main`` is the CLI entry point used by the ``attune-rag-measure``
# console_script + ``python -m attune_rag.measure_corpus``. Not in
# ``__all__`` because it's a CLI runner, not part of the Python API
# surface; the symbol-budget gate only counts ``__all__`` membership.

# Versioned independently from attune_rag.__version__ so the report's
# reproducibility metadata records *which* report shape produced these
# bytes — bumped on any change to the rendered report.
_HARNESS_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasureResult:
    """Result of one ``measure()`` invocation against a corpus + queries.

    Aggregate scalars (``p1``, ``r3``, optionally ``paraphrased_p1`` /
    ``paraphrased_r3``) are the primary signal. The per-query records
    + per-difficulty breakdown let callers drill into where the corpus
    succeeds or fails. ``report_markdown()`` renders the same shape
    documented in [`USER_CORPUS_GUIDE.md`](https://github.com/Smart-AI-Memory/attune-rag/blob/main/docs/USER_CORPUS_GUIDE.md)
    §6.2 — the bytes are byte-identical across reruns given the same
    ``frozen_timestamp`` (deterministic on the keyword path).

    Attributes:
        p1: Aggregate Precision@1 against ``queries_path``.
        r3: Aggregate Recall@3 against ``queries_path``.
        n: Number of baseline queries scored.
        paraphrased_p1: Aggregate P@1 on ``paraphrased_path`` (``None``
            if not provided).
        paraphrased_r3: Aggregate R@3 on ``paraphrased_path``.
        paraphrased_n: Number of paraphrased queries scored.
        per_query_table: Baseline query results, in YAML input order.
        paraphrased_per_query: Paraphrased query results, in YAML input
            order (``None`` when no paraphrased set provided).
        per_difficulty_breakdown: ``{difficulty: {"p1", "r3", "n"}}``
            grouped per the queries' ``difficulty`` field. Combines
            baseline + paraphrased (use ``per_query_table`` and
            ``paraphrased_per_query`` for split detail).
        corpus_label: Human-readable corpus identifier (path or
            ``"bundled (AttuneHelpCorpus)"``).
        queries_path: Path string as supplied (forward-slash on every
            platform via ``Path.as_posix()``).
        queries_sha: 12-char SHA-256 of the canonical (LF-normalized)
            queries file bytes.
        paraphrased_path: Path string for paraphrased set, if any.
        paraphrased_sha: SHA-256 of the paraphrased queries file.
        rerank: Whether Run B (rerank-on) was included.
        harness_version: ``_HARNESS_VERSION`` constant at measurement
            time (carried so re-running an old report tells you which
            report shape produced the bytes).
    """

    p1: float
    r3: float
    n: int
    paraphrased_p1: float | None
    paraphrased_r3: float | None
    paraphrased_n: int | None
    per_query_table: tuple[QueryScore, ...]
    paraphrased_per_query: tuple[QueryScore, ...] | None
    per_difficulty_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    corpus_label: str = ""
    queries_path: str = ""
    queries_sha: str = ""
    paraphrased_path: str | None = None
    paraphrased_sha: str | None = None
    rerank: bool = False
    harness_version: str = _HARNESS_VERSION

    def watermark_failures(
        self,
        *,
        p1_floor: float | None = None,
        r3_floor: float | None = None,
    ) -> list[str]:
        """Return human-readable watermark failures (empty = pass).

        Checks both baseline and paraphrased sets when applicable.
        """
        failures: list[str] = []
        if r3_floor is not None and self.r3 < r3_floor:
            failures.append(f"[baseline] R@3 {self.r3:.4f} < watermark {r3_floor:.4f}")
        if p1_floor is not None and self.p1 < p1_floor:
            failures.append(f"[baseline] P@1 {self.p1:.4f} < watermark {p1_floor:.4f}")
        if (
            self.paraphrased_r3 is not None
            and r3_floor is not None
            and self.paraphrased_r3 < r3_floor
        ):
            failures.append(
                f"[paraphrased] R@3 {self.paraphrased_r3:.4f} < watermark {r3_floor:.4f}"
            )
        if (
            self.paraphrased_p1 is not None
            and p1_floor is not None
            and self.paraphrased_p1 < p1_floor
        ):
            failures.append(
                f"[paraphrased] P@1 {self.paraphrased_p1:.4f} < watermark {p1_floor:.4f}"
            )
        return failures

    def report_markdown(self, *, frozen_timestamp: str | None = None) -> str:
        """Render the markdown report.

        The shape matches what
        [``scripts/measure_corpus.py``](https://github.com/Smart-AI-Memory/attune-rag/blob/main/scripts/measure_corpus.py)
        emitted at v0 — preserved byte-identical so the bundled-corpus
        golden snapshot test continues to pin the regression net.

        Args:
            frozen_timestamp: When given, written into the report
                header instead of ``datetime.utcnow()``. Use this for
                deterministic golden-snapshot tests.

        Returns:
            Report markdown. Use ``write_bytes(s.encode("utf-8"))`` to
            write to disk on Windows without ``\\n → \\r\\n`` translation.
        """
        timestamp = frozen_timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines: list[str] = []
        lines.append("# Corpus measurement report")
        lines.append("")
        lines.append(f"- Corpus: `{self.corpus_label}`")
        lines.append(f"- Harness version: `{self.harness_version}`")
        lines.append(f"- Timestamp: `{timestamp}`")
        lines.append(
            f"- Query set `baseline`: `{self.queries_path}` (sha256: `{self.queries_sha}`)"
        )
        if self.paraphrased_path is not None:
            lines.append(
                f"- Query set `paraphrased`: `{self.paraphrased_path}` "
                f"(sha256: `{self.paraphrased_sha}`)"
            )
        if self.rerank:
            lines.append("- Mode: `keyword + LLM rerank (opt-in)`")
        else:
            lines.append("- Mode: `keyword-only`")
        lines.append("")

        # Aggregate
        lines.append("## Aggregate")
        lines.append("")
        lines.append("| Set | n | P@1 | R@3 |")
        lines.append("|-----|---|-----|-----|")
        lines.append(f"| baseline | {self.n} | {self.p1:.4f} | {self.r3:.4f} |")
        if self.paraphrased_p1 is not None:
            lines.append(
                f"| paraphrased | {self.paraphrased_n} | "
                f"{self.paraphrased_p1:.4f} | {self.paraphrased_r3:.4f} |"
            )
        lines.append("")

        # Per-query tables (one per set)
        def _per_query_section(label: str, rows: Sequence[QueryScore]) -> None:
            lines.append(f"## Per-query — {label}")
            lines.append("")
            lines.append("| qid | difficulty | P@1 | R@3 |")
            lines.append("|-----|-----------|-----|-----|")
            for q in rows:
                lines.append(
                    f"| {q.qid} | {q.difficulty or '—'} | "
                    f"{'✓' if q.p1 else '✗'} | {'✓' if q.r3 else '✗'} |"
                )
            lines.append("")

        _per_query_section("baseline", self.per_query_table)
        if self.paraphrased_per_query is not None:
            _per_query_section("paraphrased", self.paraphrased_per_query)

        # Footer — advertise rerank only when not used.
        if not self.rerank:
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

    def to_json(self) -> str:
        """Render the result as JSON (alternate to markdown).

        The shape is intentionally narrow — primary scalars + per-query
        records + difficulty breakdown. Use ``report_markdown()`` for
        the human-readable form.
        """

        def _q(q: QueryScore) -> dict[str, Any]:
            return {
                "qid": q.qid,
                "query": q.query,
                "difficulty": q.difficulty,
                "expected": list(q.expected),
                "hit_paths": list(q.hit_paths),
                "p1": q.p1,
                "r3": q.r3,
            }

        payload: dict[str, Any] = {
            "harness_version": self.harness_version,
            "corpus_label": self.corpus_label,
            "queries_path": self.queries_path,
            "queries_sha": self.queries_sha,
            "rerank": self.rerank,
            "aggregate": {"p1": self.p1, "r3": self.r3, "n": self.n},
            "per_query": [_q(q) for q in self.per_query_table],
            "per_difficulty": self.per_difficulty_breakdown,
        }
        if self.paraphrased_p1 is not None:
            payload["paraphrased"] = {
                "path": self.paraphrased_path,
                "sha": self.paraphrased_sha,
                "p1": self.paraphrased_p1,
                "r3": self.paraphrased_r3,
                "n": self.paraphrased_n,
                "per_query": [_q(q) for q in (self.paraphrased_per_query or ())],
            }
        return json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_queries(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load queries YAML; return (list, SHA-256 of canonical bytes).

    CRLF is normalized to LF before hashing so the SHA describes the
    canonical content, not the on-disk line-ending policy. This makes
    the metadata block byte-stable across Linux / macOS / Windows.
    """
    raw = path.read_bytes()
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
    corpus_path: Path | str | None,
    bundled: bool,
    with_rerank: bool,
    candidate_multiplier: int,
    extra_aliases_file: Path | str | None,
) -> RagPipeline:
    """Build a RagPipeline from either a user corpus path or the bundled corpus."""
    corpus = None
    if not bundled:
        from .corpus import DirectoryCorpus

        assert corpus_path is not None
        corpus = DirectoryCorpus(Path(corpus_path), extra_aliases_file=extra_aliases_file)
    reranker = None
    if with_rerank:
        from .reranker import LLMReranker

        reranker = LLMReranker(candidate_multiplier=candidate_multiplier)
    if corpus is None:
        return RagPipeline(reranker=reranker)
    return RagPipeline(corpus=corpus, reranker=reranker)


def _per_difficulty_breakdown(
    baseline: Sequence[QueryScore],
    paraphrased: Sequence[QueryScore] | None,
) -> dict[str, dict[str, float]]:
    """Group queries by ``difficulty`` and compute per-difficulty P@1 / R@3.

    Combines baseline + paraphrased so callers see one breakdown across
    all the queries they measured. Empty difficulty bucket → omitted.
    """
    buckets: dict[str, list[QueryScore]] = {}
    for q in baseline:
        buckets.setdefault(q.difficulty or "unspecified", []).append(q)
    if paraphrased:
        for q in paraphrased:
            buckets.setdefault(q.difficulty or "unspecified", []).append(q)
    out: dict[str, dict[str, float]] = {}
    for diff, qs in sorted(buckets.items()):
        n = len(qs)
        if not n:
            continue
        p1_hits = sum(1 for q in qs if q.p1)
        r3_hits = sum(1 for q in qs if q.r3)
        out[diff] = {"p1": p1_hits / n, "r3": r3_hits / n, "n": float(n)}
    return out


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def measure(
    *,
    corpus_path: Path | str | None = None,
    bundled: bool = False,
    queries_path: Path | str,
    paraphrased_path: Path | str | None = None,
    rerank: bool = False,
    candidate_multiplier: int = 3,
    extra_aliases_file: Path | str | None = None,
) -> MeasureResult:
    """Score a corpus + query set; return a :class:`MeasureResult`.

    Args:
        corpus_path: Filesystem path to the corpus root (a directory of
            markdown files). Mutually exclusive with ``bundled``;
            exactly one must be set.
        bundled: When ``True``, score against the bundled
            ``AttuneHelpCorpus`` (requires the ``[attune-help]`` extra).
            Mutually exclusive with ``corpus_path``.
        queries_path: Path to a YAML file shaped
            ``{queries: [{id, query, expected_in_top_3, difficulty?}, ...]}``.
        paraphrased_path: Optional second query set (typically the
            no-token-overlap variants — see
            [`USER_CORPUS_GUIDE.md`](https://github.com/Smart-AI-Memory/attune-rag/blob/main/docs/USER_CORPUS_GUIDE.md)
            §2.2).
        rerank: When ``True``, runs through ``LLMReranker`` after the
            keyword retriever. Default ``False`` per D5's verdict
            (``reranker-evaluation/diagnostic-1.md``). Costs Anthropic
            tokens.
        candidate_multiplier: Reranker over-fetch factor (ignored when
            ``rerank=False``).
        extra_aliases_file: Optional JSON file of path-keyed alias
            overrides for the corpus (the same kwarg as
            ``DirectoryCorpus(extra_aliases_file=...)``).

    Returns:
        A :class:`MeasureResult` with aggregate + per-query data.

    Raises:
        ValueError: ``corpus_path`` and ``bundled`` both set or both
            unset; queries YAML malformed; ``paraphrased`` YAML
            malformed.
        FileNotFoundError: A given path doesn't exist.
    """
    if (corpus_path is None) == (not bundled):
        raise ValueError("measure() requires exactly one of corpus_path= or bundled=True.")

    pipeline = _build_pipeline(
        corpus_path=corpus_path,
        bundled=bundled,
        with_rerank=rerank,
        candidate_multiplier=candidate_multiplier,
        extra_aliases_file=extra_aliases_file,
    )

    qp = Path(queries_path)
    queries, qsha = _load_queries(qp)
    base_per, base_agg = score_queries(pipeline, queries, k=3)

    para_per: tuple[QueryScore, ...] | None = None
    para_agg = None
    psha: str | None = None
    p_path_str: str | None = None
    if paraphrased_path is not None:
        pp = Path(paraphrased_path)
        para_queries, psha = _load_queries(pp)
        per, agg = score_queries(pipeline, para_queries, k=3)
        para_per = tuple(per)
        para_agg = agg
        p_path_str = pp.as_posix()

    corpus_label = "bundled (AttuneHelpCorpus)" if bundled else Path(corpus_path).as_posix()  # type: ignore[arg-type]

    return MeasureResult(
        p1=base_agg.p1,
        r3=base_agg.r3,
        n=base_agg.n,
        paraphrased_p1=(para_agg.p1 if para_agg else None),
        paraphrased_r3=(para_agg.r3 if para_agg else None),
        paraphrased_n=(para_agg.n if para_agg else None),
        per_query_table=tuple(base_per),
        paraphrased_per_query=para_per,
        per_difficulty_breakdown=_per_difficulty_breakdown(base_per, para_per),
        corpus_label=corpus_label,
        queries_path=qp.as_posix(),
        queries_sha=qsha,
        paraphrased_path=p_path_str,
        paraphrased_sha=psha,
        rerank=rerank,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="attune-rag-measure",
        description="Measure retrieval quality on an attune-rag corpus.",
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
        help=(
            "Path to a queries YAML (shape: "
            "{'queries': [{'id', 'query', 'expected_in_top_3', 'difficulty'?}, ...]})."
        ),
    )
    parser.add_argument(
        "--paraphrased",
        type=Path,
        default=None,
        help="Optional second query set (e.g. paraphrased variants).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the report. Default: stdout.",
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
        help=(
            "Run with LLMReranker (requires ANTHROPIC_API_KEY + the [claude] "
            "extra). Default off per D5's verdict. Costs ~$0.05 per 80-query set."
        ),
    )
    parser.add_argument(
        "--extra-aliases-file",
        type=Path,
        default=None,
        help="Optional path-keyed extra-aliases JSON for the corpus.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. Default: markdown.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress 'wrote PATH' confirmation on --output. Errors still print.",
    )
    parser.add_argument(
        "--frozen-timestamp",
        type=str,
        default=None,
        help="ISO timestamp for the report header (deterministic mode). Default: now.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        result = measure(
            corpus_path=args.corpus_path,
            bundled=bool(args.corpus_bundled),
            queries_path=args.queries,
            paraphrased_path=args.paraphrased,
            rerank=args.with_rerank,
            candidate_multiplier=args.candidate_multiplier,
            extra_aliases_file=args.extra_aliases_file,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        out_text = result.to_json()
    else:
        out_text = result.report_markdown(frozen_timestamp=args.frozen_timestamp)

    if args.output is None:
        sys.stdout.write(out_text)
        if not out_text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        # write_bytes → LF on every platform (Path.write_text translates
        # \n → \r\n on Windows).
        args.output.write_bytes(out_text.encode("utf-8"))
        if not args.quiet:
            print(f"wrote {args.output}", file=sys.stderr)

    failures = result.watermark_failures(
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
