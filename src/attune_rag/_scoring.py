"""Internal scoring helper for golden-query evaluation.

Underscore-private: used by ``tests/golden/test_golden.py``,
``scripts/measure_corpus.py``, and (eventually)
``scripts/measure_reranker.py``. Not part of the public API
surface — does not count against the v1.0.0 symbol budget and
is not re-exported from ``attune_rag.__init__``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .pipeline import RagPipeline


@dataclass(frozen=True)
class QueryScore:
    """Per-query scoring result."""

    qid: str
    query: str
    difficulty: str
    expected: tuple[str, ...]
    hit_paths: tuple[str, ...]
    p1: bool
    r3: bool


@dataclass(frozen=True)
class AggregateScore:
    """Aggregate scoring result over a query set."""

    n: int
    p1_hits: int
    r3_hits: int

    @property
    def p1(self) -> float:
        return self.p1_hits / self.n if self.n else 0.0

    @property
    def r3(self) -> float:
        return self.r3_hits / self.n if self.n else 0.0


def score_queries(
    pipeline: RagPipeline,
    queries: Sequence[Mapping[str, Any]],
    *,
    k: int = 3,
) -> tuple[list[QueryScore], AggregateScore]:
    """Score ``queries`` against ``pipeline`` and return per-query + aggregate.

    Each entry in ``queries`` must carry ``id``, ``query``, and
    ``expected_in_top_3`` keys (the bundled YAML shape). ``difficulty``
    is optional and defaults to the empty string.

    P@1 is true iff the rank-1 hit's ``template_path`` is in
    ``expected_in_top_3``. R@3 is true iff any of the top-``k`` hits'
    paths overlap with ``expected_in_top_3``.
    """
    per_query: list[QueryScore] = []
    p1_hits = 0
    r3_hits = 0
    for entry in queries:
        expected = tuple(entry.get("expected_in_top_3", []))
        result = pipeline.run(entry["query"], k=k)
        hit_paths = tuple(h.template_path for h in result.citation.hits)
        expected_set = set(expected)
        p1 = bool(hit_paths) and hit_paths[0] in expected_set
        r3 = bool(expected_set & set(hit_paths))
        if p1:
            p1_hits += 1
        if r3:
            r3_hits += 1
        per_query.append(
            QueryScore(
                qid=str(entry["id"]),
                query=str(entry["query"]),
                difficulty=str(entry.get("difficulty", "")),
                expected=expected,
                hit_paths=hit_paths,
                p1=p1,
                r3=r3,
            )
        )
    aggregate = AggregateScore(n=len(per_query), p1_hits=p1_hits, r3_hits=r3_hits)
    return per_query, aggregate
