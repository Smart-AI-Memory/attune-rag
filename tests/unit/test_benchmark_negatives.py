"""Tests for Phase 1 eval-hardening additions to the benchmark.

Covers:
- ``_aggregate_by_difficulty`` — per-tier precision/recall rollup.
- ``_run_negative_benchmark`` — abstention / false-answer measurement
  on out-of-corpus queries.
- ``_run_benchmark`` now carries ``by_difficulty`` in its report.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attune_rag.benchmark import (
    _aggregate_by_difficulty,
    _run_benchmark,
    _run_negative_benchmark,
    main,
)


def _hit(template_path: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(template_path=template_path, score=score)


def _result(*hits: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(citation=SimpleNamespace(hits=list(hits)))


class _FakePipeline:
    """Stub RagPipeline returning scripted (path, score) hits per query."""

    def __init__(self, scripted: dict[str, list[tuple[str, float]]]) -> None:
        self._scripted = scripted
        self.retriever = SimpleNamespace()
        self.corpus = SimpleNamespace(name="fake-corpus")

    def run(self, query: str, k: int = 3) -> SimpleNamespace:
        hits = self._scripted.get(query, [])
        return _result(*[_hit(p, s) for p, s in hits])


# ---------------------------------------------------------------------------
# _aggregate_by_difficulty
# ---------------------------------------------------------------------------


def test_aggregate_by_difficulty_counts_and_rates() -> None:
    per_query = [
        {"difficulty": "easy", "top1_match": True, "topk_match": True},
        {"difficulty": "easy", "top1_match": True, "topk_match": True},
        {"difficulty": "hard", "top1_match": False, "topk_match": True},
        {"difficulty": "hard", "top1_match": False, "topk_match": False},
    ]
    agg = _aggregate_by_difficulty(per_query)
    assert agg["easy"]["total"] == 2
    assert agg["easy"]["precision_at_1"] == 1.0
    assert agg["easy"]["recall_at_k"] == 1.0
    assert agg["hard"]["total"] == 2
    assert agg["hard"]["precision_at_1"] == 0.0
    assert agg["hard"]["recall_at_k"] == 0.5


def test_aggregate_by_difficulty_missing_tier_label_buckets_unknown() -> None:
    agg = _aggregate_by_difficulty([{"top1_match": True, "topk_match": True}])
    assert "unknown" in agg
    assert agg["unknown"]["total"] == 1


def test_run_benchmark_report_includes_by_difficulty() -> None:
    queries = [
        {"id": "q1", "query": "a", "expected_in_top_3": ["a.md"], "difficulty": "easy"},
        {"id": "q2", "query": "b", "expected_in_top_3": ["b.md"], "difficulty": "hard"},
    ]
    pipeline = _FakePipeline({"a": [("a.md", 9.0)], "b": [("z.md", 9.0)]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        report = _run_benchmark(queries, k=3)
    assert "by_difficulty" in report
    assert report["by_difficulty"]["easy"]["recall_at_k"] == 1.0
    assert report["by_difficulty"]["hard"]["recall_at_k"] == 0.0


# ---------------------------------------------------------------------------
# _run_negative_benchmark
# ---------------------------------------------------------------------------


def test_negative_all_abstain_is_perfect() -> None:
    negs = [{"id": "n1", "query": "x"}, {"id": "n2", "query": "y"}]
    pipeline = _FakePipeline({})  # every query → no hits
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rep = _run_negative_benchmark(negs, k=3)
    assert rep["total_negatives"] == 2
    assert rep["false_answers"] == 0
    assert rep["false_answer_rate"] == 0.0
    assert rep["abstention_rate"] == 1.0
    assert all(q["abstained"] for q in rep["per_query"])


def test_negative_all_answered_is_worst() -> None:
    negs = [{"id": "n1", "query": "x"}, {"id": "n2", "query": "y"}]
    pipeline = _FakePipeline({"x": [("a.md", 3.0)], "y": [("b.md", 5.0)]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rep = _run_negative_benchmark(negs, k=3)
    assert rep["false_answers"] == 2
    assert rep["false_answer_rate"] == 1.0
    assert rep["abstention_rate"] == 0.0
    assert not any(q["abstained"] for q in rep["per_query"])
    # leaked hits + top score recorded for triage
    leaked = {q["id"]: q for q in rep["per_query"]}
    assert leaked["n1"]["leaked_hits"] == ["a.md"]
    assert leaked["n2"]["top_score"] == 5.0


def test_negative_mixed_rate() -> None:
    negs = [{"id": "n1", "query": "x"}, {"id": "n2", "query": "y"}]
    pipeline = _FakePipeline({"x": [("a.md", 4.0)]})  # x answers, y abstains
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rep = _run_negative_benchmark(negs, k=3)
    assert rep["false_answer_rate"] == 0.5
    assert rep["abstention_rate"] == 0.5


def test_negative_empty_set_no_divide_by_zero() -> None:
    pipeline = _FakePipeline({})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rep = _run_negative_benchmark([], k=3)
    assert rep["total_negatives"] == 0
    assert rep["false_answer_rate"] == 0.0
    assert rep["abstention_rate"] == 0.0


# ---------------------------------------------------------------------------
# main() integration — exercises --negatives / --extended wiring + JSON dump
# ---------------------------------------------------------------------------


def _yaml(path: Path, queries: list[dict]) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({"version": 1, "queries": queries}), encoding="utf-8")
    return path


def test_main_includes_negatives_and_extended_in_json(tmp_path: Path) -> None:
    q = _yaml(
        tmp_path / "q.yaml",
        [{"id": "mq", "query": "main", "expected_in_top_3": ["a.md"], "difficulty": "easy"}],
    )
    neg = _yaml(tmp_path / "neg.yaml", [{"id": "n1", "query": "offtopic"}])
    ext = _yaml(
        tmp_path / "ext.yaml",
        [{"id": "e1", "query": "hardish", "expected_in_top_3": ["c.md"], "difficulty": "hard"}],
    )
    out = tmp_path / "dump.json"
    pipeline = _FakePipeline(
        {
            "main": [("a.md", 9.0)],  # precision hit
            "offtopic": [("x.md", 4.0)],  # negative leaks (false answer)
            "hardish": [("c.md", 8.0)],  # extended hit
        }
    )
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(
            [
                "--queries",
                str(q),
                "--negatives",
                str(neg),
                "--extended",
                str(ext),
                "--json",
                str(out),
                "--min-precision",
                "0.5",
                "--verbose",
            ]
        )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["retrieval"]["by_difficulty"]["easy"]["recall_at_k"] == 1.0
    assert payload["negatives"]["false_answer_rate"] == 1.0
    assert payload["extended"]["recall_at_k"] == 1.0
