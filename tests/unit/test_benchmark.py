"""Tests for attune_rag.benchmark — CLI exit codes, aggregation, helpers.

Targets the highest coverage gap identified in the test-strategy audit:
``benchmark.py`` was at 10% line coverage. These tests exercise the pure
helpers (`_percentile`, `_load_queries`) and the CLI happy + error paths
without spending API tokens (`--with-faithfulness` is gated behind a
real ANTHROPIC_API_KEY check we don't satisfy here).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from attune_rag.benchmark import (
    _default_queries_path,
    _load_queries,
    _percentile,
    _run_benchmark,
    main,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_queries(path: Path, queries: list[dict]) -> Path:
    """Write a queries.yaml-shaped file to ``path``."""
    path.write_text(yaml.safe_dump({"queries": queries}), encoding="utf-8")
    return path


def _hit(template_path: str) -> SimpleNamespace:
    """Minimal RagPipeline.run hit shape — only template_path used by benchmark."""
    return SimpleNamespace(template_path=template_path)


def _result(*hits: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(citation=SimpleNamespace(hits=list(hits)))


class _FakeRetriever:
    pass


class _FakePipeline:
    """Stub RagPipeline that returns scripted results per query string."""

    def __init__(self, scripted: dict[str, list[str]]) -> None:
        self._scripted = scripted
        self.retriever = _FakeRetriever()
        self.corpus = SimpleNamespace(name="fake-corpus")

    def run(self, query: str, k: int = 3) -> SimpleNamespace:
        paths = self._scripted.get(query, [])
        return _result(*[_hit(p) for p in paths])


# ---------------------------------------------------------------------------
# _default_queries_path
# ---------------------------------------------------------------------------


def test_default_queries_path_resolves_inside_repo() -> None:
    path = _default_queries_path()
    assert path.name == "queries.yaml"
    assert "tests/golden" in str(path)


# ---------------------------------------------------------------------------
# _load_queries
# ---------------------------------------------------------------------------


def test_load_queries_returns_query_list(tmp_path: Path) -> None:
    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "hello", "expected_in_top_3": ["a.md"]}],
    )
    out = _load_queries(p)
    assert len(out) == 1
    assert out[0]["id"] == "q1"


def test_load_queries_raises_when_no_queries_key(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("queries: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No queries"):
        _load_queries(p)


def test_load_queries_raises_when_top_level_missing_queries(tmp_path: Path) -> None:
    p = tmp_path / "junk.yaml"
    p.write_text("not_queries: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_queries(p)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


def test_percentile_empty_list_returns_zero() -> None:
    assert _percentile([], 0.95) == 0.0


@pytest.mark.parametrize(
    "values,pct,expected",
    [
        ([1.0], 0.5, 1.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 0.0, 1.0),  # min
        ([1.0, 2.0, 3.0, 4.0, 5.0], 1.0, 5.0),  # max
        ([10.0, 20.0, 30.0, 40.0, 50.0], 0.5, 30.0),  # median
    ],
)
def test_percentile_typical_values(values: list[float], pct: float, expected: float) -> None:
    assert _percentile(values, pct) == expected


def test_percentile_handles_unsorted_input() -> None:
    assert _percentile([5.0, 2.0, 9.0, 1.0, 3.0], 0.0) == 1.0


# ---------------------------------------------------------------------------
# _run_benchmark — aggregation math
# ---------------------------------------------------------------------------


def test_run_benchmark_perfect_precision_and_recall() -> None:
    queries = [
        {"id": "q1", "query": "auth", "expected_in_top_3": ["concepts/auth.md"]},
        {"id": "q2", "query": "memory", "expected_in_top_3": ["concepts/memory.md"]},
    ]
    pipeline = _FakePipeline(
        {
            "auth": ["concepts/auth.md"],
            "memory": ["concepts/memory.md"],
        }
    )
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        report = _run_benchmark(queries, k=3)
    assert report["total_queries"] == 2
    assert report["precision_at_1"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["k"] == 3


def test_run_benchmark_zero_precision_when_top1_misses() -> None:
    queries = [
        {"id": "q1", "query": "auth", "expected_in_top_3": ["concepts/auth.md"]},
    ]
    pipeline = _FakePipeline(
        {"auth": ["concepts/wrong.md", "concepts/auth.md"]},
    )
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        report = _run_benchmark(queries, k=3)
    assert report["precision_at_1"] == 0.0
    # But recall@3 still counts since auth.md is in the top-k set
    assert report["recall_at_k"] == 1.0


def test_run_benchmark_zero_recall_when_no_match() -> None:
    queries = [
        {"id": "q1", "query": "auth", "expected_in_top_3": ["concepts/auth.md"]},
    ]
    pipeline = _FakePipeline({"auth": ["concepts/elsewhere.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        report = _run_benchmark(queries, k=3)
    assert report["precision_at_1"] == 0.0
    assert report["recall_at_k"] == 0.0


def test_run_benchmark_records_per_query_detail() -> None:
    queries = [
        {
            "id": "q1",
            "query": "auth",
            "expected_in_top_3": ["a.md"],
            "difficulty": "easy",
        },
        {
            "id": "q2",
            "query": "memory",
            "expected_in_top_3": ["m.md"],
            "difficulty": "hard",
        },
    ]
    pipeline = _FakePipeline({"auth": ["a.md"], "memory": ["wrong.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        report = _run_benchmark(queries, k=3)
    by_id = {q["id"]: q for q in report["per_query"]}
    assert by_id["q1"]["top1_match"] is True
    assert by_id["q2"]["top1_match"] is False
    assert by_id["q1"]["difficulty"] == "easy"
    assert by_id["q2"]["difficulty"] == "hard"


def test_run_benchmark_empty_queries_yields_zero_metrics() -> None:
    """Defensive: total=0 must not divide-by-zero."""
    pipeline = _FakePipeline({})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        report = _run_benchmark([], k=3)
    assert report["total_queries"] == 0
    assert report["precision_at_1"] == 0.0
    assert report["recall_at_k"] == 0.0
    assert report["mean_latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# main() — exit codes
# ---------------------------------------------------------------------------


def test_main_exits_2_when_queries_file_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["--queries", str(tmp_path / "nope.yaml")])
    assert rc == 2
    assert "Queries file not found" in capsys.readouterr().err


def test_main_exits_0_on_perfect_precision(tmp_path: Path) -> None:
    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}],
    )
    pipeline = _FakePipeline({"auth": ["a.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(["--queries", str(p), "--min-precision", "0.5"])
    assert rc == 0


def test_main_exits_1_when_precision_below_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}],
    )
    pipeline = _FakePipeline({"auth": ["wrong.md"]})  # 0% precision
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(["--queries", str(p), "--min-precision", "0.5"])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err


def test_main_with_faithfulness_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--with-faithfulness without ANTHROPIC_API_KEY exits 2."""
    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}],
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pipeline = _FakePipeline({"auth": ["a.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(
            ["--queries", str(p), "--min-precision", "0.0", "--with-faithfulness"],
        )
    assert rc == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
