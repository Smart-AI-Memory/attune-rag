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
    # Path-component check (OS-independent — Windows uses backslashes).
    assert path.parent.name == "golden"
    assert path.parent.parent.name == "tests"


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


# ---------------------------------------------------------------------------
# --compare-thinking + --json validation
# ---------------------------------------------------------------------------


def test_main_compare_thinking_requires_with_faithfulness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}],
    )
    pipeline = _FakePipeline({"auth": ["a.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(["--queries", str(p), "--compare-thinking"])
    assert rc == 2
    assert "--compare-thinking requires --with-faithfulness" in capsys.readouterr().err


def test_main_compare_thinking_rejects_explicit_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--compare-thinking already runs both sides; --thinking is ambiguous."""
    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}],
    )
    monkeypatch.delenv("ATTUNE_RAG_FAITHFULNESS_THINKING", raising=False)
    pipeline = _FakePipeline({"auth": ["a.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(
            [
                "--queries",
                str(p),
                "--with-faithfulness",
                "--compare-thinking",
                "--thinking",
            ]
        )
    assert rc == 2
    assert "redundant" in capsys.readouterr().err


def test_main_compare_thinking_rejects_native_citations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """4-way comparison would be too expensive; force separate runs."""
    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}],
    )
    pipeline = _FakePipeline({"auth": ["a.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(
            [
                "--queries",
                str(p),
                "--with-faithfulness",
                "--compare-thinking",
                "--native-citations",
            ]
        )
    assert rc == 2
    assert "cannot" in capsys.readouterr().err


def test_main_json_without_faithfulness_dumps_retrieval_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--json` without `--with-faithfulness` now emits a retrieval-only dump.

    Enables the CI quality gate to dump retrieval metrics on PRs
    that don't qualify for the (expensive) faithfulness pass.
    The dump shape is additive: `retrieval` + `queries_path`, no
    `faithfulness_legacy`.
    """
    import json

    p = _write_queries(
        tmp_path / "q.yaml",
        [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}],
    )
    out_path = tmp_path / "out.json"
    pipeline = _FakePipeline({"auth": ["a.md"]})
    with patch("attune_rag.RagPipeline", return_value=pipeline):
        rc = main(["--queries", str(p), "--json", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert "retrieval" in payload
    assert "queries_path" in payload
    assert "faithfulness_legacy" not in payload
    assert payload["retrieval"]["precision_at_1"] == 1.0


# ---------------------------------------------------------------------------
# Print + dump helpers
# ---------------------------------------------------------------------------


def _faithfulness_report(
    *,
    mean: float = 1.0,
    refusal: float = 0.0,
    hallu: float = 0.0,
    cite: float = 0.0,
    mean_lat: float = 100.0,
    p95_lat: float = 200.0,
    per_query: list[dict] | None = None,
) -> dict:
    """Build a faithfulness-report dict with the shape _score_faithfulness emits."""
    return {
        "mean_faithfulness": mean,
        "refusal_rate": refusal,
        "hallucination_rate": hallu,
        "citation_emit_rate": cite,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "per_query": per_query or [],
    }


def _per_query(
    qid: str,
    *,
    score: float = 1.0,
    supported: int = 1,
    unsupported: int = 0,
    reasoning: str = "",
) -> dict:
    return {
        "id": qid,
        "query": f"q for {qid}",
        "score": score,
        "supported": supported,
        "unsupported": unsupported,
        "supported_claims": [f"s{i}" for i in range(supported)],
        "unsupported_claims": [f"u{i}" for i in range(unsupported)],
        "reasoning": reasoning,
        "latency_ms": 100.0,
        "claim_citation_count": 0,
        "used_native_citations": False,
        "thinking_used": False,
    }


def test_print_side_by_side_with_custom_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from attune_rag.benchmark import _print_side_by_side

    a = _faithfulness_report(mean=0.80)
    b = _faithfulness_report(mean=0.95)
    _print_side_by_side(a, b, a_label="off", b_label="on")
    out = capsys.readouterr().out
    assert "off" in out and "on" in out
    assert "0.800" in out and "0.950" in out
    assert "+0.150" in out


def test_print_per_query_compare_counts_verdict_shifts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from attune_rag.benchmark import _print_per_query_compare

    a = _faithfulness_report(
        per_query=[
            _per_query("q1", score=1.0, supported=2, unsupported=0),
            _per_query("q2", score=0.5, supported=1, unsupported=1),
            _per_query("q3", score=1.0, supported=1, unsupported=0),
        ]
    )
    b = _faithfulness_report(
        per_query=[
            _per_query("q1", score=1.0, supported=2, unsupported=0),  # same
            _per_query("q2", score=1.0, supported=2, unsupported=0),  # shifted
            _per_query("q3", score=0.5, supported=1, unsupported=1),  # shifted
        ]
    )
    _print_per_query_compare(a, b, a_label="off", b_label="on")
    out = capsys.readouterr().out
    assert "Verdict-shift rate: 2/3" in out


def test_print_per_query_compare_handles_no_overlap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guard: returns silently if A and B share no query IDs."""
    from attune_rag.benchmark import _print_per_query_compare

    a = _faithfulness_report(per_query=[_per_query("q1")])
    b = _faithfulness_report(per_query=[_per_query("q2")])
    _print_per_query_compare(a, b, a_label="off", b_label="on")
    assert capsys.readouterr().out == ""


def test_dump_json_writes_indented_payload(tmp_path: Path) -> None:
    import json

    from attune_rag.benchmark import _dump_json

    out = tmp_path / "subdir" / "report.json"
    _dump_json(out, {"x": 1, "y": [2, 3]})
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == {"x": 1, "y": [2, 3]}
