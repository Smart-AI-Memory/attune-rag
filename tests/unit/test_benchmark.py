"""Tests for attune_rag.benchmark — CLI exit codes, aggregation, helpers.

Targets the highest coverage gap identified in the test-strategy audit:
``benchmark.py`` was at 10% line coverage. These tests exercise the pure
helpers (`_percentile`, `_load_queries`) and the CLI happy + error paths
without spending API tokens (`--with-faithfulness` is gated behind a
real ANTHROPIC_API_KEY check we don't satisfy here).
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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


# ---------------------------------------------------------------------------
# --retriever transformer (usability audit 2026-06-10, step 3)
# ---------------------------------------------------------------------------


def test_main_retriever_transformer_missing_extra_clean_error(
    monkeypatch, tmp_path, capsys
) -> None:
    """--retriever transformer without the extra exits 2 with the install
    hint, not a traceback."""
    import yaml

    from attune_rag import benchmark as bench_mod
    from attune_rag.embedding import EmbeddingRetriever

    queries = tmp_path / "queries.yaml"
    queries.write_text(
        yaml.safe_dump(
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "anything",
                        "expected_feature": "pipeline",
                        "expected_in_top_3": ["concepts/pipeline.md"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def _raise(self, corpus):
        raise RuntimeError(
            "TransformerRetriever requires the [transformers] extra. "
            "Install with: pip install 'attune-rag[transformers]'"
        )

    monkeypatch.setattr(EmbeddingRetriever, "_corpus_matrix", _raise)
    rc = bench_mod.main(["--queries", str(queries), "--retriever", "transformer"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "[transformers] extra" in err


def test_main_default_queries_missing_explains_pip_install(monkeypatch, tmp_path, capsys) -> None:
    """When the REPO-DEFAULT queries path is absent (pip install), the
    error explains where the golden sets live; an explicit --queries
    miss keeps the short message."""
    from attune_rag import benchmark as bench_mod

    missing = tmp_path / "golden" / "queries.yaml"
    monkeypatch.setattr(bench_mod, "_default_queries_path", lambda: missing)
    rc = bench_mod.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Queries file not found" in err
    assert "repo checkout" in err
    assert "attune-rag-measure" in err


def test_main_explicit_queries_missing_keeps_short_error(tmp_path, capsys) -> None:
    from attune_rag import benchmark as bench_mod

    rc = bench_mod.main(["--queries", str(tmp_path / "nope.yaml")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Queries file not found" in err
    assert "repo checkout" not in err


@pytest.fixture
def cli_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A real retrieval-report shape and a provider stub; no outbound calls."""
    from attune_rag import benchmark

    queries = [{"id": "q1", "query": "auth", "expected_in_top_3": ["a.md"]}]
    path = _write_queries(tmp_path / "queries.yaml", queries)
    output = tmp_path / "benchmark.json"
    with patch("attune_rag.RagPipeline", return_value=_FakePipeline({"auth": ["a.md"]})):
        retrieval = _run_benchmark(queries, 3)
    monkeypatch.setattr(benchmark, "_run_benchmark", lambda *a, **kw: retrieval)
    for name in ("_default_negatives_path", "_default_extended_path", "_default_corpus_b_path"):
        monkeypatch.setattr(benchmark, name, lambda: tmp_path / "unused-advisory-input")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-test-key")
    primary = _faithfulness_report(per_query=[_per_query("q1")])
    scorer = AsyncMock(return_value=primary)
    monkeypatch.setattr(benchmark, "_score_faithfulness", scorer)
    return SimpleNamespace(
        args=["--queries", str(path), "--json", str(output)],
        path=path,
        output=output,
        retrieval=retrieval,
        primary=primary,
        scorer=scorer,
    )


@pytest.mark.parametrize(("precision", "exit_code"), [(0.50, 1), (0.75, 0)])
def test_cli_records_retrieval_on_both_sides_of_its_cutoff(
    cli_measurement: SimpleNamespace, precision: float, exit_code: int
) -> None:
    cli_measurement.retrieval["precision_at_1"] = precision

    assert main(cli_measurement.args) == exit_code

    report = json.loads(cli_measurement.output.read_text())
    assert report["retrieval"]["precision_at_1"] == precision
    assert report["outcomes"]["retrieval"] == "completed"
    assert report["outcomes"]["faithfulness"] == "skipped"
    if exit_code == 1:
        assert report["outcomes"]["reason"] == "retrieval_regression"
    cli_measurement.scorer.assert_not_awaited()


def test_retrieval_regression_prevents_requested_provider_calls(
    cli_measurement: SimpleNamespace,
) -> None:
    cli_measurement.retrieval["precision_at_1"] = 0.50

    assert main([*cli_measurement.args, "--with-faithfulness"]) == 1

    report = json.loads(cli_measurement.output.read_text())
    assert report["outcomes"] == {
        "retrieval": "completed",
        "faithfulness": "skipped",
        "reason": "retrieval_regression",
    }
    cli_measurement.scorer.assert_not_awaited()


@pytest.mark.parametrize(("faithfulness", "exit_code"), [(0.50, 1), (0.90, 0)])
def test_completed_primary_result_survives_internal_faithfulness_cutoff(
    cli_measurement: SimpleNamespace, faithfulness: float, exit_code: int
) -> None:
    cli_measurement.primary["mean_faithfulness"] = faithfulness
    cli_measurement.primary["per_query"][0]["score"] = faithfulness

    assert main([*cli_measurement.args, "--with-faithfulness"]) == exit_code

    report = json.loads(cli_measurement.output.read_text())
    assert report["outcomes"]["faithfulness"] == "completed"
    assert report["faithfulness_legacy"]["mean_faithfulness"] == faithfulness
    assert report["retrieval"] == cli_measurement.retrieval
    assert cli_measurement.scorer.await_count == 1


def test_credential_failure_preserves_completed_retrieval(
    cli_measurement: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    assert main([*cli_measurement.args, "--with-faithfulness"]) == 2

    report = json.loads(cli_measurement.output.read_text())
    assert report["retrieval"] == cli_measurement.retrieval
    assert report["outcomes"] == {
        "retrieval": "completed",
        "faithfulness": "failed",
        "reason": "missing_credentials",
    }
    cli_measurement.scorer.assert_not_awaited()


def _provider_exception(kind: str) -> Exception:
    anthropic = pytest.importorskip("anthropic")
    import httpx

    request = httpx.Request("POST", "https://provider.invalid", headers={"x-api-key": "private"})
    secret = "SENSITIVE_PROVIDER_PAYLOAD"
    if kind == "connection":
        return anthropic.APIConnectionError(message=secret, request=request)
    if kind == "timeout":
        return anthropic.APITimeoutError(request=request)
    if kind == "runtime":
        return RuntimeError(f"503 overloaded timeout rate limit: {secret}")
    if kind == "valueerror":
        return ValueError(secret)
    if kind == "schema":
        return anthropic.APIResponseValidationError(
            response=httpx.Response(200, request=request), body={"secret": secret}, message=secret
        )
    status = int(kind)
    error_type = anthropic.RateLimitError if status == 429 else anthropic.APIStatusError
    return error_type(
        secret, response=httpx.Response(status, request=request), body={"secret": secret}
    )


@pytest.mark.parametrize(
    ("kind", "exit_code"),
    [
        ("connection", 3),
        ("timeout", 3),
        ("429", 3),
        ("500", 3),
        ("503", 3),
        ("599", 3),
        ("400", 2),
        ("401", 2),
        ("403", 2),
        ("schema", 2),
        ("runtime", 2),
        ("valueerror", 2),
    ],
)
def test_only_typed_primary_provider_outages_are_retryable(
    cli_measurement: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    exit_code: int,
) -> None:
    cli_measurement.scorer.side_effect = _provider_exception(kind)

    assert main([*cli_measurement.args, "--with-faithfulness"]) == exit_code

    raw = cli_measurement.output.read_text()
    report = json.loads(raw)
    assert report["retrieval"] == cli_measurement.retrieval
    assert report["outcomes"]["retrieval"] == "completed"
    assert report["outcomes"]["faithfulness"] == ("unavailable" if exit_code == 3 else "failed")
    assert "faithfulness_legacy" not in report
    assert "SENSITIVE_PROVIDER_PAYLOAD" not in raw + capsys.readouterr().err
    assert cli_measurement.scorer.await_count == 1


@pytest.mark.parametrize("comparison", ["--native-citations", "--compare-thinking"])
def test_secondary_outage_preserves_primary_checkpoint_and_is_not_retryable(
    cli_measurement: SimpleNamespace, comparison: str
) -> None:
    primary_key = (
        "faithfulness_thinking_off" if comparison == "--compare-thinking" else "faithfulness_legacy"
    )
    calls = []

    async def score(*args, **kwargs):
        calls.append(kwargs)
        checkpoint = json.loads(cli_measurement.output.read_text())
        assert checkpoint["outcomes"]["retrieval"] == "completed"
        if len(calls) == 1:
            assert checkpoint["outcomes"]["faithfulness"] == "pending"
            return cli_measurement.primary
        assert checkpoint["outcomes"]["faithfulness"] == "completed"
        assert checkpoint[primary_key] == cli_measurement.primary
        raise _provider_exception("503")

    cli_measurement.scorer.side_effect = score

    assert main([*cli_measurement.args, "--with-faithfulness", comparison]) == 2

    report = json.loads(cli_measurement.output.read_text())
    assert report[primary_key] == cli_measurement.primary
    assert report["outcomes"]["faithfulness"] == "completed"
    assert report["outcomes"]["reason"].startswith("secondary_failed:")
    assert len(calls) == 2


@pytest.mark.parametrize("comparison", ["--native-citations", "--compare-thinking"])
def test_primary_regression_is_gated_before_optional_comparison(
    cli_measurement: SimpleNamespace, comparison: str
) -> None:
    cli_measurement.primary["mean_faithfulness"] = 0.5
    cli_measurement.primary["per_query"][0]["score"] = 0.5
    cli_measurement.scorer.side_effect = [cli_measurement.primary, _provider_exception("503")]

    assert main([*cli_measurement.args, "--with-faithfulness", comparison]) == 1

    report = json.loads(cli_measurement.output.read_text())
    assert report["outcomes"]["faithfulness"] == "completed"
    assert report["outcomes"]["reason"] == "faithfulness_regression"
    assert cli_measurement.scorer.await_count == 1


@pytest.mark.parametrize("phase", ["retrieval", "primary"])
@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1, True, "0.9"]
)
def test_invalid_required_metrics_fail_without_claiming_completion(
    cli_measurement: SimpleNamespace, phase: str, invalid
) -> None:
    if phase == "retrieval":
        cli_measurement.retrieval["precision_at_1"] = invalid
    else:
        cli_measurement.primary["mean_faithfulness"] = invalid

    assert main([*cli_measurement.args, "--with-faithfulness"]) == 2

    report = json.loads(cli_measurement.output.read_text())
    assert report["outcomes"]["faithfulness"] == "failed"
    assert "faithfulness_legacy" not in report
    if phase == "retrieval":
        assert report["outcomes"]["retrieval"] == "failed"
        assert "retrieval" not in report
    else:
        assert report["outcomes"]["retrieval"] == "completed"
        assert report["retrieval"] == cli_measurement.retrieval


@pytest.mark.parametrize("defect", ["missing-metric", "missing-row", "nonfinite-row"])
def test_incomplete_or_invalid_primary_data_is_a_hard_failure(
    cli_measurement: SimpleNamespace, defect: str
) -> None:
    if defect == "missing-metric":
        del cli_measurement.primary["mean_faithfulness"]
    elif defect == "missing-row":
        cli_measurement.primary["per_query"].clear()
    else:
        cli_measurement.primary["per_query"][0]["score"] = float("nan")

    assert main([*cli_measurement.args, "--with-faithfulness"]) == 2

    report = json.loads(cli_measurement.output.read_text())
    assert report["outcomes"]["faithfulness"] == "failed"
    assert "faithfulness_legacy" not in report


@pytest.mark.parametrize(
    "defect", ["missing-queries", "malformed-yaml", "bad-options", "local-exception"]
)
def test_new_failed_run_cannot_leave_a_stale_passing_report(
    cli_measurement: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    from attune_rag import benchmark

    cli_measurement.output.write_text('{"outcomes":{"retrieval":"completed"},"old_run":true}')
    args = list(cli_measurement.args)
    if defect == "missing-queries":
        cli_measurement.path.unlink()
    elif defect == "malformed-yaml":
        cli_measurement.path.write_text("queries: [\n")
    elif defect == "bad-options":
        args.append("--compare-thinking")
    else:

        def fail(*args, **kwargs):
            raise ValueError("local computation failure")

        monkeypatch.setattr(benchmark, "_run_benchmark", fail)

    assert main(args) == 2

    report = json.loads(cli_measurement.output.read_text())
    assert "old_run" not in report
    assert report["outcomes"]["retrieval"] == "failed"


def test_actual_variance_parser_still_consumes_benchmark_stdout(
    cli_measurement: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_measurement.primary["mean_faithfulness"] = 0.97
    cli_measurement.primary["per_query"][0]["score"] = 0.97

    assert main([*cli_measurement.args, "--with-faithfulness"]) == 0

    output = capsys.readouterr().out
    parser = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/measure_baseline_variance.py")
    )["parse_metrics"]
    assert parser(output) == {
        "precision_at_1": 1.0,
        "recall_at_3": 1.0,
        "mean_faithfulness": 0.97,
    }


@pytest.mark.parametrize(
    ("option", "value", "diagnostic"),
    [
        ("--min-precision", "nan", "min-precision"),
        ("--min-precision", "-0.1", "min-precision"),
        ("--min-precision", "1.1", "min-precision"),
        ("--min-faithfulness", "inf", "min-faithfulness"),
        ("--min-faithfulness", "-0.1", "min-faithfulness"),
        ("--min-faithfulness", "1.1", "min-faithfulness"),
        ("-k", "0", "k must be positive"),
        ("-k", "-1", "k must be positive"),
    ],
)
def test_cli_invalid_limits_keep_actionable_safe_diagnostics(
    cli_measurement: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
    diagnostic: str,
) -> None:
    assert main([*cli_measurement.args, option, value]) == 2

    report = json.loads(cli_measurement.output.read_text())
    detail = report["outcomes"]["detail"]
    assert diagnostic in detail
    if option != "-k":
        assert "finite number between 0 and 1" in detail
    assert detail in capsys.readouterr().err
    assert report["outcomes"]["retrieval"] == "failed"
    cli_measurement.scorer.assert_not_awaited()


@pytest.mark.parametrize(
    ("document", "diagnostic"),
    [
        ({}, "No queries found"),
        ({"queries": []}, "No queries found"),
        ({"queries": ["PRIVATE_QUERY_TEXT"]}, "nonempty string id and query"),
        ({"queries": [{"id": "q1", "query": 123}]}, "nonempty string id and query"),
        (
            {"queries": [{"id": "q1", "query": "PRIVATE_QUERY_TEXT"}] * 2},
            "Query IDs must be unique",
        ),
        (
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "PRIVATE_QUERY_TEXT",
                        "expected_in_top_3": "PRIVATE_EXPECTED_PATH",
                    }
                ]
            },
            "Expected retrieval paths must be a list of nonempty strings",
        ),
    ],
)
def test_cli_query_schema_errors_describe_the_fix_without_echoing_data(
    cli_measurement: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    document: dict,
    diagnostic: str,
) -> None:
    cli_measurement.path.write_text(yaml.safe_dump(document))

    assert main([*cli_measurement.args, "--with-faithfulness"]) == 2

    raw = cli_measurement.output.read_text()
    report = json.loads(raw)
    assert diagnostic in report["outcomes"]["detail"]
    stderr = capsys.readouterr().err
    assert diagnostic in stderr
    assert "PRIVATE_QUERY_TEXT" not in raw + stderr
    assert "PRIVATE_EXPECTED_PATH" not in raw + stderr
    cli_measurement.scorer.assert_not_awaited()


@pytest.mark.parametrize(
    ("content", "diagnostic"),
    [
        (b"queries: [PRIVATE_QUERY_TEXT\n", "Invalid YAML"),
        (b"queries: PRIVATE_QUERY_TEXT\xff", "must be UTF-8 text"),
    ],
)
def test_query_parse_errors_show_file_context_without_raw_input(
    cli_measurement: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    content: bytes,
    diagnostic: str,
) -> None:
    cli_measurement.path.write_bytes(content)

    assert main(cli_measurement.args) == 2

    raw = cli_measurement.output.read_text()
    report = json.loads(raw)
    detail = report["outcomes"]["detail"]
    assert diagnostic in detail and repr(str(cli_measurement.path)) in detail
    stderr = capsys.readouterr().err
    assert detail in stderr
    assert "PRIVATE_QUERY_TEXT" not in raw + stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    ("field", "value"), [("precision_at_1", float("nan")), ("mean_latency_ms", 10**400)]
)
def test_numeric_measurement_diagnostics_name_the_invalid_field(
    cli_measurement: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value,
) -> None:
    cli_measurement.retrieval[field] = value

    assert main(cli_measurement.args) == 2

    report = json.loads(cli_measurement.output.read_text())
    detail = report["outcomes"]["detail"]
    assert field in detail and "finite number" in detail
    assert detail in capsys.readouterr().err
    assert "retrieval" not in report


@pytest.mark.parametrize("phase", ["retrieval", "secondary"])
def test_arbitrary_value_errors_remain_redacted(
    cli_measurement: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    from attune_rag import benchmark

    secret = "PRIVATE_REQUEST_PAYLOAD"
    args = list(cli_measurement.args)
    if phase == "retrieval":

        def fail(*args, **kwargs):
            raise ValueError(f"No queries found: {secret}")

        monkeypatch.setattr(benchmark, "_run_benchmark", fail)
    else:
        cli_measurement.scorer.side_effect = [cli_measurement.primary, ValueError(secret)]
        args.extend(["--with-faithfulness", "--native-citations"])

    assert main(args) == 2

    raw = cli_measurement.output.read_text()
    outcomes = json.loads(raw)["outcomes"]
    assert outcomes["reason"].endswith(":ValueError")
    assert "detail" not in outcomes
    assert secret not in raw + capsys.readouterr().err
    if phase == "secondary":
        assert outcomes["faithfulness"] == "completed"


def test_calibration_skips_both_benchmark_stages_even_when_faithfulness_requested(
    cli_measurement: SimpleNamespace, tmp_path: Path
) -> None:
    negatives = _write_queries(
        tmp_path / "negative.yaml", [{"id": "n1", "query": "unrelated question"}]
    )
    cli_measurement.scorer.side_effect = AssertionError("Calibration must not call providers")
    # Run the actual calibration and top-score collection against deterministic
    # local retrieval results; neither benchmark stage should be marked pending.
    with patch("attune_rag.RagPipeline") as pipeline:
        pipeline.return_value.run.side_effect = lambda query, k: SimpleNamespace(
            citation=SimpleNamespace(hits=[SimpleNamespace(score=5.0 if query == "auth" else 0.0)])
        )
        rc = main(
            [
                *cli_measurement.args,
                "--calibrate-abstention",
                "--negatives",
                str(negatives),
                "--with-faithfulness",
            ]
        )

    assert rc == 0
    report = json.loads(cli_measurement.output.read_text())
    assert report["outcomes"] == {
        "retrieval": "skipped",
        "faithfulness": "skipped",
        "reason": "calibration_only",
    }
    assert report["calibration"]["recommended_threshold"] == 1.0
    assert "retrieval" not in report and "faithfulness_legacy" not in report
    cli_measurement.scorer.assert_not_awaited()
