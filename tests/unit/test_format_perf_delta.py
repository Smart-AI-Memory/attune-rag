"""Tests for ``scripts/format_perf_delta.py`` — Phase 4 W0.5 helper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "format_perf_delta.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("format_perf_delta", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["format_perf_delta"] = module
    spec.loader.exec_module(module)
    return module


fpd = _load_module()


def _metric(mean: float, stdev: float, threshold: float, n: int = 30) -> dict[str, float]:
    return {"mean": mean, "stdev": stdev, "threshold": threshold, "n": n}


def _payload(metrics: dict[str, dict[str, float]]) -> dict:
    return {
        "measured_at": "2026-05-19T00:00:00Z",
        "commit": "deadbeef",
        "runs": 30,
        "sigma": 2.0,
        "environment": {"platform": "Linux"},
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


def test_compare_classifies_ok_when_current_below_threshold() -> None:
    baseline = {"bench.cpu": _metric(0.001, 0.0001, 0.0012)}
    current = {"bench.cpu": _metric(0.0011, 0.0001, 0.0013)}
    comps = fpd.compare(baseline, current)
    assert len(comps) == 1
    assert comps[0].status == "ok"
    assert comps[0].delta_pct == pytest.approx(10.0, abs=0.01)


def test_compare_classifies_regression_when_current_above_threshold() -> None:
    baseline = {"bench.cpu": _metric(0.001, 0.0001, 0.0012)}
    current = {"bench.cpu": _metric(0.0013, 0.0001, 0.0015)}  # above 0.0012
    comps = fpd.compare(baseline, current)
    assert comps[0].status == "regression"


def test_compare_classifies_new_when_only_in_current() -> None:
    baseline: dict[str, dict[str, float]] = {}
    current = {"new_bench.wall": _metric(0.005, 0.0005, 0.006)}
    comps = fpd.compare(baseline, current)
    assert comps[0].status == "new"
    assert comps[0].baseline_mean is None
    assert comps[0].delta_pct is None


def test_compare_skips_metrics_only_in_baseline() -> None:
    """A benchmark removed from current isn't a regression — out of scope."""
    baseline = {"removed.cpu": _metric(0.001, 0.0001, 0.0012)}
    current: dict[str, dict[str, float]] = {}
    comps = fpd.compare(baseline, current)
    assert comps == []


def test_compare_preserves_alphabetical_order_in_comparisons() -> None:
    baseline = {
        "z_bench.cpu": _metric(0.001, 0.0001, 0.0012),
        "a_bench.cpu": _metric(0.001, 0.0001, 0.0012),
    }
    current = {
        "z_bench.cpu": _metric(0.001, 0.0001, 0.0012),
        "a_bench.cpu": _metric(0.001, 0.0001, 0.0012),
    }
    comps = fpd.compare(baseline, current)
    assert [c.metric for c in comps] == ["a_bench.cpu", "z_bench.cpu"]


def test_delta_pct_handles_zero_baseline_mean_safely() -> None:
    """Division-by-zero guard — if baseline.mean is 0, delta_pct is None."""
    baseline = {"bench.cpu": _metric(0.0, 0.0, 0.0)}
    current = {"bench.cpu": _metric(0.001, 0.0001, 0.0012)}
    comps = fpd.compare(baseline, current)
    assert comps[0].delta_pct is None


# ---------------------------------------------------------------------------
# render_comparison_comment
# ---------------------------------------------------------------------------


def test_render_advisory_softens_phrasing() -> None:
    comps = [
        fpd.MetricComparison(
            metric="bench.cpu",
            baseline_mean=0.001,
            baseline_threshold=0.0012,
            current_mean=0.0015,
            status="regression",
        )
    ]
    advisory = fpd.render_comparison_comment(comps, advisory=True)
    blocking = fpd.render_comparison_comment(comps, advisory=False)
    assert "Advisory only" in advisory
    assert "REGRESSION" in blocking
    assert "Advisory only" not in blocking


def test_render_orders_regressions_first() -> None:
    comps = [
        fpd.MetricComparison(
            metric="z_ok",
            baseline_mean=0.001,
            baseline_threshold=0.0012,
            current_mean=0.0011,
            status="ok",
        ),
        fpd.MetricComparison(
            metric="a_regression",
            baseline_mean=0.001,
            baseline_threshold=0.0012,
            current_mean=0.0020,
            status="regression",
        ),
    ]
    body = fpd.render_comparison_comment(comps, advisory=True)
    # a_regression should appear before z_ok in the table.
    a_idx = body.index("a_regression")
    z_idx = body.index("z_ok")
    assert a_idx < z_idx


def test_render_includes_stable_marker() -> None:
    comps = [
        fpd.MetricComparison(
            metric="bench.cpu",
            baseline_mean=0.001,
            baseline_threshold=0.0012,
            current_mean=0.0011,
            status="ok",
        )
    ]
    body = fpd.render_comparison_comment(comps, advisory=True)
    # Marker present at top AND bottom — workflow uses it to find the
    # existing comment for in-place updates.
    assert body.count(fpd.COMMENT_MARKER) == 2


def test_render_is_deterministic() -> None:
    comps = [
        fpd.MetricComparison(
            metric="bench.cpu",
            baseline_mean=0.001,
            baseline_threshold=0.0012,
            current_mean=0.0011,
            status="ok",
        )
    ]
    first = fpd.render_comparison_comment(comps, advisory=True)
    second = fpd.render_comparison_comment(list(comps), advisory=True)
    assert first == second


def test_render_baseline_pending_comment_is_self_contained() -> None:
    body = fpd.render_baseline_pending_comment()
    assert fpd.COMMENT_MARKER in body
    assert "Baseline" in body
    assert "W0.4" in body


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_exit_0_when_clean(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(json.dumps(_payload({"bench.cpu": _metric(0.001, 0.0001, 0.0012)})))
    current.write_text(json.dumps(_payload({"bench.cpu": _metric(0.0011, 0.0001, 0.0013)})))

    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
        ]
    )
    assert rc == 0
    body = out.read_text(encoding="utf-8")
    assert "within baseline" in body


def test_main_exit_1_on_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(json.dumps(_payload({"bench.cpu": _metric(0.001, 0.0001, 0.0012)})))
    current.write_text(json.dumps(_payload({"bench.cpu": _metric(0.0020, 0.0001, 0.0022)})))

    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
        ]
    )
    assert rc == 1


def test_main_exit_0_when_baseline_missing(tmp_path: Path) -> None:
    """Pre-W0.4 state: baseline file doesn't exist yet. The workflow
    should comment 'baseline pending' and stay green."""
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    current.write_text(json.dumps(_payload({"bench.cpu": _metric(0.001, 0.0001, 0.0012)})))

    rc = fpd.main(
        [
            "--baseline",
            str(tmp_path / "missing.json"),
            "--current",
            str(current),
            "--comment-out",
            str(out),
        ]
    )
    assert rc == 0
    body = out.read_text(encoding="utf-8")
    assert "Baseline" in body
    assert "W0.4" in body


def test_main_exit_2_when_current_missing(tmp_path: Path) -> None:
    """Current file MUST exist — without it we can't compare anything,
    which is a validation error (exit 2), not 'green'."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_payload({})))

    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(tmp_path / "missing.json"),
            "--comment-out",
            str(tmp_path / "comment.md"),
        ]
    )
    assert rc == 2


def test_main_exit_2_when_current_is_malformed_json(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_payload({})))
    current.write_text("{ not valid json")

    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(tmp_path / "comment.md"),
        ]
    )
    assert rc == 2


def test_main_advisory_flag_propagates(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(json.dumps(_payload({"bench.cpu": _metric(0.001, 0.0001, 0.0012)})))
    current.write_text(json.dumps(_payload({"bench.cpu": _metric(0.0020, 0.0001, 0.0022)})))

    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
            "--advisory",
        ]
    )
    # exit code still reflects regression (1) — advisory only softens
    # the comment text; the workflow uses continue-on-error to absorb
    # the exit.
    assert rc == 1
    body = out.read_text(encoding="utf-8")
    assert "Advisory only" in body


# ---------------------------------------------------------------------------
# --gate-metric (W3.1 selective gating)
# ---------------------------------------------------------------------------


def test_gate_metric_scopes_exit_to_named_metric(tmp_path: Path) -> None:
    """When --gate-metric is set, only regressions in the named metric
    return exit 1. Other regressions still appear in the comment but
    don't fail the gate."""
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.001, 0.0001, 0.0012),
                    "llm_reranker_rerank.wall": _metric(0.5, 0.1, 0.8),
                }
            )
        )
    )
    # Both regress; only the gated one should affect exit code.
    current.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.0020, 0.0001, 0.0022),
                    "llm_reranker_rerank.wall": _metric(1.5, 0.1, 1.7),
                }
            )
        )
    )
    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
            "--gate-metric",
            "keyword_retriever_retrieve.cpu",
        ]
    )
    assert rc == 1


def test_gate_metric_returns_zero_when_only_non_gated_regresses(tmp_path: Path) -> None:
    """A regression in a non-gated metric doesn't trip the exit code."""
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.001, 0.0001, 0.0012),
                    "llm_reranker_rerank.wall": _metric(0.5, 0.1, 0.8),
                }
            )
        )
    )
    current.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.0011, 0.0001, 0.0013),
                    "llm_reranker_rerank.wall": _metric(1.5, 0.1, 1.7),
                }
            )
        )
    )
    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
            "--gate-metric",
            "keyword_retriever_retrieve.cpu",
        ]
    )
    assert rc == 0


def test_gate_metric_renders_blocking_icon_for_gated_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(
        json.dumps(_payload({"rag_pipeline_run.cpu": _metric(0.001, 0.0001, 0.0012)}))
    )
    current.write_text(
        json.dumps(_payload({"rag_pipeline_run.cpu": _metric(0.0020, 0.0001, 0.0022)}))
    )
    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
            "--gate-metric",
            "rag_pipeline_run.cpu",
        ]
    )
    assert rc == 1
    body = out.read_text(encoding="utf-8")
    assert "blocking" in body
    assert "REGRESSION (gating)" in body
    assert "rag_pipeline_run.cpu" in body


def test_gate_metric_renders_advisory_icon_for_non_gated_regression(tmp_path: Path) -> None:
    """Non-gated regressions in the same report still show ⚠️, not ⛔."""
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(
        json.dumps(
            _payload(
                {
                    "rag_pipeline_run.cpu": _metric(0.001, 0.0001, 0.0012),
                    "llm_reranker_rerank.wall": _metric(0.5, 0.1, 0.8),
                }
            )
        )
    )
    current.write_text(
        json.dumps(
            _payload(
                {
                    "rag_pipeline_run.cpu": _metric(0.0011, 0.0001, 0.0013),
                    "llm_reranker_rerank.wall": _metric(1.5, 0.1, 1.7),
                }
            )
        )
    )
    fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
            "--gate-metric",
            "rag_pipeline_run.cpu",
        ]
    )
    body = out.read_text(encoding="utf-8")
    # llm_reranker regressed but isn't gated — should NOT be marked blocking.
    reranker_row = next(line for line in body.splitlines() if "llm_reranker_rerank.wall" in line)
    assert "blocking" not in reranker_row
    assert "over threshold" in reranker_row


def test_gate_metric_title_softens_when_only_advisory_metric_regresses(tmp_path: Path) -> None:
    """When --gate-metric is set and the only regression is in a
    non-gated metric, the title should not claim 'gating' — the gate
    in fact stayed green."""
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.001, 0.0001, 0.0012),
                    "llm_reranker_rerank.wall": _metric(0.5, 0.1, 0.8),
                }
            )
        )
    )
    current.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.0011, 0.0001, 0.0013),
                    "llm_reranker_rerank.wall": _metric(2.0, 0.1, 2.2),
                }
            )
        )
    )
    rc = fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
            "--gate-metric",
            "keyword_retriever_retrieve.cpu",
        ]
    )
    assert rc == 0
    body = out.read_text(encoding="utf-8")
    assert "possible regression" in body
    assert "REGRESSION (gating)" not in body


def test_gate_metric_intro_lists_gated_metrics(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    baseline.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.001, 0.0001, 0.0012),
                    "rag_pipeline_run.cpu": _metric(0.001, 0.0001, 0.0012),
                }
            )
        )
    )
    current.write_text(
        json.dumps(
            _payload(
                {
                    "keyword_retriever_retrieve.cpu": _metric(0.0011, 0.0001, 0.0013),
                    "rag_pipeline_run.cpu": _metric(0.0011, 0.0001, 0.0013),
                }
            )
        )
    )
    fpd.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--comment-out",
            str(out),
            "--gate-metric",
            "keyword_retriever_retrieve.cpu",
            "--gate-metric",
            "rag_pipeline_run.cpu",
        ]
    )
    body = out.read_text(encoding="utf-8")
    assert "Blocking on regression" in body
    assert "keyword_retriever_retrieve.cpu" in body
    assert "rag_pipeline_run.cpu" in body


@pytest.fixture
def perf_cli(tmp_path: Path):
    files = {
        "baseline": tmp_path / "baseline.json",
        "current": tmp_path / "current.json",
        "comment": tmp_path / "comment.md",
    }
    for side in ("baseline", "current"):
        files[side].write_text(
            json.dumps(_payload({"bench.cpu": _metric(1, 0, 2)})), encoding="utf-8"
        )
    files["comment"].write_text("STALE PASS: within baseline", encoding="utf-8")
    args = [
        "--baseline",
        str(files["baseline"]),
        "--current",
        str(files["current"]),
        "--comment-out",
        str(files["comment"]),
        "--gate-metric",
        "bench.cpu",
    ]
    return files, args


@pytest.mark.parametrize("side", ["baseline", "current"])
def test_required_gate_missing_from_measurement_fails(perf_cli, capsys, side: str) -> None:
    files, args = perf_cli
    files[side].write_text(json.dumps(_payload({"other.cpu": _metric(1, 0, 2)})), encoding="utf-8")

    assert fpd.main(args) == 2
    error = capsys.readouterr().err
    assert side in error and "bench.cpu" in error and "missing" in error
    comment = files["comment"].read_text(encoding="utf-8")
    assert "validation error" in comment and "STALE PASS" not in comment


@pytest.mark.parametrize("side", ["baseline", "current"])
def test_required_gate_missing_file_fails(perf_cli, capsys, side: str) -> None:
    files, args = perf_cli
    files[side].unlink()

    assert fpd.main(args) == 2
    assert side in capsys.readouterr().err
    assert "validation error" in files["comment"].read_text(encoding="utf-8")


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize(
    "payload",
    [None, [], 0, "data", {}, {"metrics": None}, {"metrics": []}, {"metrics": 1}],
)
def test_invalid_document_shapes_fail(perf_cli, capsys, side: str, payload) -> None:
    files, args = perf_cli
    files[side].write_text(json.dumps(payload), encoding="utf-8")

    assert fpd.main(args) == 2
    error = capsys.readouterr().err
    assert side in error and "JSON object" in error
    assert "STALE PASS" not in files["comment"].read_text(encoding="utf-8")


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize("entry", [None, [], True, 1, "metric"])
def test_invalid_metric_shapes_fail(perf_cli, capsys, side: str, entry) -> None:
    files, args = perf_cli
    files[side].write_text(json.dumps(_payload({"bench.cpu": entry})), encoding="utf-8")

    assert fpd.main(args) == 2
    error = capsys.readouterr().err
    assert side in error and "bench.cpu" in error and "JSON object" in error


@pytest.mark.parametrize(
    ("side", "field"),
    [("current", "mean"), ("baseline", "mean"), ("baseline", "threshold")],
)
@pytest.mark.parametrize(
    "value",
    [None, True, False, "1", -1, float("nan"), float("inf"), -float("inf"), 10**400],
    ids=["null", "true", "false", "string", "negative", "nan", "inf", "-inf", "huge-int"],
)
def test_invalid_verdict_numbers_fail(perf_cli, capsys, side: str, field: str, value) -> None:
    files, args = perf_cli
    payload = json.loads(files[side].read_text(encoding="utf-8"))
    payload["metrics"]["bench.cpu"][field] = value
    files[side].write_text(json.dumps(payload), encoding="utf-8")

    assert fpd.main(args) == 2
    error = capsys.readouterr().err
    assert all(text in error for text in (side, "bench.cpu", field, "finite nonnegative"))
    assert "validation error" in files["comment"].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("side", "field"),
    [("current", "mean"), ("baseline", "mean"), ("baseline", "threshold")],
)
def test_missing_verdict_fields_fail(perf_cli, capsys, side: str, field: str) -> None:
    files, args = perf_cli
    payload = json.loads(files[side].read_text(encoding="utf-8"))
    del payload["metrics"]["bench.cpu"][field]
    files[side].write_text(json.dumps(payload), encoding="utf-8")

    assert fpd.main(args) == 2
    assert field in capsys.readouterr().err


def test_zero_means_and_threshold_are_valid(perf_cli) -> None:
    files, args = perf_cli
    files["baseline"].write_text(
        json.dumps(_payload({"bench.cpu": {"mean": 0, "threshold": 0}})), encoding="utf-8"
    )
    # A current threshold is not used to decide the verdict and isn't required.
    files["current"].write_text(json.dumps(_payload({"bench.cpu": {"mean": 0}})), encoding="utf-8")

    assert fpd.main(args) == 0
    assert "within baseline" in files["comment"].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "failure", ["baseline-missing", "baseline-json", "current-shape", "current-io", "utf8"]
)
def test_cli_validation_failures_exit_two_without_traceback(perf_cli, failure: str) -> None:
    files, args = perf_cli
    if failure == "baseline-missing":
        files["baseline"].unlink()
    elif failure == "baseline-json":
        files["baseline"].write_text("{bad json", encoding="utf-8")
    elif failure == "current-shape":
        files["current"].write_text("[]", encoding="utf-8")
    elif failure == "current-io":
        files["current"].unlink()
        files["current"].mkdir()
    else:
        files["current"].write_bytes(b"\xff")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "error:" in result.stderr and "Traceback" not in result.stderr
    assert "validation error" in files["comment"].read_text(encoding="utf-8")


def test_cli_output_io_failure_is_validation_error(perf_cli, capsys) -> None:
    files, args = perf_cli
    files["comment"].unlink()
    files["comment"].mkdir()

    assert fpd.main(args) == 2
    assert "could not write validation comment" in capsys.readouterr().err


CPU_GATES = frozenset({"keyword_retriever_retrieve.cpu", "rag_pipeline_run.cpu"})


@pytest.mark.parametrize("regression", [None, *sorted(CPU_GATES), "all-advisory"])
def test_active_thresholds_gate_only_the_selected_cpu_axes(
    tmp_path: Path, regression: str | None
) -> None:
    baseline_bytes = (
        REPO_ROOT / "docs/specs/downstream-validation/perf-thresholds.json"
    ).read_bytes()
    payload = json.loads(baseline_bytes)
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(baseline_bytes)
    current = tmp_path / "current.json"
    out = tmp_path / "comment.md"
    for name, metric in payload["metrics"].items():
        metric["mean"] = metric["threshold"]
        if name == regression or (regression == "all-advisory" and name not in CPU_GATES):
            metric["mean"] *= 2
    current_bytes = json.dumps(payload).encode()
    current.write_bytes(current_bytes)
    args = ["--baseline", str(baseline), "--current", str(current), "--comment-out", str(out)]
    for gate in sorted(CPU_GATES):
        args.extend(["--gate-metric", gate])

    assert fpd.main(args) == (1 if regression in CPU_GATES else 0)
    assert baseline.read_bytes() == baseline_bytes
    assert current.read_bytes() == current_bytes
    if regression == "all-advisory":
        body = out.read_text(encoding="utf-8")
        assert "possible regression" in body and "⛔" not in body
        for name in payload["metrics"].keys() - CPU_GATES:
            row = next(line for line in body.splitlines() if line.startswith(f"| `{name}`"))
            assert "over threshold" in row
