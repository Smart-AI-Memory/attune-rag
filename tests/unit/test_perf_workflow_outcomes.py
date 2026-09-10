"""Execute the perf workflow's delta gate without measuring or calling providers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_BASELINE = REPO_ROOT / "docs/specs/downstream-validation/perf-thresholds.json"
SELECTED_CPU = ("keyword_retriever_retrieve.cpu", "rag_pipeline_run.cpu")
ADVISORY = (
    "keyword_retriever_retrieve.wall",
    "rag_pipeline_run.wall",
    "directory_corpus_load.cpu",
    "directory_corpus_load.wall",
    "llm_reranker_rerank.cpu",
    "llm_reranker_rerank.wall",
)

# Resolve `python` to this test interpreter. Every formatter call remains real.
_REAL_PYTHON = 'python() { "$PERF_REAL_PYTHON" "$@"; }\n'


def _job() -> dict[str, Any]:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/perf.yml").read_text(encoding="utf-8")
    )
    return workflow["jobs"]["delta-check"]


def _step(step_id: str) -> dict[str, Any]:
    return next(step for step in _job()["steps"] if step.get("id") == step_id)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def perf_inputs(tmp_path: Path) -> dict[str, str]:
    baseline, current = tmp_path / "baseline.json", tmp_path / "current.json"
    # Use the actual locked schema and values, copied into a disposable fixture.
    # Exact baseline means are below their thresholds and make a passing control.
    _write(baseline, _read(LOCKED_BASELINE))
    _write(current, _read(LOCKED_BASELINE))
    return {
        "BASELINE_JSON": str(baseline),
        "CURRENT_JSON": str(current),
        "CURRENT_MD": str(tmp_path / "current.md"),
        "COMMENT_PATH": str(tmp_path / "comment.md"),
        "GITHUB_OUTPUT": str(tmp_path / "delta-output"),
    }


def _run_shell(
    script: str,
    env: dict[str, str],
    *,
    prelude: str = _REAL_PYTHON,
) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Replaying GitHub bash steps requires bash")
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", prelude + script],
        cwd=REPO_ROOT,
        env={**os.environ, **env, "PERF_REAL_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=20,
    )


def _run_delta(env: dict[str, str], outcome: str = "success") -> subprocess.CompletedProcess[str]:
    result = _run_shell(_step("delta")["run"], {**env, "MEASURE_OUTCOME": outcome})
    output = Path(env["GITHUB_OUTPUT"]).read_text(encoding="utf-8")
    assert output.splitlines() == [f"rc={result.returncode}"], result.stderr
    return result


def test_successful_measurement_and_cpu_metrics_pass(perf_inputs: dict[str, str]) -> None:
    result = _run_delta(perf_inputs)
    assert result.returncode == 0, result.stderr
    comment = Path(perf_inputs["COMMENT_PATH"]).read_text(encoding="utf-8")
    assert "within baseline" in comment
    assert all(metric in comment for metric in SELECTED_CPU)


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", "unknown", ""])
def test_unsuccessful_measurement_cannot_use_stale_passing_data(
    perf_inputs: dict[str, str], outcome: str
) -> None:
    stale_bytes = Path(perf_inputs["CURRENT_JSON"]).read_bytes()
    result = _run_delta(perf_inputs, outcome)
    assert result.returncode == 2, result.stderr
    assert Path(perf_inputs["CURRENT_JSON"]).read_bytes() == stale_bytes
    comment = Path(perf_inputs["COMMENT_PATH"]).read_text(encoding="utf-8")
    assert "measurement failed" in comment
    assert "validation failed" in comment


def test_measurement_process_crash_removes_previous_outputs_and_blocks_delta(
    perf_inputs: dict[str, str],
) -> None:
    Path(perf_inputs["CURRENT_MD"]).write_text("old measurement", encoding="utf-8")
    # Execute the real measure step; replace only its measurement process with
    # an observed crash so there are no expensive timings or provider calls.
    measure = _run_shell(
        _step("measure")["run"],
        perf_inputs,
        prelude="""
python() {
  if [ "$1" != "scripts/measure_perf_baseline.py" ]; then return 97; fi
  if [ -e "$CURRENT_JSON" ] || [ -e "$CURRENT_MD" ]; then return 98; fi
  echo "Synthetic measurement process crash" >&2
  return 23
}
""",
    )
    assert measure.returncode == 23, measure.stderr
    assert not Path(perf_inputs["CURRENT_JSON"]).exists()
    assert not Path(perf_inputs["CURRENT_MD"]).exists()
    assert _run_delta(perf_inputs, "failure").returncode == 2


@pytest.mark.parametrize("metric", SELECTED_CPU)
@pytest.mark.parametrize("target", ["CURRENT_JSON", "BASELINE_JSON"])
def test_missing_selected_cpu_metric_fails_validation(
    perf_inputs: dict[str, str], metric: str, target: str
) -> None:
    payload = _read(perf_inputs[target])
    del payload["metrics"][metric]
    _write(perf_inputs[target], payload)
    result = _run_delta(perf_inputs)
    assert result.returncode == 2, result.stderr
    assert metric in result.stderr


@pytest.mark.parametrize("metric", SELECTED_CPU)
@pytest.mark.parametrize(
    ("target", "field"),
    [("CURRENT_JSON", "mean"), ("BASELINE_JSON", "mean"), ("BASELINE_JSON", "threshold")],
)
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -0.01, "0.001", True, False],
    ids=["nan", "infinity", "negative", "numeric-string", "true", "false"],
)
def test_invalid_selected_cpu_values_cannot_pass(
    perf_inputs: dict[str, str], metric: str, target: str, field: str, value: object
) -> None:
    payload = _read(perf_inputs[target])
    payload["metrics"][metric][field] = value
    _write(perf_inputs[target], payload)
    result = _run_delta(perf_inputs)
    assert result.returncode == 2, result.stderr
    assert metric in result.stderr


@pytest.mark.parametrize("metric", SELECTED_CPU)
def test_genuine_selected_cpu_regression_fails_job(
    perf_inputs: dict[str, str], metric: str
) -> None:
    payload = _read(perf_inputs["CURRENT_JSON"])
    payload["metrics"][metric]["mean"] = payload["metrics"][metric]["threshold"] * 2
    _write(perf_inputs["CURRENT_JSON"], payload)
    result = _run_delta(perf_inputs)
    assert result.returncode == 1, result.stderr
    comment = Path(perf_inputs["COMMENT_PATH"]).read_text(encoding="utf-8")
    assert "REGRESSION (gating)" in comment
    assert f"`{metric}`" in comment
    assert "⛔ blocking" in comment


@pytest.mark.parametrize("metric", ADVISORY)
def test_other_performance_regressions_remain_advisory(
    perf_inputs: dict[str, str], metric: str
) -> None:
    payload = _read(perf_inputs["CURRENT_JSON"])
    payload["metrics"][metric]["mean"] = payload["metrics"][metric]["threshold"] * 2
    _write(perf_inputs["CURRENT_JSON"], payload)
    result = _run_delta(perf_inputs)
    assert result.returncode == 0, result.stderr
    comment = Path(perf_inputs["COMMENT_PATH"]).read_text(encoding="utf-8")
    assert "possible regression" in comment
    assert f"`{metric}`" in comment
    assert "⚠️ over threshold" in comment
    assert "⛔ blocking" not in comment


@pytest.mark.parametrize("target", ["CURRENT_JSON", "BASELINE_JSON"])
@pytest.mark.parametrize("defect", ["missing", "corrupt", "non-object", "no-metrics"])
def test_required_input_failure_cannot_become_advisory_success(
    perf_inputs: dict[str, str], target: str, defect: str
) -> None:
    path = Path(perf_inputs[target])
    if defect == "missing":
        path.unlink()
    else:
        path.write_text({"corrupt": "{", "non-object": "[]", "no-metrics": "{}"}[defect])
    result = _run_delta(perf_inputs)
    assert result.returncode == 2, result.stderr


def test_workflow_keeps_exact_two_cpu_gates_and_propagates_delta_failure() -> None:
    job, delta = _job(), _step("delta")
    assert re.findall(r"--gate-metric\s+(\S+)", delta["run"]) == list(SELECTED_CPU)
    assert delta["env"]["MEASURE_OUTCOME"] == "${{ steps.measure.outcome }}"
    assert delta["if"] == "always()"
    assert _step("measure")["continue-on-error"] is True
    assert delta.get("continue-on-error", False) is False
    assert job.get("continue-on-error", False) is False
