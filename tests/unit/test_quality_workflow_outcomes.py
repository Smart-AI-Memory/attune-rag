"""Replay the quality workflow with fake benchmark results and real validation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = REPO_ROOT / "docs/specs/release-quality-baseline/thresholds.json"

# Intercept only the potentially paid benchmark command. Inspectors and the
# threshold checker execute with the real interpreter and original arguments.
# Record retry delays without making the unit suite sleep for thirty seconds.
_SHELL_HARNESS = """
python() {
  if [ "$#" -ge 2 ] && [ "$1" = "-m" ] && [ "$2" = "attune_rag.benchmark" ]; then
    "$GATE_REAL_PYTHON" "$GATE_BENCHMARK_STUB" "$@"
  else
    "$GATE_REAL_PYTHON" "$@"
  fi
}
sleep() { printf '%s\\n' "$*" >> "$GATE_SLEEP_LOG"; }
"""

_BENCHMARK_STUB = """
import json
import os
import sys
from pathlib import Path

entries = json.loads(Path(os.environ["GATE_SCENARIO"]).read_text(encoding="utf-8"))
log = Path(os.environ["GATE_ATTEMPTS"])
history = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
attempt = len(history)
dump = Path(sys.argv[sys.argv.index("--json") + 1])
history.append({"dump_existed": dump.exists(), "args": sys.argv[1:]})
log.write_text(json.dumps(history), encoding="utf-8")
if attempt >= len(entries):
    raise SystemExit(98)
entry = entries[attempt]
if "dump" in entry:
    dump.write_text(json.dumps(entry["dump"]), encoding="utf-8")
elif "raw" in entry:
    dump.write_text(entry["raw"], encoding="utf-8")
raise SystemExit(entry["rc"])
"""


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8")
    )
    return workflow["jobs"]["gate"]["steps"]


def _step(step_id: str) -> dict[str, Any]:
    return next(step for step in _steps() if step.get("id") == step_id)


def _dump(
    *,
    p1: object = 1.0,
    recall: object = 1.0,
    faith: object = 0.98,
    status: str = "completed",
    reason: str | None = None,
) -> dict[str, Any]:
    report = {
        "retrieval": {"precision_at_1": p1, "recall_at_k": recall, "k": 3},
        "queries_path": str(REPO_ROOT / "tests/golden/queries.yaml"),
        "outcomes": {"retrieval": "completed", "faithfulness": status},
    }
    if status == "completed":
        report["faithfulness_legacy"] = {"mean_faithfulness": faith}
    if reason is not None:
        report["outcomes"]["reason"] = reason
    return report


def _outputs(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line
    )


def _run_shell(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Replaying GitHub bash steps requires bash")
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", _SHELL_HARNESS + script],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=20,
    )


@dataclass
class GateResult:
    bench: subprocess.CompletedProcess[str]
    check: subprocess.CompletedProcess[str]
    outputs: dict[str, str]
    attempts: list[dict[str, Any]]
    delays: list[str]


def _run_gate(
    tmp_path: Path,
    entries: list[dict[str, Any]],
    *,
    mode: str = "full",
    stale_dump: bool = False,
) -> GateResult:
    stub = tmp_path / "benchmark_stub.py"
    stub.write_text(_BENCHMARK_STUB, encoding="utf-8")
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(entries), encoding="utf-8")
    dump = tmp_path / "dump.json"
    if stale_dump:
        dump.write_text(json.dumps(_dump()), encoding="utf-8")
    bench_output, check_output = tmp_path / "bench-output", tmp_path / "check-output"
    attempts, delays = tmp_path / "attempts.json", tmp_path / "delays"
    env = {
        "GATE_REAL_PYTHON": sys.executable,
        "GATE_BENCHMARK_STUB": str(stub),
        "GATE_SCENARIO": str(scenario),
        "GATE_ATTEMPTS": str(attempts),
        "GATE_SLEEP_LOG": str(delays),
        "MODE": mode,
        "DUMP_PATH": str(dump),
        "THRESHOLDS_PATH": str(THRESHOLDS),
        "COMMENT_PATH": str(tmp_path / "comment.md"),
        "GITHUB_OUTPUT": str(bench_output),
    }
    bench = _run_shell(_step("bench")["run"], env)
    assert bench.returncode == 0, bench.stderr
    env.update(BENCH_RC=_outputs(bench_output)["rc"], GITHUB_OUTPUT=str(check_output))
    check = _run_shell(_step("check")["run"], env)
    outputs = _outputs(check_output)
    assert outputs["rc"] == str(check.returncode), check.stderr
    return GateResult(
        bench,
        check,
        outputs,
        json.loads(attempts.read_text(encoding="utf-8")),
        delays.read_text(encoding="utf-8").splitlines() if delays.exists() else [],
    )


@pytest.mark.parametrize(
    ("mode", "entry", "expected_rc"),
    [
        ("retrieval-only", {"rc": 0, "dump": _dump(status="skipped")}, 0),
        ("full", {"rc": 0, "dump": _dump()}, 0),
        (
            "full",
            {"rc": 1, "dump": _dump(p1=0.50, status="skipped", reason="retrieval_regression")},
            1,
        ),
        ("full", {"rc": 0, "dump": _dump(p1=0.90)}, 1),
        ("full", {"rc": 0, "dump": _dump(faith=0.90)}, 1),
        ("full", {"rc": 1, "dump": _dump(faith=0.50)}, 1),
        ("full", {"rc": 2, "dump": _dump(status="failed")}, 2),
        ("full", {"rc": 42, "dump": _dump()}, 2),
        ("full", {"rc": 2, "dump": _dump(faith=0.90)}, 1),
        ("full", {"rc": 0, "dump": _dump(status="skipped")}, 2),
        ("full", {"rc": 0, "dump": _dump(status="pending")}, 2),
        ("retrieval-only", {"rc": 3, "dump": _dump(status="unavailable")}, 2),
        ("full", {"rc": 3, "dump": _dump(status="failed")}, 2),
        ("full", {"rc": 3, "dump": _dump()}, 2),
    ],
    ids=[
        "intentional-retrieval-only",
        "full-pass",
        "severe-retrieval-regression",
        "locked-retrieval-regression",
        "locked-faithfulness-regression",
        "severe-faithfulness-regression",
        "local-error",
        "unexpected-exit",
        "completed-faithfulness-before-error",
        "full-cannot-skip",
        "full-cannot-pend",
        "retrieval-only-cannot-claim-outage",
        "unclassified-failure-cannot-retry",
        "completed-result-cannot-claim-outage",
    ],
)
def test_workflow_preserves_real_failures_without_retrying(
    tmp_path: Path, mode: str, entry: dict[str, Any], expected_rc: int
) -> None:
    result = _run_gate(tmp_path, [entry], mode=mode)
    assert result.check.returncode == expected_rc, result.check.stderr
    assert len(result.attempts) == 1
    assert result.delays == []
    assert ("--with-faithfulness" in result.attempts[0]["args"]) == (mode == "full")


@pytest.mark.parametrize("recovered", [True, False])
def test_only_classified_outage_with_passing_retrieval_gets_one_retry(
    tmp_path: Path, recovered: bool
) -> None:
    outage = {"rc": 3, "dump": _dump(status="unavailable")}
    result = _run_gate(tmp_path, [outage, {"rc": 0, "dump": _dump()} if recovered else outage])
    assert result.check.returncode == 0, result.check.stderr
    assert len(result.attempts) == 2
    assert result.delays == ["30"]
    assert result.outputs["faithfulness"] == ("completed" if recovered else "unavailable")
    assert all(not attempt["dump_existed"] for attempt in result.attempts)
    if not recovered:
        assert "unavailable" in result.check.stdout.lower()
        assert "retrieval passed" in result.check.stdout.lower()


@pytest.mark.parametrize("p1", [0.50, 0.90])
def test_outage_cannot_retry_or_hide_retrieval_regression(tmp_path: Path, p1: float) -> None:
    result = _run_gate(tmp_path, [{"rc": 3, "dump": _dump(p1=p1, status="unavailable")}])
    assert result.check.returncode == 1
    assert "FAIL precision_at_1" in result.check.stderr
    assert len(result.attempts) == 1
    assert result.delays == []


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
@pytest.mark.parametrize("metric", ["precision", "recall", "faithfulness"])
def test_invalid_measured_values_never_pass_or_trigger_retry(
    tmp_path: Path, invalid: object, metric: str
) -> None:
    report = _dump(status="unavailable")
    rc = 3
    if metric == "faithfulness":
        report, rc = _dump(faith=invalid), 0
    else:
        report["retrieval"]["precision_at_1" if metric == "precision" else "recall_at_k"] = invalid
    result = _run_gate(tmp_path, [{"rc": rc, "dump": report}])
    assert result.check.returncode == 2, result.check.stderr
    assert len(result.attempts) == 1
    assert result.delays == []


@pytest.mark.parametrize("entry", [{"rc": 2}, {"rc": 0}, {"rc": 3}, {"rc": 3, "raw": "{"}])
def test_missing_or_corrupt_dump_cannot_reuse_stale_pass(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    result = _run_gate(tmp_path, [entry], stale_dump=True)
    assert result.check.returncode == 2
    assert len(result.attempts) == 1
    assert result.attempts[0]["dump_existed"] is False
    assert result.delays == []


def test_retry_that_produces_no_dump_cannot_reuse_first_outage_report(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        [{"rc": 3, "dump": _dump(status="unavailable")}, {"rc": 3}],
        stale_dump=True,
    )
    assert result.check.returncode == 2
    assert len(result.attempts) == 2
    assert all(not attempt["dump_existed"] for attempt in result.attempts)


@pytest.mark.parametrize("defect", ["missing-retrieval", "invalid-outcomes", "missing-sha-path"])
def test_outage_requires_complete_retrieval_evidence(tmp_path: Path, defect: str) -> None:
    report = _dump(status="unavailable")
    if defect == "missing-retrieval":
        del report["retrieval"]
    elif defect == "invalid-outcomes":
        report["outcomes"]["retrieval"] = "failed"
    else:
        del report["queries_path"]
    result = _run_gate(tmp_path, [{"rc": 3, "dump": report}])
    assert result.check.returncode == 2
    assert len(result.attempts) == 1
    assert result.delays == []


@pytest.mark.parametrize("has_key", ["true", "false"])
def test_mode_selection_discloses_missing_key_without_running_provider(
    tmp_path: Path, has_key: str
) -> None:
    output = tmp_path / "mode-output"
    result = _run_shell(
        _step("mode")["run"],
        {"EVENT": "push", "HAS_KEY": has_key, "PR_TITLE": "", "GITHUB_OUTPUT": str(output)},
    )
    assert result.returncode == 0, result.stderr
    assert _outputs(output)["mode"] == ("full" if has_key == "true" else "retrieval-only")
    if has_key == "false":
        assert "SKIPPED" in result.stdout
        assert "ANTHROPIC_API_KEY" in result.stdout


def test_checker_always_runs_and_job_restores_failed_check_status() -> None:
    check = _step("check")
    assert check.get("if") in (None, "always()", "${{ always() }}")
    assert check["env"]["BENCH_RC"] == "${{ steps.bench.outputs.rc }}"
    failures = {
        step["name"]: step for step in _steps() if step.get("name", "").startswith("Fail on")
    }
    assert failures["Fail on regression"]["if"] == "steps.check.outputs.rc == '1'"
    validation_if = failures["Fail on validation error"]["if"]
    assert "always()" in validation_if
    assert "steps.check.outputs.rc != '0'" in validation_if
    assert "steps.check.outputs.rc != '1'" in validation_if
    for failure in failures.values():
        assert failure.get("continue-on-error", False) is False
        assert _run_shell(failure["run"], {}).returncode != 0


@pytest.mark.parametrize("check_rc", ["", "2", "42"])
def test_final_failure_distinguishes_incomplete_validation_from_invalid_evidence(
    check_rc: str,
) -> None:
    step = next(step for step in _steps() if step.get("name") == "Fail on validation error")
    assert step["env"]["CHECK_RC"] == "${{ steps.check.outputs.rc }}"

    result = _run_shell(step["run"], {"CHECK_RC": check_rc})

    assert result.returncode == 1
    if check_rc == "":
        assert (
            "Required quality validation did not complete; inspect earlier steps" in result.stdout
        )
        assert "hit a validation error" not in result.stdout
    else:
        assert "Quality gate hit a validation error" in result.stdout
        assert "queries.yaml SHA mismatch" in result.stdout
        assert "did not complete" not in result.stdout
