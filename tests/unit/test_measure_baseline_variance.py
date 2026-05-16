"""Tests for scripts/measure_baseline_variance.py.

The script invokes the benchmark via subprocess. These tests
inject a fake runner so no real benchmark is executed and no API
tokens are spent. Coverage targets the spec's M1.2 list:

- happy path: N canned stdouts → deterministic stats
- N < 10 → exit 2, no files written
- one run fails partway → exit 1, no files written
- --sigma override flows through to thresholds.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import measure_baseline_variance as mbv  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────────


def _stdout_with(
    p1: float,
    r3: float,
    faith: float | None = None,
    *,
    faith_label: str = "legacy",
) -> str:
    """Build a stdout payload shaped like attune-rag-benchmark output.

    ``faith_label`` mirrors the real benchmark, which calls
    ``_print_faithfulness(..., label="legacy")`` for the default
    single-pass run. Pass ``faith_label=""`` to simulate an
    unlabeled run (kept for completeness).
    """
    lines = [
        "Retriever:  KeywordRetriever",
        "Corpus:     attune_help",
        "Queries:    40",
        f"Precision@1: {p1 * 100:.2f}% (33/40)",
        f"Recall@3:    {r3 * 100:.2f}% (38/40)",
        "Mean latency: 12.50ms",
        "Max latency:  47.00ms",
    ]
    if faith is not None:
        suffix = f" ({faith_label})" if faith_label else ""
        lines.append("")
        lines.append(f"Mean faithfulness{suffix}:   {faith:.3f}")
        lines.append("Refusal rate:        2.5%")
    return "\n".join(lines) + "\n"


def _fake_runner(stdouts: list[str], *, returncode: int = 0):
    """Build a subprocess.run replacement that services *benchmark* calls only.

    Git invocations (``git rev-parse HEAD``) are answered with a
    canned SHA so they don't consume the stdout queue. Anything
    else falls through to an empty response with the configured
    returncode.
    """
    queue = list(stdouts)

    def runner(cmd, capture_output=True, text=True, check=False, **kwargs):
        if cmd and cmd[0] == "git":
            return SimpleNamespace(stdout="deadbeef\n", stderr="", returncode=0)
        out = queue.pop(0) if queue else ""
        return SimpleNamespace(stdout=out, stderr="", returncode=returncode)

    return runner


# ── parse_metrics ────────────────────────────────────────────────────────────


def test_parse_metrics_extracts_all_three():
    out = mbv.parse_metrics(_stdout_with(0.825, 0.94, 0.781))
    assert out == {
        "precision_at_1": 0.825,
        "recall_at_3": 0.94,
        "mean_faithfulness": 0.781,
    }


def test_parse_metrics_retrieval_only():
    out = mbv.parse_metrics(_stdout_with(0.80, 0.92))
    assert out == {"precision_at_1": 0.80, "recall_at_3": 0.92}


def test_parse_metrics_returns_empty_on_unrelated_text():
    assert mbv.parse_metrics("nothing useful here\n") == {}


def test_parse_metrics_handles_labeled_faithfulness_line():
    """The real benchmark prints 'Mean faithfulness (legacy):' for a single-pass run."""
    for label in ("legacy", "thinking off", "native"):
        out = mbv.parse_metrics(_stdout_with(0.825, 0.94, 0.781, faith_label=label))
        assert out["mean_faithfulness"] == 0.781, f"failed for label={label!r}"


def test_parse_metrics_handles_unlabeled_faithfulness_line():
    out = mbv.parse_metrics(_stdout_with(0.825, 0.94, 0.781, faith_label=""))
    assert out["mean_faithfulness"] == 0.781


# ── compute_stats ────────────────────────────────────────────────────────────


def test_compute_stats_deterministic_math():
    out = mbv.compute_stats([0.78, 0.80, 0.82], sigma=2.0)
    # mean = 0.80, stdev (sample) = 0.02, threshold = 0.76
    assert out["mean"] == 0.8
    assert out["stdev"] == 0.02
    assert out["threshold"] == 0.76
    assert out["raw"] == [0.78, 0.80, 0.82]


def test_compute_stats_single_value_has_zero_stdev():
    out = mbv.compute_stats([0.5], sigma=2.0)
    assert out["mean"] == 0.5
    assert out["stdev"] == 0.0
    assert out["threshold"] == 0.5


def test_compute_stats_sigma_one_tighter_threshold():
    out = mbv.compute_stats([0.78, 0.80, 0.82], sigma=1.0)
    assert out["threshold"] == 0.78


def test_compute_stats_empty_raises():
    with pytest.raises(ValueError):
        mbv.compute_stats([], sigma=2.0)


# ── run_benchmark_once ───────────────────────────────────────────────────────


def test_run_benchmark_once_parses_with_faithfulness():
    runner = _fake_runner([_stdout_with(0.825, 0.94, 0.781)])
    out = mbv.run_benchmark_once(queries=None, with_faithfulness=True, runner=runner)
    assert out == {
        "precision_at_1": 0.825,
        "recall_at_3": 0.94,
        "mean_faithfulness": 0.781,
    }


def test_run_benchmark_once_parses_without_faithfulness():
    runner = _fake_runner([_stdout_with(0.80, 0.92)])
    out = mbv.run_benchmark_once(queries=None, with_faithfulness=False, runner=runner)
    assert out == {"precision_at_1": 0.80, "recall_at_3": 0.92}


def test_run_benchmark_once_raises_on_nonzero_exit():
    def runner(cmd, **_):
        return SimpleNamespace(stdout="", stderr="boom", returncode=2)

    with pytest.raises(RuntimeError, match="exit 2"):
        mbv.run_benchmark_once(queries=None, with_faithfulness=False, runner=runner)


def test_run_benchmark_once_raises_on_missing_metric():
    def runner(cmd, **_):
        return SimpleNamespace(stdout="Retriever: x\n", stderr="", returncode=0)

    with pytest.raises(RuntimeError, match="missing metrics"):
        mbv.run_benchmark_once(queries=None, with_faithfulness=False, runner=runner)


# ── main: end-to-end with mocked subprocess ──────────────────────────────────


def _run_main_with_canned(
    tmp_path: Path,
    stdouts: list[str],
    *,
    extra_args: list[str] | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, Path, Path]:
    out_md = tmp_path / "baseline.md"
    out_json = tmp_path / "thresholds.json"
    monkeypatch.setattr(mbv.subprocess, "run", _fake_runner(stdouts))
    rc = mbv.main(
        [
            "--runs",
            str(len(stdouts)),
            "--skip-faithfulness",
            "--out",
            str(out_md),
            "--thresholds-out",
            str(out_json),
            *(extra_args or []),
        ]
    )
    return rc, out_md, out_json


def test_main_happy_path_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stdouts = [_stdout_with(0.80 + 0.01 * i, 0.92) for i in range(10)]
    rc, out_md, out_json = _run_main_with_canned(tmp_path, stdouts, monkeypatch=monkeypatch)
    assert rc == 0
    assert out_md.exists()
    assert out_json.exists()

    payload = json.loads(out_json.read_text())
    assert payload["runs"] == 10
    assert payload["sigma"] == 2.0
    assert "precision_at_1" in payload["metrics"]
    assert "recall_at_3" in payload["metrics"]
    # mean_faithfulness is absent because we used --skip-faithfulness
    assert "mean_faithfulness" not in payload["metrics"]
    p1 = payload["metrics"]["precision_at_1"]
    assert p1["mean"] == pytest.approx(0.845, abs=1e-3)
    # threshold = mean - 2*stdev; stdev > 0 since values vary
    assert p1["threshold"] < p1["mean"]

    md = out_md.read_text()
    assert "Baseline measurement" in md
    assert "precision_at_1" in md
    assert "run  1:" in md  # raw runs section


def test_main_runs_below_minimum_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_md = tmp_path / "baseline.md"
    out_json = tmp_path / "thresholds.json"
    monkeypatch.setattr(mbv.subprocess, "run", _fake_runner([]))
    rc = mbv.main(
        [
            "--runs",
            "5",
            "--skip-faithfulness",
            "--out",
            str(out_md),
            "--thresholds-out",
            str(out_json),
        ]
    )
    assert rc == 2
    assert not out_md.exists()
    assert not out_json.exists()


def test_main_aborts_on_benchmark_failure_no_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_md = tmp_path / "baseline.md"
    out_json = tmp_path / "thresholds.json"

    call_count = {"n": 0}

    def flaky_runner(cmd, **_):
        if cmd and cmd[0] == "git":
            return SimpleNamespace(stdout="deadbeef\n", stderr="", returncode=0)
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return SimpleNamespace(stdout="", stderr="kaboom", returncode=1)
        return SimpleNamespace(
            stdout=_stdout_with(0.80, 0.92),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(mbv.subprocess, "run", flaky_runner)
    rc = mbv.main(
        [
            "--runs",
            "10",
            "--skip-faithfulness",
            "--out",
            str(out_md),
            "--thresholds-out",
            str(out_json),
        ]
    )
    assert rc == 1
    assert not out_md.exists()
    assert not out_json.exists()


def test_main_sigma_override_flows_to_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdouts = [_stdout_with(0.80 + 0.01 * i, 0.92) for i in range(10)]
    rc, _, out_json = _run_main_with_canned(
        tmp_path,
        stdouts,
        extra_args=["--sigma", "1.0"],
        monkeypatch=monkeypatch,
    )
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["sigma"] == 1.0
    p1 = payload["metrics"]["precision_at_1"]
    # With sigma=1.0, threshold should be exactly mean - 1*stdev
    assert p1["threshold"] == pytest.approx(p1["mean"] - p1["stdev"], abs=1e-4)
