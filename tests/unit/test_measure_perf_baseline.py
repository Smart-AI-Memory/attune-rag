"""Tests for ``scripts/measure_perf_baseline.py`` — Phase 4 W0.3.

Two-axis (wall + CPU) capture introduces a wider test surface
than the Phase 1 single-axis variance script. Coverage per spec:

- wall-only / cpu-only / both-present aggregation cases.
- TimedResult captures both perf_counter and process_time deltas.
- Threshold = mean + sigma * stdev per axis (latency upper bound).
- JSON payload + markdown render are deterministic.
- main() honours --runs minimum, --include-llm gating, exit codes.

Real hot-path factories are NOT exercised here (they hit the real
attune_rag package + optional Anthropic). Their wiring is covered
by a smoke-style monkey-patched main() call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_perf_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("measure_perf_baseline", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_perf_baseline"] = module
    spec.loader.exec_module(module)
    return module


mpb = _load_module()


# ---------------------------------------------------------------------------
# time_call + run_benchmark_n
# ---------------------------------------------------------------------------


def test_time_call_captures_both_axes() -> None:
    """A no-op callable produces small but non-negative wall + cpu deltas."""

    def noop() -> None:
        pass

    result = mpb.time_call(noop)
    assert result.wall >= 0.0
    assert result.cpu >= 0.0


def test_time_call_wall_includes_sleep_cpu_does_not() -> None:
    """time.sleep is wall-time but not CPU time. That asymmetry is the
    whole reason both axes are tracked — gating CPU avoids false-firing
    on Anthropic network variance."""
    import time

    def slept() -> None:
        time.sleep(0.02)  # 20ms

    result = mpb.time_call(slept)
    # Wall time should reflect the sleep (with slack for scheduler).
    assert result.wall >= 0.015
    # CPU time during sleep should be effectively zero. Allow a small
    # tolerance for interpreter overhead on slow machines.
    assert result.cpu < result.wall


def test_run_benchmark_n_returns_n_samples_per_axis() -> None:
    def noop() -> None:
        pass

    raw = mpb.run_benchmark_n(noop, 5)
    assert set(raw) == {"wall", "cpu"}
    assert len(raw["wall"]) == 5
    assert len(raw["cpu"]) == 5


# ---------------------------------------------------------------------------
# compute_stats — threshold direction
# ---------------------------------------------------------------------------


def test_compute_stats_uses_upper_threshold() -> None:
    """Latencies are higher-is-worse, so threshold = mean + sigma*stdev
    (the opposite sign of Phase 1's retrieval-quality threshold)."""
    values = [0.10, 0.12, 0.11, 0.13, 0.14]
    s = mpb.compute_stats(values, sigma=2.0)
    assert s["mean"] == pytest.approx(0.12, abs=1e-6)
    assert s["threshold"] > s["mean"]
    assert s["n"] == 5


def test_compute_stats_single_value_has_zero_stdev() -> None:
    s = mpb.compute_stats([0.10], sigma=2.0)
    assert s["stdev"] == 0.0
    assert s["threshold"] == s["mean"]


def test_compute_stats_empty_raises() -> None:
    with pytest.raises(ValueError):
        mpb.compute_stats([], sigma=2.0)


# ---------------------------------------------------------------------------
# aggregate_results — wall-only / cpu-only / both-present
# ---------------------------------------------------------------------------


def test_aggregate_both_axes_present() -> None:
    raw = {
        "bench_a": {"wall": [0.1, 0.12, 0.11], "cpu": [0.09, 0.10, 0.095]},
    }
    out = mpb.aggregate_results(raw, sigma=2.0)
    assert set(out) == {"bench_a.wall", "bench_a.cpu"}
    assert out["bench_a.wall"]["n"] == 3
    assert out["bench_a.cpu"]["n"] == 3


def test_aggregate_wall_only() -> None:
    """A benchmark that only recorded wall-clock (cpu axis absent or
    empty) appears in output with just the .wall row."""
    raw = {"bench_a": {"wall": [0.1, 0.12]}}
    out = mpb.aggregate_results(raw, sigma=2.0)
    assert set(out) == {"bench_a.wall"}


def test_aggregate_cpu_only() -> None:
    raw = {"bench_a": {"cpu": [0.05, 0.06]}}
    out = mpb.aggregate_results(raw, sigma=2.0)
    assert set(out) == {"bench_a.cpu"}


def test_aggregate_skips_empty_axes() -> None:
    """An axis present in the dict but with an empty list is silently
    skipped — downstream consumers can trust presence to mean data."""
    raw = {"bench_a": {"wall": [0.1, 0.12], "cpu": []}}
    out = mpb.aggregate_results(raw, sigma=2.0)
    assert set(out) == {"bench_a.wall"}


def test_aggregate_multiple_benchmarks_yields_two_rows_each() -> None:
    """The spec's 'two threshold rows per benchmark' guarantee."""
    raw = {
        "a": {"wall": [0.1, 0.2], "cpu": [0.05, 0.06]},
        "b": {"wall": [0.3, 0.4], "cpu": [0.15, 0.16]},
    }
    out = mpb.aggregate_results(raw, sigma=2.0)
    assert set(out) == {"a.wall", "a.cpu", "b.wall", "b.cpu"}


# ---------------------------------------------------------------------------
# environment_fingerprint
# ---------------------------------------------------------------------------


def test_environment_fingerprint_keys() -> None:
    env = mpb.environment_fingerprint()
    # Don't pin values (they vary by host); pin the schema.
    for key in (
        "platform",
        "machine",
        "processor",
        "python_version",
        "python_implementation",
        "ci_runner",
    ):
        assert key in env, f"missing key: {key}"


def test_environment_fingerprint_picks_up_runner_os(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNNER_OS", "Linux")
    env = mpb.environment_fingerprint()
    assert env["ci_runner"] == "Linux"


# ---------------------------------------------------------------------------
# build_thresholds_payload + write_thresholds_json + render_markdown
# ---------------------------------------------------------------------------


def _sample_payload() -> dict:
    stats = mpb.aggregate_results(
        {"bench_a": {"wall": [0.10, 0.11], "cpu": [0.05, 0.06]}}, sigma=2.0
    )
    return mpb.build_thresholds_payload(
        measured_at="2026-05-19T20:00:00Z",
        commit="deadbeef",
        runs=2,
        sigma=2.0,
        env={"platform": "Linux-x86_64", "python_version": "3.11.0"},
        stats_by_metric=stats,
        include_llm=False,
    )


def test_build_thresholds_payload_shape() -> None:
    payload = _sample_payload()
    assert payload["commit"] == "deadbeef"
    assert payload["runs"] == 2
    assert payload["include_llm"] is False
    assert set(payload["metrics"]) == {"bench_a.wall", "bench_a.cpu"}


def test_write_thresholds_json_round_trips(tmp_path: Path) -> None:
    payload = _sample_payload()
    out_path = tmp_path / "subdir" / "perf-thresholds.json"
    mpb.write_thresholds_json(out_path, payload)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded == payload


def test_write_thresholds_json_is_deterministic(tmp_path: Path) -> None:
    """sort_keys=True so two equivalent payloads write byte-identical files."""
    payload = _sample_payload()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    mpb.write_thresholds_json(a, payload)
    mpb.write_thresholds_json(b, payload)
    assert a.read_text() == b.read_text()


def test_render_markdown_contains_per_axis_rows() -> None:
    payload = _sample_payload()
    md = mpb.render_markdown(payload)
    assert "# Perf baseline" in md
    assert "`bench_a.wall`" in md
    assert "`bench_a.cpu`" in md
    assert "deadbeef" in md
    assert "Mean (s)" in md


def test_render_markdown_keys_sorted() -> None:
    """Sorted keys → diff stability when the metric set is unchanged."""
    payload = mpb.build_thresholds_payload(
        measured_at="2026-05-19T20:00:00Z",
        commit="deadbeef",
        runs=2,
        sigma=2.0,
        env={"platform": "Linux"},
        stats_by_metric=mpb.aggregate_results(
            {
                "z_bench": {"wall": [0.1, 0.2]},
                "a_bench": {"wall": [0.3, 0.4]},
            },
            sigma=2.0,
        ),
        include_llm=False,
    )
    md = mpb.render_markdown(payload)
    # a_bench should appear before z_bench in the rendered table.
    a_idx = md.index("a_bench.wall")
    z_idx = md.index("z_bench.wall")
    assert a_idx < z_idx


# ---------------------------------------------------------------------------
# main() — CLI exit codes
# ---------------------------------------------------------------------------


def _patch_timings(monkeypatch: pytest.MonkeyPatch, *, include_llm_seen_value: list[bool]) -> None:
    """Stub collect_raw_timings so main() doesn't actually run hot paths.

    Captures the include_llm arg in a 1-element list so the caller can
    assert on it after main() returns.
    """

    def stub(runs: int, *, include_llm: bool) -> dict[str, dict[str, list[float]]]:
        include_llm_seen_value.append(include_llm)
        # Two benchmarks, both axes, n=runs samples each.
        return {
            "fake_bench": {
                "wall": [0.1] * runs,
                "cpu": [0.05] * runs,
            },
        }

    monkeypatch.setattr(mpb, "collect_raw_timings", stub)


def test_main_writes_both_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[bool] = []
    _patch_timings(monkeypatch, include_llm_seen_value=seen)

    out_md = tmp_path / "perf-baseline.md"
    out_json = tmp_path / "perf-thresholds.json"
    rc = mpb.main(
        [
            "--runs",
            "10",
            "--out",
            str(out_md),
            "--thresholds-out",
            str(out_json),
        ]
    )
    assert rc == 0
    assert seen == [False]  # --include-llm not set
    assert out_md.exists()
    assert out_json.exists()

    payload = json.loads(out_json.read_text())
    assert payload["runs"] == 10
    assert payload["include_llm"] is False
    assert set(payload["metrics"]) == {"fake_bench.wall", "fake_bench.cpu"}


def test_main_runs_below_minimum_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rc = mpb.main(
        [
            "--runs",
            "5",  # below MIN_RUNS=10
            "--out",
            str(tmp_path / "x.md"),
            "--thresholds-out",
            str(tmp_path / "x.json"),
        ]
    )
    assert rc == 2


def test_main_include_llm_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--include-llm flag should reach collect_raw_timings."""
    seen: list[bool] = []
    _patch_timings(monkeypatch, include_llm_seen_value=seen)

    rc = mpb.main(
        [
            "--runs",
            "10",
            "--out",
            str(tmp_path / "x.md"),
            "--thresholds-out",
            str(tmp_path / "x.json"),
            "--include-llm",
        ]
    )
    assert rc == 0
    assert seen == [True]


def test_main_runtime_error_in_collect_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A RuntimeError from collect_raw_timings (e.g. missing API key
    under --include-llm) surfaces as exit 2 — same validation-error
    convention as check_freeze.py / check_thresholds.py."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("ANTHROPIC_API_KEY required for --include-llm benchmarks")

    monkeypatch.setattr(mpb, "collect_raw_timings", boom)

    rc = mpb.main(
        [
            "--runs",
            "10",
            "--out",
            str(tmp_path / "x.md"),
            "--thresholds-out",
            str(tmp_path / "x.json"),
            "--include-llm",
        ]
    )
    assert rc == 2


def test_main_sigma_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[bool] = []
    _patch_timings(monkeypatch, include_llm_seen_value=seen)

    out_json = tmp_path / "x.json"
    rc = mpb.main(
        [
            "--runs",
            "10",
            "--out",
            str(tmp_path / "x.md"),
            "--thresholds-out",
            str(out_json),
            "--sigma",
            "3.0",
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["sigma"] == 3.0


# ---------------------------------------------------------------------------
# _resolve_benchmark_corpus_dir
# ---------------------------------------------------------------------------


def test_resolve_benchmark_corpus_dir_returns_existing_dir() -> None:
    """Either .help/templates or tests/golden must exist in this repo."""
    found = mpb._resolve_benchmark_corpus_dir()
    assert found.is_dir()
    assert found.name in {"templates", "golden"}
