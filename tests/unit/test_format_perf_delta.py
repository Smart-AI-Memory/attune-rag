"""Tests for ``scripts/format_perf_delta.py`` — Phase 4 W0.5 helper."""

from __future__ import annotations

import importlib.util
import json
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
    body = out.read_text()
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
    body = out.read_text()
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
    body = out.read_text()
    assert "Advisory only" in body
