"""Unit tests for scripts/aggregate_perf_baseline.py.

Synthetic per-invocation JSONs; no subprocess/IO beyond tmp_path.
Validates the M1 aggregation math: mean-of-means, stdev-of-means,
intra-run-stdev averaging, and the schema-additions contract.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "aggregate_perf_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("aggregate_perf_baseline", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_perf_baseline"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def aggmod():
    return _load_module()


def _make_per_invocation(
    tmp_path: Path,
    *,
    raw: dict[str, list[float]],
    index: int,
    invocations: int = 5,
    runs_per_invocation: int = 20,
    commit: str = "deadbeef",
    include_llm: bool = False,
) -> Path:
    payload = {
        "methodology_version": 2,
        "invocation_index": index,
        "invocations": invocations,
        "runs_per_invocation": runs_per_invocation,
        "sigma": 2.0,
        "include_llm": include_llm,
        "commit": commit,
        "measured_at": "2026-05-22T00:00:00Z",
        "environment": {"platform": "linux", "python_version": "3.10.11"},
        "raw_timings": raw,
    }
    p = tmp_path / f"inv-{index}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_aggregate_metric_math(aggmod) -> None:
    """Mean-of-means, stdev-of-means, intra-run-stdev — pin the math."""
    # Two invocations with known per-trial values for clarity.
    inv0 = [1.0, 2.0, 3.0]  # mean=2.0, stdev=1.0
    inv1 = [5.0, 6.0, 7.0]  # mean=6.0, stdev=1.0
    result = aggmod.aggregate_metric([inv0, inv1], sigma=2.0)

    # mean-of-means
    assert result["mean"] == pytest.approx(4.0)
    # inter_run_stdev = stdev of [2.0, 6.0] = sqrt(8) ≈ 2.828427
    assert result["inter_run_stdev"] == pytest.approx(statistics.stdev([2.0, 6.0]))
    # intra_run_stdev = mean of [stdev(inv0), stdev(inv1)] = mean([1.0, 1.0]) = 1.0
    assert result["intra_run_stdev"] == pytest.approx(1.0)
    # threshold = mean + sigma * inter
    assert result["threshold"] == pytest.approx(4.0 + 2.0 * statistics.stdev([2.0, 6.0]))
    # backward-compat: stdev aliases inter_run_stdev
    assert result["stdev"] == result["inter_run_stdev"]


def test_aggregate_metric_rejects_single_invocation(aggmod) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        aggmod.aggregate_metric([[1.0, 2.0, 3.0]], sigma=2.0)


def test_aggregate_all_skips_metric_missing_in_any_invocation(aggmod) -> None:
    """A metric present in some invocations but not others is dropped,
    not silently averaged over a partial set."""
    payloads = [
        {
            "raw_timings": {"a.wall": [1.0, 2.0], "b.wall": [10.0, 11.0]},
            "invocation_index": 0,
            "invocations": 2,
            "runs_per_invocation": 2,
            "methodology_version": 2,
        },
        {
            "raw_timings": {"a.wall": [3.0, 4.0]},  # b.wall missing
            "invocation_index": 1,
            "invocations": 2,
            "runs_per_invocation": 2,
            "methodology_version": 2,
        },
    ]
    result = aggmod.aggregate_all(payloads, sigma=2.0)
    assert "a.wall" in result
    assert "b.wall" not in result


def test_validate_consistent_detects_commit_mismatch(aggmod, tmp_path: Path) -> None:
    p0 = _make_per_invocation(tmp_path, raw={"x.wall": [1.0, 2.0]}, index=0, commit="aaa")
    p1 = _make_per_invocation(tmp_path, raw={"x.wall": [1.5, 2.5]}, index=1, commit="bbb")
    payloads = aggmod._load_per_invocation([p0, p1])
    with pytest.raises(ValueError, match="inconsistent 'commit'"):
        aggmod._validate_consistent(payloads)


def test_validate_consistent_detects_duplicate_index(aggmod, tmp_path: Path) -> None:
    p0 = _make_per_invocation(tmp_path, raw={"x.wall": [1.0]}, index=0)
    p1 = _make_per_invocation(tmp_path, raw={"x.wall": [2.0]}, index=0)  # dup
    payloads = aggmod._load_per_invocation([p0, p1])
    with pytest.raises(ValueError, match="contiguous range"):
        aggmod._validate_consistent(payloads)


def test_load_per_invocation_rejects_v1_payload(aggmod, tmp_path: Path) -> None:
    bad = tmp_path / "v1.json"
    bad.write_text(json.dumps({"methodology_version": 1, "raw_timings": {}}), encoding="utf-8")
    bad2 = tmp_path / "v1b.json"
    bad2.write_text(json.dumps({"methodology_version": 1, "raw_timings": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="methodology_version == 2"):
        aggmod._load_per_invocation([bad, bad2])


def test_load_per_invocation_rejects_too_few(aggmod, tmp_path: Path) -> None:
    p = _make_per_invocation(tmp_path, raw={"x.wall": [1.0]}, index=0)
    with pytest.raises(ValueError, match="at least 2"):
        aggmod._load_per_invocation([p])


def test_load_per_invocation_missing_file(aggmod, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        aggmod._load_per_invocation([tmp_path / "nope.json", tmp_path / "nada.json"])


def test_build_payload_schema_v2(aggmod, tmp_path: Path) -> None:
    """Output payload carries methodology_version: 2 + new keys."""
    p0 = _make_per_invocation(tmp_path, raw={"a.wall": [1.0, 2.0, 3.0]}, index=0, invocations=2)
    p1 = _make_per_invocation(tmp_path, raw={"a.wall": [5.0, 6.0, 7.0]}, index=1, invocations=2)
    payloads = aggmod._load_per_invocation([p0, p1])
    aggmod._validate_consistent(payloads)
    metrics = aggmod.aggregate_all(payloads, sigma=2.0)
    payload = aggmod.build_payload(payloads=payloads, sigma=2.0, metrics=metrics)
    assert payload["methodology_version"] == 2
    assert payload["invocations"] == 2
    assert payload["runs_per_invocation"] == 20
    assert payload["sigma"] == 2.0
    # New v2 fields are present per-metric
    a = payload["metrics"]["a.wall"]
    assert "intra_run_stdev" in a
    assert "inter_run_stdev" in a
    assert "invocations" in a
    assert "runs_per_invocation" in a
    # Backward-compat keys held
    assert "mean" in a and "stdev" in a and "threshold" in a


def test_render_markdown_includes_dual_stdev(aggmod) -> None:
    payload = {
        "methodology_version": 2,
        "measured_at": "2026-05-22T00:00:00Z",
        "commit": "deadbeef",
        "invocations": 5,
        "runs_per_invocation": 20,
        "sigma": 2.0,
        "include_llm": True,
        "environment": {},
        "metrics": {
            "a.wall": {
                "mean": 0.1,
                "intra_run_stdev": 0.01,
                "inter_run_stdev": 0.02,
                "stdev": 0.02,
                "threshold": 0.14,
                "invocations": 5,
                "runs_per_invocation": 20,
            }
        },
    }
    md = aggmod.render_markdown(payload)
    assert "intra_run_stdev" in md
    assert "inter_run_stdev" in md
    assert "v2" in md
    assert "deadbeef" in md
