"""Unit tests for scripts/measure_reranker.py.

Mock the actual reranker — no live Anthropic calls. Live diagnostic
run is M2's job per the spec.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("attune_help")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "measure_reranker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("measure_reranker", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_reranker"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mr():
    return _load_module()


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


def test_load_queries_parses_bundled_yaml(mr) -> None:
    """SHA-256 normalization + queries-key contract."""
    queries, sha = mr._load_queries(_REPO_ROOT / "tests" / "golden" / "queries.yaml")
    assert isinstance(queries, list) and queries
    assert isinstance(sha, str) and len(sha) == 12


def test_load_queries_rejects_missing_queries_key(mr, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_queries: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="queries"):
        mr._load_queries(bad)


def test_check_r1_pass(mr) -> None:
    """R1 passes when Run A reproduces the locked numbers."""

    class _Agg:
        def __init__(self, p1: float, r3: float) -> None:
            self.p1 = p1
            self.r3 = r3

    result = {
        "baseline_agg": _Agg(1.0, 1.0),
        "paraphrased_agg": _Agg(0.9, 0.9875),
    }
    ok, msg = mr.check_r1(result)
    assert ok, msg


def test_check_r1_fail_surfaces_actual_vs_expected(mr) -> None:
    class _Agg:
        def __init__(self, p1: float, r3: float) -> None:
            self.p1 = p1
            self.r3 = r3

    drifted = {
        "baseline_agg": _Agg(1.0, 1.0),
        "paraphrased_agg": _Agg(0.80, 0.95),  # drift!
    }
    ok, msg = mr.check_r1(drifted)
    assert not ok
    assert "0.95" in msg or "0.8" in msg


def test_aggregate_run_b_mean_p50_p95(mr) -> None:
    """Mean / p50 / p95 across N runs."""
    # Build 5 stub results with predictable values
    results = [
        mr._RunBResult(
            baseline_p1=1.0,
            baseline_r3=1.0,
            paraphrased_p1=p,
            paraphrased_r3=1.0,
            baseline_per=(),
            paraphrased_per=(),
        )
        for p in [0.85, 0.90, 0.92, 0.95, 0.95]
    ]
    agg = mr._aggregate_run_b(results)
    assert agg["paraphrased_p1"]["mean"] == pytest.approx(sum([0.85, 0.90, 0.92, 0.95, 0.95]) / 5)
    assert agg["paraphrased_p1"]["p50"] == pytest.approx(0.92)
    # All baseline values are 1.0 → all stats are 1.0
    assert agg["baseline_p1"]["mean"] == 1.0


def test_residual_stability_counts_lifts(mr) -> None:
    """For each Run-A paraphrased P@1 miss, count how many of N Run-B
    passes lifted it to ✓ and how many held R@3."""
    from attune_rag._scoring import QueryScore

    # Run A had one paraphrased miss: gqp-X
    qa_miss = QueryScore(
        qid="gqp-X",
        query="test query",
        difficulty="medium",
        expected=("a.md",),
        hit_paths=("b.md", "a.md"),  # R@3 yes, P@1 no
        p1=False,
        r3=True,
    )
    qa_hit = QueryScore(
        qid="gqp-Y",
        query="other",
        difficulty="easy",
        expected=("c.md",),
        hit_paths=("c.md",),
        p1=True,
        r3=True,
    )
    run_a = {"paraphrased_per": [qa_miss, qa_hit]}

    # Build 5 Run-B passes: gqp-X lifts to ✓ in 4 of them; R@3 holds in all 5
    def _b_record(qid: str, p1: bool, r3: bool) -> QueryScore:
        return QueryScore(
            qid=qid,
            query="x",
            difficulty="medium",
            expected=("a.md",),
            hit_paths=(),
            p1=p1,
            r3=r3,
        )

    run_b_results = [
        mr._RunBResult(
            baseline_p1=1.0,
            baseline_r3=1.0,
            paraphrased_p1=0.5,
            paraphrased_r3=1.0,
            baseline_per=(),
            paraphrased_per=(_b_record("gqp-X", p1=lifts, r3=True), _b_record("gqp-Y", True, True)),
        )
        for lifts in [True, True, True, False, True]  # 4 of 5 lifts
    ]
    residuals = mr._residual_stability(run_a, run_b_results)
    assert len(residuals) == 1
    r = residuals[0]
    assert r["qid"] == "gqp-X"
    assert r["p1_lifts"] == 4
    assert r["n"] == 5
    assert r["p1_stability"] == "4/5"
    assert r["r3_stability"] == "5/5"
    # ⌈0.8 * 5⌉ = 4 → 4/5 counts as stable lift
    assert r["stable_lift"] is True


def test_residual_stability_empty_when_run_a_perfect(mr) -> None:
    """When Run A's paraphrased P@1 = 100%, there are no residuals."""
    from attune_rag._scoring import QueryScore

    run_a = {
        "paraphrased_per": [
            QueryScore(
                qid="gqp-Z",
                query="q",
                difficulty="easy",
                expected=("x.md",),
                hit_paths=("x.md",),
                p1=True,
                r3=True,
            )
        ]
    }
    residuals = mr._residual_stability(run_a, [])
    assert residuals == []


def test_residual_stability_stable_threshold(mr) -> None:
    """⌈0.8 * 5⌉ = 4 → 3/5 lifts is NOT a stable lift."""
    from attune_rag._scoring import QueryScore

    run_a = {
        "paraphrased_per": [
            QueryScore(
                qid="q",
                query="x",
                difficulty="medium",
                expected=("a.md",),
                hit_paths=("b.md",),
                p1=False,
                r3=False,
            )
        ]
    }

    def _rec(p1: bool) -> QueryScore:
        return QueryScore(
            qid="q",
            query="x",
            difficulty="medium",
            expected=("a.md",),
            hit_paths=(),
            p1=p1,
            r3=False,
        )

    runs = [
        mr._RunBResult(
            baseline_p1=1.0,
            baseline_r3=1.0,
            paraphrased_p1=0.0,
            paraphrased_r3=0.0,
            baseline_per=(),
            paraphrased_per=(_rec(p1),),
        )
        for p1 in [True, True, True, False, False]  # 3/5 lifts
    ]
    residuals = mr._residual_stability(run_a, runs)
    assert residuals[0]["p1_lifts"] == 3
    assert residuals[0]["stable_lift"] is False  # 3/5 < 4/5


def test_render_report_includes_metadata_block(mr) -> None:
    class _Agg:
        def __init__(self, p1: float, r3: float) -> None:
            self.p1 = p1
            self.r3 = r3

    run_a = {
        "baseline_agg": _Agg(1.0, 1.0),
        "paraphrased_agg": _Agg(0.9, 0.9875),
    }
    metadata = {
        "harness_version": "0.1.0",
        "commit_sha": "deadbeef",
        "baseline_queries_sha": "abc123",
        "paraphrased_queries_sha": "def456",
        "timestamp": "2026-05-22T00:00:00Z",
    }
    md = mr.render_report(
        run_a_result=run_a,
        run_b_aggregated=None,
        run_b_n=0,
        metadata=metadata,
    )
    assert "## Reproducibility metadata" in md
    assert "commit_sha" in md
    assert "deadbeef" in md
    assert "Run A" in md
    assert "Run B" in md
    assert "Verdict" in md


def test_cli_skip_run_b_no_live_calls(tmp_path: Path) -> None:
    """--skip-run-b runs Run A only; no Anthropic import attempted."""
    out = tmp_path / "report.md"
    result = _run_script(
        "--baseline-queries",
        "tests/golden/queries.yaml",
        "--paraphrased-queries",
        "tests/golden/queries_paraphrased.yaml",
        "--skip-run-b",
        "--out",
        str(out),
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    )
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "R1 reproduction OK" in text
    assert "Skipped (M1 ships the script" in text


def test_cli_skip_run_b_deterministic_metadata(tmp_path: Path) -> None:
    """Two --skip-run-b runs with --frozen-timestamp produce byte-identical reports."""
    out_a = tmp_path / "a.md"
    out_b = tmp_path / "b.md"
    common = [
        "--baseline-queries",
        "tests/golden/queries.yaml",
        "--paraphrased-queries",
        "tests/golden/queries_paraphrased.yaml",
        "--skip-run-b",
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    ]
    assert _run_script(*common, "--out", str(out_a)).returncode == 0
    assert _run_script(*common, "--out", str(out_b)).returncode == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def test_anthropic_sdk_version_returns_string(mr) -> None:
    """Reproducibility metadata can call the SDK-version lookup safely."""
    v = mr._anthropic_sdk_version()
    assert isinstance(v, str)
    assert v in {"not-installed", "unknown"} or "." in v
