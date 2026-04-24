"""Unit tests for attune_rag.eval.bench_prompts.

Covers the three HIGH-risk coverage gaps flagged by the
0.1.4 deep review:

- ``main()`` argument / environment validation (exit codes).
- ``_aggregate()`` metric math, including the empty-runs
  and all-errors branches added in 0.1.4.
- ``_validate_read_path`` / ``_validate_write_path`` — the
  new path guards on ``--queries`` / ``--output``.

The Anthropic + pipeline paths are never exercised here —
these are pure-Python tests that never hit the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from attune_rag.eval import bench_prompts

# The Unix-style _SYSTEM_DIRS list (/etc, /sys, /proc, /dev, /bin) is only
# meaningful on POSIX. On Windows, a path like /etc/passwd resolves to
# C:\etc\passwd, which isn't caught by the system-dir check. The validator
# is designed to block obvious Unix footguns; Windows users simply aren't
# exposed to the Unix paths it's guarding.
_unix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-style system-dir guards don't apply on Windows",
)

# ---------- _aggregate ----------


def _make_run(
    *,
    variant: str = "baseline",
    top1: bool = False,
    topk: bool = False,
    faith: float = 1.0,
    supported: int = 1,
    unsupported: int = 0,
    total: int | None = None,
    gen_ms: float = 100.0,
    judge_ms: float = 200.0,
    error: str | None = None,
) -> bench_prompts._QueryRun:
    return bench_prompts._QueryRun(
        id="test",
        query="q",
        difficulty="easy",
        variant=variant,
        answer="a",
        retrieval_top1_match=top1,
        retrieval_topk_match=topk,
        faithfulness_score=faith,
        supported_claims=supported,
        unsupported_claims=unsupported,
        total_claims=total if total is not None else supported + unsupported,
        generate_ms=gen_ms,
        judge_ms=judge_ms,
        error=error,
    )


def test_aggregate_empty_runs_returns_zero_report() -> None:
    report = bench_prompts._aggregate("baseline", [])
    assert report.total_queries == 0
    assert report.precision_at_1 == 0.0
    assert report.recall_at_k == 0.0
    assert report.mean_faithfulness == 0.0
    assert report.refusal_rate == 0.0
    assert report.hallucination_rate == 0.0
    assert report.error_count == 0


def test_aggregate_computes_metrics_correctly() -> None:
    runs = [
        _make_run(top1=True, topk=True, faith=1.0, supported=3, unsupported=0),
        _make_run(top1=True, topk=True, faith=1.0, supported=2, unsupported=0),
        _make_run(top1=False, topk=True, faith=0.5, supported=1, unsupported=1),
        _make_run(top1=False, topk=False, faith=0.0, supported=0, unsupported=2),
    ]
    r = bench_prompts._aggregate("baseline", runs)
    assert r.total_queries == 4
    assert r.precision_at_1 == pytest.approx(2 / 4)
    assert r.recall_at_k == pytest.approx(3 / 4)
    assert r.mean_faithfulness == pytest.approx((1.0 + 1.0 + 0.5 + 0.0) / 4)
    assert r.refusal_rate == 0.0  # every run has ≥1 claim
    assert r.hallucination_rate == pytest.approx(2 / 4)  # 2 runs with unsupported > 0
    assert r.error_count == 0


def test_aggregate_refusal_rate_counts_zero_claim_runs() -> None:
    runs = [
        _make_run(supported=0, unsupported=0, total=0),
        _make_run(supported=1, unsupported=0),
    ]
    r = bench_prompts._aggregate("baseline", runs)
    assert r.refusal_rate == pytest.approx(0.5)


def test_aggregate_excludes_errored_runs_from_means() -> None:
    runs = [
        _make_run(top1=True, topk=True, faith=1.0, supported=1, unsupported=0),
        _make_run(top1=True, topk=True, faith=1.0, supported=1, unsupported=0),
        _make_run(error="RateLimitError: 429"),
    ]
    r = bench_prompts._aggregate("baseline", runs)
    assert r.total_queries == 3
    assert r.error_count == 1
    # Means should be computed over the 2 successful runs, not diluted
    # by the third run's zeros.
    assert r.precision_at_1 == pytest.approx(1.0)
    assert r.mean_faithfulness == pytest.approx(1.0)


def test_aggregate_all_errors_returns_zero_metrics() -> None:
    runs = [_make_run(error="oops") for _ in range(3)]
    r = bench_prompts._aggregate("baseline", runs)
    assert r.total_queries == 3
    assert r.error_count == 3
    assert r.precision_at_1 == 0.0
    assert r.mean_faithfulness == 0.0


# ---------- _validate_read_path / _validate_write_path ----------


def test_validate_read_path_accepts_existing(tmp_path: Path) -> None:
    f = tmp_path / "queries.yaml"
    f.write_text("queries: []\n")
    out = bench_prompts._validate_read_path(f, kind="queries")
    assert out == f.resolve()


def test_validate_read_path_rejects_null_bytes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="null bytes"):
        bench_prompts._validate_read_path(str(tmp_path / "bad\x00.yaml"), kind="queries")


def test_validate_write_path_accepts_tmp(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    out = bench_prompts._validate_write_path(target)
    assert out == target.resolve()


@_unix_only
@pytest.mark.parametrize(
    "sysdir",
    ["/etc/passwd", "/sys/kernel/foo", "/proc/self/status", "/dev/null", "/bin/sh"],
)
def test_validate_write_path_refuses_system_dirs(sysdir: str) -> None:
    with pytest.raises(ValueError, match="system directory"):
        bench_prompts._validate_write_path(sysdir)


def test_validate_write_path_rejects_null_bytes() -> None:
    with pytest.raises(ValueError, match="null bytes"):
        bench_prompts._validate_write_path("/tmp/bad\x00.json")


# ---------- main() argument + env validation ----------


def test_main_missing_api_key_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = bench_prompts.main(["--queries", "/does/not/matter"])
    assert rc == 2


def test_main_missing_queries_file_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")  # pragma: allowlist secret
    missing = tmp_path / "nope.yaml"
    rc = bench_prompts.main(["--queries", str(missing)])
    assert rc == 2


def test_main_default_queries_missing_hints_about_wheel(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """When no --queries is passed and the default path doesn't
    exist (pip-install case), the error must tell users how to
    recover."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")  # pragma: allowlist secret
    # Force the default path to point at a nonexistent location.
    monkeypatch.setattr(
        bench_prompts,
        "_default_queries_path",
        lambda: Path("/does/not/exist/queries.yaml"),
    )
    rc = bench_prompts.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not shipped in the attune-rag wheel" in err
    assert "--queries" in err


def test_main_unknown_variant_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")  # pragma: allowlist secret
    queries = tmp_path / "q.yaml"
    queries.write_text("queries: [{id: x, query: hi, expected_in_top_3: [], difficulty: easy}]\n")
    rc = bench_prompts.main(["--queries", str(queries), "--variants", "does-not-exist"])
    assert rc == 2


def test_main_null_byte_in_queries_exits_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")  # pragma: allowlist secret
    rc = bench_prompts.main(["--queries", "a\x00b.yaml"])
    assert rc == 2


@_unix_only
def test_main_output_path_refuses_system_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")  # pragma: allowlist secret
    queries = tmp_path / "q.yaml"
    queries.write_text("queries: [{id: x, query: hi, expected_in_top_3: [], difficulty: easy}]\n")
    rc = bench_prompts.main(["--queries", str(queries), "--output", "/etc/attune-report.json"])
    assert rc == 2


# ---------- _load_queries ----------


def test_load_queries_empty_list_raises(tmp_path: Path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text("queries: []\n")
    with pytest.raises(ValueError, match="No queries found"):
        bench_prompts._load_queries(f)
