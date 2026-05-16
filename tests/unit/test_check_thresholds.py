"""Tests for scripts/check_thresholds.py.

Coverage targets the spec's M2.2 list:

- All metrics pass: exit 0, no stderr.
- One metric fails: exit 1, FAIL line with metric / measured / threshold.
- ``queries_sha256`` mismatch: exit 2 with re-measure hint.
- Missing metric in the dump: exit 2.

Plus structural cases worth catching: the ``recall_at_k`` →
``recall_at_<k>`` translation, the ``--skip-queries-sha-check``
bypass, missing-file handling, and the "threshold expects a
metric the dump doesn't have" path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_thresholds as ct  # noqa: E402

# ── fixtures ─────────────────────────────────────────────────────────────────


_REAL_QUERIES = REPO_ROOT / "tests" / "golden" / "queries.yaml"
_REAL_QUERIES_SHA = "f47486df87c61e2391a4326f9ffe68659373f6837063cf70c5b990a639f2290c"


def _good_dump(
    p1: float = 0.95,
    rk: float = 1.0,
    k: int = 3,
    faith: float | None = 0.98,
    queries_path: str | None = None,
) -> dict:
    out: dict = {
        "retrieval": {
            "precision_at_1": p1,
            "recall_at_k": rk,
            "k": k,
        },
        "queries_path": queries_path or str(_REAL_QUERIES),
    }
    if faith is not None:
        out["faithfulness_legacy"] = {"mean_faithfulness": faith}
    return out


def _thresholds(
    *,
    p1: float = 0.95,
    r3: float = 1.0,
    faith: float | None = 0.9686,
    queries_sha: str | None = _REAL_QUERIES_SHA,
) -> dict:
    metrics: dict = {
        "precision_at_1": {
            "mean": p1,
            "stdev": 0.0,
            "threshold": p1,
        },
        "recall_at_3": {
            "mean": r3,
            "stdev": 0.0,
            "threshold": r3,
        },
    }
    if faith is not None:
        metrics["mean_faithfulness"] = {
            "mean": 0.979,
            "stdev": 0.0052,
            "threshold": faith,
        }
    return {
        "commit": "deadbeef",
        "measured_at": "2026-05-16T08:11:25Z",
        "metrics": metrics,
        "queries_path": "tests/golden/queries.yaml",
        "queries_sha256": queries_sha,
        "runs": 20,
        "sigma": 2.0,
    }


def _write_pair(tmp_path: Path, dump: dict, thresholds: dict) -> tuple[Path, Path]:
    dump_path = tmp_path / "dump.json"
    thr_path = tmp_path / "thresholds.json"
    dump_path.write_text(json.dumps(dump))
    thr_path.write_text(json.dumps(thresholds))
    return dump_path, thr_path


# ── happy path ───────────────────────────────────────────────────────────────


def test_all_metrics_pass_exit_zero_no_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(), _thresholds())
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out == ""


# ── regression (exit 1) ──────────────────────────────────────────────────────


def test_faithfulness_below_threshold_exit_one(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(faith=0.85), _thresholds())
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL mean_faithfulness" in captured.err
    assert "measured=0.8500" in captured.err
    assert "threshold=0.9686" in captured.err
    assert "delta=-0.1186" in captured.err


def test_precision_below_threshold_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(p1=0.90), _thresholds())
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL precision_at_1" in captured.err


def test_multiple_failures_all_reported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(p1=0.90, faith=0.80), _thresholds())
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 1
    # Both failures should be on stderr; one FAIL line each.
    assert captured.err.count("FAIL ") == 2
    assert "FAIL precision_at_1" in captured.err
    assert "FAIL mean_faithfulness" in captured.err


# ── validation errors (exit 2) ───────────────────────────────────────────────


def test_queries_sha_mismatch_exit_two_with_remeasure_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dump_path, thr_path = _write_pair(
        tmp_path,
        _good_dump(),
        _thresholds(queries_sha="0" * 64),
    )
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "SHA-256 mismatch" in captured.err
    assert "measure_baseline_variance.py" in captured.err


def test_missing_metric_in_dump_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    bad = _good_dump()
    del bad["retrieval"]["precision_at_1"]
    dump_path, thr_path = _write_pair(tmp_path, bad, _thresholds())
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "precision_at_1" in captured.err
    # The cosmetic-fix path: no stray quotes from KeyError.__str__.
    assert '"' not in captured.err


def test_threshold_expects_metric_dump_lacks_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """thresholds.json names faithfulness but dump is retrieval-only."""
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(faith=None), _thresholds())
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "mean_faithfulness" in captured.err


def test_missing_dump_file_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _, thr_path = _write_pair(tmp_path, _good_dump(), _thresholds())
    rc = ct.main(
        [
            "--dump",
            str(tmp_path / "does-not-exist.json"),
            "--thresholds",
            str(thr_path),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "not found" in captured.err


def test_malformed_thresholds_json_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dump_path = tmp_path / "dump.json"
    thr_path = tmp_path / "thresholds.json"
    dump_path.write_text(json.dumps(_good_dump()))
    thr_path.write_text("{not valid json")
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "valid JSON" in captured.err


# ── structural / translation ─────────────────────────────────────────────────


def test_recall_at_k_translates_to_recall_at_k_value(
    tmp_path: Path,
) -> None:
    """k=5 in dump should look up recall_at_5 in thresholds."""
    dump = _good_dump(rk=0.99, k=5)
    thr = _thresholds()
    thr["metrics"]["recall_at_5"] = thr["metrics"].pop("recall_at_3")
    thr["metrics"]["recall_at_5"]["threshold"] = 0.95
    dump_path, thr_path = _write_pair(tmp_path, dump, thr)
    rc = ct.main(["--dump", str(dump_path), "--thresholds", str(thr_path)])
    assert rc == 0


def test_skip_queries_sha_check_bypasses_mismatch(
    tmp_path: Path,
) -> None:
    dump_path, thr_path = _write_pair(
        tmp_path,
        _good_dump(),
        _thresholds(queries_sha="0" * 64),
    )
    rc = ct.main(
        [
            "--dump",
            str(dump_path),
            "--thresholds",
            str(thr_path),
            "--skip-queries-sha-check",
        ]
    )
    assert rc == 0


def test_extract_metrics_returns_flat_dict():
    dump = _good_dump(p1=0.9, rk=0.95, k=3, faith=0.88)
    out = ct.extract_metrics(dump)
    assert out == {
        "precision_at_1": 0.9,
        "recall_at_3": 0.95,
        "mean_faithfulness": 0.88,
    }


def test_extract_metrics_retrieval_only_no_faithfulness_key():
    dump = _good_dump(faith=None)
    out = ct.extract_metrics(dump)
    assert "mean_faithfulness" not in out
    assert set(out.keys()) == {"precision_at_1", "recall_at_3"}


def test_metric_failure_delta_sign_negative_for_regression():
    f = ct.MetricFailure(metric="x", measured=0.85, threshold=0.95)
    assert f.delta == pytest.approx(-0.10, abs=1e-6)


# ── format_failure_comment (M2.3) ────────────────────────────────────────────


_EXPECTED_SINGLE_FAILURE = """\
<!-- attune-rag-quality-gate -->
## Quality gate failed

This PR's benchmark run did not meet the locked thresholds at \
`docs/specs/release-quality-baseline/thresholds.json`.

| Metric | Measured | Threshold | Delta |
|---|---:|---:|---:|
| `mean_faithfulness` | 0.8500 | 0.9686 | -0.1186 |

### What to do

- If the regression is real, fix it before merging.
- If this PR intentionally changes the corpus, the judge, or \
the prompts, re-measure the baseline:
  ```
  python scripts/measure_baseline_variance.py --runs 20 \\
      --out docs/specs/release-quality-baseline/baseline-N.md \\
      --thresholds-out docs/specs/release-quality-baseline/thresholds.json
  ```
  and commit the updated baseline in this same PR with \
`[baseline-update]` in the title.

<!-- attune-rag-quality-gate -->
"""


def test_format_failure_comment_golden_single_failure():
    body = ct.format_failure_comment(
        [
            ct.MetricFailure(
                metric="mean_faithfulness",
                measured=0.85,
                threshold=0.9686,
            )
        ]
    )
    assert body == _EXPECTED_SINGLE_FAILURE


def test_format_failure_comment_orders_by_metric_name():
    """Two failures in any order render the same body — deterministic."""
    a = ct.MetricFailure(metric="precision_at_1", measured=0.90, threshold=0.95)
    b = ct.MetricFailure(metric="mean_faithfulness", measured=0.85, threshold=0.9686)
    assert ct.format_failure_comment([a, b]) == ct.format_failure_comment([b, a])


def test_format_failure_comment_lists_every_failure():
    body = ct.format_failure_comment(
        [
            ct.MetricFailure(
                metric="mean_faithfulness",
                measured=0.85,
                threshold=0.9686,
            ),
            ct.MetricFailure(
                metric="precision_at_1",
                measured=0.90,
                threshold=0.95,
            ),
        ]
    )
    assert "`mean_faithfulness`" in body
    assert "`precision_at_1`" in body
    # Alphabetical: mean_ before precision_.
    assert body.find("`mean_faithfulness`") < body.find("`precision_at_1`")


def test_format_failure_comment_has_marker_at_both_ends():
    body = ct.format_failure_comment([ct.MetricFailure(metric="x", measured=0.0, threshold=0.5)])
    assert body.startswith(ct.COMMENT_MARKER + "\n")
    assert body.rstrip("\n").endswith(ct.COMMENT_MARKER)


def test_format_failure_comment_empty_raises():
    with pytest.raises(ValueError, match="no failures"):
        ct.format_failure_comment([])


# ── --comment-out CLI flag ───────────────────────────────────────────────────


def test_comment_out_writes_body_on_regression(
    tmp_path: Path,
) -> None:
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(faith=0.85), _thresholds())
    comment_path = tmp_path / "comment.md"
    rc = ct.main(
        [
            "--dump",
            str(dump_path),
            "--thresholds",
            str(thr_path),
            "--comment-out",
            str(comment_path),
        ]
    )
    assert rc == 1
    assert comment_path.exists()
    body = comment_path.read_text()
    assert ct.COMMENT_MARKER in body
    assert "mean_faithfulness" in body


def test_comment_out_not_written_on_green_run(
    tmp_path: Path,
) -> None:
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(), _thresholds())
    comment_path = tmp_path / "comment.md"
    rc = ct.main(
        [
            "--dump",
            str(dump_path),
            "--thresholds",
            str(thr_path),
            "--comment-out",
            str(comment_path),
        ]
    )
    assert rc == 0
    assert not comment_path.exists()


def test_skip_metric_drops_faithfulness_gate(
    tmp_path: Path,
) -> None:
    """Retrieval-only PR: dump has no faithfulness, --skip-metric makes that fine."""
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(faith=None), _thresholds())
    rc = ct.main(
        [
            "--dump",
            str(dump_path),
            "--thresholds",
            str(thr_path),
            "--skip-metric",
            "mean_faithfulness",
        ]
    )
    assert rc == 0


def test_skip_metric_does_not_silence_real_regression(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Skipping faithfulness must still let precision regressions fail."""
    dump_path, thr_path = _write_pair(tmp_path, _good_dump(p1=0.85, faith=None), _thresholds())
    rc = ct.main(
        [
            "--dump",
            str(dump_path),
            "--thresholds",
            str(thr_path),
            "--skip-metric",
            "mean_faithfulness",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL precision_at_1" in captured.err
    assert "mean_faithfulness" not in captured.err


def test_skip_metric_repeatable(
    tmp_path: Path,
) -> None:
    """Repeatable flag drops multiple gates."""
    dump = _good_dump(faith=None)
    # Even precision is "absent" in this hypothetical
    del dump["retrieval"]["precision_at_1"]
    dump_path, thr_path = _write_pair(tmp_path, dump, _thresholds())
    rc = ct.main(
        [
            "--dump",
            str(dump_path),
            "--thresholds",
            str(thr_path),
            "--skip-metric",
            "mean_faithfulness",
            "--skip-metric",
            "precision_at_1",
        ]
    )
    # Now: precision is missing from dump (extract_metrics raises)
    # before skip_metrics is even consulted. We expect exit 2.
    assert rc == 2


def test_comment_out_not_written_on_validation_error(
    tmp_path: Path,
) -> None:
    bad = _good_dump()
    del bad["retrieval"]["precision_at_1"]
    dump_path, thr_path = _write_pair(tmp_path, bad, _thresholds())
    comment_path = tmp_path / "comment.md"
    rc = ct.main(
        [
            "--dump",
            str(dump_path),
            "--thresholds",
            str(thr_path),
            "--comment-out",
            str(comment_path),
        ]
    )
    assert rc == 2
    assert not comment_path.exists()
