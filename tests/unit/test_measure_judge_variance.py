"""Tests for scripts/measure_judge_variance.py.

The script re-runs the FaithfulnessJudge against captured
answer + context from a calibration artifact. These tests
inject a fake judge so no real API calls are made.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import measure_judge_variance as mjv  # noqa: E402

# ---------------------------------------------------------------------------
# Fake judge — deterministic scores via canned values
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, score: float) -> None:
        self.score = score


class _FakeJudge:
    """Stand-in for FaithfulnessJudge.

    Returns scores from per-(qid, condition) queues. The
    script calls ``.score(query=, answer=, passages=,
    use_thinking=)``; we look up the qid by matching the
    query string back to the test's seeded queries.
    """

    def __init__(self, plan: dict[tuple[str, str], list[float]]) -> None:
        self._plan = {k: list(v) for k, v in plan.items()}
        self.model = "fake-judge-v1"

    async def score(
        self,
        *,
        query: str,
        answer: str,
        passages: str,
        use_thinking: bool = False,
        **_unused: Any,
    ) -> _FakeResult:
        condition = "on" if use_thinking else "off"
        # The test seeds each query string as just its qid.
        qid = query
        scores = self._plan.get((qid, condition))
        if not scores:
            raise AssertionError(f"No more canned scores for ({qid}, {condition})")
        return _FakeResult(scores.pop(0))


def _artifact_for(query_specs: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Build an artifact with the minimum fields the script reads.

    Each tuple is (qid, answer, context); the query string
    is just the qid (so the fake judge can route by it).
    """
    records_off = [
        {
            "id": qid,
            "query": qid,
            "score": 1.0,
            "supported": 1,
            "unsupported": 0,
            "answer": ans,
            "context": ctx,
        }
        for qid, ans, ctx in query_specs
    ]
    return {
        "faithfulness_thinking_off": {"per_query": records_off},
        "faithfulness_thinking_on": {"per_query": list(records_off)},
    }


# ---------------------------------------------------------------------------
# _aggregate math
# ---------------------------------------------------------------------------


def test_aggregate_pooled_stdev_matches_flat_list() -> None:
    """Pooled stdev = stdev of the flattened raw scores list."""
    q1_off = [0.9, 0.95, 0.92]
    q1_on = [0.85, 0.80, 0.88]
    q2_off = [1.0, 1.0, 1.0]
    q2_on = [0.95, 0.90, 0.92]
    results = {
        "q1": {
            "off": {"mean": statistics.fmean(q1_off), "stdev": 0.0, "raw": q1_off},
            "on": {"mean": statistics.fmean(q1_on), "stdev": 0.0, "raw": q1_on},
        },
        "q2": {
            "off": {"mean": statistics.fmean(q2_off), "stdev": 0.0, "raw": q2_off},
            "on": {"mean": statistics.fmean(q2_on), "stdev": 0.0, "raw": q2_on},
        },
    }
    agg = mjv._aggregate(results)
    expected_off = statistics.stdev(q1_off + q2_off)
    expected_on = statistics.stdev(q1_on + q2_on)
    expected_margin = statistics.stdev(
        [
            statistics.fmean(q1_off) - statistics.fmean(q1_on),
            statistics.fmean(q2_off) - statistics.fmean(q2_on),
        ]
    )
    assert agg["off_stdev_pooled"] == pytest.approx(expected_off)
    assert agg["on_stdev_pooled"] == pytest.approx(expected_on)
    assert agg["margin_stdev"] == pytest.approx(expected_margin)


def test_aggregate_handles_single_query() -> None:
    """One query → margin_stdev = 0 (only one sample for the margin)."""
    results = {
        "q1": {
            "off": {"mean": 1.0, "stdev": 0.0, "raw": [1.0, 0.9, 0.95]},
            "on": {"mean": 0.9, "stdev": 0.0, "raw": [0.85, 0.92, 0.93]},
        }
    }
    agg = mjv._aggregate(results)
    assert agg["margin_stdev"] == 0.0
    assert agg["off_stdev_pooled"] == pytest.approx(statistics.stdev([1.0, 0.9, 0.95]))


# ---------------------------------------------------------------------------
# _run_query — fake judge integration
# ---------------------------------------------------------------------------


def test_run_query_records_M_runs_per_condition() -> None:
    plan = {
        ("gq-1", "off"): [0.9, 0.95, 0.93],
        ("gq-1", "on"): [0.85, 0.90, 0.88],
    }
    judge = _FakeJudge(plan)
    result = asyncio.run(
        mjv._run_query(
            judge,
            qid="gq-1",
            query="gq-1",
            answer="...",
            context="...",
            runs=3,
        )
    )
    assert result["off"]["raw"] == [0.9, 0.95, 0.93]
    assert result["on"]["raw"] == [0.85, 0.90, 0.88]
    assert result["off"]["mean"] == pytest.approx(statistics.fmean([0.9, 0.95, 0.93]))
    assert result["on"]["stdev"] == pytest.approx(statistics.stdev([0.85, 0.90, 0.88]))


# ---------------------------------------------------------------------------
# main() — argparse, validation, end-to-end output
# ---------------------------------------------------------------------------


def test_main_rejects_runs_below_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(json.dumps(_artifact_for([("gq-1", "a", "c")])), encoding="utf-8")
    rc = mjv.main(
        [
            "--artifact",
            str(artifact_path),
            "--query-ids",
            "gq-1",
            "--runs",
            "1",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    assert "stdev needs at least 2 samples" in capsys.readouterr().err


def test_main_rejects_missing_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = mjv.main(
        [
            "--artifact",
            str(tmp_path / "nope.json"),
            "--query-ids",
            "gq-1",
            "--runs",
            "3",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    assert "Artifact not found" in capsys.readouterr().err


def test_main_rejects_unknown_query_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(json.dumps(_artifact_for([("gq-1", "a", "c")])), encoding="utf-8")
    rc = mjv.main(
        [
            "--artifact",
            str(artifact_path),
            "--query-ids",
            "gq-999",
            "--runs",
            "3",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    assert "not in artifact" in capsys.readouterr().err


def test_main_rejects_legacy_artifact_missing_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Build an artifact with NO answer / context fields.
    legacy = {
        "faithfulness_thinking_off": {
            "per_query": [
                {"id": "gq-1", "query": "gq-1", "score": 1.0, "supported": 1, "unsupported": 0}
            ]
        },
        "faithfulness_thinking_on": {
            "per_query": [
                {"id": "gq-1", "query": "gq-1", "score": 1.0, "supported": 1, "unsupported": 0}
            ]
        },
    }
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(json.dumps(legacy), encoding="utf-8")
    rc = mjv.main(
        [
            "--artifact",
            str(artifact_path),
            "--query-ids",
            "gq-1",
            "--runs",
            "3",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    assert "predates PR #26" in capsys.readouterr().err


def test_main_writes_valid_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full happy path with a stubbed FaithfulnessJudge."""
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(
        json.dumps(_artifact_for([("gq-1", "ans1", "ctx1"), ("gq-2", "ans2", "ctx2")])),
        encoding="utf-8",
    )
    out_path = tmp_path / "out.json"
    plan = {
        ("gq-1", "off"): [0.9, 0.95, 0.92],
        ("gq-1", "on"): [0.85, 0.80, 0.88],
        ("gq-2", "off"): [1.0, 1.0, 1.0],
        ("gq-2", "on"): [0.95, 0.90, 0.92],
    }

    # Patch the FaithfulnessJudge import the script does internally.
    import attune_rag.eval.faithfulness as faith_mod

    monkeypatch.setattr(faith_mod, "FaithfulnessJudge", lambda: _FakeJudge(plan))

    rc = mjv.main(
        [
            "--artifact",
            str(artifact_path),
            "--query-ids",
            "gq-1,gq-2",
            "--runs",
            "3",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    out = json.loads(out_path.read_text(encoding="utf-8"))

    # Schema match against design.md.
    assert out["judge_model"] == "fake-judge-v1"
    assert out["runs"] == 3
    assert out["query_ids"] == ["gq-1", "gq-2"]
    assert set(out["queries"].keys()) == {"gq-1", "gq-2"}
    for qid in ("gq-1", "gq-2"):
        for cond in ("off", "on"):
            assert set(out["queries"][qid][cond].keys()) == {"mean", "stdev", "raw"}
    # Numbers.
    assert out["queries"]["gq-1"]["off"]["raw"] == [0.9, 0.95, 0.92]
    assert out["queries"]["gq-2"]["on"]["mean"] == pytest.approx(
        statistics.fmean([0.95, 0.90, 0.92])
    )
    # Aggregate block present.
    assert set(out["aggregate"].keys()) == {
        "off_stdev_pooled",
        "on_stdev_pooled",
        "margin_stdev",
    }


def test_main_handles_whitespace_in_query_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'gq-1, gq-2,gq-3' should parse cleanly."""
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(
        json.dumps(_artifact_for([("gq-1", "a", "c"), ("gq-2", "a", "c"), ("gq-3", "a", "c")])),
        encoding="utf-8",
    )
    plan = {(qid, cond): [0.9, 0.9] for qid in ("gq-1", "gq-2", "gq-3") for cond in ("off", "on")}
    import attune_rag.eval.faithfulness as faith_mod

    monkeypatch.setattr(faith_mod, "FaithfulnessJudge", lambda: _FakeJudge(plan))

    rc = mjv.main(
        [
            "--artifact",
            str(artifact_path),
            "--query-ids",
            " gq-1, gq-2 ,gq-3 ",
            "--runs",
            "2",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 0
