"""Tests for scripts/build_calibration_labeling_kit.py and
scripts/score_against_ground_truth.py.

The scripts live under scripts/ (not in the package), so we
import them through a path injection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_calibration_labeling_kit as kit  # noqa: E402
import score_against_ground_truth as score  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(qid: str, score: float, supported: int = 1, unsupported: int = 0) -> dict[str, Any]:
    return {
        "id": qid,
        "query": f"q for {qid}",
        "score": score,
        "supported": supported,
        "unsupported": unsupported,
        "supported_claims": [f"s{i}" for i in range(supported)],
        "unsupported_claims": [f"u{i}" for i in range(unsupported)],
        "reasoning": f"reasoning for {qid}",
        "latency_ms": 100.0,
        "claim_citation_count": 0,
        "used_native_citations": False,
        "thinking_used": False,
    }


def _artifact(pairs: list[tuple[str, float, float]]) -> dict[str, Any]:
    """Build an artifact dict from (id, score_off, score_on) triples."""
    off_records = [_q(qid, sc_off) for qid, sc_off, _ in pairs]
    on_records = [_q(qid, sc_on) for qid, _, sc_on in pairs]
    return {
        "queries_path": "fake.yaml",
        "retrieval": {},
        "faithfulness_thinking_off": {
            "mean_faithfulness": 0.0,
            "refusal_rate": 0.0,
            "hallucination_rate": 0.0,
            "citation_emit_rate": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "per_query": off_records,
        },
        "faithfulness_thinking_on": {
            "mean_faithfulness": 0.0,
            "refusal_rate": 0.0,
            "hallucination_rate": 0.0,
            "citation_emit_rate": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "per_query": on_records,
        },
    }


# ---------------------------------------------------------------------------
# build_calibration_labeling_kit._select_queries
# ---------------------------------------------------------------------------


def test_select_picks_largest_shifts_then_controls() -> None:
    artifact = _artifact(
        [
            ("big-shift", 0.5, 1.0),  # |Δ| = 0.50
            ("med-shift", 0.8, 0.9),  # |Δ| = 0.10
            ("tiny-shift", 1.0, 1.01),  # |Δ| < threshold
            ("stable-a", 0.7, 0.7),  # unchanged + same counts (control)
            ("stable-b", 1.0, 1.0),  # unchanged + same counts (control)
        ]
    )
    shifted, controls, random_picks = kit._select_queries(
        artifact, n_shifted=2, n_controls=2, shift_threshold=0.05
    )
    assert shifted == ["big-shift", "med-shift"]
    assert set(controls) == {"stable-a", "stable-b"}
    assert random_picks == []


def test_select_skips_shifts_below_threshold() -> None:
    artifact = _artifact([("only-tiny", 0.50, 0.51), ("stable", 0.9, 0.9)])
    shifted, _, _ = kit._select_queries(artifact, n_shifted=5, n_controls=0, shift_threshold=0.05)
    assert "only-tiny" not in shifted  # 0.01 < 0.05 threshold


def test_select_empty_when_no_overlap() -> None:
    """If off and on report different IDs, fall back to empty."""
    artifact = {
        "faithfulness_thinking_off": {"per_query": [_q("a", 1.0)]},
        "faithfulness_thinking_on": {"per_query": [_q("b", 1.0)]},
    }
    assert kit._select_queries(artifact, n_shifted=5, n_controls=5, shift_threshold=0.05) == (
        [],
        [],
        [],
    )


# ---------------------------------------------------------------------------
# build_calibration_labeling_kit._select_queries — --n-random bucket
# ---------------------------------------------------------------------------


def _artifact_n(n: int) -> dict[str, Any]:
    """Build an artifact with ``n`` queries — all stable (no shift), no controls.

    Every query has score 1.0 in both passes BUT distinct claim
    counts so they don't qualify as 'controls' under the
    unchanged-score-AND-unchanged-claim-count rule. This leaves
    the entire set available for the random bucket.
    """
    import random as _random

    rng = _random.Random(0)
    artifact = {
        "faithfulness_thinking_off": {"per_query": []},
        "faithfulness_thinking_on": {"per_query": []},
    }
    for i in range(n):
        qid = f"gq-{i:03d}"
        sup = rng.randint(5, 20)
        artifact["faithfulness_thinking_off"]["per_query"].append(_q(qid, 1.0, supported=sup))
        # Different claim count on the 'on' side keeps it out of the control bucket.
        artifact["faithfulness_thinking_on"]["per_query"].append(_q(qid, 1.0, supported=sup + 1))
    return artifact


def test_n_random_is_disjoint_from_shifted_and_controls() -> None:
    """Random picks come from common queries minus shifted minus controls."""
    import random as _random

    artifact = _artifact(
        [
            ("big-shift", 0.5, 1.0),  # shifted
            ("med-shift", 0.8, 0.9),  # shifted
            ("stable-a", 0.7, 0.7),  # control
            ("stable-b", 1.0, 1.0),  # control
            (
                "random-pool-1",
                0.5,
                0.5,
            ),  # eligible for random (score same, claims same → also control)
            ("random-pool-2", 0.5, 0.5),
        ]
    )
    # Make pool entries NOT eligible as controls by varying claim count.
    for rec in artifact["faithfulness_thinking_on"]["per_query"]:
        if rec["id"].startswith("random-pool"):
            rec["supported"] = 99
            rec["supported_claims"] = [f"s{i}" for i in range(99)]

    shifted, controls, random_picks = kit._select_queries(
        artifact,
        n_shifted=2,
        n_controls=2,
        shift_threshold=0.05,
        n_random=2,
        rng=_random.Random(42),
    )
    assert set(shifted).isdisjoint(controls)
    assert set(random_picks).isdisjoint(shifted)
    assert set(random_picks).isdisjoint(controls)
    assert len(random_picks) == 2
    assert set(random_picks) <= {"random-pool-1", "random-pool-2"}


def test_n_random_is_reproducible_under_seed() -> None:
    """Same seed → same draw; different seed → different draw."""
    import random as _random

    artifact = _artifact_n(20)

    def draw(seed: int) -> list[str]:
        _, _, picks = kit._select_queries(
            artifact,
            n_shifted=0,
            n_controls=0,
            shift_threshold=0.05,
            n_random=5,
            rng=_random.Random(seed),
        )
        return picks

    assert draw(42) == draw(42)
    # Different seeds *almost certainly* produce different draws at n=5 of 20.
    assert draw(42) != draw(7)


def test_n_random_zero_skips_bucket() -> None:
    """n_random=0 → no random picks, no rng required."""
    artifact = _artifact([("a", 0.5, 1.0), ("b", 0.5, 0.5)])
    shifted, _, random_picks = kit._select_queries(
        artifact, n_shifted=1, n_controls=0, shift_threshold=0.05, n_random=0, rng=None
    )
    assert shifted == ["a"]
    assert random_picks == []


def test_n_random_errors_when_rng_missing() -> None:
    """Asking for random picks without an rng is a programmer error."""
    artifact = _artifact_n(10)
    with pytest.raises(ValueError, match="rng is required"):
        kit._select_queries(
            artifact,
            n_shifted=0,
            n_controls=0,
            shift_threshold=0.05,
            n_random=3,
            rng=None,
        )


def test_n_random_errors_when_exceeds_pool() -> None:
    """N > remaining pool surfaces a clear, user-facing error."""
    import random as _random

    artifact = _artifact_n(3)  # pool of 3
    with pytest.raises(ValueError, match="exceeds remaining pool"):
        kit._select_queries(
            artifact,
            n_shifted=0,
            n_controls=0,
            shift_threshold=0.05,
            n_random=10,  # impossible
            rng=_random.Random(0),
        )


# ---------------------------------------------------------------------------
# build_calibration_labeling_kit.main — --n-random CLI integration
# ---------------------------------------------------------------------------


def test_cli_n_random_writes_blocks_for_all_buckets(tmp_path: Path) -> None:
    """End-to-end: --n-random N adds N more blocks past shift+control."""
    artifact = _artifact_n(10)
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    out = tmp_path / "kit.md"
    rc = kit.main(
        [
            "--artifact",
            str(artifact_path),
            "--out",
            str(out),
            "--n-shifted",
            "0",
            "--n-controls",
            "0",
            "--n-random",
            "4",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "0 shifted + 0 controls + 4 random" in text
    assert "seed = 42" in text
    # Four `## gq-NNN —` headings present.
    assert text.count("## gq-") == 4


def test_cli_n_random_overflow_returns_two(tmp_path: Path) -> None:
    """N > pool → exit code 2, error message on stderr."""
    artifact = _artifact_n(3)
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    out = tmp_path / "kit.md"
    rc = kit.main(
        [
            "--artifact",
            str(artifact_path),
            "--out",
            str(out),
            "--n-shifted",
            "0",
            "--n-controls",
            "0",
            "--n-random",
            "99",
            "--seed",
            "42",
        ]
    )
    assert rc == 2
    assert not out.exists()


def test_cli_negative_counts_rejected(tmp_path: Path) -> None:
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(json.dumps(_artifact_n(5)), encoding="utf-8")
    out = tmp_path / "kit.md"
    rc = kit.main(
        [
            "--artifact",
            str(artifact_path),
            "--out",
            str(out),
            "--n-shifted",
            "-1",
        ]
    )
    assert rc == 2
    assert not out.exists()


# ---------------------------------------------------------------------------
# build_calibration_labeling_kit.main (end-to-end CLI)
# ---------------------------------------------------------------------------


def test_kit_writes_one_block_per_selected_query(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact = _artifact(
        [
            ("shift-1", 0.5, 1.0),
            ("stable-1", 0.9, 0.9),
        ]
    )
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    out = tmp_path / "kit.md"
    rc = kit.main(
        [
            "--artifact",
            str(artifact_path),
            "--out",
            str(out),
            "--n-shifted",
            "1",
            "--n-controls",
            "1",
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    # Both selected queries should have a `## <id> —` heading.
    assert "## shift-1 —" in text
    assert "## stable-1 —" in text
    # Each query block carries a YAML label stub.
    assert text.count("```yaml") == 2


def test_kit_returns_nonzero_when_artifact_missing(tmp_path: Path) -> None:
    out = tmp_path / "kit.md"
    rc = kit.main(["--artifact", str(tmp_path / "nope.json"), "--out", str(out)])
    assert rc == 2


def _q_with_answer_context(qid: str, score: float, answer: str, context: str) -> dict[str, Any]:
    """``_q`` plus the ``answer``/``context`` fields added 2026-05-15."""
    rec = _q(qid, score)
    rec["answer"] = answer
    rec["context"] = context
    return rec


def test_kit_embeds_answer_and_context_when_present(tmp_path: Path) -> None:
    """Modern artifacts include answer + context — the kit should embed them."""
    artifact_path = tmp_path / "artifact.json"
    off = _q_with_answer_context(
        "shift-1", 0.5, answer="THE ANSWER TEXT", context="P1 PASSAGE TEXT"
    )
    on = _q_with_answer_context("shift-1", 1.0, answer="THE ANSWER TEXT", context="P1 PASSAGE TEXT")
    artifact = {
        "faithfulness_thinking_off": {"per_query": [off]},
        "faithfulness_thinking_on": {"per_query": [on]},
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    out = tmp_path / "kit.md"
    rc = kit.main(
        [
            "--artifact",
            str(artifact_path),
            "--out",
            str(out),
            "--n-shifted",
            "1",
            "--n-controls",
            "0",
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    # Both fields show up under their headings.
    assert "### Retrieved context" in text
    assert "P1 PASSAGE TEXT" in text
    assert "### Answer" in text
    assert "THE ANSWER TEXT" in text
    # Legacy warning should NOT appear when fields are present.
    assert "predates answer/context capture" not in text


def test_kit_falls_back_to_legacy_warning_when_fields_absent(tmp_path: Path) -> None:
    """Older artifacts (pre-2026-05-15) lack answer/context — show warning."""
    artifact_path = tmp_path / "artifact.json"
    artifact = {
        "faithfulness_thinking_off": {"per_query": [_q("shift-1", 0.5)]},
        "faithfulness_thinking_on": {"per_query": [_q("shift-1", 1.0)]},
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    out = tmp_path / "kit.md"
    rc = kit.main(
        [
            "--artifact",
            str(artifact_path),
            "--out",
            str(out),
            "--n-shifted",
            "1",
            "--n-controls",
            "0",
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "predates answer/context capture" in text
    assert "### Retrieved context" not in text  # not embedded
    assert "### Answer" not in text


# ---------------------------------------------------------------------------
# score_against_ground_truth._extract_labels
# ---------------------------------------------------------------------------


def test_extract_labels_skips_unmarked_yaml_blocks() -> None:
    md = """\
Some prose.

```yaml
unrelated: value
```

## q1

```yaml
id: q1
faithfulness_score: 0.8
verdict: faithful
```

```yaml
id: q2
faithfulness_score: TBD
verdict: TBD
```
"""
    labels = score._extract_labels(md)
    ids = {lbl["id"] for lbl in labels}
    assert ids == {"q1", "q2"}  # both have `id` keys
    # But only q1 is usable.
    usable = [lbl for lbl in labels if score._is_labeled(lbl)]
    assert len(usable) == 1
    assert usable[0]["id"] == "q1"


# ---------------------------------------------------------------------------
# score_against_ground_truth._classify
# ---------------------------------------------------------------------------


def test_classify_picks_closer_pass() -> None:
    # Label 0.9, off 0.5, on 0.85 → on is closer (Δoff=0.40, Δon=0.05).
    assert score._classify(0.9, off_score=0.5, on_score=0.85, tied_window=0.02) == "on"
    # Reverse the gap.
    assert score._classify(0.5, off_score=0.5, on_score=0.95, tied_window=0.02) == "off"


def test_classify_tied_when_within_window() -> None:
    # Δoff = 0.10, Δon = 0.11 → diff is 0.01 < window 0.02.
    assert score._classify(1.0, off_score=0.9, on_score=0.89, tied_window=0.02) == "tied"


# ---------------------------------------------------------------------------
# score_against_ground_truth.main
# ---------------------------------------------------------------------------


def test_score_main_reports_aggregate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = _artifact([("q1", 0.5, 0.9), ("q2", 0.6, 0.7)])
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    labels = tmp_path / "labels.md"
    labels.write_text(
        """\
```yaml
id: q1
faithfulness_score: 0.9
verdict: faithful
```

```yaml
id: q2
faithfulness_score: 0.6
verdict: partial
```
""",
        encoding="utf-8",
    )
    rc = score.main(["--labels", str(labels), "--artifact", str(artifact_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # q1: label 0.9, off 0.5, on 0.9 → on closer.
    # q2: label 0.6, off 0.6, on 0.7 → off closer.
    assert "off-closer" in out
    assert "on-closer" in out
    assert "Signal: tied" in out  # 1 vs 1


def test_score_main_returns_nonzero_when_no_usable_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_path = tmp_path / "a.json"
    artifact_path.write_text(json.dumps(_artifact([("q1", 0.5, 0.6)])), encoding="utf-8")

    labels = tmp_path / "labels.md"
    labels.write_text(
        """\
```yaml
id: q1
faithfulness_score: TBD
```
""",
        encoding="utf-8",
    )
    rc = score.main(["--labels", str(labels), "--artifact", str(artifact_path)])
    assert rc == 1
    assert "No usable labels" in capsys.readouterr().err
