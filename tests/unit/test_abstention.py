"""Tests for Phase 5 abstention: configurable min_score + calibration tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from attune_rag import benchmark as b
from attune_rag.corpus import DirectoryCorpus
from attune_rag.retrieval import KeywordRetriever

# ---------------------------------------------------------------------------
# KeywordRetriever(min_score=...) — the abstention knob
# ---------------------------------------------------------------------------


def _one_doc_corpus(tmp_path: Path) -> DirectoryCorpus:
    (tmp_path / "security-audit.md").write_text(
        "# Security audit\n\nrun a security audit scan", encoding="utf-8"
    )
    return DirectoryCorpus(tmp_path)


def test_default_min_score_unchanged() -> None:
    assert KeywordRetriever().MIN_SCORE == 2.0  # backward compatible


def test_min_score_override_is_per_instance() -> None:
    assert KeywordRetriever(min_score=5).MIN_SCORE == 5
    assert KeywordRetriever().MIN_SCORE == 2.0  # class default not mutated


def test_high_min_score_abstains_low_keeps(tmp_path: Path) -> None:
    corpus = _one_doc_corpus(tmp_path)
    q = "security audit"
    assert list(KeywordRetriever().retrieve(q, corpus, k=3))  # default finds it
    assert list(KeywordRetriever(min_score=1000).retrieve(q, corpus, k=3)) == []  # abstains


# ---------------------------------------------------------------------------
# _top1_scores
# ---------------------------------------------------------------------------


def test_top1_scores_zero_when_no_hits() -> None:
    from types import SimpleNamespace

    pipe = SimpleNamespace(
        run=lambda query, k=3: SimpleNamespace(citation=SimpleNamespace(hits=[]))
    )
    with patch("attune_rag.RagPipeline", return_value=pipe):
        assert b._top1_scores([{"query": "x"}], k=3) == [0.0]


# ---------------------------------------------------------------------------
# _calibrate_abstention — recommendation logic
# ---------------------------------------------------------------------------


def test_calibrate_recommends_separating_threshold(monkeypatch) -> None:
    legit_q = [{"query": "L"}]
    neg_q = [{"query": "N"}]

    def fake_top1(queries, k, corpus=None):
        return [10.0] if queries is legit_q else [3.0]

    monkeypatch.setattr(b, "_top1_scores", fake_top1)
    cal = b._calibrate_abstention(legit_q, neg_q, k=3)
    # A threshold above the negative (3) but at/below the legit (10) is best.
    assert 3 < cal["recommended_threshold"] <= 10
    assert cal["recommended_legit_kept"] == 1.0
    assert cal["recommended_negatives_abstained"] == 1.0
    assert cal["recommended_false_answer_rate"] == 0.0


def test_calibrate_recommends_no_abstention_when_no_separation(monkeypatch) -> None:
    # legit and negatives have identical scores -> no threshold abstains
    # negatives without dropping legit, so the recommendation is T=0 (don't
    # abstain) — the honest answer when there's no separating signal.
    monkeypatch.setattr(b, "_top1_scores", lambda q, k, corpus=None: [5.0])
    cal = b._calibrate_abstention([{"query": "L"}], [{"query": "N"}], k=3)
    assert cal["recommended_threshold"] == 0
    assert cal["recommended_negatives_abstained"] == 0.0


# ---------------------------------------------------------------------------
# main --calibrate-abstention
# ---------------------------------------------------------------------------


def _yaml(path: Path, queries: list[dict]) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({"version": 1, "queries": queries}), encoding="utf-8")
    return path


def test_main_calibrate_abstention_runs_and_exits(tmp_path, monkeypatch, capsys) -> None:
    q = _yaml(tmp_path / "q.yaml", [{"id": "l1", "query": "legit"}])
    neg = _yaml(tmp_path / "n.yaml", [{"id": "n1", "query": "offtopic"}])
    monkeypatch.setattr(
        b, "_top1_scores", lambda qs, k, corpus=None: [10.0] if qs[0]["id"] == "l1" else [2.0]
    )
    rc = b.main(["--queries", str(q), "--negatives", str(neg), "--calibrate-abstention"])
    assert rc == 0
    assert "Abstention calibration" in capsys.readouterr().out


def test_main_calibrate_abstention_requires_negatives(tmp_path, capsys) -> None:
    q = _yaml(tmp_path / "q.yaml", [{"id": "l1", "query": "legit"}])
    rc = b.main(
        [
            "--queries",
            str(q),
            "--negatives",
            str(tmp_path / "missing.yaml"),
            "--calibrate-abstention",
        ]
    )
    assert rc == 2
    assert "needs a --negatives set" in capsys.readouterr().err
