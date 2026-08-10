"""safe-abstention-defaults M3+M4 — the bundled default and the BYO path.

Spec: docs/specs/safe-abstention-defaults/ (scoping locked 2026-07-17).
R1 (gate-set recall) is enforced by tests/golden/; these tests pin the
wiring contracts: who gets the calibrated threshold, who keeps the
conservative class default, and that the escape hatches survive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus, RagPipeline
from attune_rag.pipeline import _BUNDLED_MIN_SCORE
from attune_rag.retrieval import KeywordRetriever


@pytest.fixture
def byo_corpus(tmp_path: Path) -> DirectoryCorpus:
    (tmp_path / "doc.md").write_text(
        "---\ntype: concept\nname: widget\n---\n# Widget guide\n"
        "Widgets are assembled from sprockets and flanges.\n"
    )
    return DirectoryCorpus(tmp_path)


def test_bundled_default_gets_calibrated_threshold() -> None:
    # RagPipeline() with nothing passed → bundled corpus → calibrated 5.0.
    p = RagPipeline()
    assert p.retriever.MIN_SCORE == _BUNDLED_MIN_SCORE == 5.0


def test_byo_corpus_keeps_conservative_class_default(byo_corpus) -> None:
    # A caller-supplied corpus must NOT inherit the bundled calibration
    # (R3: no unsafe universal constant — 5.0 over-abstains lean corpora).
    p = RagPipeline(corpus=byo_corpus)
    assert p.retriever.MIN_SCORE == KeywordRetriever.MIN_SCORE == 2.0


def test_explicit_retriever_is_untouched(byo_corpus) -> None:
    # Escape hatch: explicit construction wins on every path.
    r = KeywordRetriever(min_score=7)
    assert RagPipeline(retriever=r).retriever is r
    assert RagPipeline(corpus=byo_corpus, retriever=r).retriever is r


def test_raw_keyword_retriever_class_default_unchanged() -> None:
    # The class default is part of the public contract (POLICY §7 levers);
    # the bundled calibration must not leak into it.
    assert KeywordRetriever().MIN_SCORE == 2.0


def test_bundled_pipeline_abstains_on_out_of_corpus_query() -> None:
    # The behavior R2 buys: an out-of-corpus query on the bundled
    # pipeline abstains (fallback prompt) instead of answering wrongly.
    p = RagPipeline()
    result = p.run("best sourdough starter hydration ratio")
    assert result.fallback_used is True
    assert result.confidence == 0.0


def test_bundled_pipeline_still_answers_gate_query() -> None:
    # R1 spot check (full enforcement lives in tests/golden/): a
    # gate-set query still retrieves at the calibrated threshold.
    p = RagPipeline()
    result = p.run("SAST scan my repository")
    assert result.fallback_used is False


def test_calibrated_classmethod_derives_corpus_specific_threshold(
    byo_corpus,
) -> None:
    # M4: the honest BYO path — threshold from the corpus's own data.
    p = RagPipeline.calibrated(
        byo_corpus,
        queries=["widget sprocket flange guide"],
        negatives=["marathon training plan for beginners"],
    )
    assert isinstance(p.retriever, KeywordRetriever)
    # Derived, corpus-specific: keeps the legit query answerable...
    assert p.run("widget sprocket flange guide").fallback_used is False
    # ...and abstains on the out-of-corpus one.
    assert p.run("marathon training plan for beginners").fallback_used is True


def test_calibrated_with_no_separation_recommends_zero(byo_corpus) -> None:
    # When legit and negative sets don't separate, the sweep lands on
    # T=0 — "don't abstain", the honest no-signal answer.
    p = RagPipeline.calibrated(
        byo_corpus,
        queries=["quantum chromodynamics lattice"],  # scores ~0 (out-of-corpus)
        negatives=["another unrelated astrophysics query"],
    )
    assert p.retriever.MIN_SCORE == 0.0
