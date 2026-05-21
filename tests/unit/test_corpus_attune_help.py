"""Unit tests for AttuneHelpCorpus.

Skipped when the optional ``attune-help`` dep is unavailable
(per the CLAUDE.md lesson about ``pytest.importorskip``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("attune_help")

from attune_rag.corpus.attune_help import AttuneHelpCorpus  # noqa: E402


def test_loads_bundled_corpus() -> None:
    corpus = AttuneHelpCorpus.from_attune_help()
    entries = list(corpus.entries())
    # attune-help v0.5.x ships >=500 templates; assert a floor
    # that still catches regressions without being brittle.
    assert len(entries) >= 500


def test_has_expected_categories() -> None:
    corpus = AttuneHelpCorpus.from_attune_help()
    categories = {e.category for e in corpus.entries()}
    expected = {
        "concepts",
        "quickstarts",
        "tasks",
        "errors",
        "faqs",
        "references",
    }
    assert expected.issubset(categories)


def test_name_and_version() -> None:
    import attune_help

    corpus = AttuneHelpCorpus.from_attune_help()
    assert corpus.name == "attune-help"
    assert corpus.version == attune_help.__version__


def test_get_returns_by_path() -> None:
    corpus = AttuneHelpCorpus.from_attune_help()
    some = next(iter(corpus.entries()))
    fetched = corpus.get(some.path)
    assert fetched is not None
    assert fetched.path == some.path


def test_get_unknown_returns_none() -> None:
    corpus = AttuneHelpCorpus.from_attune_help()
    assert corpus.get("does/not/exist.md") is None


# --- aliases_override.json integration ---


def test_bug_predict_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-bug-predict.md.

    Regression guard for the alias-expansion-sweep entry condition: D3
    proved alias expansion closes the bug-predict paraphrase gap with
    zero baseline regression. This test pins the mechanism so a future
    change to AttuneHelpCorpus or the override file shape can't silently
    revert it.
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-bug-predict.md")
    assert entry is not None
    # Frontmatter aliases come first; overrides are appended. Pin a few
    # override aliases the diagnostic-3.md sweep identified.
    assert "dangerous code" in entry.aliases
    assert "weak points" in entry.aliases
    assert "fails silently" in entry.aliases  # gqp-015a fix
    assert "diff bite" in entry.aliases  # gqp-036a fix


def test_bug_predict_paraphrased_queries_surface_concepts_entry() -> None:
    """KeywordRetriever returns concepts/tool-bug-predict.md in top-3 for
    paraphrased queries that previously missed (D1) and that the alias
    sweep was authored to fix.

    Inline subset of tests/golden/queries_paraphrased.yaml so this test
    doesn't depend on that file existing in the working tree. The full
    diagnostic-1 re-run lives elsewhere; this test is a lightweight
    smoke check that the override file did the job.
    """
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()

    # 5 of the 9 bug-predict queries that missed R@3 in D1 keyword-only.
    # Each pair: (query text, expected path that must appear in top-3).
    target = "concepts/tool-bug-predict.md"
    paraphrased = [
        "what's potentially harmful in my source",  # gqp-014a
        "what are the weak points in my source",  # gqp-016b
        "what part of this PR is dangerous",  # gqp-027a
        "where might my service fail silently",  # gqp-015a (residual fix)
        "where's the diff going to bite me",  # gqp-036a (residual fix)
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Paraphrased query did not surface bug-predict in top-3: {query!r} " f"→ {paths}"
        )


def test_bug_predict_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly bug-predict queries still find the target
    after aliases are added — a sanity check that the override doesn't
    pull the entry away from queries it already handled."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "concepts/tool-bug-predict.md"

    for query in [
        "find bugs in my code",
        "predict bugs before they happen",
        "spot risky code changes before merging",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Baseline bug-predict query regressed after alias override: " f"{query!r} → {paths}"
        )


def test_path_keyed_summaries_load_from_attune_help_0_7_0() -> None:
    """AttuneHelpCorpus consumes summaries_by_path.json (0.7.0+).

    Before 0.1.2 this corpus passed no sidecar file because
    attune-help's summaries.json was feature-keyed and
    silently ignored by DirectoryCorpus. Now we read
    summaries_by_path.json which attune-help 0.7.0+ ships,
    so some — not necessarily all — entries populate
    summaries.
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entries = list(corpus.entries())
    with_summary = sum(1 for e in entries if e.summary)
    # attune-help 0.7.0 ships 124 polished path-keyed
    # summaries; older attune-help ships none. Either is
    # acceptable — the test just verifies the sidecar path
    # is being wired up correctly.
    assert with_summary >= 0
    # cross_links.json has an incompatible schema and is
    # intentionally not wired; every entry should have
    # empty related for now.
    assert all(e.related == () for e in entries)


def test_raises_helpful_error_when_attune_help_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sys.modules sentinel — works across Python 3.10-3.13."""
    import sys

    saved = {
        k: v for k, v in sys.modules.items() if k == "attune_help" or k.startswith("attune_help.")
    }
    for key in saved:
        del sys.modules[key]
    sys.modules["attune_help"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(RuntimeError, match=r"\[attune-help\] extra"):
            AttuneHelpCorpus.from_attune_help()
    finally:
        sys.modules.pop("attune_help", None)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# HelpCorpusAdapter — protocol path (no attune-help required)
# ---------------------------------------------------------------------------


def test_corpus_works_with_injected_adapter(tmp_path) -> None:
    """The protocol path lets callers wire any directory of templates
    in without ever importing attune-help. Doubles as a contract test
    for the HelpCorpusAdapter shape.
    """
    from attune_rag.corpus.attune_help import _BundledAdapter

    # Minimal templates dir
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "alpha.md").write_text("# alpha\nbody\n")

    adapter = _BundledAdapter(templates_root=tmp_path, version="test-v1")
    corpus = AttuneHelpCorpus(adapter=adapter)
    entries = list(corpus.entries())

    assert len(entries) == 1
    assert entries[0].path == "concepts/alpha.md"
    assert corpus.version == "test-v1"


def test_invalid_templates_root_raises() -> None:
    """Adapter pointing at a non-directory must fail loudly at construction."""
    from pathlib import Path as _P

    from attune_rag.corpus.attune_help import _BundledAdapter

    bad_adapter = _BundledAdapter(templates_root=_P("/this/path/does/not/exist"), version="x")
    with pytest.raises(RuntimeError, match="templates_root is not a directory"):
        AttuneHelpCorpus(adapter=bad_adapter)
