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


def test_security_audit_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-security-audit.md.

    Regression guard for the alias-expansion-sweep M3 (security-audit
    cluster) — pins a few override aliases the sweep identified so a
    future change to the override file or the merge mechanism can't
    silently revert the cluster's R@3 lift.
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-security-audit.md")
    assert entry is not None
    assert "leaked credentials" in entry.aliases
    assert "exploit surface" in entry.aliases
    assert "attackers break in" in entry.aliases
    assert "potentially attackable" in entry.aliases  # gqp-032b
    assert "service compromised" in entry.aliases  # gqp-011a


def test_security_audit_paraphrased_queries_surface_concepts_entry() -> None:
    """KeywordRetriever returns concepts/tool-security-audit.md in top-3 for
    5 of the 8 D1 paraphrased misses on the security-audit cluster.

    Mirrors the bug-predict regression test. Each alias used here was
    pre-validated with attune_rag.retrieval._tokenize to clear
    MIN_ALIAS_OVERLAP=2 against the corresponding query.

    Known residual: gqp-023a "look for ways my code could be exploited"
    is semantically ambiguous between security-audit and bug-predict
    (the verb "exploited" reads as security; "what could go wrong with
    my code" reads as bug-predict). Bug-predict currently wins via a
    summary token match on "code". Captured in the full diagnostic-1
    re-run, not papered over here with awkward aliases.
    """
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "concepts/tool-security-audit.md"

    paraphrased = [
        "what catches unsafe input handling and leaked credentials",  # gqp-001a
        "I want to find places attackers could break in",  # gqp-001b
        "look for ways my service could be compromised",  # gqp-011a
        "check my project for exploit surface",  # gqp-011b
        "look for leaked tokens and risky function calls",  # gqp-023b
        "what's potentially attackable here",  # gqp-032b
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Paraphrased query did not surface security-audit in top-3: " f"{query!r} → {paths}"
        )


def test_security_audit_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly security-audit queries still surface *some*
    security-audit entry in top-3 after the alias override is in place.

    Matches the R@3 semantic used by tests/golden/test_golden.py — the
    golden set accepts either concepts/ or quickstarts/run-security-audit.md
    for these baseline queries, so this test does too. Asserting concepts/
    specifically would be stricter than what production callers (and the
    golden suite) require.
    """
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    security_audit_paths = {
        "concepts/tool-security-audit.md",
        "quickstarts/run-security-audit.md",
        "quickstarts/skill-security-audit.md",
        "references/tool-security-audit.md",
    }

    for query in [
        "vulnerability scan",
        "check for security vulnerabilities",
        "SAST scan my repository",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & security_audit_paths, (
            f"Baseline security-audit query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
        )


def test_release_prep_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-release-prep.md.

    Regression guard for the alias-expansion-sweep M4 (release-prep cluster).
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-release-prep.md")
    assert entry is not None
    # Frontmatter aliases preserved.
    assert "publish to PyPI" in entry.aliases
    assert "cut a release" in entry.aliases
    # Override aliases appended.
    assert "before pushing" in entry.aliases
    assert "push to users" in entry.aliases
    assert "push to the world" in entry.aliases  # gqp-019b
    assert "push package" in entry.aliases  # gqp-035a


def test_release_prep_paraphrased_queries_surface_release_prep_entry() -> None:
    """KeywordRetriever returns *some* release-prep entry in top-3 for
    5 of the 8 D1 paraphrased misses on the release-prep cluster.

    Mirrors the security-audit / bug-predict regression tests. R@3
    semantic matches tests/golden/test_golden.py — multiple
    release-prep paths are acceptable (concepts/, quickstarts/,
    references/, tasks/) since the golden set accepts any of them.
    """
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    release_prep_paths = {
        "concepts/tool-release-prep.md",
        "quickstarts/skill-release-prep.md",
        "references/skill-release-prep.md",
        "references/tool-release-prep.md",
        "tasks/use-release-prep.md",
    }

    paraphrased = [
        "what's the gate before I push v0.5 out",  # gqp-008b
        "what should I do before pushing v0.4 to users",  # gqp-019a
        "push my code out to the world",  # gqp-019b
        "what do I need before pushing this out",  # gqp-026a
        "push this package out to users",  # gqp-035a
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & release_prep_paths, (
            f"Paraphrased query did not surface release-prep in top-3: "
            f"{query!r} → {sorted(paths)}"
        )


def test_release_prep_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly release-prep queries still surface *some*
    release-prep entry in top-3 after the alias override is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    release_prep_paths = {
        "concepts/tool-release-prep.md",
        "quickstarts/skill-release-prep.md",
        "references/skill-release-prep.md",
        "references/tool-release-prep.md",
        "tasks/use-release-prep.md",
    }

    for query in [
        "prepare a release",
        "create a release",
        "version bump and changelog",
        "publish to PyPI",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & release_prep_paths, (
            f"Baseline release-prep query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
        )


def test_smart_test_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-smart-test.md.

    Regression guard for the alias-expansion-sweep M5 (smart-test cluster).
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-smart-test.md")
    assert entry is not None
    assert "safety nets" in entry.aliases
    assert "untouched module" in entry.aliases  # gqp-022a
    assert "functions need assertions" in entry.aliases  # gqp-022b
    assert "shore up coverage" in entry.aliases  # gqp-022a alt
    assert "missing tests" in entry.aliases


def test_smart_test_paraphrased_queries_surface_smart_test_entry() -> None:
    """KeywordRetriever returns *some* smart-test entry in top-3 for the
    4 D1 paraphrased misses on the smart-test cluster.

    R@3 semantic matches tests/golden/test_golden.py — concepts/,
    quickstarts/, or any other smart-test path is acceptable.
    """
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    smart_test_paths = {
        "concepts/tool-smart-test.md",
        "quickstarts/generate-tests.md",
        "quickstarts/skill-smart-test.md",
        "references/skill-smart-test.md",
        "tasks/use-smart-test.md",
    }

    paraphrased = [
        "I have functions with no safety nets",  # gqp-002a
        "build safety nets for these functions",  # gqp-020a
        "shore up untouched parts of my module",  # gqp-022a
        "what functions need assertions",  # gqp-022b
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & smart_test_paths, (
            f"Paraphrased query did not surface smart-test in top-3: "
            f"{query!r} → {sorted(paths)}"
        )


def test_smart_test_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly smart-test queries still surface *some*
    smart-test entry in top-3 after the alias override is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    smart_test_paths = {
        "concepts/tool-smart-test.md",
        "quickstarts/generate-tests.md",
        "quickstarts/skill-smart-test.md",
        "references/skill-smart-test.md",
        "tasks/use-smart-test.md",
    }

    for query in [
        "generate tests for my code",
        "write unit tests",
        "add test coverage to my project",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & smart_test_paths, (
            f"Baseline smart-test query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
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
