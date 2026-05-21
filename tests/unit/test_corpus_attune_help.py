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


def test_fix_test_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-fix-test.md.

    Regression guard for the alias-expansion-sweep M6 (fix-test cluster).
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-fix-test.md")
    assert entry is not None
    # Frontmatter aliases preserved.
    assert "CI pipeline failing" in entry.aliases
    # Override aliases appended.
    assert "suite is red" in entry.aliases
    assert "red after merge" in entry.aliases  # gqp-003a
    assert "figure out why tests fail" in entry.aliases  # gqp-024a
    assert "blocking my merges" in entry.aliases  # gqp-031b


def test_fix_test_paraphrased_queries_surface_fix_test_entry() -> None:
    """KeywordRetriever returns *some* fix-test entry in top-3 for the
    3 D1 paraphrased misses on the fix-test cluster."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    fix_test_paths = {
        "concepts/tool-fix-test.md",
        "quickstarts/skill-fix-test.md",
        "references/skill-fix-test.md",
        "tasks/use-fix-test.md",
    }

    paraphrased = [
        "suite went red after last merge",  # gqp-003a
        "my suite is red — figure out why",  # gqp-024a
        "what's blocking my merges",  # gqp-031b
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & fix_test_paths, (
            f"Paraphrased query did not surface fix-test in top-3: " f"{query!r} → {sorted(paths)}"
        )


def test_fix_test_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly fix-test queries still surface *some*
    fix-test entry in top-3 after the alias override is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    fix_test_paths = {
        "concepts/tool-fix-test.md",
        "quickstarts/skill-fix-test.md",
        "references/skill-fix-test.md",
        "tasks/use-fix-test.md",
    }

    for query in [
        "fix failing tests",
        "debug failing tests",
        "my CI pipeline keeps failing",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & fix_test_paths, (
            f"Baseline fix-test query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
        )


def test_code_quality_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-code-quality.md.

    Regression guard for the alias-expansion-sweep M7 (code-quality cluster).
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-code-quality.md")
    assert entry is not None
    assert "once-over module" in entry.aliases  # gqp-004a / 018a
    assert "evaluate craftsmanship" in entry.aliases  # gqp-004b
    assert "raise the bar" in entry.aliases  # gqp-029a
    assert "clean module" in entry.aliases  # gqp-029b
    assert "code craftsmanship" in entry.aliases


def test_code_quality_paraphrased_queries_surface_code_quality_entry() -> None:
    """KeywordRetriever returns *some* code-quality entry in top-3 for the
    5 D1 paraphrased misses on the code-quality cluster."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    code_quality_paths = {
        "concepts/tool-code-quality.md",
        "quickstarts/skill-code-quality.md",
        "references/skill-code-quality.md",
        "tasks/use-code-quality.md",
    }

    paraphrased = [
        "give me a once-over on this module",  # gqp-004a
        "evaluate craftsmanship of my changes",  # gqp-004b
        "give my module a once-over",  # gqp-018a
        "raise the bar on this codebase",  # gqp-029a
        "what's keeping this module from being clean",  # gqp-029b
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & code_quality_paths, (
            f"Paraphrased query did not surface code-quality in top-3: "
            f"{query!r} → {sorted(paths)}"
        )


def test_code_quality_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly code-quality queries still surface *some*
    code-quality entry in top-3 after the alias override is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    code_quality_paths = {
        "concepts/tool-code-quality.md",
        "quickstarts/skill-code-quality.md",
        "references/skill-code-quality.md",
        "tasks/use-code-quality.md",
    }

    for query in [
        "review code quality",
        "check code quality",
        "improve code quality metrics",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & code_quality_paths, (
            f"Baseline code-quality query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
        )


def test_refactor_plan_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-refactor-plan.md.

    Regression guard for the alias-expansion-sweep M8 (refactor-plan cluster).
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-refactor-plan.md")
    assert entry is not None
    # Frontmatter aliases preserved.
    assert "technical debt" in entry.aliases
    assert "code smells" in entry.aliases
    # Override aliases appended.
    assert "untangle module" in entry.aliases  # gqp-007a
    assert "rework tangled" in entry.aliases  # gqp-007b
    assert "accumulated junk" in entry.aliases  # gqp-021a
    assert "prune codebase" in entry.aliases  # gqp-021b
    assert "simplify gnarly" in entry.aliases  # gqp-030a
    assert "too many branches" in entry.aliases  # gqp-030b


def test_refactor_plan_paraphrased_queries_surface_refactor_plan_entry() -> None:
    """KeywordRetriever returns *some* refactor-plan entry in top-3 for the
    6 D1 paraphrased misses on the refactor-plan cluster."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    refactor_paths = {
        "concepts/tool-refactor-plan.md",
        "quickstarts/skill-refactor-plan.md",
        "references/skill-refactor-plan.md",
        "references/tool-refactor-plan.md",
        "tasks/use-refactor-plan.md",
    }

    paraphrased = [
        "untangle this module",  # gqp-007a
        "rework the tangled bits in my project",  # gqp-007b
        "the codebase has accumulated junk over time",  # gqp-021a
        "what should I prune from this codebase",  # gqp-021b
        "simplify this gnarly module",  # gqp-030a
        "this module has too many branches and nesting",  # gqp-030b
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & refactor_paths, (
            f"Paraphrased query did not surface refactor-plan in top-3: "
            f"{query!r} → {sorted(paths)}"
        )


def test_refactor_plan_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly refactor-plan queries still surface *some*
    refactor-plan entry in top-3 after the alias override is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    refactor_paths = {
        "concepts/tool-refactor-plan.md",
        "quickstarts/skill-refactor-plan.md",
        "references/skill-refactor-plan.md",
        "references/tool-refactor-plan.md",
        "tasks/use-refactor-plan.md",
    }

    for query in [
        "refactor my code",
        "reduce code complexity",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & refactor_paths, (
            f"Baseline refactor-plan query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
        )


def test_planning_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-planning.md.

    Regression guard for the alias-expansion-sweep M9 (planning cluster).
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-planning.md")
    assert entry is not None
    # Frontmatter aliases preserved.
    assert "sprint planning" in entry.aliases
    assert "feature roadmap" in entry.aliases
    # Override aliases appended.
    assert "design before code" in entry.aliases  # gqp-028a
    assert "what to tackle next" in entry.aliases  # gqp-039a
    assert "next two weeks" in entry.aliases  # gqp-039a alt


def test_planning_paraphrased_queries_surface_planning_entry() -> None:
    """KeywordRetriever returns *some* planning entry in top-3 for the
    2 D1 paraphrased misses on the planning cluster (gqp-028a, gqp-039a).
    The other 4 planning paraphrased queries (gqp-010a/b, 028b, 039b)
    already passed under existing frontmatter aliases."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    planning_paths = {
        "concepts/tool-planning.md",
        "quickstarts/skill-planning.md",
        "references/skill-planning.md",
        "tasks/use-planning.md",
    }

    paraphrased = [
        "I need a design pass before I write code",  # gqp-028a
        "what should I tackle in the next two weeks",  # gqp-039a
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & planning_paths, (
            f"Paraphrased query did not surface planning in top-3: " f"{query!r} → {sorted(paths)}"
        )


def test_planning_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly planning queries still surface *some*
    planning entry in top-3 after the alias override is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    planning_paths = {
        "concepts/tool-planning.md",
        "quickstarts/skill-planning.md",
        "references/skill-planning.md",
        "tasks/use-planning.md",
    }

    for query in [
        "plan a new feature",
        "architect a new feature",
        "scope out next sprint tasks",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & planning_paths, (
            f"Baseline planning query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
        )


def test_doc_gen_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into concepts/tool-doc-gen.md.

    Regression guard for the alias-expansion-sweep M10 (doc-gen cluster).
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("concepts/tool-doc-gen.md")
    assert entry is not None
    assert "spin up docs" in entry.aliases  # gqp-017a
    assert "module explainer" in entry.aliases
    assert "explain functions" in entry.aliases
    assert "annotate functions" in entry.aliases
    assert "human-readable docs" in entry.aliases


def test_doc_gen_paraphrased_queries_surface_doc_gen_entry() -> None:
    """KeywordRetriever returns *some* doc-gen entry in top-3 for the
    1 D1 paraphrased miss on the doc-gen cluster (gqp-017a). The other
    5 doc-gen paraphrased queries already passed under content/summary
    matches before this PR."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    doc_gen_paths = {
        "concepts/tool-doc-gen.md",
        "quickstarts/skill-doc-gen.md",
        "references/skill-doc-gen.md",
        "references/tool-doc-gen.md",
        "tasks/use-doc-gen.md",
    }

    paraphrased = [
        "spin up an explainer for my module",  # gqp-017a
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & doc_gen_paths, (
            f"Paraphrased query did not surface doc-gen in top-3: " f"{query!r} → {sorted(paths)}"
        )


def test_doc_gen_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly doc-gen queries still surface *some*
    doc-gen entry in top-3 after the alias override is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    doc_gen_paths = {
        "concepts/tool-doc-gen.md",
        "quickstarts/skill-doc-gen.md",
        "references/skill-doc-gen.md",
        "references/tool-doc-gen.md",
        "tasks/use-doc-gen.md",
    }

    for query in [
        "write documentation for my module",
        "create documentation for my code",
        "add docstrings to all my functions",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = {h.entry.path for h in hits}
        assert paths & doc_gen_paths, (
            f"Baseline doc-gen query regressed after alias override: "
            f"{query!r} → {sorted(paths)}"
        )


def test_doc_orchestrator_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into
    references/tool-doc-orchestrator.md.

    Regression guard for the alias-expansion-sweep M11 (doc-orchestrator
    cluster). Note: doc-orchestrator only has the references/ entry —
    no concepts/, quickstarts/, etc. — so aliases land there.
    """
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("references/tool-doc-orchestrator.md")
    assert entry is not None
    # Frontmatter aliases preserved.
    assert "orchestrate documentation workflow" in entry.aliases
    assert "doc pipeline" in entry.aliases
    # Override aliases appended.
    assert "wire up readme jobs" in entry.aliases  # gqp-013a / 038a
    assert "readme tasks back-to-back" in entry.aliases  # gqp-013b
    assert "wire up readme flow" in entry.aliases  # gqp-034a
    assert "reference material jobs" in entry.aliases  # gqp-034b
    assert "reference material tasks" in entry.aliases  # gqp-038b


def test_doc_orchestrator_paraphrased_queries_surface_doc_orchestrator_entry() -> None:
    """KeywordRetriever returns references/tool-doc-orchestrator.md in top-3
    for all 6 D1 paraphrased misses on the doc-orchestrator cluster."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "references/tool-doc-orchestrator.md"

    paraphrased = [
        "wire up all my readme-related jobs as one process",  # gqp-013a
        "run readme tasks back-to-back across the repo",  # gqp-013b
        "wire up readme tasks as one flow",  # gqp-034a
        "run all my reference-material jobs in sequence",  # gqp-034b
        "wire up readme jobs across all modules",  # gqp-038a
        "run my reference-material tasks in sequence repo-wide",  # gqp-038b
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Paraphrased query did not surface doc-orchestrator in top-3: " f"{query!r} → {paths}"
        )


def test_doc_orchestrator_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly doc-orchestrator queries still surface
    references/tool-doc-orchestrator.md in top-3 after the alias override
    is in place."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "references/tool-doc-orchestrator.md"

    for query in [
        "orchestrate documentation workflow",
        "manage the documentation pipeline",
        "coordinate documentation updates across the project",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Baseline doc-orchestrator query regressed after alias override: "
            f"{query!r} → {paths}"
        )


def test_deep_review_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into
    references/tool-deep-review.md (M12 cluster)."""
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("references/tool-deep-review.md")
    assert entry is not None
    assert "end-to-end review" in entry.aliases  # frontmatter preserved
    assert "go over branch" in entry.aliases  # gqp-005a
    assert "audit before landing" in entry.aliases  # gqp-005b
    assert "every changed file" in entry.aliases  # gqp-037b


def test_deep_review_paraphrased_queries_surface_deep_review_entry() -> None:
    """KeywordRetriever returns references/tool-deep-review.md in top-3
    for the 3 D1 paraphrased misses on the deep-review cluster
    (gqp-005a, gqp-005b, gqp-037b). gqp-037a already passed."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "references/tool-deep-review.md"

    paraphrased = [
        "go over everything in this branch before I ship",  # gqp-005a
        "audit all touched files before landing",  # gqp-005b
        "look at every changed file before landing",  # gqp-037b
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Paraphrased query did not surface deep-review in top-3: " f"{query!r} → {paths}"
        )


def test_deep_review_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly deep-review queries still surface
    references/tool-deep-review.md in top-3."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "references/tool-deep-review.md"

    for query in [
        "deep review my PR",
        "end-to-end review before merging a PR",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Baseline deep-review query regressed after alias override: " f"{query!r} → {paths}"
        )


def test_doc_audit_carries_override_aliases() -> None:
    """The aliases_override.json mechanism merges into
    references/tool-doc-audit.md (M12 cluster)."""
    corpus = AttuneHelpCorpus.from_attune_help()
    entry = corpus.get("references/tool-doc-audit.md")
    assert entry is not None
    assert "stale documentation" in entry.aliases  # frontmatter preserved
    assert "readme lies" in entry.aliases  # gqp-009a
    assert "readme is wrong" in entry.aliases  # gqp-009b
    assert "no longer accurate" in entry.aliases  # gqp-025a


def test_doc_audit_paraphrased_queries_surface_doc_audit_entry() -> None:
    """KeywordRetriever returns references/tool-doc-audit.md in top-3
    for the 3 D1 paraphrased misses on the doc-audit cluster
    (gqp-009a, gqp-009b, gqp-025a). gqp-025b already passed."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "references/tool-doc-audit.md"

    paraphrased = [
        "find places where my readme lies about the code",  # gqp-009a
        "what readme bits are wrong about how the app works now",  # gqp-009b
        "what guidance is no longer accurate",  # gqp-025a
    ]
    for query in paraphrased:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Paraphrased query did not surface doc-audit in top-3: " f"{query!r} → {paths}"
        )


def test_doc_audit_baseline_queries_still_pass() -> None:
    """Baseline keyword-friendly doc-audit queries still surface
    references/tool-doc-audit.md in top-3."""
    from attune_rag.retrieval import KeywordRetriever

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()
    target = "references/tool-doc-audit.md"

    for query in [
        "audit documentation for staleness",
        "find stale documentation",
    ]:
        hits = retriever.retrieve(query, corpus, k=3)
        paths = [h.entry.path for h in hits]
        assert target in paths, (
            f"Baseline doc-audit query regressed after alias override: " f"{query!r} → {paths}"
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
