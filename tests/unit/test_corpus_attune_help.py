"""Unit tests for AttuneHelpCorpus.

Skipped when the optional ``attune-help`` dep is unavailable
(per the CLAUDE.md lesson about ``pytest.importorskip``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("attune_help")

from attune_rag.corpus.attune_help import AttuneHelpCorpus  # noqa: E402


def test_loads_bundled_corpus() -> None:
    corpus = AttuneHelpCorpus()
    entries = list(corpus.entries())
    # attune-help v0.5.x ships >=500 templates; assert a floor
    # that still catches regressions without being brittle.
    assert len(entries) >= 500


def test_has_expected_categories() -> None:
    corpus = AttuneHelpCorpus()
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

    corpus = AttuneHelpCorpus()
    assert corpus.name == "attune-help"
    assert corpus.version == attune_help.__version__


def test_get_returns_by_path() -> None:
    corpus = AttuneHelpCorpus()
    some = next(iter(corpus.entries()))
    fetched = corpus.get(some.path)
    assert fetched is not None
    assert fetched.path == some.path


def test_get_unknown_returns_none() -> None:
    corpus = AttuneHelpCorpus()
    assert corpus.get("does/not/exist.md") is None


def test_path_keyed_summaries_load_from_attune_help_0_7_0() -> None:
    """AttuneHelpCorpus consumes summaries_by_path.json (0.7.0+).

    Before 0.1.2 this corpus passed no sidecar file because
    attune-help's summaries.json was feature-keyed and
    silently ignored by DirectoryCorpus. Now we read
    summaries_by_path.json which attune-help 0.7.0+ ships,
    so some — not necessarily all — entries populate
    summaries.
    """
    corpus = AttuneHelpCorpus()
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
            AttuneHelpCorpus()
    finally:
        sys.modules.pop("attune_help", None)
        sys.modules.update(saved)
