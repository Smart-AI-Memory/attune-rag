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


def test_sidecars_deferred_to_v020() -> None:
    """v0.1.0 does not wire attune-help's summaries/cross_links.

    attune-help's sidecar schemas do not match DirectoryCorpus's
    path-keyed expectation (see module docstring in
    corpus/attune_help.py). v0.1.0 ships templates only;
    summaries and related metadata land in v0.2.0 when a
    schema adapter is added.
    """
    corpus = AttuneHelpCorpus()
    assert all(e.summary is None for e in corpus.entries())
    assert all(e.related == () for e in corpus.entries())


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
