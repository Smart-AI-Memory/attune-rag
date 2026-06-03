"""Build-time single-token-alias warning (alias-overlap-remediation).

Covers the observability-only warning added to DirectoryCorpus: it fires
when a meaningful share of aliased entries have only single-token aliases
that can never satisfy KeywordRetriever.MIN_ALIAS_OVERLAP. The warning
does NOT change retrieval — the bundled golden snapshot byte-identical
test (tests/golden/) is the companion proof of that.

See docs/specs/alias-overlap-remediation/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from attune_rag.corpus.directory import DirectoryCorpus

_LOGGER = "attune_rag.corpus.directory"


def _write(root: Path, aliases_per_file: list[list[str]]) -> None:
    for i, aliases in enumerate(aliases_per_file):
        fm = "---\naliases:\n" + "".join(f"  - {a}\n" for a in aliases) + "---\n"
        (root / f"doc{i}.md").write_text(fm + f"# Doc {i}\n\nbody {i}\n", encoding="utf-8")


def _build(corpus: DirectoryCorpus) -> None:
    # entries() triggers the lazy build (and thus the warning).
    list(corpus.entries())


def test_single_token_aliases_warn_once(tmp_path, caplog):
    _write(tmp_path, [["security", "api"], ["mcp"], ["redis"]])
    corpus = DirectoryCorpus(root=tmp_path)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _build(corpus)
    recs = [r for r in caplog.records if r.name == _LOGGER]
    assert len(recs) == 1, f"expected exactly one warning, got {len(recs)}"
    msg = recs[0].getMessage()
    assert "single-token aliases" in msg
    assert "MIN_ALIAS_OVERLAP=2" in msg
    assert "3 of 3" in msg  # all three aliased entries are degraded


def test_multi_token_aliases_silent(tmp_path, caplog):
    _write(tmp_path, [["ship a release", "publish to pypi"], ["run the tests"]])
    corpus = DirectoryCorpus(root=tmp_path)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _build(corpus)
    assert not [r for r in caplog.records if r.name == _LOGGER]


def test_min_overlap_one_silent(tmp_path, caplog, monkeypatch):
    # When the floor is inert (<=1), single-token aliases count, so no warning.
    from attune_rag.retrieval import KeywordRetriever

    monkeypatch.setattr(KeywordRetriever, "MIN_ALIAS_OVERLAP", 1)
    _write(tmp_path, [["security"], ["api"]])
    corpus = DirectoryCorpus(root=tmp_path)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _build(corpus)
    assert not [r for r in caplog.records if r.name == _LOGGER]


def test_warn_disabled_flag_silent(tmp_path, caplog):
    _write(tmp_path, [["security"], ["api"]])
    corpus = DirectoryCorpus(root=tmp_path, warn_alias_overlap=False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _build(corpus)
    assert not [r for r in caplog.records if r.name == _LOGGER]


def test_entries_without_aliases_not_degraded(tmp_path, caplog):
    # Files with zero aliases never relied on alias signal — not degraded.
    (tmp_path / "plain.md").write_text("# Plain\n\nno frontmatter aliases\n", encoding="utf-8")
    corpus = DirectoryCorpus(root=tmp_path)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _build(corpus)
    assert not [r for r in caplog.records if r.name == _LOGGER]


def test_warning_latched_once_per_instance(tmp_path, caplog):
    # cache=False rebuilds on every entries() call; the latch keeps it to one warning.
    _write(tmp_path, [["security"], ["api"]])
    corpus = DirectoryCorpus(root=tmp_path, cache=False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _build(corpus)
        _build(corpus)
    recs = [r for r in caplog.records if r.name == _LOGGER]
    assert len(recs) == 1, f"latch failed: {len(recs)} warnings"


def test_bundled_corpus_silent(caplog):
    # The bundled AttuneHelpCorpus is multi-token alias-tuned; it must NOT warn.
    try:
        from attune_rag.corpus import AttuneHelpCorpus
    except Exception:  # noqa: BLE001
        pytest.skip("AttuneHelpCorpus unavailable")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        corpus = AttuneHelpCorpus.from_attune_help()
        list(corpus.entries())
    assert not [r for r in caplog.records if r.name == _LOGGER]
