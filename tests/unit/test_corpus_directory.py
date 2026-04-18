"""Unit tests for DirectoryCorpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus, RetrievalEntry


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    _write(root / "concepts" / "alpha.md", "# alpha\nalpha body")
    _write(root / "concepts" / "beta.md", "# beta\nbeta body")
    _write(root / "quickstarts" / "getting-started.md", "# start\nbody")
    _write(root / "readme.md", "# root\nroot body")
    _write(
        root / "summaries.json",
        json.dumps(
            {
                "concepts/alpha.md": "one line about alpha",
                "quickstarts/getting-started.md": "how to start",
            }
        ),
    )
    _write(
        root / "cross_links.json",
        json.dumps({"concepts/alpha.md": ["concepts/beta.md"]}),
    )
    return root


def test_entries_enumerates_markdown_files(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(
        tiny_corpus,
        summaries_file="summaries.json",
        cross_links_file="cross_links.json",
    )
    entries = list(corpus.entries())
    paths = {e.path for e in entries}
    assert paths == {
        "concepts/alpha.md",
        "concepts/beta.md",
        "quickstarts/getting-started.md",
        "readme.md",
    }


def test_category_is_first_segment(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(tiny_corpus)
    by_path = {e.path: e for e in corpus.entries()}
    assert by_path["concepts/alpha.md"].category == "concepts"
    assert by_path["quickstarts/getting-started.md"].category == "quickstarts"
    assert by_path["readme.md"].category == ""


def test_summary_and_related_are_wired(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(
        tiny_corpus,
        summaries_file="summaries.json",
        cross_links_file="cross_links.json",
    )
    alpha = corpus.get("concepts/alpha.md")
    assert alpha is not None
    assert alpha.summary == "one line about alpha"
    assert alpha.related == ("concepts/beta.md",)
    # File without a summary entry gets None
    assert corpus.get("concepts/beta.md").summary is None


def test_missing_sidecar_files_are_tolerated(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(
        tiny_corpus,
        summaries_file="does-not-exist.json",
        cross_links_file="also-missing.json",
    )
    entries = list(corpus.entries())
    assert len(entries) == 4
    assert all(e.summary is None for e in entries)
    assert all(e.related == () for e in entries)


def test_non_directory_root_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir.md"
    file_path.write_text("# nope", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        DirectoryCorpus(file_path)


def test_cache_returns_same_entries_across_calls(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(tiny_corpus, cache=True)
    first = tuple(corpus.entries())
    second = tuple(corpus.entries())
    assert first == second


def test_cache_disabled_reflects_new_files(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(tiny_corpus, cache=False)
    before = {e.path for e in corpus.entries()}
    _write(tiny_corpus / "concepts" / "gamma.md", "# gamma")
    after = {e.path for e in corpus.entries()}
    assert "concepts/gamma.md" not in before
    assert "concepts/gamma.md" in after


def test_name_and_version_populated(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(tiny_corpus)
    assert corpus.name == "directory:docs"
    v1 = corpus.version
    assert isinstance(v1, str) and len(v1) == 16
    # Version stable while content unchanged (cached)
    assert corpus.version == v1


def test_version_changes_when_content_changes(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(tiny_corpus, cache=False)
    v1 = corpus.version
    _write(tiny_corpus / "concepts" / "alpha.md", "# alpha\nchanged body")
    v2 = corpus.version
    assert v1 != v2


def test_retrievalentry_is_frozen_and_hashable(tiny_corpus: Path) -> None:
    from dataclasses import FrozenInstanceError

    corpus = DirectoryCorpus(tiny_corpus)
    entry = next(iter(corpus.entries()))
    assert isinstance(entry, RetrievalEntry)
    with pytest.raises(FrozenInstanceError):
        entry.path = "tampered"  # type: ignore[misc]
    # Related is tuple (hashable)
    hash((entry.path, entry.related))


def test_malformed_sidecar_json_is_tolerated(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _write(root / "a.md", "content")
    _write(root / "summaries.json", "{ not valid json")
    corpus = DirectoryCorpus(root, summaries_file="summaries.json")
    entries = list(corpus.entries())
    assert len(entries) == 1
    assert entries[0].summary is None


def test_deterministic_ordering(tiny_corpus: Path) -> None:
    corpus = DirectoryCorpus(tiny_corpus, cache=False)
    first = [e.path for e in corpus.entries()]
    second = [e.path for e in corpus.entries()]
    assert first == second
    assert first == sorted(first)
