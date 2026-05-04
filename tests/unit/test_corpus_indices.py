"""Tests for DirectoryCorpus path_index + alias_index (M1 task #2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus
from attune_rag.corpus import AliasInfo, DuplicateAliasError


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _template(name: str, *, type_: str = "concept", aliases: list[str] | None = None) -> str:
    fm = [f"type: {type_}", f"name: {name}"]
    if aliases:
        rendered = ", ".join(aliases)
        fm.append(f"aliases: [{rendered}]")
    return "---\n" + "\n".join(fm) + "\n---\n\n# " + name + "\n\nbody text\n"


@pytest.fixture
def aliased_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    _write(
        root / "concepts" / "alpha.md",
        _template("Alpha", aliases=["a", "alpha-spec"]),
    )
    _write(
        root / "concepts" / "beta.md",
        _template("Beta", aliases=["b"]),
    )
    _write(
        root / "tasks" / "build.md",
        _template("Build", type_="task"),  # no aliases
    )
    return root


def test_path_index_keys_match_entries(aliased_corpus: Path) -> None:
    corpus = DirectoryCorpus(aliased_corpus)
    keys = set(corpus.path_index)
    assert keys == {
        "concepts/alpha.md",
        "concepts/beta.md",
        "tasks/build.md",
    }


def test_path_index_returns_loaded_entries(aliased_corpus: Path) -> None:
    corpus = DirectoryCorpus(aliased_corpus)
    alpha = corpus.path_index["concepts/alpha.md"]
    assert alpha.path == "concepts/alpha.md"
    assert alpha.aliases == ("a", "alpha-spec")
    assert alpha.metadata["frontmatter"]["type"] == "concept"
    assert alpha.metadata["frontmatter"]["name"] == "Alpha"


def test_path_index_is_a_copy(aliased_corpus: Path) -> None:
    """Mutating the returned dict must not corrupt internal state."""
    corpus = DirectoryCorpus(aliased_corpus)
    snapshot = corpus.path_index
    snapshot.pop("concepts/alpha.md")
    # Internal state intact
    assert "concepts/alpha.md" in corpus.path_index


def test_alias_index_built_from_frontmatter(aliased_corpus: Path) -> None:
    corpus = DirectoryCorpus(aliased_corpus)
    idx = corpus.alias_index
    assert set(idx) == {"a", "alpha-spec", "b"}

    info: AliasInfo = idx["a"]
    assert info["alias"] == "a"
    assert info["template_path"] == "concepts/alpha.md"
    assert info["template_name"] == "Alpha"

    assert idx["b"]["template_path"] == "concepts/beta.md"
    assert idx["b"]["template_name"] == "Beta"


def test_alias_index_is_a_copy(aliased_corpus: Path) -> None:
    corpus = DirectoryCorpus(aliased_corpus)
    snapshot = corpus.alias_index
    snapshot.clear()
    assert corpus.alias_index  # rebuilt from cache; not empty


def test_template_without_aliases_contributes_nothing(aliased_corpus: Path) -> None:
    corpus = DirectoryCorpus(aliased_corpus)
    # tasks/build.md declares no aliases — confirm none of its data
    # leaked into alias_index.
    paths = {info["template_path"] for info in corpus.alias_index.values()}
    assert "tasks/build.md" not in paths


def test_template_name_falls_back_to_path_stem(tmp_path: Path) -> None:
    """If frontmatter has no `name`, alias_index uses the file stem."""
    root = tmp_path / "docs"
    _write(
        root / "stub.md",
        "---\ntype: concept\naliases: [stub-alias]\n---\nbody\n",
    )
    corpus = DirectoryCorpus(root)
    assert corpus.alias_index["stub-alias"]["template_name"] == "stub"


def test_duplicate_alias_raises_with_both_paths(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _write(root / "first.md", _template("First", aliases=["shared"]))
    _write(root / "second.md", _template("Second", aliases=["shared"]))
    corpus = DirectoryCorpus(root)
    with pytest.raises(DuplicateAliasError) as exc_info:
        _ = corpus.alias_index
    err = exc_info.value
    assert err.alias == "shared"
    # Sort order is stable — first.md is loaded before second.md (sorted glob).
    assert err.first_path == "first.md"
    assert err.second_path == "second.md"
    # Message should mention both paths so the editor can show the conflict.
    assert "first.md" in str(err)
    assert "second.md" in str(err)


def test_malformed_frontmatter_is_tolerated(tmp_path: Path) -> None:
    """A single broken template must not break the whole corpus load."""
    root = tmp_path / "docs"
    _write(root / "good.md", _template("Good", aliases=["good"]))
    _write(
        root / "bad.md",
        "---\ntype: concept\nname: [unterminated\n---\nbody\n",
    )
    corpus = DirectoryCorpus(root)
    # Both files are loaded; bad.md has no aliases (parse failure → empty).
    assert set(corpus.path_index) == {"good.md", "bad.md"}
    assert "good" in corpus.alias_index
    assert corpus.path_index["bad.md"].aliases == ()
