"""Tests for DirectoryCorpus(extra_aliases_file=...) and load_aliases_from_file().

Covers M2.3 of docs/specs/user-corpus-onboarding/tasks.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attune_rag.corpus import DirectoryCorpus, load_aliases_from_file


def _write_corpus(tmp_path: Path) -> Path:
    """Build a tiny 2-doc corpus."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(
        "---\naliases: [alpha-feature]\n---\n# Alpha\n", encoding="utf-8"
    )
    (root / "beta.md").write_text("---\naliases: [beta-feature]\n---\n# Beta\n", encoding="utf-8")
    return root


def test_load_aliases_from_file_basic(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps({"alpha.md": ["a one", "a two"], "beta.md": ["b one"]}),
        encoding="utf-8",
    )
    result = load_aliases_from_file(path)
    assert result == {"alpha.md": ["a one", "a two"], "beta.md": ["b one"]}


def test_load_aliases_from_file_drops_underscore_keys(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "_comment": "ignored",
                "_version": "ignored too",
                "alpha.md": ["a one"],
            }
        ),
        encoding="utf-8",
    )
    result = load_aliases_from_file(path)
    assert result == {"alpha.md": ["a one"]}


def test_load_aliases_from_file_missing_file_raises_with_path(tmp_path: Path) -> None:
    bad = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError) as exc:
        load_aliases_from_file(bad)
    assert str(bad) in str(exc.value)


def test_load_aliases_from_file_malformed_json_raises_with_path(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_aliases_from_file(path)
    assert "malformed JSON" in str(exc.value)
    assert str(path) in str(exc.value)


def test_load_aliases_from_file_top_level_not_dict_raises(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_aliases_from_file(path)
    assert "top-level JSON must be an object" in str(exc.value)


def test_load_aliases_from_file_non_list_value_raises(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"alpha.md": "not a list"}), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_aliases_from_file(path)
    assert "alpha.md" in str(exc.value)
    assert "must be a list" in str(exc.value)


def test_load_aliases_from_file_non_string_alias_raises(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"alpha.md": ["ok", 42]}), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_aliases_from_file(path)
    assert "must be a string" in str(exc.value)


def test_directory_corpus_loads_extra_aliases_file(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path)
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"alpha.md": ["alpha override one"]}), encoding="utf-8")
    corpus = DirectoryCorpus(root=root, extra_aliases_file=aliases)
    entries = {e.path: e for e in corpus.entries()}
    assert "alpha override one" in entries["alpha.md"].aliases
    # Frontmatter alias still present.
    assert "alpha-feature" in entries["alpha.md"].aliases


def test_directory_corpus_inline_wins_on_collision(tmp_path: Path) -> None:
    """When both extra_aliases (inline) and extra_aliases_file carry the
    same path, the inline dict's values replace the file's for that path."""
    root = _write_corpus(tmp_path)
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps({"alpha.md": ["from file"], "beta.md": ["file b"]}),
        encoding="utf-8",
    )
    corpus = DirectoryCorpus(
        root=root,
        extra_aliases={"alpha.md": ["from inline"]},
        extra_aliases_file=aliases,
    )
    entries = {e.path: e for e in corpus.entries()}
    # alpha.md: inline wins
    assert "from inline" in entries["alpha.md"].aliases
    assert "from file" not in entries["alpha.md"].aliases
    # beta.md: file-only path still applies
    assert "file b" in entries["beta.md"].aliases


def test_directory_corpus_missing_file_propagates(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path)
    bad = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        DirectoryCorpus(root=root, extra_aliases_file=bad)


def test_directory_corpus_malformed_file_propagates(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        DirectoryCorpus(root=root, extra_aliases_file=bad)
