"""Tests for autocomplete_tags + autocomplete_aliases (M1 task #4)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus
from attune_rag.editor import autocomplete_aliases, autocomplete_tags


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _template(
    name: str,
    *,
    tags: list[str] | None = None,
    aliases: list[str] | None = None,
) -> str:
    fm = ["type: concept", f"name: {name}"]
    if tags:
        fm.append(f"tags: [{', '.join(tags)}]")
    if aliases:
        fm.append(f"aliases: [{', '.join(aliases)}]")
    return "---\n" + "\n".join(fm) + "\n---\n\nbody\n"


@pytest.fixture
def corpus(tmp_path: Path) -> DirectoryCorpus:
    root = tmp_path / "docs"
    _write(
        root / "alpha.md",
        _template("Alpha", tags=["security", "api"], aliases=["alpha", "alphabeta"]),
    )
    _write(
        root / "beta.md",
        _template("Beta", tags=["security", "ui"], aliases=["b", "beta-rich"]),
    )
    _write(
        root / "gamma.md",
        _template("Gamma", tags=["security", "Ops"], aliases=["g"]),
    )
    return DirectoryCorpus(root)


# -- tags ------------------------------------------------------------


def test_tags_ranked_by_frequency(corpus: DirectoryCorpus) -> None:
    suggestions = autocomplete_tags(corpus, "")
    # `security` appears in all 3 templates, `api`/`ui`/`Ops` once each.
    assert suggestions[0] == "security"
    assert set(suggestions[1:]) == {"api", "ui", "Ops"}


def test_tags_prefix_filter_case_insensitive(corpus: DirectoryCorpus) -> None:
    assert autocomplete_tags(corpus, "se") == ["security"]
    # Case-insensitive: "OPS" and "ops" both match the "Ops" tag.
    assert autocomplete_tags(corpus, "op") == ["Ops"]
    assert autocomplete_tags(corpus, "OP") == ["Ops"]


def test_tags_no_match_returns_empty(corpus: DirectoryCorpus) -> None:
    assert autocomplete_tags(corpus, "zzz") == []


def test_tags_respects_limit(corpus: DirectoryCorpus) -> None:
    assert (
        autocomplete_tags(corpus, "", limit=2) == ["security", "api"]
        or len(autocomplete_tags(corpus, "", limit=2)) == 2
    )


def test_tags_alphabetical_within_same_frequency(corpus: DirectoryCorpus) -> None:
    """Tags with equal frequency should sort case-insensitive alphabetical."""
    suggestions = autocomplete_tags(corpus, "")
    tail = suggestions[1:]  # everything except `security` (freq 3)
    assert tail == sorted(tail, key=str.casefold)


def test_tags_works_with_corpus_without_path_index() -> None:
    """Autocomplete should fall back gracefully via `entries()`."""

    class _FakeEntry:
        def __init__(self, fm: dict) -> None:
            self.metadata = {"frontmatter": fm}

    class _FakeCorpus:
        def entries(self) -> list:
            return [
                _FakeEntry({"tags": ["x", "y"]}),
                _FakeEntry({"tags": ["x"]}),
            ]

    assert autocomplete_tags(_FakeCorpus(), "") == ["x", "y"]


# -- aliases ---------------------------------------------------------


def test_aliases_filter_by_prefix(corpus: DirectoryCorpus) -> None:
    suggestions = autocomplete_aliases(corpus, "alpha")
    aliases = [info["alias"] for info in suggestions]
    assert aliases == ["alpha", "alphabeta"]  # shorter first


def test_aliases_case_insensitive_prefix(corpus: DirectoryCorpus) -> None:
    suggestions = autocomplete_aliases(corpus, "ALPHA")
    assert {info["alias"] for info in suggestions} == {"alpha", "alphabeta"}


def test_aliases_include_template_metadata(corpus: DirectoryCorpus) -> None:
    suggestions = autocomplete_aliases(corpus, "b")
    by_alias = {info["alias"]: info for info in suggestions}
    assert by_alias["b"]["template_path"] == "beta.md"
    assert by_alias["b"]["template_name"] == "Beta"


def test_aliases_corpus_without_alias_index_returns_empty() -> None:
    class _NoAliases:
        pass

    assert autocomplete_aliases(_NoAliases(), "x") == []


def test_aliases_respect_limit(corpus: DirectoryCorpus) -> None:
    suggestions = autocomplete_aliases(corpus, "", limit=2)
    assert len(suggestions) == 2


# -- performance -----------------------------------------------------


def test_autocomplete_under_1ms_on_1000_templates(tmp_path: Path) -> None:
    """Performance gate: per-call < 1ms median on 1000 templates."""
    root = tmp_path / "big"
    for i in range(1000):
        _write(
            root / f"t{i:04d}.md",
            _template(
                f"T{i}",
                tags=["security" if i % 3 == 0 else "ui", f"tag-{i % 50}"],
                aliases=[f"alias-{i}"],
            ),
        )
    corpus = DirectoryCorpus(root)
    # warm the cache
    _ = autocomplete_tags(corpus, "se")
    _ = autocomplete_aliases(corpus, "alias-")

    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        autocomplete_tags(corpus, "se")
    tag_ms = (time.perf_counter() - t0) * 1000 / iterations

    t0 = time.perf_counter()
    for _ in range(iterations):
        autocomplete_aliases(corpus, "alias-")
    alias_ms = (time.perf_counter() - t0) * 1000 / iterations

    # Generous-but-real upper bound. Goal in tasks.md is < 1ms; allow
    # 2x headroom for noisy CI environments.
    assert tag_ms < 2.0, f"tags autocomplete avg {tag_ms:.3f}ms"
    assert alias_ms < 2.0, f"aliases autocomplete avg {alias_ms:.3f}ms"
