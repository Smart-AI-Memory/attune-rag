"""Tests for find_references (M1 task #5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus
from attune_rag.editor import find_references


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> DirectoryCorpus:
    """Three-template corpus with aliases, tags, and cross-links."""
    root = tmp_path / "docs"
    _write(
        root / "alpha.md",
        (
            "---\n"
            "type: concept\n"
            "name: Alpha\n"
            "aliases: [a, alpha-spec]\n"
            "tags: [security, api]\n"
            "---\n\n"
            "# Alpha\n\n"
            "See also [[beta-spec]] for the partner doc.\n"
            "And again [[beta-spec]] in another sentence.\n"
            "```\n"
            "Code: [[beta-spec]] (should be ignored)\n"
            "```\n"
            "Escape: \\[[beta-spec]] (should be ignored).\n"
        ),
    )
    _write(
        root / "beta.md",
        (
            "---\n"
            "type: concept\n"
            "name: Beta\n"
            "aliases:\n"
            "  - beta-spec\n"
            "  - b\n"
            "tags: [security]\n"
            "---\n\n"
            "Refers to [[a]] for context.\n"
        ),
    )
    _write(
        root / "gamma.md",
        (
            "---\n"
            "type: concept\n"
            "name: Gamma\n"
            "tags: [api, ops]\n"
            "---\n\n"
            "no body refs here\n"
        ),
    )
    _write(
        root / "cross_links.json",
        json.dumps({"alpha.md": ["beta.md"], "gamma.md": ["alpha.md"]}),
    )
    return DirectoryCorpus(root, cross_links_file="cross_links.json")


# -- alias references ------------------------------------------------


def test_alias_body_refs_collected(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "beta-spec", kind="alias")
    body_refs = [r for r in refs if r.context == "body"]
    # alpha.md mentions [[beta-spec]] twice in body (excluding fenced + escaped).
    assert all(r.template_path == "alpha.md" for r in body_refs)
    assert len(body_refs) == 2
    assert body_refs[0].line < body_refs[1].line


def test_alias_body_refs_exclude_fenced_and_escaped(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "beta-spec", kind="alias")
    body_refs = [r for r in refs if r.context == "body" and r.template_path == "alpha.md"]
    # Exactly 2 body refs (the two prose mentions); fenced + escaped excluded.
    assert len(body_refs) == 2


def test_alias_frontmatter_decl_flow_style(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "alpha-spec", kind="alias")
    decl = [r for r in refs if r.context == "frontmatter.alias"]
    assert len(decl) == 1
    assert decl[0].template_path == "alpha.md"


def test_alias_frontmatter_decl_block_style(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "beta-spec", kind="alias")
    decl = [r for r in refs if r.context == "frontmatter.alias"]
    assert len(decl) == 1
    assert decl[0].template_path == "beta.md"
    # Block-style: `  - beta-spec` is on its own line; column points
    # past `- `.
    assert decl[0].col >= 4


def test_alias_combines_decls_and_body_refs(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "a", kind="alias")
    by_context = {r.context for r in refs}
    # `a` is declared in alpha.md frontmatter and referenced in beta.md body.
    assert "frontmatter.alias" in by_context
    assert "body" in by_context


# -- tag references --------------------------------------------------


def test_tag_refs_flow_style(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "api", kind="tag")
    paths = {r.template_path for r in refs}
    assert paths == {"alpha.md", "gamma.md"}
    assert all(r.context == "frontmatter.tag" for r in refs)


def test_tag_refs_count(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "security", kind="tag")
    # alpha.md and beta.md both tag `security`.
    assert {r.template_path for r in refs} == {"alpha.md", "beta.md"}


def test_tag_no_match(corpus: DirectoryCorpus) -> None:
    assert find_references(corpus, "ghost-tag", kind="tag") == []


# -- template_path references ---------------------------------------


def test_template_path_via_cross_links(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "beta.md", kind="template_path")
    # alpha.md cross-links to beta.md.
    assert len(refs) == 1
    ref = refs[0]
    assert ref.template_path == "alpha.md"
    assert ref.context == "cross_links"


def test_template_path_no_match(corpus: DirectoryCorpus) -> None:
    assert find_references(corpus, "nonexistent.md", kind="template_path") == []


# -- error handling --------------------------------------------------


def test_unsupported_kind_raises(corpus: DirectoryCorpus) -> None:
    with pytest.raises(ValueError, match="Unsupported reference kind"):
        find_references(corpus, "x", kind="bogus")  # type: ignore[arg-type]


def test_to_dict_round_trip(corpus: DirectoryCorpus) -> None:
    refs = find_references(corpus, "api", kind="tag")
    dumped = [r.to_dict() for r in refs]
    assert all(set(d) == {"template_path", "line", "col", "context"} for d in dumped)
