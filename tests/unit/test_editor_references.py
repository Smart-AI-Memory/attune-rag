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


# -- W3.3 coverage gap (from W2.1 deep-review) ----------------------


def test_find_references_skips_aliases_inside_fenced_code_blocks(tmp_path: Path) -> None:
    """``[[alias]]`` references that appear ONLY inside fenced code blocks
    must not be reported as body refs.

    Covers ``references.py:101-107`` — the ``in_fence`` toggle and the
    ``if in_fence: continue`` skip in ``_alias_body_refs``. The
    existing exclusion test interleaves fenced and prose refs in the
    same fixture; this one isolates the fenced-only case so the
    exclusion is the sole reason the count is zero, and exercises
    both fence flavors (``\\`\\`\\``` and ``~~~``).
    """
    root = tmp_path / "docs"
    root.mkdir(parents=True)
    # Two fenced blocks, one backtick and one tilde, both containing
    # the same alias reference. No prose references anywhere.
    (root / "alpha.md").write_text(
        "---\n"
        "type: concept\n"
        "name: Alpha\n"
        "aliases: [a]\n"
        "---\n\n"
        "# Body\n\n"
        "```\n"
        "Backtick-fenced: [[beta-spec]] (must be skipped)\n"
        "Still fenced:    [[beta-spec]] (must be skipped)\n"
        "```\n\n"
        "Plain prose with no refs at all.\n\n"
        "~~~\n"
        "Tilde-fenced: [[beta-spec]] (must be skipped)\n"
        "~~~\n",
        encoding="utf-8",
    )
    corpus = DirectoryCorpus(root)
    refs = find_references(corpus, "beta-spec", kind="alias")
    body_refs = [r for r in refs if r.context == "body"]
    assert body_refs == [], f"expected zero body refs, got {body_refs!r}"


# -- W3.3 corpus-shape / entry-shape edge cases ---------------------
#
# These cover branches that DirectoryCorpus doesn't naturally exercise:
# entries returned via the `entries()` callable fallback, entries with
# empty content/path, docs with no frontmatter at all, and block-style
# alias scans terminated by blank lines or dedents. Each test uses a
# tiny duck-typed corpus or a hand-crafted markdown fixture rather than
# DirectoryCorpus, so the failure surface stays scoped to the code path
# under test.


class _Entry:
    """Tiny duck-typed entry for the `corpus.entries()` shape."""

    def __init__(self, content: str = "", path: str = "", related: tuple[str, ...] = ()):
        self.content = content
        self.path = path
        self.related = related


class _EntriesCallableCorpus:
    """Duck-typed corpus that exposes `entries()` (not `path_index`)."""

    def __init__(self, entries: list[_Entry]) -> None:
        self._entries = entries

    def entries(self) -> list[_Entry]:
        return self._entries


class _NoEntryProtocolCorpus:
    """Corpus with neither `path_index` nor `entries` — `_iter_entries`
    falls through to an empty iterator, find_references returns []."""


def test_iter_entries_via_entries_callable_fallback(tmp_path: Path) -> None:
    """`_iter_entries` prefers `path_index` but falls back to `entries()`
    when the dict-shape attr is absent. Covers references.py:167-169."""
    entries = [
        _Entry(
            content=("---\n" "aliases: [foo]\n" "---\n\n" "Body has [[bar]] reference.\n"),
            path="x.md",
        ),
    ]
    corpus = _EntriesCallableCorpus(entries)
    refs = find_references(corpus, "bar", kind="alias")
    assert any(r.context == "body" for r in refs)


def test_iter_entries_empty_fallback_when_neither_attr_present() -> None:
    """No `path_index` and no `entries` → `iter(())`. Covers L170."""
    corpus = _NoEntryProtocolCorpus()
    assert find_references(corpus, "anything", kind="alias") == []
    assert find_references(corpus, "anything", kind="tag") == []
    assert find_references(corpus, "anything", kind="template_path") == []


def test_alias_refs_skip_entry_with_empty_content() -> None:
    """An entry with empty `content` is skipped — covers L76."""
    corpus = _EntriesCallableCorpus([_Entry(content="", path="empty.md")])
    assert find_references(corpus, "x", kind="alias") == []


def test_alias_refs_skip_entry_with_empty_path() -> None:
    """An entry with empty `path` is skipped — same L76 branch, other axis."""
    corpus = _EntriesCallableCorpus([_Entry(content="some markdown body", path="")])
    assert find_references(corpus, "x", kind="alias") == []


def test_alias_decl_returns_empty_on_doc_with_no_frontmatter() -> None:
    """A doc without a `---\\n...\\n---` block produces no alias-decl
    refs. Covers references.py:86 (early return) and references.py:184
    (body_start_line falls through to 1)."""
    entries = [
        _Entry(
            content="Just prose, no frontmatter. Mentions [[target]] in body.\n",
            path="bare.md",
        )
    ]
    corpus = _EntriesCallableCorpus(entries)
    refs = find_references(corpus, "target", kind="alias")
    # No frontmatter decls, but body refs still found from line 1.
    assert all(r.context != "frontmatter.alias" for r in refs)
    body = [r for r in refs if r.context == "body"]
    assert len(body) == 1
    assert body[0].line == 1


def test_tag_refs_skip_entry_with_no_content_or_no_frontmatter() -> None:
    """Tag scan covers two early-skip branches:
    - L129: empty content/path → continue
    - L132: present content but no frontmatter → continue"""
    entries = [
        _Entry(content="", path="empty.md"),
        _Entry(content="body text only, no fm", path="no_fm.md"),
    ]
    corpus = _EntriesCallableCorpus(entries)
    assert find_references(corpus, "security", kind="tag") == []


def test_block_style_alias_blank_line_continues_then_dedent_terminates() -> None:
    """Block-style aliases interleaved with a blank line keep scanning;
    a dedent at the same or lower indent than the key ends the block.
    Covers references.py:221 (blank-line continue) and L223 (dedent break)."""
    entries = [
        _Entry(
            content=(
                "---\n"
                "name: Alpha\n"
                "aliases:\n"
                "  - first-alias\n"
                "\n"  # blank line inside the block → L221 continue
                "  - target-alias\n"
                "tags:\n"  # dedent back to top-level key → L223 break
                "  - sec\n"
                "---\n\n"
                "Body.\n"
            ),
            path="alpha.md",
        )
    ]
    corpus = _EntriesCallableCorpus(entries)
    decl = [
        r
        for r in find_references(corpus, "target-alias", kind="alias")
        if r.context == "frontmatter.alias"
    ]
    assert len(decl) == 1


def test_template_path_refs_skip_entries_without_path() -> None:
    """`_find_path_refs` requires both `related` membership AND a
    truthy path; an entry with an empty path is skipped silently."""
    entries = [_Entry(content="x", path="", related=("target.md",))]
    corpus = _EntriesCallableCorpus(entries)
    assert find_references(corpus, "target.md", kind="template_path") == []
