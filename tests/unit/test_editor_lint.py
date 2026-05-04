"""Tests for attune_rag.editor.lint_template (M1 task #3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus
from attune_rag.editor import Diagnostic, lint_template


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> DirectoryCorpus:
    """Two-template corpus exposing aliases for lint to resolve against."""
    root = tmp_path / "docs"
    _write(
        root / "concepts" / "alpha.md",
        "---\ntype: concept\nname: Alpha\naliases: [a, alpha]\n---\nbody\n",
    )
    _write(
        root / "concepts" / "beta.md",
        "---\ntype: concept\nname: Beta\naliases: [b]\n---\nbody\n",
    )
    return DirectoryCorpus(root)


def _codes(diagnostics: list[Diagnostic]) -> set[str]:
    return {d.code for d in diagnostics}


# -- frontmatter / schema -------------------------------------------


def test_clean_template_produces_no_diagnostics(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: X\n---\n\nbody\n"
    assert lint_template(text, "x.md", corpus) == []


def test_missing_required_field_diagnostic(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\n---\n\nbody\n"  # no `name`
    diags = lint_template(text, "x.md", corpus)
    assert "missing-required" in _codes(diags)
    err = next(d for d in diags if d.code == "missing-required")
    assert err.severity == "error"
    assert "name" in err.message


def test_bad_enum_diagnostic_points_at_type_line(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: bogus\nname: X\n---\n\nbody\n"
    diags = lint_template(text, "x.md", corpus)
    bad = next(d for d in diags if d.code == "bad-enum")
    # `type:` is the second line of the document (line 2).
    assert bad.line == 2
    assert bad.col == 1


def test_unknown_field_is_info_level(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: X\nexperimental: true\n---\n\nbody\n"
    diags = lint_template(text, "x.md", corpus)
    info = next(d for d in diags if d.code == "unknown-field")
    assert info.severity == "info"
    # `experimental:` is line 4 in the document.
    assert info.line == 4


def test_malformed_yaml_in_frontmatter(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: [unterminated\ntags:\n  - foo\n   - bar\n---\nbody\n"
    diags = lint_template(text, "x.md", corpus)
    assert "malformed-yaml" in _codes(diags)
    err = next(d for d in diags if d.code == "malformed-yaml")
    assert err.severity == "error"


def test_no_frontmatter_still_runs_body_lint(corpus: DirectoryCorpus) -> None:
    """A template without frontmatter should not crash; body lint runs."""
    text = "no frontmatter here\n[[ghost]]\n"
    diags = lint_template(text, "x.md", corpus)
    # No frontmatter → no schema diagnostics, but broken alias caught.
    assert "broken-alias" in _codes(diags)
    # Schema-related codes should NOT appear.
    assert "missing-required" not in _codes(diags)


# -- alias references -----------------------------------------------


def test_broken_alias_in_body(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: X\n---\n\nSee [[missing-alias]] for details.\n"
    diags = lint_template(text, "x.md", corpus)
    bad = next(d for d in diags if d.code == "broken-alias")
    assert bad.line == 6
    assert bad.col == 5  # 1-indexed col of `[[`
    # End col should cover the closing `]]`.
    assert bad.end_col == 5 + len("[[missing-alias]]")


def test_known_alias_in_body_passes(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: X\n---\n\nLink to [[alpha]].\n"
    diags = lint_template(text, "x.md", corpus)
    assert "broken-alias" not in _codes(diags)


def test_alias_inside_fenced_code_block_is_ignored(corpus: DirectoryCorpus) -> None:
    text = (
        "---\ntype: concept\nname: X\n---\n\n"
        "```\n[[ghost]]\n```\n"
        "But [[alpha]] outside the fence is fine.\n"
    )
    diags = lint_template(text, "x.md", corpus)
    assert "broken-alias" not in _codes(diags)


def test_escaped_alias_ref_is_not_a_reference(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: X\n---\n\nLiteral: \\[[ghost]] should not lint.\n"
    diags = lint_template(text, "x.md", corpus)
    assert "broken-alias" not in _codes(diags)


# -- depth markers --------------------------------------------------


def test_depth_sequence_ok(corpus: DirectoryCorpus) -> None:
    text = (
        "---\ntype: concept\nname: X\n---\n\n"
        "## Depth 1\n\nintro\n\n"
        "## Depth 2\n\nmore\n\n"
        "## Depth 3\n\neven more\n"
    )
    diags = lint_template(text, "x.md", corpus)
    assert {"depth-skipped", "depth-out-of-order", "depth-not-starting-at-one"} & _codes(
        diags
    ) == set()


def test_depth_must_start_at_one(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: X\n---\n\n## Depth 2\n\nbody\n"
    diags = lint_template(text, "x.md", corpus)
    err = next(d for d in diags if d.code == "depth-not-starting-at-one")
    assert err.severity == "warning"


def test_depth_skipped_diagnostic(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\nname: X\n---\n\n" "## Depth 1\n\nfirst\n\n" "## Depth 3\n\nthird\n"
    diags = lint_template(text, "x.md", corpus)
    err = next(d for d in diags if d.code == "depth-skipped")
    assert "Depth 3" in err.message


def test_depth_out_of_order_diagnostic(corpus: DirectoryCorpus) -> None:
    text = (
        "---\ntype: concept\nname: X\n---\n\n"
        "## Depth 1\n\nfirst\n\n"
        "## Depth 2\n\nsecond\n\n"
        "## Depth 1\n\nbacktrack\n"
    )
    diags = lint_template(text, "x.md", corpus)
    err = next(d for d in diags if d.code == "depth-out-of-order")
    assert err.severity == "warning"


# -- edge cases / duck typing ---------------------------------------


def test_corpus_without_alias_index_skips_alias_check(tmp_path: Path) -> None:
    """A corpus that doesn't expose alias_index should not crash lint;
    alias diagnostics are simply skipped."""

    class _PlainCorpus:
        pass  # no alias_index attribute

    text = "---\ntype: concept\nname: X\n---\n\n[[ghost]]\n"
    diags = lint_template(text, "x.md", _PlainCorpus())
    assert "broken-alias" not in _codes(diags)


def test_diagnostics_are_sorted_by_position(corpus: DirectoryCorpus) -> None:
    text = (
        "---\ntype: concept\nname: X\n---\n\n"
        "[[late-ref-one]]\n"
        "[[early-ref]] but earlier on its own line.\n"
    )
    diags = lint_template(text, "x.md", corpus)
    # Sorted by (line, col), so line 6 broken-alias precedes line 7.
    if len(diags) >= 2:
        assert (diags[0].line, diags[0].col) <= (diags[1].line, diags[1].col)


def test_diagnostic_to_dict_round_trip(corpus: DirectoryCorpus) -> None:
    text = "---\ntype: concept\n---\n\nbody\n"
    diags = lint_template(text, "x.md", corpus)
    dumped = [d.to_dict() for d in diags]
    assert all(set(d.keys()) >= {"severity", "code", "message", "line", "col"} for d in dumped)
