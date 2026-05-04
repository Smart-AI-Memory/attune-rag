"""Tests for plan_rename + apply_rename (M1 task #6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus
from attune_rag.editor import (
    FileEdit,
    RenameCollisionError,
    apply_rename,
    plan_rename,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def trio(tmp_path: Path) -> Path:
    """Three-template corpus with `beta-spec` referenced from two templates."""
    root = tmp_path / "docs"
    _write(
        root / "alpha.md",
        (
            "---\n"
            "type: concept\n"
            "name: Alpha\n"
            "aliases: [a, beta-spec]\n"
            "---\n\n"
            "Alpha body.\n"
        ),
    )
    _write(
        root / "gamma.md",
        (
            "---\n"
            "type: concept\n"
            "name: Gamma\n"
            "tags: [api]\n"
            "---\n\n"
            "Gamma references [[beta-spec]] and [[beta-spec]] again.\n"
            "```\n"
            "Code: [[beta-spec]] (should NOT be rewritten)\n"
            "```\n"
        ),
    )
    _write(
        root / "delta.md",
        (
            "---\n"
            "type: concept\n"
            "name: Delta\n"
            "aliases:\n"
            "  - delta-spec\n"
            "---\n\n"
            "no refs here\n"
        ),
    )
    return root


# -- alias rename: end-to-end ---------------------------------------


def test_plan_alias_rename_finds_three_files(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "beta-spec", "beta", kind="alias")
    paths = {edit.path for edit in plan.edits}
    # alpha.md (declaration) + gamma.md (body refs); delta.md untouched.
    assert paths == {"alpha.md", "gamma.md"}


def test_apply_alias_rename_writes_disk_and_refreshes_indices(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    # Prime indices so we can verify they refresh.
    assert "beta-spec" in corpus.alias_index
    plan = plan_rename(corpus, "beta-spec", "beta", kind="alias")

    written = apply_rename(corpus, plan)
    assert set(written) == {"alpha.md", "gamma.md"}

    alpha_text = (trio / "alpha.md").read_text(encoding="utf-8")
    gamma_text = (trio / "gamma.md").read_text(encoding="utf-8")
    delta_text = (trio / "delta.md").read_text(encoding="utf-8")

    # Frontmatter declaration rewritten.
    assert "beta-spec" not in alpha_text
    assert "beta" in alpha_text
    # Body refs rewritten outside the fence; fenced ref preserved.
    assert gamma_text.count("[[beta]]") == 2
    assert "[[beta-spec]]" in gamma_text  # the fenced one survives
    # Untouched file unchanged.
    assert "delta-spec" in delta_text

    # Indices refreshed.
    refreshed_aliases = set(corpus.alias_index)
    assert "beta-spec" not in refreshed_aliases
    assert "beta" in refreshed_aliases


def test_alias_rename_collision_rejects(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    # `a` is already declared by alpha.md; renaming `delta-spec` -> `a`
    # must raise.
    with pytest.raises(RenameCollisionError) as exc:
        plan_rename(corpus, "delta-spec", "a", kind="alias")
    err = exc.value
    assert err.name == "a"
    assert err.owning_path == "alpha.md"


def test_no_op_rename_returns_empty_plan(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "ghost-name", "still-ghost", kind="alias")
    # No references → no edits, but a valid plan.
    assert plan.edits == []
    assert plan.kind == "alias"
    assert apply_rename(corpus, plan) == []


def test_rename_to_same_name_is_noop(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "beta-spec", "beta-spec", kind="alias")
    assert plan.edits == []


# -- tag rename ------------------------------------------------------


def test_plan_tag_rename_rewrites_frontmatter(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "api", "API", kind="tag")
    assert {e.path for e in plan.edits} == {"gamma.md"}

    apply_rename(corpus, plan)
    assert "tags: [API]" in (trio / "gamma.md").read_text(encoding="utf-8")


# -- safety / atomicity ----------------------------------------------


def test_apply_detects_drifted_base(trio: Path) -> None:
    """If the file changed on disk between plan and apply, refuse to write."""
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "beta-spec", "beta", kind="alias")

    # Drift: rewrite alpha.md externally between plan and apply.
    (trio / "alpha.md").write_text("totally different content\n", encoding="utf-8")

    from attune_rag.editor import RenameError

    with pytest.raises(RenameError):
        apply_rename(corpus, plan)
    # The other planned file should NOT have been written either,
    # because alpha.md was the first staged file (sorted load order)
    # and drift is detected during staging before any rename.
    gamma_text = (trio / "gamma.md").read_text(encoding="utf-8")
    assert "[[beta-spec]]" in gamma_text
    assert "[[beta]]" not in gamma_text


def test_template_path_rename_is_not_implemented(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    with pytest.raises(NotImplementedError):
        plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")


# -- diff structure --------------------------------------------------


def test_plan_includes_unified_diff_hunks(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "beta-spec", "beta", kind="alias")
    edit: FileEdit = next(e for e in plan.edits if e.path == "alpha.md")
    assert edit.hunks
    h = edit.hunks[0]
    assert h.header.startswith("@@")
    assert h.hunk_id  # non-empty stable id
    # Diff body has at least one '-' and one '+' line.
    has_minus = any(line.startswith("-") for line in h.lines)
    has_plus = any(line.startswith("+") for line in h.lines)
    assert has_minus and has_plus


def test_plan_to_dict_round_trip(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "beta-spec", "beta", kind="alias")
    dumped = plan.to_dict()
    assert dumped["old"] == "beta-spec"
    assert dumped["new"] == "beta"
    assert dumped["kind"] == "alias"
    assert isinstance(dumped["edits"], list)
    assert all("hunks" in e for e in dumped["edits"])
