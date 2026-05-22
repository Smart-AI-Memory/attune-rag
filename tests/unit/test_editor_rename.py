"""Tests for plan_rename + apply_rename (M1 task #6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus
from attune_rag.editor import (
    FileEdit,
    FileMove,
    RenameCollisionError,
    RenameError,
    RenamePlan,
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

    with pytest.raises(RenameError):
        apply_rename(corpus, plan)
    # The other planned file should NOT have been written either,
    # because alpha.md was the first staged file (sorted load order)
    # and drift is detected during staging before any rename.
    gamma_text = (trio / "gamma.md").read_text(encoding="utf-8")
    assert "[[beta-spec]]" in gamma_text
    assert "[[beta]]" not in gamma_text


# -- template_path rename -------------------------------------------


def test_template_path_rename_moves_file_and_emits_move(trio: Path) -> None:
    """Plan returns one FileMove; apply replaces the file at the new path."""
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")
    assert plan.kind == "template_path"
    assert len(plan.moves) == 1
    move = plan.moves[0]
    assert move.old_path == "alpha.md"
    assert move.new_path == "alpha-renamed.md"
    assert plan.edits == []  # no sidecar in this fixture

    written = apply_rename(corpus, plan)
    assert "alpha-renamed.md" in written
    assert (trio / "alpha-renamed.md").is_file()
    assert not (trio / "alpha.md").exists()

    # Corpus indexes reflect the new path.
    paths = set(corpus.path_index)
    assert "alpha-renamed.md" in paths
    assert "alpha.md" not in paths
    # The alias still belongs to the renamed template.
    assert corpus.alias_index["a"]["template_path"] == "alpha-renamed.md"


def test_template_path_rename_into_new_subdir(trio: Path) -> None:
    """Apply creates missing parent directories of the target path."""
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "alpha.md", "concepts/alpha.md", kind="template_path")
    apply_rename(corpus, plan)
    assert (trio / "concepts" / "alpha.md").is_file()
    assert not (trio / "alpha.md").exists()


def test_template_path_rename_rejects_existing_target(trio: Path) -> None:
    """Collide with an existing file at the new path."""
    corpus = DirectoryCorpus(trio)
    with pytest.raises(RenameCollisionError) as exc:
        plan_rename(corpus, "alpha.md", "gamma.md", kind="template_path")
    err = exc.value
    assert err.name == "gamma.md"
    assert err.owning_path == "gamma.md"


def test_template_path_rename_to_same_path_is_noop(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "alpha.md", "alpha.md", kind="template_path")
    assert plan.moves == []
    assert plan.edits == []
    assert apply_rename(corpus, plan) == []


def test_template_path_rename_rejects_escape(trio: Path) -> None:
    """Reject ``..`` walks that point outside the corpus root."""
    corpus = DirectoryCorpus(trio)
    with pytest.raises(ValueError, match="escapes corpus root"):
        plan_rename(corpus, "alpha.md", "../escapee.md", kind="template_path")


def test_template_path_rename_rejects_missing_source(trio: Path) -> None:
    corpus = DirectoryCorpus(trio)

    with pytest.raises(RenameError, match="Source template does not exist"):
        plan_rename(corpus, "no-such.md", "elsewhere.md", kind="template_path")


def test_template_path_rename_updates_summaries_sidecar(trio: Path) -> None:
    """``summaries.json`` entry gets its key renamed; other keys preserved."""
    import json as _json

    sidecar = trio / "summaries.json"
    sidecar.write_text(
        _json.dumps(
            {
                "alpha.md": "Alpha summary.",
                "gamma.md": "Gamma summary.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    corpus = DirectoryCorpus(trio, summaries_file="summaries.json")
    plan = plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")
    sidecar_edit = next(e for e in plan.edits if e.path == "summaries.json")
    assert "alpha-renamed.md" in sidecar_edit.new_text
    assert "alpha.md" not in sidecar_edit.new_text.split("alpha-renamed.md")[0]
    assert "Gamma summary." in sidecar_edit.new_text  # untouched

    apply_rename(corpus, plan)
    on_disk = _json.loads((trio / "summaries.json").read_text(encoding="utf-8"))
    assert "alpha-renamed.md" in on_disk
    assert "alpha.md" not in on_disk
    assert on_disk["gamma.md"] == "Gamma summary."


def test_template_path_rename_no_sidecar_is_not_an_error(trio: Path) -> None:
    """Missing summaries.json is normal; planning still succeeds."""
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")
    assert plan.edits == []  # no sidecar to update
    apply_rename(corpus, plan)
    assert (trio / "alpha-renamed.md").is_file()


def test_template_path_rename_rolls_back_when_target_appears_late(
    trio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the target appears between plan and apply, raise and leave source intact.

    Simulates a TOCTOU window: planning sees the target as free, then
    another process creates the file before apply. apply_rename's
    re-check should catch it and not move the source.
    """
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")

    # Drift: someone else writes the target between plan and apply.
    (trio / "alpha-renamed.md").write_text("intruder", encoding="utf-8")

    with pytest.raises(RenameCollisionError):
        apply_rename(corpus, plan)
    # Source preserved.
    assert (trio / "alpha.md").is_file()
    # Intruder preserved.
    assert (trio / "alpha-renamed.md").read_text(encoding="utf-8") == "intruder"


def test_template_path_rename_rolls_back_move_when_edit_drifts(trio: Path) -> None:
    """If a sidecar edit's drift check fails post-move, the move is reversed."""
    import json as _json

    sidecar = trio / "summaries.json"
    sidecar.write_text(
        _json.dumps({"alpha.md": "Alpha summary."}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    corpus = DirectoryCorpus(trio, summaries_file="summaries.json")
    plan = plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")

    # Drift the sidecar contents between plan and apply.
    sidecar.write_text(
        _json.dumps({"alpha.md": "Tampered summary."}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RenameError, match="drifted"):
        apply_rename(corpus, plan)
    # Move reversed.
    assert (trio / "alpha.md").is_file()
    assert not (trio / "alpha-renamed.md").exists()
    # Sidecar untouched (still the tampered value).
    assert "Tampered" in (trio / "summaries.json").read_text(encoding="utf-8")


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


# -- W3.3 coverage gaps (from W2.1 deep-review) ---------------------


def test_apply_rolls_back_edits_when_commit_replace_fails(
    trio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-3 commit (``os.replace``) failure mid-loop restores prior edits.

    Covers ``rename.py:498-512``: when the sequential rename of a
    staged tempfile fails after at least one earlier rename has
    already succeeded, the earlier target must be rewritten with its
    pre-apply text and any remaining staged tempfiles must be
    unlinked. Re-raises the original ``OSError``.
    """
    import attune_rag.editor.rename as rmod

    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "beta-spec", "beta", kind="alias")
    # The fixture is constructed so two files participate (alpha.md +
    # gamma.md). The mid-flight rollback path only fires when the
    # failing index is > 0, so the test's whole premise depends on this.
    assert len(plan.edits) == 2

    originals = {e.path: (trio / e.path).read_text(encoding="utf-8") for e in plan.edits}

    real_replace = rmod.os.replace
    calls: list[tuple[str, str]] = []

    def failing_replace(src, dst):
        calls.append((str(src), str(dst)))
        if len(calls) >= 2:
            raise OSError("simulated disk fault on second commit")
        return real_replace(src, dst)

    monkeypatch.setattr(rmod.os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated disk fault"):
        apply_rename(corpus, plan)

    for path, original_text in originals.items():
        assert (trio / path).read_text(
            encoding="utf-8"
        ) == original_text, f"{path} was not restored to its pre-apply state after rollback"
    # No staging tempfiles left behind in the corpus root.
    leftovers = [p for p in trio.iterdir() if p.name.startswith(".") and p.suffix == ".tmp"]
    assert leftovers == [], f"leftover tempfiles: {leftovers}"


def test_rewrite_yaml_block_value_handles_quoted_list_items() -> None:
    """Block-style list items in single/double quotes are detected and rewritten.

    Covers the ``token.strip("'\\"")`` branch in
    ``_rewrite_yaml_block_value`` (``rename.py:323-329``). The current
    contract drops the surrounding quotes in the rewritten output —
    pinned here so a future refactor that "helpfully" preserves the
    quote style is caught.
    """
    from attune_rag.editor.rename import _rewrite_yaml_block_value

    fm_body = (
        "type: concept\n"
        "aliases:\n"
        "  - 'beta-spec'\n"
        '  - "beta-spec"\n'
        "  - beta-spec\n"
        "tags: [api]\n"
    )
    rewritten = _rewrite_yaml_block_value(fm_body, "aliases", "beta-spec", "beta")
    # All three list items rewrote to the bare new value, quote chars dropped.
    assert "beta-spec" not in rewritten
    assert rewritten.count("  - beta") == 3
    # Block terminator (the next top-level key) is preserved verbatim.
    assert "tags: [api]" in rewritten


def test_sidecar_corrupt_json_is_skipped_silently(trio: Path) -> None:
    """Malformed sidecars are skipped without raising or emitting an edit.

    Covers ``_plan_sidecar_path_rename_edits`` lines 245-254 in
    ``rename.py``: the ``except (OSError, json.JSONDecodeError):
    continue`` arm and the ``if not isinstance(data, dict) or old_rel
    not in data: continue`` arm. Both must leave the move intact and
    produce no sidecar edit.
    """
    import json as _json

    # Case 1: malformed JSON → JSONDecodeError → skipped.
    sidecar = trio / "summaries.json"
    sidecar.write_text("{this is not valid JSON", encoding="utf-8")
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")
    assert all(e.path != "summaries.json" for e in plan.edits)
    assert len(plan.moves) == 1
    # Corrupt sidecar untouched on disk.
    assert sidecar.read_text(encoding="utf-8") == "{this is not valid JSON"

    # Case 2: valid JSON but ``old_rel`` not a top-level key → skipped.
    sidecar.write_text(
        _json.dumps({"someone-else.md": "summary"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    corpus2 = DirectoryCorpus(trio)
    plan2 = plan_rename(corpus2, "alpha.md", "alpha-renamed.md", kind="template_path")
    assert all(e.path != "summaries.json" for e in plan2.edits)

    # Case 3: valid JSON but not a dict (a list at the top level) → skipped.
    sidecar.write_text(_json.dumps(["alpha.md"]) + "\n", encoding="utf-8")
    corpus3 = DirectoryCorpus(trio)
    plan3 = plan_rename(corpus3, "alpha.md", "alpha-renamed.md", kind="template_path")
    assert all(e.path != "summaries.json" for e in plan3.edits)


# -- W3.3 coverage push: error/edge paths in rename.py --------------
#
# These cover branches that the existing end-to-end tests don't hit
# naturally: unsupported kinds, corpus shapes without _root, normalize
# rejections (empty / absolute / escape), apply-time failure modes
# (source missing, planned target missing), entries with empty content,
# and the FileMove.to_dict serialization path.


class _Entry:
    """Tiny duck-typed entry for the `corpus.entries()` shape."""

    def __init__(self, content: str = "", path: str = ""):
        self.content = content
        self.path = path


class _EntriesCallableCorpus:
    """Duck-typed corpus with `entries()` and no `_root` — for plan-only
    paths that don't require a filesystem root."""

    def __init__(self, entries: list[_Entry]) -> None:
        self._entries = entries

    def entries(self) -> list[_Entry]:
        return self._entries


def test_plan_rename_rejects_unsupported_kind(trio: Path) -> None:
    """Covers rename.py:149 — the ValueError raised by `plan_rename`
    when ``kind`` isn't one of the three supported reference kinds."""
    corpus = DirectoryCorpus(trio)
    with pytest.raises(ValueError, match="Unsupported rename kind"):
        plan_rename(corpus, "x", "y", kind="bogus")  # type: ignore[arg-type]


def test_plan_template_path_rename_raises_when_corpus_has_no_root() -> None:
    """Covers rename.py:194-198 — corpus without a `_root` attribute
    can't accept a template_path rename (the move needs a filesystem
    anchor)."""
    corpus = _EntriesCallableCorpus([])
    with pytest.raises(RenameError, match="no resolvable root path"):
        plan_rename(corpus, "x.md", "y.md", kind="template_path")


def test_plan_template_path_rename_rejects_empty_template_path(trio: Path) -> None:
    """Covers rename.py:222-223 — empty `raw` in `_normalize_corpus_relpath`."""
    corpus = DirectoryCorpus(trio)
    with pytest.raises(ValueError, match="Empty template path"):
        plan_rename(corpus, "", "y.md", kind="template_path")


def test_plan_template_path_rename_rejects_absolute_path(trio: Path, tmp_path: Path) -> None:
    """Covers rename.py:225-226 — absolute path rejected by normalizer."""
    corpus = DirectoryCorpus(trio)
    absolute = str(tmp_path / "elsewhere" / "doc.md")
    with pytest.raises(ValueError, match="must be relative to the corpus root"):
        plan_rename(corpus, absolute, "y.md", kind="template_path")


def test_plan_alias_rename_skips_entries_with_empty_content() -> None:
    """Covers rename.py:163-164 — `_plan_alias_rename` skip when an
    entry has empty content or path. Uses a duck-typed corpus so we
    can construct entries without going through DirectoryCorpus's
    on-disk filter."""
    corpus = _EntriesCallableCorpus(
        [
            _Entry(content="", path="empty.md"),
            _Entry(content="some content but...", path=""),
        ]
    )
    plan = plan_rename(corpus, "x", "y", kind="alias")
    assert plan.edits == []
    assert plan.moves == []


def test_plan_tag_rename_skips_entries_with_empty_content() -> None:
    """Covers rename.py:177-178 — same skip in `_plan_tag_rename`."""
    corpus = _EntriesCallableCorpus(
        [
            _Entry(content="", path="empty.md"),
            _Entry(content="body", path=""),
        ]
    )
    plan = plan_rename(corpus, "old-tag", "new-tag", kind="tag")
    assert plan.edits == []


def test_file_move_to_dict_round_trip() -> None:
    """Covers rename.py:108-109 — `FileMove.to_dict` short return.
    The end-to-end `test_plan_to_dict_round_trip` uses an alias plan
    with zero moves, so the FileMove.to_dict line was never executed."""
    move = FileMove(old_path="old.md", new_path="new.md")
    assert move.to_dict() == {"old_path": "old.md", "new_path": "new.md"}
    # Through a RenamePlan with a real move:
    plan = RenamePlan(old="old.md", new="new.md", kind="template_path", moves=[move])
    dumped = plan.to_dict()
    assert dumped["moves"] == [{"old_path": "old.md", "new_path": "new.md"}]


def test_apply_rename_raises_when_corpus_has_no_root() -> None:
    """Covers rename.py:455-457 — `apply_rename` mirrors the planner's
    root-required check. A plan with edits can't apply against a
    corpus that has no `_root`."""
    corpus = _EntriesCallableCorpus([])
    plan = RenamePlan(
        old="x",
        new="y",
        kind="alias",
        edits=[FileEdit(path="x.md", old_text="a", new_text="b", hunks=[])],
    )
    with pytest.raises(RenameError, match="apply is not supported"):
        apply_rename(corpus, plan)


def test_apply_rename_raises_when_move_source_missing(trio: Path) -> None:
    """Covers rename.py:508-509 — `_apply_moves` raises if the source
    file disappeared between plan and apply."""
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "alpha.md", "alpha-renamed.md", kind="template_path")
    # Delete the source after planning, before applying.
    (trio / "alpha.md").unlink()
    with pytest.raises(RenameError, match="Move source missing at apply time"):
        apply_rename(corpus, plan)


def test_apply_rename_raises_when_edit_target_missing(trio: Path) -> None:
    """Covers rename.py:535-536 — `_stage_edits` raises if a planned
    edit target disappeared between plan and apply."""
    corpus = DirectoryCorpus(trio)
    plan = plan_rename(corpus, "beta-spec", "beta", kind="alias")
    # Pick a path in the plan and delete it before applying.
    target_path = trio / plan.edits[0].path
    target_path.unlink()
    with pytest.raises(RenameError, match="Planned target does not exist"):
        apply_rename(corpus, plan)


def test_apply_rename_noop_plan_returns_empty_list(trio: Path) -> None:
    """Covers the early-return at rename.py:451-453 — an empty plan
    short-circuits without touching disk."""
    corpus = DirectoryCorpus(trio)
    plan = RenamePlan(old="x", new="y", kind="alias", edits=[], moves=[])
    written = apply_rename(corpus, plan)
    assert written == []


def test_plan_template_path_rename_rejects_path_escaping_root(trio: Path) -> None:
    """Covers rename.py:233-237 — normalizer rejects ``..`` escapes
    even though the path looks relative."""
    corpus = DirectoryCorpus(trio)
    with pytest.raises(ValueError, match="escapes corpus root"):
        plan_rename(corpus, "../escape.md", "y.md", kind="template_path")
