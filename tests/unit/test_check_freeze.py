"""Tests for ``scripts/check_freeze.py`` — Phase 4 feature-freeze enforcer.

Three check axes plus the override path. Tests feed synthetic
"base" and "head" strings via a monkey-patched ``git_show``, so no
real git repo is required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_freeze.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_freeze", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_freeze"] = module
    spec.loader.exec_module(module)
    return module


check_freeze = _load_module()


# ---------------------------------------------------------------------------
# Synthetic ref/path table for tests
# ---------------------------------------------------------------------------


class FakeRefs:
    """In-memory ``{(ref, path): content}`` map for stubbing ``git_show``."""

    def __init__(self, mapping: dict[tuple[str, str], str | None]):
        self._m = mapping

    def get(self, ref: str, path: str, repo: Path) -> str | None:
        return self._m.get((ref, path))


def _patch_git_show(monkeypatch: pytest.MonkeyPatch, refs: FakeRefs) -> None:
    monkeypatch.setattr(check_freeze, "git_show", refs.get)


# ---------------------------------------------------------------------------
# Check 1 — __all__ diff
# ---------------------------------------------------------------------------


def _module_src(symbols: list[str]) -> str:
    quoted = ", ".join(f'"{s}"' for s in symbols)
    return f"__all__ = [{quoted}]\n"


def test_all_diff_clean_when_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    src = _module_src(["A", "B"])
    refs = FakeRefs(
        {(ref, path): src for ref in ("base", "head") for _, path in check_freeze.PUBLIC_MODULES}
    )
    _patch_git_show(monkeypatch, refs)

    violations, errors = check_freeze.check_all_diff("base", "head", Path("."))
    assert violations == []
    assert errors == []


def test_all_diff_flags_added_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _module_src(["RagPipeline"])
    head = _module_src(["RagPipeline", "NewClass"])
    mapping: dict[tuple[str, str], str | None] = {}
    for _, path in check_freeze.PUBLIC_MODULES:
        mapping[("base", path)] = base
        mapping[("head", path)] = head
    _patch_git_show(monkeypatch, FakeRefs(mapping))

    violations, errors = check_freeze.check_all_diff("base", "head", Path("."))
    assert errors == []
    # Every module reports the same diff in this fixture.
    details = [v.detail for v in violations]
    assert all(v.kind == "all" for v in violations)
    assert all("'NewClass'" in d for d in details)
    assert len(violations) == len(check_freeze.PUBLIC_MODULES)


def test_all_diff_ignores_removed_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _module_src(["A", "B"])
    head = _module_src(["A"])  # B removed — removal is not a freeze violation
    mapping: dict[tuple[str, str], str | None] = {}
    for _, path in check_freeze.PUBLIC_MODULES:
        mapping[("base", path)] = base
        mapping[("head", path)] = head
    _patch_git_show(monkeypatch, FakeRefs(mapping))

    violations, _ = check_freeze.check_all_diff("base", "head", Path("."))
    assert violations == []


def test_all_diff_missing_at_head_is_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _module_src(["A"])
    mapping: dict[tuple[str, str], str | None] = {}
    for i, (_, path) in enumerate(check_freeze.PUBLIC_MODULES):
        mapping[("base", path)] = base
        # First module disappears at head; rest unchanged.
        mapping[("head", path)] = None if i == 0 else base
    _patch_git_show(monkeypatch, FakeRefs(mapping))

    violations, errors = check_freeze.check_all_diff("base", "head", Path("."))
    assert violations == []
    assert len(errors) == 1
    assert "missing at head" in errors[0]


def test_all_diff_new_module_flags_every_symbol_as_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _module_src(["A", "B"])
    mapping: dict[tuple[str, str], str | None] = {}
    for i, (_, path) in enumerate(check_freeze.PUBLIC_MODULES):
        # First module is brand-new at head (no base version).
        mapping[("base", path)] = None if i == 0 else head
        mapping[("head", path)] = head
    _patch_git_show(monkeypatch, FakeRefs(mapping))

    violations, errors = check_freeze.check_all_diff("base", "head", Path("."))
    assert errors == []
    new_module_name = check_freeze.PUBLIC_MODULES[0][0]
    new_module_violations = [v for v in violations if new_module_name in v.detail]
    assert len(new_module_violations) == 2  # A and B both new
    assert all("new public module" in v.detail for v in new_module_violations)


def test_parse_all_handles_tuple_and_list() -> None:
    list_form = '__all__ = ["A", "B"]\n'
    tuple_form = '__all__ = ("A", "B")\n'
    assert check_freeze.parse_all(list_form) == frozenset({"A", "B"})
    assert check_freeze.parse_all(tuple_form) == frozenset({"A", "B"})


def test_parse_all_returns_none_on_dynamic_construction() -> None:
    # Dynamic constructions are correctly rejected so we don't
    # misread them as empty.
    dynamic = "__all__ = ['A'] + extras\n"
    assert check_freeze.parse_all(dynamic) is None


def test_parse_all_returns_none_when_no_all_present() -> None:
    assert check_freeze.parse_all("x = 1\n") is None


# ---------------------------------------------------------------------------
# Check 2 — CHANGELOG [Unreleased] ### Added
# ---------------------------------------------------------------------------


def _changelog(unreleased_added: list[str] | None) -> str:
    """Build a minimal CHANGELOG.md with optional [Unreleased] ### Added bullets."""
    lines = ["# Changelog", ""]
    lines.append("## [Unreleased]")
    lines.append("")
    if unreleased_added is not None:
        lines.append("### Added")
        lines.append("")
        for bullet in unreleased_added:
            lines.append(f"- {bullet}")
        lines.append("")
    lines.append("## [0.1.19] - 2026-05-16")
    lines.append("")
    lines.append("### Changed")
    lines.append("")
    lines.append("- Some older change.")
    lines.append("")
    return "\n".join(lines)


def test_changelog_clean_when_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _changelog(None)
    refs = FakeRefs({("base", "CHANGELOG.md"): text, ("head", "CHANGELOG.md"): text})
    _patch_git_show(monkeypatch, refs)

    violations, errors = check_freeze.check_changelog_added("base", "head", Path("."))
    assert violations == []
    assert errors == []


def test_changelog_flags_new_added_bullet(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _changelog(None)
    head = _changelog(["New public function `foo()`."])
    refs = FakeRefs({("base", "CHANGELOG.md"): base, ("head", "CHANGELOG.md"): head})
    _patch_git_show(monkeypatch, refs)

    violations, errors = check_freeze.check_changelog_added("base", "head", Path("."))
    assert errors == []
    assert len(violations) == 1
    assert violations[0].kind == "changelog"
    assert "foo()" in violations[0].detail


def test_changelog_unchanged_bullet_is_not_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bullet that's identical between base and head is not new."""
    bullets = ["Existing Added entry that was already there."]
    text = _changelog(bullets)
    refs = FakeRefs({("base", "CHANGELOG.md"): text, ("head", "CHANGELOG.md"): text})
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_changelog_added("base", "head", Path("."))
    assert violations == []


def test_changelog_ignores_added_in_released_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`### Added` under a tagged release isn't a freeze violation —
    only the `[Unreleased]` block matters during the freeze."""
    head = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [0.1.19] - 2026-05-16\n\n"
        "### Added\n\n"
        "- Historical entry from the 0.1.19 release.\n"
    )
    refs = FakeRefs({("base", "CHANGELOG.md"): "", ("head", "CHANGELOG.md"): head})
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_changelog_added("base", "head", Path("."))
    assert violations == []


def test_changelog_missing_at_head_is_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = FakeRefs({("base", "CHANGELOG.md"): "", ("head", "CHANGELOG.md"): None})
    _patch_git_show(monkeypatch, refs)

    violations, errors = check_freeze.check_changelog_added("base", "head", Path("."))
    assert violations == []
    assert errors and "missing" in errors[0]


def test_extract_unreleased_added_handles_multiline_bullets() -> None:
    text = (
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- First bullet, first line.\n"
        "  Continuation of first bullet.\n"
        "  Another continuation line.\n"
        "- Second bullet on its own.\n"
        "\n"
        "## [0.1.19] - 2026-05-16\n"
    )
    bullets = check_freeze.extract_unreleased_added(text)
    assert len(bullets) == 2
    assert "First bullet, first line." in bullets[0]
    assert "Continuation of first bullet." in bullets[0]
    assert bullets[1].startswith("Second bullet")


# ---------------------------------------------------------------------------
# Check 3 — template_schema.json backward-compat
# ---------------------------------------------------------------------------


def _schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema)


def test_schema_clean_when_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _schema_text({"type": "object", "additionalProperties": True, "required": ["a"]})
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): schema, ("head", check_freeze.SCHEMA_PATH): schema}
    )
    _patch_git_show(monkeypatch, refs)

    violations, errors = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert violations == []
    assert errors == []


def test_schema_additional_properties_tightening_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _schema_text({"type": "object", "additionalProperties": True})
    head = _schema_text({"type": "object", "additionalProperties": False})
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): base, ("head", check_freeze.SCHEMA_PATH): head}
    )
    _patch_git_show(monkeypatch, refs)

    violations, errors = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert errors == []
    assert any(
        v.kind == "schema" and "additionalProperties tightened" in v.detail for v in violations
    )


def test_schema_loosening_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Going from strict additionalProperties to permissive is fine."""
    base = _schema_text({"type": "object", "additionalProperties": False})
    head = _schema_text({"type": "object", "additionalProperties": True})
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): base, ("head", check_freeze.SCHEMA_PATH): head}
    )
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert violations == []


def test_schema_enum_narrowing_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _schema_text(
        {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["a", "b", "c"]}},
        }
    )
    head = _schema_text(
        {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["a", "b"]}},
        }
    )
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): base, ("head", check_freeze.SCHEMA_PATH): head}
    )
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert any(v.kind == "schema" and "enum removes 'c'" in v.detail for v in violations)


def test_schema_enum_widening_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a new enum value is loosening, not tightening."""
    base = _schema_text(
        {"type": "object", "properties": {"kind": {"type": "string", "enum": ["a"]}}}
    )
    head = _schema_text(
        {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["a", "b"]}},
        }
    )
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): base, ("head", check_freeze.SCHEMA_PATH): head}
    )
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert violations == []


def test_schema_required_field_addition_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _schema_text({"type": "object", "required": ["a"]})
    head = _schema_text({"type": "object", "required": ["a", "b"]})
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): base, ("head", check_freeze.SCHEMA_PATH): head}
    )
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert any(v.kind == "schema" and "required adds 'b'" in v.detail for v in violations)


def test_schema_required_field_removal_not_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a required field loosens the contract."""
    base = _schema_text({"type": "object", "required": ["a", "b"]})
    head = _schema_text({"type": "object", "required": ["a"]})
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): base, ("head", check_freeze.SCHEMA_PATH): head}
    )
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert violations == []


def test_schema_nested_recursion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward-incompat changes deep in `properties.*.properties.*` get caught."""
    base = _schema_text(
        {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string", "enum": ["x", "y"]}},
                }
            },
        }
    )
    head = _schema_text(
        {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string", "enum": ["x"]}},
                }
            },
        }
    )
    refs = FakeRefs(
        {("base", check_freeze.SCHEMA_PATH): base, ("head", check_freeze.SCHEMA_PATH): head}
    )
    _patch_git_show(monkeypatch, refs)

    violations, _ = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert any("$.properties.outer.properties.inner" in v.detail for v in violations)


def test_schema_malformed_json_is_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = FakeRefs(
        {
            ("base", check_freeze.SCHEMA_PATH): "{}",
            ("head", check_freeze.SCHEMA_PATH): "{ not valid json",
        }
    )
    _patch_git_show(monkeypatch, refs)

    violations, errors = check_freeze.check_schema_backward_compat("base", "head", Path("."))
    assert violations == []
    assert errors and "not valid JSON" in errors[0]


# ---------------------------------------------------------------------------
# main() — exit codes + override path
# ---------------------------------------------------------------------------


def _all_clean_refs() -> FakeRefs:
    """Build a refs table where every check passes."""
    mapping: dict[tuple[str, str], str | None] = {}
    src = _module_src(["A"])
    for _, path in check_freeze.PUBLIC_MODULES:
        mapping[("base", path)] = src
        mapping[("head", path)] = src
    cl = _changelog(None)
    mapping[("base", "CHANGELOG.md")] = cl
    mapping[("head", "CHANGELOG.md")] = cl
    schema = _schema_text({"type": "object", "additionalProperties": True})
    mapping[("base", check_freeze.SCHEMA_PATH)] = schema
    mapping[("head", check_freeze.SCHEMA_PATH)] = schema
    return FakeRefs(mapping)


def test_main_exit_0_on_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_git_show(monkeypatch, _all_clean_refs())
    rc = check_freeze.main(["--base", "base", "--head", "head"])
    assert rc == 0


def test_main_exit_1_on_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = _all_clean_refs()
    refs._m[("head", "CHANGELOG.md")] = _changelog(["New thing."])
    _patch_git_show(monkeypatch, refs)

    rc = check_freeze.main(["--base", "base", "--head", "head"])
    assert rc == 1


def test_main_exit_2_on_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = _all_clean_refs()
    refs._m[("head", check_freeze.SCHEMA_PATH)] = "{ not valid json"
    _patch_git_show(monkeypatch, refs)

    rc = check_freeze.main(["--base", "base", "--head", "head"])
    assert rc == 2


def test_main_allow_override_returns_0_on_violation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--allow-override mirrors the freeze-override PR label: violations
    still print to stderr, but exit code is 0."""
    refs = _all_clean_refs()
    refs._m[("head", "CHANGELOG.md")] = _changelog(["New thing under override."])
    _patch_git_show(monkeypatch, refs)

    rc = check_freeze.main(["--base", "base", "--head", "head", "--allow-override"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "FAIL changelog" in captured.err
    assert "--allow-override set" in captured.err


def test_main_allow_override_still_blocks_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override flips violations to warnings but does NOT mask validation
    errors — malformed input should always exit 2, override or not."""
    refs = _all_clean_refs()
    refs._m[("head", check_freeze.SCHEMA_PATH)] = "{ not valid json"
    _patch_git_show(monkeypatch, refs)

    rc = check_freeze.main(["--base", "base", "--head", "head", "--allow-override"])
    assert rc == 2


def test_main_comment_out_writes_markdown_on_violation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    refs = _all_clean_refs()
    refs._m[("head", "CHANGELOG.md")] = _changelog(["A new public thing."])
    _patch_git_show(monkeypatch, refs)
    comment_path = tmp_path / "out" / "comment.md"

    rc = check_freeze.main(
        [
            "--base",
            "base",
            "--head",
            "head",
            "--comment-out",
            str(comment_path),
        ]
    )
    assert rc == 1
    assert comment_path.exists()
    body = comment_path.read_text()
    assert check_freeze.COMMENT_MARKER in body
    assert "Feature-freeze gate failed" in body
    assert "A new public thing." in body


def test_main_comment_out_not_written_on_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_git_show(monkeypatch, _all_clean_refs())
    comment_path = tmp_path / "comment.md"
    rc = check_freeze.main(["--base", "base", "--head", "head", "--comment-out", str(comment_path)])
    assert rc == 0
    assert not comment_path.exists()


def test_format_violation_comment_is_deterministic() -> None:
    """Two equivalent violation lists produce byte-identical comments —
    the stable-marker workflow depends on this."""
    vs = [
        check_freeze.Violation(kind="changelog", detail="entry b"),
        check_freeze.Violation(kind="all", detail="adds 'X'"),
        check_freeze.Violation(kind="changelog", detail="entry a"),
    ]
    first = check_freeze.format_violation_comment(vs)
    second = check_freeze.format_violation_comment(list(reversed(vs)))
    assert first == second
    assert check_freeze.COMMENT_MARKER in first


def test_format_violation_comment_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        check_freeze.format_violation_comment([])
