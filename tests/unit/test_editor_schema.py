"""Tests for attune_rag.editor frontmatter schema (M1 task #1)."""

from __future__ import annotations

import pytest

from attune_rag.editor import (
    SchemaError,
    load_schema,
    parse_frontmatter,
    validate_frontmatter,
)


def test_schema_loads_as_resource() -> None:
    schema = load_schema()
    assert schema["title"] == "Attune Template Frontmatter"
    assert schema["required"] == ["type", "name"]
    assert "concept" in schema["properties"]["type"]["enum"]
    assert schema["additionalProperties"] is True


def test_schema_is_cached() -> None:
    # load_schema is lru_cached; second call returns the same object.
    assert load_schema() is load_schema()


# Fixture 1: valid full frontmatter ----------------------------------

VALID_FULL = """\
type: concept
name: Tool security audit
tags: [security, audit, tools]
aliases: [security-audit, tool-sec]
summary: How to audit a tool for security risks before publishing.
"""


def test_valid_full_frontmatter() -> None:
    data, issues = parse_frontmatter(VALID_FULL)
    assert issues == []
    assert data["type"] == "concept"
    assert data["name"] == "Tool security audit"
    assert data["tags"] == ["security", "audit", "tools"]


# Fixture 2: valid minimal frontmatter -------------------------------

VALID_MINIMAL = """\
type: task
name: Run linter
"""


def test_valid_minimal_frontmatter() -> None:
    data, issues = parse_frontmatter(VALID_MINIMAL)
    assert issues == []
    assert data == {"type": "task", "name": "Run linter"}


# Fixture 3: missing required field ----------------------------------

MISSING_REQUIRED = """\
type: concept
tags: [foo]
"""


def test_missing_required_field() -> None:
    _, issues = parse_frontmatter(MISSING_REQUIRED)
    codes = {i.code for i in issues}
    assert "missing-required" in codes
    msg = next(i.message for i in issues if i.code == "missing-required")
    assert "name" in msg


# Fixture 4: bad enum value ------------------------------------------

BAD_ENUM = """\
type: invalid-kind
name: Some template
"""


def test_bad_enum_value() -> None:
    _, issues = parse_frontmatter(BAD_ENUM)
    codes = {i.code for i in issues}
    assert "bad-enum" in codes
    msg = next(i.message for i in issues if i.code == "bad-enum")
    assert "concept" in msg  # allowed values listed


# Fixture 5: unknown key (allowed via additionalProperties) ----------

UNKNOWN_KEY = """\
type: reference
name: API reference
category: experimental
custom_field: 42
"""


def test_unknown_keys_are_allowed() -> None:
    """additionalProperties: true → unknown keys produce no schema issues.

    Forward-compat: future fields can land in templates without breaking
    older editors. Lint surfaces them as info-level warnings later (task
    #3), not schema errors here.
    """
    data, issues = parse_frontmatter(UNKNOWN_KEY)
    assert issues == []
    assert data["category"] == "experimental"
    assert data["custom_field"] == 42


# Fixture 6: malformed YAML ------------------------------------------

MALFORMED_YAML = """\
type: concept
name: [unterminated
tags:
  - foo
   - bar
"""


def test_malformed_yaml_raises_schema_error() -> None:
    with pytest.raises(SchemaError) as exc:
        parse_frontmatter(MALFORMED_YAML)
    assert "Malformed YAML" in str(exc.value)


# Edge cases ---------------------------------------------------------


def test_non_mapping_root() -> None:
    """A YAML list at root is structurally invalid frontmatter."""
    issues = validate_frontmatter(["just", "a", "list"])
    assert any(i.code == "not-a-mapping" for i in issues)


def test_empty_string_name_caught() -> None:
    data, issues = parse_frontmatter("type: concept\nname: ''\n")
    assert any(i.code == "too-short" for i in issues)
    assert data["type"] == "concept"


def test_duplicate_aliases_caught() -> None:
    src = "type: concept\nname: x\naliases: [foo, foo]\n"
    _, issues = parse_frontmatter(src)
    assert any(i.code == "duplicate-items" for i in issues)
