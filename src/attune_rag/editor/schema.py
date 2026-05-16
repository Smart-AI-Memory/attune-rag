"""Frontmatter schema loader + validators.

The JSON Schema lives next to this module as `template_schema.json`.
It is loaded once via `importlib.resources` and cached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_SCHEMA_RESOURCE = "template_schema.json"


class SchemaError(Exception):
    """Raised when the frontmatter cannot be parsed at all (malformed YAML)."""


@dataclass(frozen=True)
class FrontmatterIssue:
    """A single schema violation. Line/column are 1-indexed within the
    frontmatter block; callers that care about absolute file coordinates
    must offset by the frontmatter's start line."""

    code: str
    message: str
    path: tuple[str | int, ...] = ()


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the parsed JSON Schema for template frontmatter."""
    text = files(__package__).joinpath(_SCHEMA_RESOURCE).read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_frontmatter(data: Any) -> list[FrontmatterIssue]:
    """Validate already-parsed frontmatter against the schema.

    Returns a list of issues; an empty list means valid. The frontmatter
    must be a mapping; non-mapping inputs produce a single
    ``not-a-mapping`` issue.
    """
    if not isinstance(data, dict):
        return [
            FrontmatterIssue(
                code="not-a-mapping",
                message=f"Frontmatter must be a YAML mapping, got {type(data).__name__}.",
            )
        ]
    issues: list[FrontmatterIssue] = []
    for err in _validator().iter_errors(data):
        issues.append(_issue_from_error(err))
    return issues


def parse_frontmatter(yaml_text: str) -> tuple[dict[str, Any], list[FrontmatterIssue]]:
    """Parse a YAML frontmatter block and validate it.

    Returns ``(data, issues)``. If the YAML is malformed, raises
    ``SchemaError`` so the caller can distinguish parse failure (which
    blocks the structured form entirely) from schema violations (which
    are inline diagnostics).
    """
    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise SchemaError(f"Malformed YAML in frontmatter: {exc}") from exc
    issues = validate_frontmatter(data)
    return data if isinstance(data, dict) else {}, issues


def _issue_from_error(err: ValidationError) -> FrontmatterIssue:
    """Map a jsonschema ValidationError to a stable issue code + message."""
    path = tuple(err.absolute_path)
    validator = err.validator
    if validator == "required":
        # err.message is e.g. "'name' is a required property"
        missing = err.message.split("'")[1] if "'" in err.message else "?"
        return FrontmatterIssue(
            code="missing-required",
            message=f"Missing required field: {missing}",
            path=path + (missing,),
        )
    if validator == "enum":
        field = path[-1] if path else "?"
        allowed = ", ".join(repr(v) for v in (err.validator_value or []))
        return FrontmatterIssue(
            code="bad-enum",
            message=f"Field {field!r} must be one of: {allowed}",
            path=path,
        )
    if validator == "type":
        field = path[-1] if path else "?"
        expected = err.validator_value
        return FrontmatterIssue(
            code="bad-type",
            message=f"Field {field!r} must be of type {expected!r}",
            path=path,
        )
    if validator == "uniqueItems":
        field = path[-1] if path else "?"
        return FrontmatterIssue(
            code="duplicate-items",
            message=f"Field {field!r} must contain unique items",
            path=path,
        )
    if validator == "minLength":
        field = path[-1] if path else "?"
        return FrontmatterIssue(
            code="too-short",
            message=f"Field {field!r} must not be empty",
            path=path,
        )
    return FrontmatterIssue(
        code="schema-violation",
        message=err.message,
        path=path,
    )
