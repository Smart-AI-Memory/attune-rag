"""Editor support for attune-rag — schema, lint, autocomplete, refactor.

Used by attune-gui's template editor. Kept separate from the retrieval
core so the editor surface can grow without bloating retrieval imports.
"""

from __future__ import annotations

from ._autocomplete import autocomplete_aliases, autocomplete_tags
from ._lint import Diagnostic, Severity, lint_template
from ._references import Reference, ReferenceContext, ReferenceKind, find_references
from ._rename import (
    FileEdit,
    Hunk,
    RenameCollisionError,
    RenameError,
    RenamePlan,
    apply_rename,
    plan_rename,
)
from ._schema import (
    SchemaError,
    load_schema,
    parse_frontmatter,
    validate_frontmatter,
)

__all__ = [
    "Diagnostic",
    "FileEdit",
    "Hunk",
    "Reference",
    "ReferenceContext",
    "ReferenceKind",
    "RenameCollisionError",
    "RenameError",
    "RenamePlan",
    "SchemaError",
    "Severity",
    "apply_rename",
    "autocomplete_aliases",
    "autocomplete_tags",
    "find_references",
    "lint_template",
    "load_schema",
    "parse_frontmatter",
    "plan_rename",
    "validate_frontmatter",
]
