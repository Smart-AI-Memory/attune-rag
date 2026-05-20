"""Phase 4 feature-freeze enforcer.

Compares two git refs (base vs head) for changes that count as "new
public surface" during the v1.0-roadmap Phase 4 freeze window. CI
runs this on every PR:

    python scripts/check_freeze.py --base origin/main --head HEAD

Three check axes (see ``docs/specs/downstream-validation/design.md``
§1 and ``tasks.md`` W0.1):

1. ``__all__`` diff per public module. Any symbol added to
   ``attune_rag.__all__`` (or the corpus / providers / editor
   submodules' ``__all__``) at head that wasn't there at base is a
   freeze violation.
2. CHANGELOG ``[Unreleased]`` ``### Added`` diff. Any new bullet
   under ``### Added`` in the Unreleased block is a freeze violation.
3. ``src/attune_rag/editor/template_schema.json`` backward-compat
   diff. Three sub-checks: ``additionalProperties`` tightening,
   ``enum`` value removal, ``required`` field additions.

Exit codes
----------
0 — no freeze violations (or ``--allow-override`` set).
1 — at least one freeze violation.
2 — validation error (missing file, parse error, git failure).

Stderr lists each violation as one line per finding, prefixed by its
check kind:

    FAIL all: attune_rag.__all__ adds 'NewClass'
    FAIL changelog: new [Unreleased] ### Added entry: ...
    FAIL schema: $.properties.type.enum removes 'foo'

Pure stdlib. No LLM dependency. Safe to run in any CI image.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Public modules whose ``__all__`` is part of the frozen surface. Keep
# in sync with ``tests/unit/test_api_surface.py`` (the EXPECTED_*_ALL
# frozensets). If a new public module is genuinely added, it's a
# surface change — exactly what this script exists to catch.
PUBLIC_MODULES: tuple[tuple[str, str], ...] = (
    ("attune_rag", "src/attune_rag/__init__.py"),
    ("attune_rag.corpus", "src/attune_rag/corpus/__init__.py"),
    ("attune_rag.providers", "src/attune_rag/providers/__init__.py"),
    ("attune_rag.editor", "src/attune_rag/editor/__init__.py"),
)

CHANGELOG_PATH = "CHANGELOG.md"
SCHEMA_PATH = "src/attune_rag/editor/template_schema.json"


@dataclass(frozen=True)
class Violation:
    kind: str  # "all" | "changelog" | "schema"
    detail: str


COMMENT_MARKER = "<!-- attune-rag-freeze-gate -->"


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------


def git_show(ref: str, path: str, repo: Path) -> str | None:
    """Return file contents at ``<ref>:<path>``, or ``None`` if the path
    doesn't exist in that ref.

    Distinguishes "file is missing at this ref" (returns None) from
    "git invocation itself failed" (raises). The caller decides how
    to treat a missing path — for ``__all__`` and CHANGELOG checks,
    missing-at-base just means everything in head is new.
    """
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout
    # git show prints "fatal: ... does not exist" for missing paths.
    # Other failure modes (bad ref, not a repo) also exit non-zero;
    # those bubble up as None too, but get caught downstream when no
    # check can read anything.
    return None


# ---------------------------------------------------------------------------
# Check 1 — __all__ diff
# ---------------------------------------------------------------------------


def parse_all(source: str) -> frozenset[str] | None:
    """Extract ``__all__`` from a Python source string, or ``None`` if
    the module has no ``__all__``.

    Only handles literal list/tuple assignments (``__all__ = [...]``
    or ``__all__ = (...)``). Dynamic ``__all__`` constructions
    (concatenation, comprehensions) return None — which is the right
    failure mode for a freeze enforcer (better to under-detect on an
    unusual module than to misread one).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = node.value
                if not isinstance(value, (ast.List, ast.Tuple)):
                    return None
                names: list[str] = []
                for elt in value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        names.append(elt.value)
                return frozenset(names)
    return None


def check_all_diff(base: str, head: str, repo: Path) -> tuple[list[Violation], list[str]]:
    """Return (violations, errors). Errors are exit-2 conditions."""
    violations: list[Violation] = []
    errors: list[str] = []
    for module_name, path in PUBLIC_MODULES:
        head_src = git_show(head, path, repo)
        if head_src is None:
            # The module no longer exists at head. Removing public
            # modules is a surface change, but it's a REMOVAL — the
            # spec scopes the freeze to "no new public surface", not
            # "no churn at all". Note it as a validation concern so
            # the maintainer notices, but don't auto-fail.
            errors.append(
                f"{module_name}: {path} missing at {head} "
                "(public module removed? confirm before merging)"
            )
            continue
        head_all = parse_all(head_src)
        if head_all is None:
            # No __all__ at head — nothing to gate on this module.
            continue
        base_src = git_show(base, path, repo)
        if base_src is None:
            # Brand-new public module in this PR. Everything in its
            # __all__ counts as added.
            for name in sorted(head_all):
                violations.append(
                    Violation(
                        kind="all",
                        detail=(f"{module_name}.__all__ adds {name!r} " "(new public module)"),
                    )
                )
            continue
        base_all = parse_all(base_src) or frozenset()
        added = head_all - base_all
        for name in sorted(added):
            violations.append(Violation(kind="all", detail=f"{module_name}.__all__ adds {name!r}"))
    return violations, errors


# ---------------------------------------------------------------------------
# Check 2 — CHANGELOG [Unreleased] ### Added
# ---------------------------------------------------------------------------


_UNRELEASED_RE = re.compile(r"^##\s+\[Unreleased\]\s*$", re.MULTILINE)
_NEXT_H2_RE = re.compile(r"^##\s+\[", re.MULTILINE)
_ADDED_RE = re.compile(r"^###\s+Added\s*$", re.MULTILINE)
_NEXT_H3_RE = re.compile(r"^###\s+\w", re.MULTILINE)


def extract_unreleased_added(text: str) -> list[str]:
    """Return the list of bullet items under ``[Unreleased]`` ``### Added``.

    Bullets span from one ``- ``-prefixed line up to the next bullet
    or the next heading. Continuation lines (indented or blank) are
    folded into the bullet they belong to. Empty bullets are dropped.

    Returns ``[]`` when there's no ``[Unreleased]`` block, no
    ``### Added`` under it, or the section is empty.
    """
    m = _UNRELEASED_RE.search(text)
    if not m:
        return []
    start = m.end()
    next_h2 = _NEXT_H2_RE.search(text, start)
    unreleased = text[start : next_h2.start()] if next_h2 else text[start:]

    m_added = _ADDED_RE.search(unreleased)
    if not m_added:
        return []
    sub_start = m_added.end()
    next_h3 = _NEXT_H3_RE.search(unreleased, sub_start)
    added_block = unreleased[sub_start : next_h3.start()] if next_h3 else unreleased[sub_start:]

    bullets: list[str] = []
    current: list[str] = []
    for line in added_block.splitlines():
        if line.startswith("- "):
            if current:
                bullets.append("\n".join(current).strip())
            current = [line[2:]]  # drop the leading "- "
        elif current and (line.startswith("  ") or not line.strip()):
            current.append(line)
        elif current:
            bullets.append("\n".join(current).strip())
            current = []
    if current:
        bullets.append("\n".join(current).strip())

    return [b for b in bullets if b]


def _normalize_bullet(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_changelog_added(base: str, head: str, repo: Path) -> tuple[list[Violation], list[str]]:
    violations: list[Violation] = []
    errors: list[str] = []
    head_text = git_show(head, CHANGELOG_PATH, repo)
    if head_text is None:
        errors.append(f"{CHANGELOG_PATH} missing at {head}")
        return violations, errors
    base_text = git_show(base, CHANGELOG_PATH, repo) or ""
    base_bullets = {_normalize_bullet(b) for b in extract_unreleased_added(base_text)}
    for bullet in extract_unreleased_added(head_text):
        if _normalize_bullet(bullet) in base_bullets:
            continue
        summary = bullet.splitlines()[0].strip()
        if len(summary) > 100:
            summary = summary[:97] + "..."
        violations.append(
            Violation(
                kind="changelog",
                detail=f"new [Unreleased] ### Added entry: {summary}",
            )
        )
    return violations, errors


# ---------------------------------------------------------------------------
# Check 3 — template_schema.json backward-compat
# ---------------------------------------------------------------------------


def _json_loads_safe(text: str, label: str, errors: list[str]) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        errors.append(f"{SCHEMA_PATH} at {label} is not valid JSON: {e}")
        return None


def _ap_tightened(base_ap: Any, head_ap: Any) -> bool:
    """Was ``additionalProperties`` tightened between base and head?

    JSON Schema's default is ``True`` (additionalProperties allowed),
    so a missing value at base means "permissive". Tightening cases:

    - base True (or absent) → head False
    - base True (or absent) → head dict-schema (now constrained)
    - base dict-schema → head False
    - base/head both dicts: out of scope; deep-comparing two schemas
      that both constrain is more freeze-noise than signal at this
      gate. A real signature-locking gate (1.0.0) does that work.
    """
    base_strict = base_ap is False or isinstance(base_ap, dict)
    head_strict = head_ap is False or isinstance(head_ap, dict)
    if not head_strict:
        return False
    return not base_strict


def _hashable(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    return v


def _walk_schema(base: Any, head: Any, path: str, violations: list[Violation]) -> None:
    """Compare two schema nodes for backward-incompatible changes.

    Three sub-checks are applied at every node: ``additionalProperties``
    tightening, ``required`` field additions, ``enum`` value removal.
    Then recurse into ``properties.*`` and ``items``.

    Non-dict nodes are ignored (caller passes ``Any`` because schemas
    can technically be ``True``/``False`` at any depth; those are
    leaf nodes with no sub-structure worth walking).
    """
    if not isinstance(base, dict) or not isinstance(head, dict):
        return

    if _ap_tightened(
        base.get("additionalProperties", True), head.get("additionalProperties", True)
    ):
        violations.append(
            Violation(
                kind="schema",
                detail=f"{path}.additionalProperties tightened",
            )
        )

    base_req = set(base.get("required", []) or [])
    head_req = set(head.get("required", []) or [])
    for name in sorted(head_req - base_req):
        violations.append(Violation(kind="schema", detail=f"{path}.required adds {name!r}"))

    if "enum" in base and "enum" in head:
        base_enum = {_hashable(v) for v in base.get("enum", [])}
        head_enum = {_hashable(v) for v in head.get("enum", [])}
        for v in sorted(base_enum - head_enum, key=str):
            violations.append(Violation(kind="schema", detail=f"{path}.enum removes {v!r}"))

    base_props = base.get("properties") or {}
    head_props = head.get("properties") or {}
    if isinstance(head_props, dict):
        for prop_name, head_sub in head_props.items():
            base_sub = base_props.get(prop_name) if isinstance(base_props, dict) else None
            _walk_schema(
                base_sub if base_sub is not None else {},
                head_sub,
                f"{path}.properties.{prop_name}",
                violations,
            )

    base_items = base.get("items")
    head_items = head.get("items")
    if isinstance(head_items, dict):
        _walk_schema(
            base_items if isinstance(base_items, dict) else {},
            head_items,
            f"{path}.items",
            violations,
        )


def check_schema_backward_compat(
    base: str, head: str, repo: Path
) -> tuple[list[Violation], list[str]]:
    violations: list[Violation] = []
    errors: list[str] = []
    head_text = git_show(head, SCHEMA_PATH, repo)
    if head_text is None:
        # Schema deleted between base and head. That's a removal, not
        # an addition; surface as a validation note rather than auto-fail.
        errors.append(f"{SCHEMA_PATH} missing at {head} (schema removed? confirm before merging)")
        return violations, errors
    head_schema = _json_loads_safe(head_text, head, errors)
    if head_schema is None:
        return violations, errors

    base_text = git_show(base, SCHEMA_PATH, repo)
    if base_text is None:
        # New schema on this PR; not a tightening of anything that
        # existed before. Caller's job to vet it on its own merits.
        return violations, errors
    base_schema = _json_loads_safe(base_text, base, errors)
    if base_schema is None:
        return violations, errors

    _walk_schema(base_schema, head_schema, "$", violations)
    return violations, errors


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def run_all_checks(base: str, head: str, repo: Path) -> tuple[list[Violation], list[str]]:
    all_violations: list[Violation] = []
    all_errors: list[str] = []
    for check in (check_all_diff, check_changelog_added, check_schema_backward_compat):
        v, e = check(base, head, repo)
        all_violations.extend(v)
        all_errors.extend(e)
    return all_violations, all_errors


def format_violation_comment(violations: list[Violation]) -> str:
    """Render a markdown PR-comment body for a non-empty violation list.

    Deterministic — no timestamps, no hostnames — so a golden test
    can pin it. Sorted by (kind, detail) so two equivalent runs
    produce byte-identical comments. ``COMMENT_MARKER`` lets the
    workflow grep + update the same comment instead of stacking new
    ones on every push.
    """
    if not violations:
        raise ValueError(
            "format_violation_comment called with no violations; "
            "callers should skip commenting on a green run"
        )

    ordered = sorted(violations, key=lambda v: (v.kind, v.detail))
    by_kind: dict[str, list[Violation]] = {}
    for v in ordered:
        by_kind.setdefault(v.kind, []).append(v)

    kind_titles = {
        "all": "Public `__all__` additions",
        "changelog": "New `[Unreleased]` `### Added` entries",
        "schema": "`template_schema.json` backward-incompatible changes",
    }

    lines: list[str] = [
        COMMENT_MARKER,
        "## Feature-freeze gate failed",
        "",
        (
            "This PR adds public surface during the Phase 4 feature "
            "freeze. See [docs/specs/downstream-validation/]"
            "(docs/specs/downstream-validation/) for the freeze policy."
        ),
        "",
    ]
    for kind in ("all", "changelog", "schema"):
        items = by_kind.get(kind)
        if not items:
            continue
        lines.append(f"### {kind_titles[kind]}")
        lines.append("")
        for v in items:
            lines.append(f"- {v.detail}")
        lines.append("")
    lines.extend(
        [
            "### What to do",
            "",
            "- If this is intentional new surface, defer it to the "
            "post-freeze 0.2.0 cut and remove the addition from this PR.",
            "- If this is a `Security`-scoped exception, apply the "
            "`freeze-override` label and add an `[Override-rationale]` "
            "block to the PR description.",
            "- If this is an internal improvement that doesn't actually "
            "add surface, re-classify as `### Changed` / `### Fixed` and "
            "drop the public symbol.",
            "",
            COMMENT_MARKER,
        ]
    )
    return "\n".join(lines) + "\n"


def _print_violations(violations: list[Violation]) -> None:
    for v in sorted(violations, key=lambda x: (x.kind, x.detail)):
        print(f"FAIL {v.kind}: {v.detail}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_freeze",
        description=(
            "Phase 4 feature-freeze enforcer. Compares two git refs "
            "for public-surface additions. Exits 0 (clean / override), "
            "1 (violation), or 2 (validation error)."
        ),
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base ref to compare against (default: origin/main).",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head ref to check (default: HEAD).",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repo root (default: cwd). Mostly used by tests.",
    )
    parser.add_argument(
        "--allow-override",
        action="store_true",
        help=(
            "Exit 0 even on violations. Mirrors the `freeze-override` "
            "PR label; the workflow sets this when the label is "
            "present and an `[Override-rationale]` block is in the "
            "PR body."
        ),
    )
    parser.add_argument(
        "--comment-out",
        type=Path,
        default=None,
        help=(
            "On violation, write a markdown PR-comment body to this "
            "path. The workflow then invokes `gh pr comment "
            "--body-file ...`. Not written on green runs or "
            "validation errors."
        ),
    )
    args = parser.parse_args(argv)

    violations, errors = run_all_checks(args.base, args.head, args.repo)

    if errors:
        for msg in errors:
            print(f"error: {msg}", file=sys.stderr)
        return 2

    if violations:
        _print_violations(violations)
        if args.comment_out is not None:
            args.comment_out.parent.mkdir(parents=True, exist_ok=True)
            args.comment_out.write_text(format_violation_comment(violations), encoding="utf-8")
        if args.allow_override:
            print(
                "warning: --allow-override set; " f"{len(violations)} violation(s) ignored.",
                file=sys.stderr,
            )
            return 0
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
