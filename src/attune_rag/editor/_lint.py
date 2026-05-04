"""Template linter — produces diagnostics for the editor.

Public entry point: :func:`lint_template`. Coordinates three layers of
checks:

1. Frontmatter YAML parse + schema validation (delegates to
   :mod:`._schema`).
2. ``[[alias]]`` body references against the corpus alias index.
3. ``## Depth N`` markers in the body — sequence + start.

Diagnostics carry 1-indexed line and column ranges referring to the
original document text (not the frontmatter-only or body-only views).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

import yaml

from ._schema import (
    FrontmatterIssue,
    SchemaError,
    load_schema,
    validate_frontmatter,
)

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Diagnostic:
    """A single lint diagnostic.

    Line/column are 1-indexed and refer to the *original* document
    text (the editor's source-of-truth coordinates). ``end_line`` and
    ``end_col`` are inclusive of the marked range.
    """

    severity: Severity
    code: str
    message: str
    line: int
    col: int
    end_line: int
    end_col: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FRONTMATTER_OPEN_RE = re.compile(r"^---\s*$")
_FRONTMATTER_CLOSE_RE = re.compile(r"^---\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
# Match [[alias]] but not \[[alias]] (escape) and not nested.
_ALIAS_REF_RE = re.compile(r"(?<!\\)\[\[([^\[\]\n]+?)\]\]")
_DEPTH_RE = re.compile(r"^(#{1,6})\s+Depth\s+(\d+)\b", re.IGNORECASE)
_TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:")

_KNOWN_FRONTMATTER_KEYS = set(load_schema()["properties"].keys())


def lint_template(text: str, rel_path: str, corpus: Any) -> list[Diagnostic]:
    """Run all lint checks against ``text``.

    ``rel_path`` is unused today but reserved for path-aware checks
    (e.g. self-references in aliases). ``corpus`` is duck-typed: if it
    exposes ``alias_index``, broken-alias diagnostics are produced;
    otherwise alias checks are skipped.
    """
    diagnostics: list[Diagnostic] = []
    lines = text.splitlines()

    fm_block, fm_start_line, fm_end_line = _extract_frontmatter(lines)
    if fm_block is not None:
        diagnostics.extend(_lint_frontmatter(fm_block, fm_start_line))

    body_start_line = (fm_end_line + 1) if fm_end_line is not None else 1
    diagnostics.extend(_lint_body(lines, body_start_line, corpus))

    diagnostics.sort(key=lambda d: (d.line, d.col, d.code))
    return diagnostics


# -- frontmatter ----------------------------------------------------


def _extract_frontmatter(
    lines: list[str],
) -> tuple[list[str] | None, int, int | None]:
    """Locate the YAML frontmatter block.

    Returns ``(fm_lines, fm_start_line, fm_end_line)``:

    - ``fm_lines`` are the lines *between* the ``---`` delimiters
      (excluding them); ``None`` if no frontmatter.
    - ``fm_start_line`` is 1-indexed line of the opening ``---``
      (or 1 if there's no frontmatter — body lint will use it as the
      base for body line numbers).
    - ``fm_end_line`` is 1-indexed line of the closing ``---``;
      ``None`` if not found.
    """
    if not lines or not _FRONTMATTER_OPEN_RE.match(lines[0]):
        return None, 1, None
    for idx in range(1, len(lines)):
        if _FRONTMATTER_CLOSE_RE.match(lines[idx]):
            return lines[1:idx], 1, idx + 1
    # Unclosed frontmatter — treat the whole prefix as fm for the
    # malformed-yaml diagnostic; body lint sees an empty body.
    return lines[1:], 1, len(lines)


def _lint_frontmatter(fm_lines: list[str], fm_start_line: int) -> list[Diagnostic]:
    """Validate frontmatter against the schema; map issues to lines."""
    fm_text = "\n".join(fm_lines)
    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        line, col = _yaml_error_position(exc, fm_start_line + 1)
        problem = getattr(exc, "problem", None) or exc
        return [
            Diagnostic(
                severity="error",
                code="malformed-yaml",
                message=f"Malformed YAML in frontmatter: {problem}",
                line=line,
                col=col,
                end_line=line,
                end_col=col + 1,
            )
        ]

    if not isinstance(data, dict):
        return [
            Diagnostic(
                severity="error",
                code="not-a-mapping",
                message=f"Frontmatter must be a YAML mapping, got {type(data).__name__}.",
                line=fm_start_line + 1,
                col=1,
                end_line=fm_start_line + 1,
                end_col=1,
            )
        ]

    diagnostics: list[Diagnostic] = []
    issues = validate_frontmatter(data)
    for issue in issues:
        diagnostics.append(_diagnostic_from_issue(issue, fm_lines, fm_start_line))

    # Info-level: unknown top-level keys (forward-compat)
    for key in data:
        if key not in _KNOWN_FRONTMATTER_KEYS:
            line = _find_key_line(fm_lines, key, fm_start_line)
            diagnostics.append(
                Diagnostic(
                    severity="info",
                    code="unknown-field",
                    message=f"Unknown frontmatter field {key!r} (preserved verbatim).",
                    line=line,
                    col=1,
                    end_line=line,
                    end_col=len(key) + 1,
                )
            )

    return diagnostics


def _diagnostic_from_issue(
    issue: FrontmatterIssue, fm_lines: list[str], fm_start_line: int
) -> Diagnostic:
    if issue.code == "missing-required":
        return Diagnostic(
            severity="error",
            code="missing-required",
            message=issue.message,
            line=fm_start_line,  # opening `---` line
            col=1,
            end_line=fm_start_line,
            end_col=4,
        )
    # All other issues — try to point at the offending key's line.
    key = next((p for p in issue.path if isinstance(p, str)), None)
    if key is None:
        line = fm_start_line + 1
        col = 1
        end_col = col + 1
    else:
        line = _find_key_line(fm_lines, key, fm_start_line)
        col = 1
        end_col = len(key) + 1
    severity: Severity = "error"
    return Diagnostic(
        severity=severity,
        code=issue.code,
        message=issue.message,
        line=line,
        col=col,
        end_line=line,
        end_col=end_col,
    )


def _find_key_line(fm_lines: list[str], key: str, fm_start_line: int) -> int:
    """Find the 1-indexed document line where ``key:`` appears in fm."""
    for idx, raw in enumerate(fm_lines):
        m = _TOP_LEVEL_KEY_RE.match(raw)
        if m and m.group(1) == key:
            return fm_start_line + 1 + idx
    return fm_start_line + 1


def _yaml_error_position(exc: yaml.YAMLError, fallback_line: int) -> tuple[int, int]:
    mark = getattr(exc, "problem_mark", None)
    if mark is None:
        return fallback_line, 1
    return fallback_line + mark.line, mark.column + 1


# -- body -----------------------------------------------------------


def _lint_body(lines: list[str], body_start_line: int, corpus: Any) -> list[Diagnostic]:
    alias_index = getattr(corpus, "alias_index", None)
    diagnostics: list[Diagnostic] = []

    in_fence = False
    depth_seen: list[tuple[int, int]] = []  # (line, depth_value)

    body_lines = lines[body_start_line - 1 :] if body_start_line - 1 < len(lines) else []
    for offset, raw in enumerate(body_lines):
        doc_line = body_start_line + offset

        if _FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        m = _DEPTH_RE.match(raw)
        if m:
            depth_seen.append((doc_line, int(m.group(2))))

        if alias_index is not None:
            for amatch in _ALIAS_REF_RE.finditer(raw):
                alias = amatch.group(1).strip()
                if alias and alias not in alias_index:
                    start_col = amatch.start() + 1
                    end_col = amatch.end() + 1
                    diagnostics.append(
                        Diagnostic(
                            severity="error",
                            code="broken-alias",
                            message=f"Unknown alias [[{alias}]] — no template declares it.",
                            line=doc_line,
                            col=start_col,
                            end_line=doc_line,
                            end_col=end_col,
                        )
                    )

    diagnostics.extend(_lint_depth_sequence(depth_seen))
    return diagnostics


def _lint_depth_sequence(depth_seen: list[tuple[int, int]]) -> list[Diagnostic]:
    if not depth_seen:
        return []
    diagnostics: list[Diagnostic] = []
    first_line, first_value = depth_seen[0]
    if first_value != 1:
        diagnostics.append(
            Diagnostic(
                severity="warning",
                code="depth-not-starting-at-one",
                message=f"Depth sections should start at 1; found Depth {first_value}.",
                line=first_line,
                col=1,
                end_line=first_line,
                end_col=1,
            )
        )
    expected = first_value + 1
    for line, value in depth_seen[1:]:
        if value < expected:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    code="depth-out-of-order",
                    message=(
                        f"Depth section out of order: "
                        f"expected {expected} or higher, found {value}."
                    ),
                    line=line,
                    col=1,
                    end_line=line,
                    end_col=1,
                )
            )
        elif value > expected:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    code="depth-skipped",
                    message=f"Depth gap: jumped from Depth {expected - 1} to Depth {value}.",
                    line=line,
                    col=1,
                    end_line=line,
                    end_col=1,
                )
            )
        expected = value + 1
    return diagnostics


__all__ = ["Diagnostic", "Severity", "lint_template", "SchemaError"]
