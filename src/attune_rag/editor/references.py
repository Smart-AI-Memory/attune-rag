"""Find references to an alias, tag, or template path across a corpus.

Used by the rename-refactor flow (task #6) and by editor UI features
that need a "where is this used?" query (planned for v2).

Public entry point: :func:`find_references`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

ReferenceContext = Literal["body", "frontmatter.alias", "frontmatter.tag", "cross_links"]
ReferenceKind = Literal["alias", "tag", "template_path"]


@dataclass(frozen=True)
class Reference:
    """A single reference to a name in a corpus.

    Line/column are 1-indexed and refer to the *original* template
    text. For ``cross_links`` references (path mentioned in a
    cross-links sidecar rather than in a template body), line and col
    are sentinel ``1`` values — the rename refactor rewrites the
    sidecar JSON, not specific lines.
    """

    template_path: str
    line: int
    col: int
    context: ReferenceContext

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FENCE_RE = re.compile(r"^(```|~~~)")
_TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:")


def find_references(corpus: Any, name: str, kind: ReferenceKind) -> list[Reference]:
    """Return every reference to ``name`` across the corpus.

    Behavior by kind:

    - ``alias``: frontmatter ``aliases:`` declarations *plus* body
      ``[[alias]]`` references (excluding fenced code blocks and
      ``\\[[`` escapes).
    - ``tag``: frontmatter ``tags:`` occurrences.
    - ``template_path``: cross-links references (entries whose
      ``related`` tuple contains the path).
    """
    if kind == "alias":
        return _find_alias_refs(corpus, name)
    if kind == "tag":
        return _find_tag_refs(corpus, name)
    if kind == "template_path":
        return _find_path_refs(corpus, name)
    raise ValueError(f"Unsupported reference kind: {kind!r}")


# -- alias ----------------------------------------------------------


_ALIAS_REF_RE = re.compile(r"(?<!\\)\[\[([^\[\]\n]+?)\]\]")


def _find_alias_refs(corpus: Any, name: str) -> list[Reference]:
    refs: list[Reference] = []
    for entry in _iter_entries(corpus):
        text = getattr(entry, "content", "")
        path = getattr(entry, "path", "")
        if not text or not path:
            continue
        refs.extend(_alias_decl_refs(text, path, name))
        refs.extend(_alias_body_refs(text, path, name))
    return refs


def _alias_decl_refs(text: str, path: str, name: str) -> list[Reference]:
    """Find frontmatter declarations of ``name`` as an alias."""
    fm_text, fm_start_line = _frontmatter_slice(text)
    if not fm_text:
        return []
    line_refs = _find_value_in_block(fm_text, "aliases", name, fm_start_line)
    return [
        Reference(template_path=path, line=line, col=col, context="frontmatter.alias")
        for line, col in line_refs
    ]


def _alias_body_refs(text: str, path: str, name: str) -> list[Reference]:
    """Find ``[[name]]`` body references, excluding fenced code blocks."""
    refs: list[Reference] = []
    body_start_line = _body_start_line(text)
    lines = text.splitlines()
    body_lines = lines[body_start_line - 1 :] if body_start_line - 1 < len(lines) else []
    in_fence = False
    for offset, raw in enumerate(body_lines):
        if _FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _ALIAS_REF_RE.finditer(raw):
            if match.group(1).strip() == name:
                refs.append(
                    Reference(
                        template_path=path,
                        line=body_start_line + offset,
                        col=match.start() + 1,
                        context="body",
                    )
                )
    return refs


# -- tag ------------------------------------------------------------


def _find_tag_refs(corpus: Any, name: str) -> list[Reference]:
    refs: list[Reference] = []
    for entry in _iter_entries(corpus):
        text = getattr(entry, "content", "")
        path = getattr(entry, "path", "")
        if not text or not path:
            continue
        fm_text, fm_start_line = _frontmatter_slice(text)
        if not fm_text:
            continue
        for line, col in _find_value_in_block(fm_text, "tags", name, fm_start_line):
            refs.append(
                Reference(template_path=path, line=line, col=col, context="frontmatter.tag")
            )
    return refs


# -- template_path --------------------------------------------------


def _find_path_refs(corpus: Any, name: str) -> list[Reference]:
    """Find cross-links pointing at ``name``.

    Uses ``entry.related`` (loaded from the corpus's cross-links
    sidecar). Line/col are sentinel ``1`` because the canonical source
    is a JSON file, not a template line; the rename refactor rewrites
    the sidecar wholesale.
    """
    refs: list[Reference] = []
    for entry in _iter_entries(corpus):
        related = getattr(entry, "related", ()) or ()
        path = getattr(entry, "path", "")
        if name in related and path:
            refs.append(Reference(template_path=path, line=1, col=1, context="cross_links"))
    return refs


# -- helpers --------------------------------------------------------


def _iter_entries(corpus: Any):
    path_index = getattr(corpus, "path_index", None)
    if isinstance(path_index, dict):
        return iter(path_index.values())
    entries_fn = getattr(corpus, "entries", None)
    if callable(entries_fn):
        return iter(entries_fn())
    return iter(())


def _frontmatter_slice(text: str) -> tuple[str, int]:
    """Return (frontmatter_text, fm_start_line). 1-indexed start line."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return "", 1
    return m.group(1), 1


def _body_start_line(text: str) -> int:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return 1
    return text[: m.end()].count("\n") + 1


def _find_value_in_block(
    fm_text: str, key: str, value: str, fm_start_line: int
) -> list[tuple[int, int]]:
    """Find lines where ``value`` appears under top-level ``key:`` in fm.

    Handles both flow style (``aliases: [a, b]``) and block style
    (``aliases:\\n  - a``). Returns list of ``(doc_line, col)`` 1-indexed.
    """
    lines = fm_text.splitlines()
    found_key = False
    key_indent: int | None = None
    results: list[tuple[int, int]] = []
    for idx, raw in enumerate(lines):
        m = _TOP_LEVEL_KEY_RE.match(raw)
        if m:
            if found_key and m.group(1) != key:
                # Reached the next top-level key; stop scanning the
                # block we were in.
                break
            if m.group(1) == key:
                found_key = True
                key_indent = len(raw) - len(raw.lstrip())
                # Flow-style values live on the same line as the key.
                tail = raw[m.end() :]
                for col_in_tail in _find_token_cols(tail, value):
                    col = m.end() + col_in_tail + 1  # +1 → 1-indexed
                    results.append((fm_start_line + idx + 1, col))
                continue
        if found_key and key_indent is not None:
            # Block-style continuation: lines deeper than the key indent.
            stripped = raw.lstrip()
            indent = len(raw) - len(stripped)
            if stripped == "":
                continue
            if indent <= key_indent:
                break
            # Match `- value` items.
            list_match = re.match(r"-\s*(.*)$", stripped)
            if list_match:
                token = list_match.group(1).strip().strip("'\"")
                if token == value:
                    col = indent + 1 + len("- ")
                    results.append((fm_start_line + idx + 1, col))
    return results


def _find_token_cols(text: str, value: str) -> list[int]:
    """Find 0-indexed columns where bare ``value`` appears as a list token.

    Strips quotes and brackets; matches whole-token only.
    """
    cols: list[int] = []
    # Tokens are separated by `,`, `[`, `]`, whitespace.
    # Walk the string, segment it.
    pattern = re.compile(r"([^,\[\]\s]+)")
    for match in pattern.finditer(text):
        token = match.group(1).strip("'\"")
        if token == value:
            cols.append(match.start())
    return cols


__all__ = ["Reference", "ReferenceContext", "ReferenceKind", "find_references"]
