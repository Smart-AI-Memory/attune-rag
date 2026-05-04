"""Cross-template rename refactor.

Two phases:

- :func:`plan_rename` builds a :class:`RenamePlan` describing every
  file that would change, with unified-diff hunks for review.
- :func:`apply_rename` writes the planned edits to disk atomically
  (best-effort: tempfile per file + sequential rename, with rollback
  to the original snapshot on partial failure) and refreshes the
  corpus index.

Supported kinds in v1: ``alias`` and ``tag``. ``template_path`` rename
(moving a file + updating cross-links) is reserved for a follow-up.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._references import ReferenceKind

_FRONTMATTER_RE = re.compile(r"^(---\s*\n)(.*?)(\n---\s*\n)", re.DOTALL)
_FENCE_RE = re.compile(r"^(```|~~~)")
_ALIAS_REF_RE = re.compile(r"(?<!\\)\[\[([^\[\]\n]+?)\]\]")


class RenameError(ValueError):
    """Base class for rename refactor failures."""


class RenameCollisionError(RenameError):
    """Raised when the proposed new name already exists.

    Carries the conflicting template path so the editor can show which
    template owns the name.
    """

    def __init__(self, name: str, owning_path: str) -> None:
        self.name = name
        self.owning_path = owning_path
        super().__init__(f"Cannot rename to {name!r}: already declared by {owning_path!r}")


@dataclass(frozen=True)
class Hunk:
    """A single unified-diff hunk."""

    hunk_id: str  # stable sha256-based identifier
    header: str  # canonical @@ hunk header
    lines: list[str] = field(default_factory=list)  # `' ' / '-' / '+'` prefixed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FileEdit:
    """A planned edit to a single template file."""

    path: str
    old_text: str
    new_text: str
    hunks: list[Hunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "hunks": [h.to_dict() for h in self.hunks],
        }


@dataclass(frozen=True)
class RenamePlan:
    old: str
    new: str
    kind: ReferenceKind
    edits: list[FileEdit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "old": self.old,
            "new": self.new,
            "kind": self.kind,
            "edits": [e.to_dict() for e in self.edits],
        }


# -- planning -------------------------------------------------------


def plan_rename(corpus: Any, old: str, new: str, kind: ReferenceKind) -> RenamePlan:
    """Compute a :class:`RenamePlan` for renaming ``old`` to ``new``.

    Returns a plan even if no references exist (a no-op plan with an
    empty ``edits`` list). Raises :class:`RenameCollisionError` if
    ``new`` is already taken (alias kind only).
    """
    if old == new:
        return RenamePlan(old=old, new=new, kind=kind, edits=[])

    if kind == "alias":
        return _plan_alias_rename(corpus, old, new)
    if kind == "tag":
        return _plan_tag_rename(corpus, old, new)
    if kind == "template_path":
        raise NotImplementedError(
            "template_path rename is reserved for a future spec; v1 supports alias + tag."
        )
    raise ValueError(f"Unsupported rename kind: {kind!r}")


def _plan_alias_rename(corpus: Any, old: str, new: str) -> RenamePlan:
    alias_index = getattr(corpus, "alias_index", None)
    if isinstance(alias_index, dict) and new in alias_index:
        owner = alias_index[new]
        owning_path = owner["template_path"] if isinstance(owner, dict) else str(owner)
        raise RenameCollisionError(new, owning_path)

    edits: list[FileEdit] = []
    for entry in _iter_entries(corpus):
        old_text = getattr(entry, "content", "")
        path = getattr(entry, "path", "")
        if not old_text or not path:
            continue
        new_text = _rewrite_alias_in_text(old_text, old, new)
        if new_text == old_text:
            continue
        edits.append(_file_edit(path, old_text, new_text))
    return RenamePlan(old=old, new=new, kind="alias", edits=edits)


def _plan_tag_rename(corpus: Any, old: str, new: str) -> RenamePlan:
    edits: list[FileEdit] = []
    for entry in _iter_entries(corpus):
        old_text = getattr(entry, "content", "")
        path = getattr(entry, "path", "")
        if not old_text or not path:
            continue
        new_text = _rewrite_tag_in_text(old_text, old, new)
        if new_text == old_text:
            continue
        edits.append(_file_edit(path, old_text, new_text))
    return RenamePlan(old=old, new=new, kind="tag", edits=edits)


# -- rewrites -------------------------------------------------------


def _rewrite_alias_in_text(text: str, old: str, new: str) -> str:
    """Rewrite both frontmatter declarations and body refs of an alias."""
    text = _rewrite_frontmatter_value(text, "aliases", old, new)
    return _rewrite_alias_body_refs(text, old, new)


def _rewrite_tag_in_text(text: str, old: str, new: str) -> str:
    return _rewrite_frontmatter_value(text, "tags", old, new)


def _rewrite_frontmatter_value(text: str, key: str, old: str, new: str) -> str:
    """Rewrite occurrences of ``old`` under top-level ``key:`` in fm.

    Only replaces standalone tokens — substring matches of ``old``
    within longer values are left alone.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text
    fm_open, fm_body, fm_close = m.group(1), m.group(2), m.group(3)
    rewritten_body = _rewrite_yaml_block_value(fm_body, key, old, new)
    if rewritten_body == fm_body:
        return text
    return fm_open + rewritten_body + fm_close + text[m.end() :]


_TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:")


def _rewrite_yaml_block_value(fm_body: str, key: str, old: str, new: str) -> str:
    """Rewrite ``old`` -> ``new`` inside the ``key:`` list (flow or block)."""
    lines = fm_body.split("\n")
    out: list[str] = []
    in_target_block = False
    target_indent: int | None = None
    for raw in lines:
        m = _TOP_LEVEL_KEY_RE.match(raw)
        if m:
            if in_target_block and m.group(1) != key:
                in_target_block = False
                target_indent = None
            if m.group(1) == key:
                in_target_block = True
                target_indent = len(raw) - len(raw.lstrip())
                # Flow-style values on the same line get rewritten too.
                head = raw[: m.end()]
                tail = raw[m.end() :]
                tail = _rewrite_flow_token(tail, old, new)
                out.append(head + tail)
                continue
        if in_target_block and target_indent is not None:
            stripped = raw.lstrip()
            indent = len(raw) - len(stripped)
            if stripped == "":
                out.append(raw)
                continue
            if indent <= target_indent:
                in_target_block = False
                target_indent = None
                out.append(raw)
                continue
            list_match = re.match(r"(-\s*)(.*)$", stripped)
            if list_match:
                token = list_match.group(2).strip().strip("'\"")
                if token == old:
                    leading = raw[: indent + len(list_match.group(1))]
                    out.append(leading + new)
                    continue
        out.append(raw)
    return "\n".join(out)


def _rewrite_flow_token(text: str, old: str, new: str) -> str:
    """Rewrite ``old`` token in a flow-style list, e.g. ``[a, old, b]``."""

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        bare = token.strip("'\"")
        return new if bare == old else token

    return re.sub(r"([^,\[\]\s]+)", _replace, text)


def _rewrite_alias_body_refs(text: str, old: str, new: str) -> str:
    """Rewrite ``[[old]]`` -> ``[[new]]`` in the body, excluding fenced
    code blocks and escaped ``\\[[`` sequences."""
    fm_match = _FRONTMATTER_RE.match(text)
    body_offset = fm_match.end() if fm_match else 0
    head = text[:body_offset]
    body = text[body_offset:]

    body_lines = body.split("\n")
    in_fence = False
    rewritten: list[str] = []
    for raw in body_lines:
        if _FENCE_RE.match(raw):
            in_fence = not in_fence
            rewritten.append(raw)
            continue
        if in_fence:
            rewritten.append(raw)
            continue

        def _replace(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            return f"[[{new}]]" if inner == old else match.group(0)

        rewritten.append(_ALIAS_REF_RE.sub(_replace, raw))
    return head + "\n".join(rewritten)


# -- diff / hunks ---------------------------------------------------


def _file_edit(path: str, old_text: str, new_text: str) -> FileEdit:
    return FileEdit(
        path=path,
        old_text=old_text,
        new_text=new_text,
        hunks=_hunks(path, old_text, new_text),
    )


_HUNK_HEADER_RE = re.compile(r"^@@.*?@@")


def _hunks(path: str, old_text: str, new_text: str) -> list[Hunk]:
    """Compute unified-diff hunks between ``old_text`` and ``new_text``."""
    diff = list(
        difflib.unified_diff(
            old_text.splitlines(keepends=False),
            new_text.splitlines(keepends=False),
            fromfile=path,
            tofile=path,
            n=3,
            lineterm="",
        )
    )
    # Skip the first two header lines ("--- a", "+++ b").
    body = diff[2:] if len(diff) > 2 else []
    hunks: list[Hunk] = []
    current: list[str] | None = None
    current_header: str | None = None
    for line in body:
        if line.startswith("@@"):
            if current is not None and current_header is not None:
                hunks.append(_make_hunk(current_header, current, path))
            current_header = line
            current = []
        elif current is not None:
            current.append(line)
    if current is not None and current_header is not None:
        hunks.append(_make_hunk(current_header, current, path))
    return hunks


def _make_hunk(header: str, lines: list[str], path: str) -> Hunk:
    digest = hashlib.sha256(("\0".join([path, header, *lines])).encode("utf-8")).hexdigest()[:16]
    return Hunk(hunk_id=digest, header=header, lines=lines)


# -- apply ----------------------------------------------------------


def apply_rename(corpus: Any, plan: RenamePlan) -> list[str]:
    """Apply ``plan`` to disk and refresh the corpus.

    Returns the list of relative paths that were actually written.
    Best-effort atomic: each file is staged to a tempfile in the same
    directory, then renamed into place sequentially. If a later rename
    fails, earlier files are restored from their in-memory snapshots.
    """
    if not plan.edits:
        _refresh_corpus(corpus)
        return []

    root = _corpus_root(corpus)
    if root is None:
        raise RenameError("Corpus has no resolvable root path; apply is not supported.")

    staged: list[tuple[Path, Path, str]] = []  # (target, tmp, original_text)
    written: list[str] = []
    for edit in plan.edits:
        target = root / edit.path
        if not target.exists():
            raise RenameError(f"Planned target does not exist: {target}")
        original = target.read_text(encoding="utf-8")
        if original != edit.old_text:
            raise RenameError(f"File {edit.path!r} drifted from the planned base; rebuild plan.")
        tmp = _stage(target, edit.new_text)
        staged.append((target, tmp, original))

    # Sequential rename; on failure, restore originals from snapshots.
    for idx, (target, tmp, _original) in enumerate(staged):
        try:
            os.replace(tmp, target)
            written.append(str(target.relative_to(root).as_posix()))
        except OSError:
            # Restore previously-renamed files from snapshots.
            for prev_target, _prev_tmp, prev_original in staged[:idx]:
                try:
                    prev_target.write_text(prev_original, encoding="utf-8")
                except OSError:
                    pass
            # Clean up untouched tempfiles.
            for _t, leftover, _o in staged[idx:]:
                leftover.unlink(missing_ok=True)
            raise

    _refresh_corpus(corpus)
    return written


def _stage(target: Path, new_text: str) -> Path:
    """Write ``new_text`` to a tempfile next to ``target`` and return it."""
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_text)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)


def _refresh_corpus(corpus: Any) -> None:
    """Best-effort cache invalidation so subsequent reads reflect disk."""
    for attr in ("_loaded", "_aliases"):
        if hasattr(corpus, attr):
            setattr(corpus, attr, None)


def _corpus_root(corpus: Any) -> Path | None:
    root = getattr(corpus, "_root", None)
    if isinstance(root, Path):
        return root
    return None


def _iter_entries(corpus: Any):
    path_index = getattr(corpus, "path_index", None)
    if isinstance(path_index, dict):
        return iter(path_index.values())
    entries_fn = getattr(corpus, "entries", None)
    if callable(entries_fn):
        return iter(entries_fn())
    return iter(())


__all__ = [
    "FileEdit",
    "Hunk",
    "RenameCollisionError",
    "RenameError",
    "RenamePlan",
    "apply_rename",
    "plan_rename",
]
