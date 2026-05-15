"""Cross-template rename refactor.

Two phases:

- :func:`plan_rename` builds a :class:`RenamePlan` describing every
  file that would change, with unified-diff hunks for review.
- :func:`apply_rename` writes the planned edits to disk atomically
  (best-effort: tempfile per file + sequential rename, with rollback
  to the original snapshot on partial failure) and refreshes the
  corpus index.

Supported kinds: ``alias`` and ``tag`` (text edits inside other
templates), and ``template_path`` (move a template file and update
the path-keyed ``summaries.json`` / ``summaries_by_path.json``
sidecar entry, if present).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._references import ReferenceKind

_FRONTMATTER_RE = re.compile(r"^(---\s*\n)(.*?)(\n---\s*\n)", re.DOTALL)
_FENCE_RE = re.compile(r"^(```|~~~)")

# Sidecar files we look for in the corpus root when planning a
# template_path rename. Both have the same flat ``rel_path -> summary``
# shape; ``summaries_by_path.json`` is the attune-help variant.
_PATH_KEYED_SIDECARS = ("summaries.json", "summaries_by_path.json")
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
class FileMove:
    """A planned file move (rename of a template's rel-path).

    The text of the file does not change. Sidecar text edits that
    follow the move (e.g. ``summaries.json``) live in
    :attr:`RenamePlan.edits`.
    """

    old_path: str
    new_path: str

    def to_dict(self) -> dict[str, Any]:
        return {"old_path": self.old_path, "new_path": self.new_path}


@dataclass(frozen=True)
class RenamePlan:
    old: str
    new: str
    kind: ReferenceKind
    edits: list[FileEdit] = field(default_factory=list)
    moves: list[FileMove] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "old": self.old,
            "new": self.new,
            "kind": self.kind,
            "edits": [e.to_dict() for e in self.edits],
            "moves": [m.to_dict() for m in self.moves],
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
        return _plan_template_path_rename(corpus, old, new)
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


def _plan_template_path_rename(corpus: Any, old: str, new: str) -> RenamePlan:
    """Plan a template-file move within the corpus root.

    Validates that the new path stays inside the corpus root and that
    no file already exists at the target. Reads path-keyed sidecars
    (``summaries.json`` / ``summaries_by_path.json``) in the corpus
    root and plans key-rename edits when present.
    """
    root = _corpus_root(corpus)
    if root is None:
        raise RenameError(
            "Corpus has no resolvable root path; template_path rename is not supported."
        )

    old_rel = _normalize_corpus_relpath(root, old)
    new_rel = _normalize_corpus_relpath(root, new)

    source = root / old_rel
    if not source.is_file():
        raise RenameError(f"Source template does not exist: {old_rel}")

    target = root / new_rel
    if target.exists():
        raise RenameCollisionError(new_rel, owning_path=new_rel)

    move = FileMove(old_path=old_rel, new_path=new_rel)
    edits = _plan_sidecar_path_rename_edits(root, old_rel, new_rel)
    return RenamePlan(old=old_rel, new=new_rel, kind="template_path", edits=edits, moves=[move])


def _normalize_corpus_relpath(root: Path, raw: str) -> str:
    """Validate ``raw`` is a non-empty corpus-relative posix path.

    Rejects absolute paths, ``..`` escapes, and empty strings. Returns
    the cleaned posix-style relative path.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty template path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError(f"Template path must be relative to the corpus root: {raw!r}")
    # Resolve relative to root WITHOUT following symlinks so we can
    # check containment. ``Path.resolve(strict=False)`` works on
    # nonexistent targets (we need that — target is a future file).
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"Template path escapes corpus root: {raw!r}") from exc
    return resolved.relative_to(root.resolve(strict=False)).as_posix()


def _plan_sidecar_path_rename_edits(root: Path, old_rel: str, new_rel: str) -> list[FileEdit]:
    """Build FileEdit entries for path-keyed sidecars that mention ``old_rel``.

    Sidecar files that don't exist, can't be parsed as JSON, or don't
    contain ``old_rel`` as a top-level key are skipped silently.
    """
    edits: list[FileEdit] = []
    for sidecar in _PATH_KEYED_SIDECARS:
        sidecar_path = root / sidecar
        if not sidecar_path.is_file():
            continue
        try:
            old_text = sidecar_path.read_text(encoding="utf-8")
            data = json.loads(old_text)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or old_rel not in data:
            continue
        renamed: dict[str, Any] = {}
        for key, value in data.items():
            renamed[new_rel if key == old_rel else key] = value
        new_text = json.dumps(renamed, indent=2, sort_keys=True) + "\n"
        if new_text == old_text:
            continue
        edits.append(_file_edit(sidecar, old_text, new_text))
    return edits


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

    Returns the list of relative paths that were actually written or
    moved. Best-effort atomic:

    - File moves run first via ``os.replace`` (atomic on the same
      filesystem). Parent directories are created as needed and
      tracked for rollback.
    - Text edits are then staged to tempfiles (drift detection runs
      here) and renamed into place sequentially.

    On any mid-flight failure, prior moves are reversed and prior
    edits are restored from in-memory snapshots before re-raising.
    """
    if not plan.edits and not plan.moves:
        _refresh_corpus(corpus)
        return []

    root = _corpus_root(corpus)
    if root is None:
        raise RenameError("Corpus has no resolvable root path; apply is not supported.")

    # 1) Apply moves first. Track what we did so we can reverse it.
    applied_moves: list[tuple[Path, Path]] = []  # (source_now_target, was_source)
    created_dirs: list[Path] = []
    written: list[str] = []
    try:
        for move in plan.moves:
            source = root / move.old_path
            target = root / move.new_path
            if not source.is_file():
                raise RenameError(f"Move source missing at apply time: {move.old_path}")
            if target.exists():
                raise RenameCollisionError(move.new_path, owning_path=move.new_path)
            created_dirs.extend(_ensure_parents(target, root))
            try:
                os.replace(source, target)
            except OSError as exc:
                raise RenameError(
                    f"Failed to move {move.old_path!r} -> {move.new_path!r}: {exc}"
                ) from exc
            applied_moves.append((target, source))
            written.append(move.new_path)
    except Exception:
        _undo_moves(applied_moves)
        _undo_created_dirs(created_dirs)
        raise

    # 2) Stage edits + drift-check. On failure here, also reverse moves.
    staged: list[tuple[Path, Path, str]] = []  # (target, tmp, original_text)
    try:
        for edit in plan.edits:
            target = root / edit.path
            if not target.exists():
                raise RenameError(f"Planned target does not exist: {target}")
            original = target.read_text(encoding="utf-8")
            if original != edit.old_text:
                raise RenameError(
                    f"File {edit.path!r} drifted from the planned base; rebuild plan."
                )
            tmp = _stage(target, edit.new_text)
            staged.append((target, tmp, original))
    except Exception:
        for _t, leftover, _o in staged:
            leftover.unlink(missing_ok=True)
        _undo_moves(applied_moves)
        _undo_created_dirs(created_dirs)
        raise

    # 3) Sequential rename of staged edits. On failure mid-loop,
    # restore prior edits AND reverse moves.
    for idx, (target, tmp, _original) in enumerate(staged):
        try:
            os.replace(tmp, target)
            written.append(str(target.relative_to(root).as_posix()))
        except OSError:
            for prev_target, _prev_tmp, prev_original in staged[:idx]:
                try:
                    prev_target.write_text(prev_original, encoding="utf-8")
                except OSError:
                    pass
            for _t, leftover, _o in staged[idx:]:
                leftover.unlink(missing_ok=True)
            _undo_moves(applied_moves)
            _undo_created_dirs(created_dirs)
            raise

    _refresh_corpus(corpus)
    return written


def _ensure_parents(target: Path, root: Path) -> list[Path]:
    """Create missing parents of ``target`` inside ``root``.

    Returns the directories created, deepest first, so they can be
    removed on rollback in reverse order.
    """
    created: list[Path] = []
    parent = target.parent
    chain: list[Path] = []
    while parent != root and not parent.exists():
        chain.append(parent)
        parent = parent.parent
    # Build from outermost-missing to innermost so mkdir succeeds.
    for d in reversed(chain):
        try:
            d.mkdir(parents=False, exist_ok=False)
            created.append(d)
        except FileExistsError:
            continue
    return list(reversed(created))


def _undo_moves(applied: list[tuple[Path, Path]]) -> None:
    """Reverse moves recorded by :func:`apply_rename`."""
    for now_at, was_at in reversed(applied):
        try:
            os.replace(now_at, was_at)
        except OSError:
            pass


def _undo_created_dirs(created: list[Path]) -> None:
    """Remove directories we created, deepest first, only if empty."""
    for d in created:
        try:
            d.rmdir()
        except OSError:
            pass


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
    "FileMove",
    "Hunk",
    "RenameCollisionError",
    "RenameError",
    "RenamePlan",
    "apply_rename",
    "plan_rename",
]
