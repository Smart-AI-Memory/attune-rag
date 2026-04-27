"""DirectoryCorpus — load markdown files from a directory."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from .base import CorpusProtocol, RetrievalEntry

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ALIASES_RE = re.compile(r"^\s*aliases\s*:\s*\[([^\]]*)\]", re.MULTILINE)


def _parse_aliases(content: str) -> tuple[str, ...]:
    """Extract aliases list from YAML frontmatter, e.g. aliases: [foo, bar]."""
    fm = _FRONTMATTER_RE.match(content)
    if not fm:
        return ()
    m = _ALIASES_RE.search(fm.group(1))
    if not m:
        return ()
    return tuple(a.strip().strip("'\"") for a in m.group(1).split(",") if a.strip())


logger = logging.getLogger(__name__)

DEFAULT_GLOB = "**/*.md"


class DirectoryCorpus(CorpusProtocol):
    """Load a directory of markdown files as a corpus.

    Walks ``root`` for ``*.md`` files and exposes them as
    ``RetrievalEntry`` records. Category is inferred from
    the first path segment relative to ``root``; paths in
    the root itself get category ``""``.

    Optionally loads a ``summaries.json`` mapping of
    relative paths to one-line summaries and a
    ``cross_links.json`` mapping of relative paths to
    related-paths lists.

    Rejects files whose resolved paths escape ``root`` to
    prevent path-traversal when summaries/cross-links feed
    arbitrary paths.
    """

    def __init__(
        self,
        root: Path,
        summaries_file: str | None = None,
        cross_links_file: str | None = None,
        extra_summaries: dict[str, str | None] | None = None,
        cache: bool = True,
        glob: str = DEFAULT_GLOB,
    ) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise ValueError(f"DirectoryCorpus root is not a directory: {self._root}")
        self._summaries_file = summaries_file
        self._cross_links_file = cross_links_file
        self._extra_summaries: dict[str, str | None] = extra_summaries or {}
        self._glob = glob
        self._cache = cache
        self._loaded: dict[str, RetrievalEntry] | None = None

    def _within_root(self, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(self._root)
            return True
        except ValueError:
            return False

    def _load_sidecar(self, filename: str | None) -> dict:
        if not filename:
            return {}
        path = self._root / filename
        if not path.is_file():
            return {}
        if not self._within_root(path):
            logger.warning("Sidecar outside corpus root: %s", path)
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
            return {}

    def _infer_category(self, relative: Path) -> str:
        parts = relative.parts
        if len(parts) <= 1:
            return ""
        return parts[0]

    def _build(self) -> dict[str, RetrievalEntry]:
        summaries = {**self._load_sidecar(self._summaries_file), **self._extra_summaries}
        cross_links = self._load_sidecar(self._cross_links_file)

        entries: dict[str, RetrievalEntry] = {}
        for md in sorted(self._root.glob(self._glob)):
            if not md.is_file():
                continue
            if not self._within_root(md):
                logger.warning("Skipping path outside root: %s", md)
                continue
            relative = md.relative_to(self._root)
            key = relative.as_posix()
            category = self._infer_category(relative)
            content = md.read_text(encoding="utf-8")
            related_raw = cross_links.get(key, ())
            related = tuple(r for r in related_raw if isinstance(r, str))
            entry = RetrievalEntry(
                path=key,
                category=category,
                content=content,
                summary=summaries.get(key),
                related=related,
                aliases=_parse_aliases(content),
            )
            entries[key] = entry
        return entries

    def _ensure_loaded(self) -> dict[str, RetrievalEntry]:
        if self._cache and self._loaded is not None:
            return self._loaded
        built = self._build()
        if self._cache:
            self._loaded = built
        return built

    def entries(self) -> Iterable[RetrievalEntry]:
        return tuple(self._ensure_loaded().values())

    def get(self, path: str) -> RetrievalEntry | None:
        return self._ensure_loaded().get(path)

    @property
    def name(self) -> str:
        return f"directory:{self._root.name}"

    @property
    def version(self) -> str:
        entries = self._ensure_loaded()
        hasher = hashlib.sha256()
        for key in sorted(entries):
            entry = entries[key]
            hasher.update(key.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(entry.content.encode("utf-8"))
            hasher.update(b"\0\0")
        return hasher.hexdigest()[:16]
