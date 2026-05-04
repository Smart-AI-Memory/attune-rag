"""DirectoryCorpus — load markdown files from a directory."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from .base import AliasInfo, CorpusProtocol, DuplicateAliasError, RetrievalEntry

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from a markdown file.

    Returns an empty dict when there's no frontmatter or the YAML is
    malformed. Malformed frontmatter is *tolerated* at corpus-load time
    (logged, not raised) so a single broken template can't break the
    whole corpus. The editor's lint pass surfaces parse errors per-file.
    """
    fm = _FRONTMATTER_RE.match(content)
    if not fm:
        return {}
    try:
        data = yaml.safe_load(fm.group(1))
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse frontmatter: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _aliases_from_frontmatter(fm: dict[str, Any]) -> tuple[str, ...]:
    raw = fm.get("aliases")
    if not isinstance(raw, list):
        return ()
    return tuple(a for a in raw if isinstance(a, str) and a)


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

    Indexes:

    - ``path_index`` — ``rel_path -> RetrievalEntry``, keyed by
      corpus-relative posix path.
    - ``alias_index`` — ``alias -> AliasInfo``, built from each
      template's ``aliases`` frontmatter list. Aliases must be globally
      unique across the corpus; duplicates raise
      :class:`DuplicateAliasError` at load time so the editor can
      surface the conflict before any retrieval happens.
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
        self._aliases: dict[str, AliasInfo] | None = None

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

    def _build(self) -> tuple[dict[str, RetrievalEntry], dict[str, AliasInfo]]:
        summaries = {**self._load_sidecar(self._summaries_file), **self._extra_summaries}
        cross_links = self._load_sidecar(self._cross_links_file)

        entries: dict[str, RetrievalEntry] = {}
        alias_index: dict[str, AliasInfo] = {}

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
            frontmatter = _parse_frontmatter(content)
            aliases = _aliases_from_frontmatter(frontmatter)
            fm_name = frontmatter.get("name")
            template_name = fm_name if isinstance(fm_name, str) else relative.stem

            for alias in aliases:
                if alias in alias_index:
                    raise DuplicateAliasError(
                        alias=alias,
                        first_path=alias_index[alias]["template_path"],
                        second_path=key,
                    )
                alias_index[alias] = AliasInfo(
                    alias=alias, template_path=key, template_name=template_name
                )

            entry = RetrievalEntry(
                path=key,
                category=category,
                content=content,
                summary=summaries.get(key),
                related=related,
                aliases=aliases,
                metadata={"frontmatter": frontmatter},
            )
            entries[key] = entry
        return entries, alias_index

    def _ensure_loaded(self) -> dict[str, RetrievalEntry]:
        if self._cache and self._loaded is not None:
            return self._loaded
        built_entries, built_aliases = self._build()
        if self._cache:
            self._loaded = built_entries
        # Even without caching the entries themselves we keep the alias
        # index in sync with the most recent build so alias_index reads
        # are coherent with the entries the caller just observed.
        self._aliases = built_aliases
        return built_entries

    def entries(self) -> Iterable[RetrievalEntry]:
        return tuple(self._ensure_loaded().values())

    def get(self, path: str) -> RetrievalEntry | None:
        return self._ensure_loaded().get(path)

    @property
    def path_index(self) -> dict[str, RetrievalEntry]:
        """``rel_path -> RetrievalEntry`` for every loaded template."""
        return dict(self._ensure_loaded())

    @property
    def alias_index(self) -> dict[str, AliasInfo]:
        """``alias -> AliasInfo`` for every alias declared in the corpus."""
        self._ensure_loaded()
        assert self._aliases is not None
        return dict(self._aliases)

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
