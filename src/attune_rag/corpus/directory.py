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

# Fraction of aliased entries that must be "alias-degraded" (every alias
# too short to ever satisfy MIN_ALIAS_OVERLAP) before the build-time
# warning fires. Tunable without an API change. A single degraded entry
# in a small corpus still warns (the max(1, ...) floor); 10%+ is the
# corpus-shape signal for larger corpora. See
# docs/specs/alias-overlap-remediation/.
_ALIAS_WARN_FRACTION = 0.10


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
        extra_aliases: dict[str, Iterable[str]] | None = None,
        extra_aliases_file: Path | str | None = None,
        cache: bool = True,
        glob: str = DEFAULT_GLOB,
        warn_alias_overlap: bool = True,
    ) -> None:
        """Configure the corpus root and optional sidecars.

        Args:
            root: Directory containing the markdown corpus.
                Must exist; resolved to an absolute path. Path
                escapes (``..``) outside this root are rejected
                at load time.
            summaries_file: Optional filename (relative to
                ``root``) of a path-keyed summaries sidecar
                (e.g. ``"summaries.json"``). Schema:
                ``{rel_path: summary_string | null}``. When
                ``None``, no summaries are loaded.
            cross_links_file: Optional filename of a path-keyed
                cross-links sidecar. Schema:
                ``{rel_path: [related_rel_path, ...]}``. Feeds
                :attr:`RetrievalEntry.related`. When ``None``,
                no cross-links are loaded.
            extra_summaries: In-memory summaries that override
                or supplement the sidecar (sidecar first, this
                dict layered on top). Useful for tests and for
                packages that ship their own summary table
                programmatically.
            extra_aliases: In-memory ``rel_path -> [alias, ...]``
                mapping that **appends** aliases to each entry's
                frontmatter aliases. Used by the alias-expansion
                override path so that aliases authored downstream
                (e.g. in ``attune-rag``) don't require an upstream
                ``attune-help`` release to take effect. Appends only
                — frontmatter aliases are the canonical source.
                Each appended alias must still be globally unique
                across the corpus per the duplicate-alias rule;
                duplicates raise :class:`DuplicateAliasError` at
                load time. Within-template duplicates (an extra
                alias that already exists in frontmatter for the
                same path) are silently deduplicated.
            cache: If True (the default), the loaded entry
                dict is memoized on the instance. Set False to
                re-scan disk on every call — useful in long-
                running processes where the corpus changes.
            glob: Shell glob for files to include. Default
                ``"**/*.md"`` (all markdown, recursive).
            warn_alias_overlap: When True (default), emit a single
                ``logging.warning`` at build time if a meaningful
                share of aliased entries have only single-token
                aliases that can never satisfy
                ``KeywordRetriever.MIN_ALIAS_OVERLAP`` (observability
                only; retrieval is unchanged). Set False to silence,
                or raise the logger ``attune_rag.corpus.directory``
                to ERROR. See docs/specs/alias-overlap-remediation/.

        Raises:
            ValueError: When ``root`` is not an existing
                directory.
        """
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise ValueError(f"DirectoryCorpus root is not a directory: {self._root}")
        self._summaries_file = summaries_file
        self._cross_links_file = cross_links_file
        self._extra_summaries: dict[str, str | None] = extra_summaries or {}
        # Merge file-sourced extra aliases with the inline dict. Inline
        # entries win on per-path collision so callers can override
        # specific paths after loading a base file. File loading uses
        # strict semantics (typed errors with path in message); the
        # tolerance bundled AttuneHelpCorpus needs for its own override
        # file is in its own wrapper.
        from ._aliases import load_aliases_from_file as _load_aliases

        file_aliases: dict[str, list[str]] = {}
        if extra_aliases_file is not None:
            file_aliases = _load_aliases(extra_aliases_file)
        merged_aliases: dict[str, Iterable[str]] = dict(file_aliases)
        if extra_aliases:
            merged_aliases.update(extra_aliases)  # inline wins on collision
        # Normalize extras to tuples of strings at construction time so
        # _build() doesn't have to re-validate per-rebuild.
        self._extra_aliases: dict[str, tuple[str, ...]] = {
            path: tuple(a for a in aliases if isinstance(a, str) and a)
            for path, aliases in merged_aliases.items()
        }
        self._glob = glob
        self._cache = cache
        # Build-time alias-overlap warning (observability only; see
        # docs/specs/alias-overlap-remediation/). Latched so it fires
        # at most once per corpus instance.
        self._warn_alias_overlap = warn_alias_overlap
        self._alias_warning_emitted = False
        self._loaded: dict[str, RetrievalEntry] | None = None
        self._aliases: dict[str, AliasInfo] | None = None
        # Cached SHA-256 fingerprint of the loaded corpus. Invalidated
        # whenever ``_loaded`` is cleared or rebuilt so it always reflects
        # the entries currently in memory.
        self._version: str | None = None

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
            frontmatter_aliases = _aliases_from_frontmatter(frontmatter)
            # Append extras from extra_aliases (appended after frontmatter
            # so the canonical source still comes first). Within-template
            # duplicates are silently dropped so an override author can
            # safely re-list an alias the frontmatter already declares.
            extras = self._extra_aliases.get(key, ())
            seen_in_template = set(frontmatter_aliases)
            extra_unique: list[str] = []
            for alias in extras:
                if alias in seen_in_template:
                    continue
                seen_in_template.add(alias)
                extra_unique.append(alias)
            aliases = frontmatter_aliases + tuple(extra_unique)
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
        self._warn_if_alias_degraded(entries)
        return entries, alias_index

    def _warn_if_alias_degraded(
        self, entries: dict[str, RetrievalEntry]
    ) -> None:
        """Warn once if aliased entries can't satisfy the overlap floor.

        Observability only — does not touch retrieval scoring. Detects the
        silent failure where a corpus's aliases are all single-token and
        therefore contribute zero alias signal under
        ``KeywordRetriever.MIN_ALIAS_OVERLAP >= 2``. See
        docs/specs/alias-overlap-remediation/.
        """
        if not self._warn_alias_overlap or self._alias_warning_emitted:
            return
        # Lazy import: retrieval has no corpus dependency, but importing
        # here keeps corpus import-light and avoids any load-order coupling.
        from ..retrieval import KeywordRetriever, _tokenize

        floor = KeywordRetriever.MIN_ALIAS_OVERLAP
        if floor <= 1:
            return  # floor is inert; single-token aliases still count

        aliased = 0
        degraded = 0
        for entry in entries.values():
            if not entry.aliases:
                continue
            aliased += 1
            reachable = any(len(_tokenize(a)) >= floor for a in entry.aliases)
            if not reachable:
                degraded += 1

        if degraded == 0:
            return
        threshold = max(1, int(_ALIAS_WARN_FRACTION * aliased))
        if degraded < threshold:
            return

        self._alias_warning_emitted = True
        logger.warning(
            "%d of %d aliased entries have only single-token aliases and "
            "will contribute zero alias signal under MIN_ALIAS_OVERLAP=%d. "
            "If your corpus uses single-word aliases, set "
            "MIN_ALIAS_OVERLAP=1 (see USER_CORPUS_GUIDE section 4.2) or "
            "author multi-token aliases.",
            degraded,
            aliased,
            floor,
        )

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
        # Any rebuild invalidates the version fingerprint — even when
        # caching is off, the previous cached hash no longer matches
        # the entries we just produced.
        self._version = None
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
        """Stable SHA-256 fingerprint of the loaded corpus.

        Cached after the first computation and invalidated whenever
        ``_ensure_loaded`` rebuilds. The previous implementation hashed
        the entire corpus on every call, which made this property an
        unwitting hot path for any consumer using ``version`` as a cache
        key (every API request in attune-gui's RAG route, for example).
        """
        entries = self._ensure_loaded()
        if self._version is not None:
            return self._version
        hasher = hashlib.sha256()
        for key in sorted(entries):
            entry = entries[key]
            hasher.update(key.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(entry.content.encode("utf-8"))
            hasher.update(b"\0\0")
        self._version = hasher.hexdigest()[:16]
        return self._version
