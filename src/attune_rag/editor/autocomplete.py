"""Autocomplete providers for the template editor.

Two functions:

- :func:`autocomplete_tags` — prefix-matched tags ranked by corpus
  frequency, then alphabetically.
- :func:`autocomplete_aliases` — prefix-matched aliases ranked by
  lexical proximity to the prefix (shortest first, then alphabetical).

Both are case-insensitive on the prefix. Suggestions preserve the
original casing they appeared with in the corpus.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..corpus import AliasInfo


def autocomplete_tags(corpus: Any, prefix: str, limit: int = 50) -> list[str]:
    """Return up to ``limit`` tag suggestions starting with ``prefix``.

    Ranking: descending corpus frequency (how many templates declare
    the tag), then alphabetical for ties. Case-insensitive prefix match
    on a case-preserving display value.
    """
    counts: Counter[str] = Counter()
    for entry in _iter_entries(corpus):
        fm = entry.metadata.get("frontmatter") or {}
        tags = fm.get("tags")
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag:
                counts[tag] += 1

    needle = prefix.casefold()
    matched = [tag for tag in counts if tag.casefold().startswith(needle)]
    matched.sort(key=lambda t: (-counts[t], t.casefold()))
    return matched[:limit]


def autocomplete_aliases(corpus: Any, prefix: str, limit: int = 50) -> list[AliasInfo]:
    """Return up to ``limit`` alias suggestions starting with ``prefix``.

    Ranking: shortest match first (closest to the prefix), then
    alphabetical for ties. Case-insensitive prefix match.

    Returns full :class:`AliasInfo` records so the editor can show the
    template name alongside the alias in autocomplete UI.
    """
    alias_index = getattr(corpus, "alias_index", None)
    if not isinstance(alias_index, dict):
        return []
    needle = prefix.casefold()
    matched = [info for alias, info in alias_index.items() if alias.casefold().startswith(needle)]
    matched.sort(key=lambda info: (len(info["alias"]), info["alias"].casefold()))
    return matched[:limit]


def _iter_entries(corpus: Any):
    """Best-effort iterator over corpus entries.

    Prefers ``path_index`` (avoids the protocol's iterable conversion)
    and falls back to ``entries()`` for any conforming
    ``CorpusProtocol``.
    """
    path_index = getattr(corpus, "path_index", None)
    if isinstance(path_index, dict):
        return iter(path_index.values())
    entries_fn = getattr(corpus, "entries", None)
    if callable(entries_fn):
        return iter(entries_fn())
    return iter(())


__all__ = ["autocomplete_aliases", "autocomplete_tags"]
