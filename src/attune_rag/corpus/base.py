"""CorpusProtocol + RetrievalEntry. Implementation in task 1.2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable


@dataclass(frozen=True)
class RetrievalEntry:
    """A single corpus entry.

    Attributes:
        path: Corpus-relative posix path (e.g.
            ``"concepts/tool-fix-test.md"``). The unique key
            used by ``CorpusProtocol.get``,
            ``Corpus.path_index``, and as the value of every
            ``AliasInfo.template_path``.
        category: First path segment (e.g. ``"concepts"``,
            ``"tasks"``). Drives the per-category scoring
            weight in :class:`KeywordRetriever`.
        content: The full body text of the markdown file
            (frontmatter is parsed out into ``metadata`` /
            ``aliases`` and is not included here).
        summary: Optional one-line summary, typically sourced
            from a corpus-root ``summaries.json`` sidecar.
            ``None`` when no summary is registered.
        related: Tuple of corpus-relative paths cross-linked
            from this entry, sourced from ``cross_links.json``.
            Used by the retriever's "related hits" signal.
        aliases: Tuple of alias strings declared in the entry's
            frontmatter ``aliases`` list. Each alias must be
            globally unique within a corpus (enforced at load
            time via :class:`DuplicateAliasError`).
        metadata: The remaining frontmatter as a plain dict.
            Free-form; consumers should treat keys as optional.

    Mutability note: the ``_tokens_cache`` field is a
    per-instance mutable cache used by
    :mod:`attune_rag.retrieval` to memoize tokenized
    representations (path / summary / content-preview /
    aliases) so the keyword retriever doesn't re-tokenize on
    every query. ``frozen=True`` prevents the field itself from
    being reassigned but doesn't stop callers from mutating its
    contents — exactly what we want for a write-once-on-first-
    access cache. Excluded from hash, equality, and repr so two
    entries with the same content compare equal even if one has
    cached tokens and the other hasn't.
    """

    path: str
    category: str
    content: str
    summary: str | None = None
    related: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    _tokens_cache: dict[Any, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
        repr=False,
    )


class AliasInfo(TypedDict):
    """An alias declared by a template, indexed for fast lookup.

    `template_name` falls back to the relative path stem when the
    template's frontmatter has no `name` field.
    """

    alias: str
    template_path: str
    template_name: str


class DuplicateAliasError(ValueError):
    """Raised when two templates declare the same alias.

    Aliases must be globally unique across a corpus. Carries both
    template paths so the editor can surface them in the diagnostic.
    """

    def __init__(self, alias: str, first_path: str, second_path: str) -> None:
        self.alias = alias
        self.first_path = first_path
        self.second_path = second_path
        super().__init__(
            f"Duplicate alias {alias!r}: declared by both " f"{first_path!r} and {second_path!r}"
        )


@runtime_checkable
class CorpusProtocol(Protocol):
    """Any object that exposes a collection of RetrievalEntry."""

    def entries(self) -> Iterable[RetrievalEntry]: ...

    def get(self, path: str) -> RetrievalEntry | None: ...

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...
