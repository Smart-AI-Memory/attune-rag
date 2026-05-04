"""CorpusProtocol + RetrievalEntry. Implementation in task 1.2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable


@dataclass(frozen=True)
class RetrievalEntry:
    """A single corpus entry. Task 1.1 shape; task 1.2 wires loaders."""

    path: str
    category: str
    content: str
    summary: str | None = None
    related: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


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
