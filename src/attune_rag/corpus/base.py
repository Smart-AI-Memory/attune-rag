"""CorpusProtocol + RetrievalEntry. Implementation in task 1.2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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


@runtime_checkable
class CorpusProtocol(Protocol):
    """Any object that exposes a collection of RetrievalEntry."""

    def entries(self) -> Iterable[RetrievalEntry]: ...

    def get(self, path: str) -> RetrievalEntry | None: ...

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...
