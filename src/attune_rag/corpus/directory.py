"""DirectoryCorpus. Implementation in task 1.2."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .base import CorpusProtocol, RetrievalEntry


class DirectoryCorpus(CorpusProtocol):
    """Load a directory of markdown files as a corpus.

    Full implementation in task 1.2 of the spec.
    """

    def __init__(
        self,
        root: Path,
        summaries_file: str | None = None,
        cross_links_file: str | None = None,
        cache: bool = True,
    ) -> None:
        self._root = Path(root).resolve()
        self._summaries_file = summaries_file
        self._cross_links_file = cross_links_file
        self._cache = cache

    def entries(self) -> Iterable[RetrievalEntry]:
        raise NotImplementedError(
            "DirectoryCorpus.entries is implemented in " "task 1.2 of the RAG grounding spec."
        )

    def get(self, path: str) -> RetrievalEntry | None:
        raise NotImplementedError(
            "DirectoryCorpus.get is implemented in task 1.2 " "of the RAG grounding spec."
        )

    @property
    def name(self) -> str:
        return f"directory:{self._root.name}"

    @property
    def version(self) -> str:
        return "0.0.0-scaffold"
