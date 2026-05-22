"""Typed protocol for plugging an attune-help (or similar) corpus into rag.

This module exists to remove the module-level
``importlib.import_module("attune_help")`` from
:mod:`attune_rag.corpus.attune_help`. With the protocol in place,
attune-rag never imports attune-help — the consumer hands an adapter
in. Static analyzers can see the real dependency graph (rag does not
depend on help; help depends on rag's protocol).

The protocol is intentionally minimal: the adapter only needs to point
to the templates directory and report a version string. attune-rag's
``AttuneHelpCorpus`` does the corpus work via the existing
``DirectoryCorpus`` against that root.

The simpler protocol differs from the shape proposed in
``specs/architecture-realignment/design.md`` (which suggested
``iter_entries`` on the adapter). Iteration is already handled by
``DirectoryCorpus``; pushing it into the adapter would duplicate that
logic in attune-help. Honest-to-the-code wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class HelpCorpusAdapter(Protocol):
    """Implemented by attune-help (or any other consumer that wants
    its bundled corpus exposed to attune-rag's keyword retriever).

    Implementations are typically a small dataclass-like object that
    points at a directory of markdown templates and exposes a version
    string. attune-help ships such an adapter as
    ``attune_help.adapters.rag.AttuneHelpAdapter``; downstream callers
    can pass any object that satisfies this protocol.
    """

    @property
    def templates_root(self) -> Path:
        """Filesystem path to the directory of ``*.md`` templates."""
        ...  # pragma: no cover -- Protocol stub; never executed

    @property
    def version(self) -> str:
        """Stable version string for the help corpus (e.g. package
        ``__version__``). Used for cache-busting and provenance."""
        ...  # pragma: no cover -- Protocol stub; never executed
