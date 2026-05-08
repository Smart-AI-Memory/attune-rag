"""AttuneHelpCorpus — thin adapter over the bundled attune-help templates.

Requires the ``attune-rag[attune-help]`` optional extra.

Sidecar behavior
----------------

As of attune-rag 0.1.2 + attune-help 0.7.0:

- ``summaries_by_path.json`` (path-keyed, keyword-rich,
  polished) is the primary summary sidecar. This is what
  ``DirectoryCorpus`` expects and what drives the summary
  signal in keyword retrieval.
- Legacy ``summaries.json`` (feature-keyed) is still
  shipped by attune-help for backwards compatibility but
  has a schema mismatch with ``DirectoryCorpus``, so it is
  silently ignored by path-keyed consumers. Passing it
  directly would produce zero summary coverage.
- ``cross_links.json`` uses a nested layout incompatible
  with ``DirectoryCorpus``'s expected ``{path: [paths]}``
  map. Leave it unset for now; a schema adapter is a
  future concern.

For attune-help < 0.7.0 the summaries_by_path.json file
doesn't exist. We still attempt to load it — the
``_load_sidecar`` helper treats a missing file as an empty
map, so older attune-help installs simply get no summary
signal (same as before).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .base import RetrievalEntry
from .directory import DirectoryCorpus
from .help_adapter import HelpCorpusAdapter

_OVERRIDES_PATH = Path(__file__).parent / "summaries_override.json"


@dataclass(frozen=True)
class _BundledAdapter:
    """Default :class:`HelpCorpusAdapter` for the bundled attune-help.

    Module-level so it doesn't fall foul of Python's class-body scoping
    rules (a class declared inside a function can't reference the
    function's locals).
    """

    templates_root: Path
    version: str


class AttuneHelpCorpus:
    """Loads an attune-help-shaped corpus of templates.

    Takes a :class:`HelpCorpusAdapter` so attune-rag never imports
    attune-help at module level. Use :meth:`from_attune_help` for the
    common case where you have attune-help installed and want the
    bundled templates without writing your own adapter.
    """

    def __init__(self, adapter: HelpCorpusAdapter) -> None:
        if not adapter.templates_root.is_dir():
            raise RuntimeError(
                f"templates_root is not a directory: {adapter.templates_root}. "
                "The corpus adapter may be misconfigured or the package "
                "layout may have changed."
            )

        self._version = adapter.version
        overrides: dict[str, str | None] = {}
        if _OVERRIDES_PATH.is_file():
            try:
                overrides = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        # attune-help 0.7.0+ ships summaries_by_path.json; older
        # versions don't have it. DirectoryCorpus._load_sidecar
        # treats a missing file as an empty map, so this is safe
        # to pass unconditionally.
        self._inner = DirectoryCorpus(
            root=adapter.templates_root,
            summaries_file="summaries_by_path.json",
            extra_summaries=overrides,
        )

    @classmethod
    def from_attune_help(cls) -> AttuneHelpCorpus:
        """Construct using the bundled attune-help package as adapter.

        Localizes the dynamic import to one factory call instead of the
        module body. Static analyzers see the rag→help boundary as a
        runtime-only dependency in this single function. Callers that
        want to avoid the implicit dep entirely (testing, alternate
        corpora) should construct the adapter themselves and pass it
        to ``__init__``.
        """
        from importlib import import_module
        from importlib.resources import as_file, files

        try:
            attune_help = import_module("attune_help")
        except ImportError as exc:
            raise RuntimeError(
                "AttuneHelpCorpus.from_attune_help() requires the "
                "[attune-help] extra. Install with: "
                "pip install 'attune-rag[attune-help]'"
            ) from exc

        templates = files("attune_help").joinpath("templates")
        with as_file(templates) as templates_path:
            root = Path(templates_path)

        version = getattr(attune_help, "__version__", "unknown")
        return cls(_BundledAdapter(templates_root=root, version=version))

    def entries(self) -> Iterable[RetrievalEntry]:
        return self._inner.entries()

    def get(self, path: str) -> RetrievalEntry | None:
        return self._inner.get(path)

    @property
    def name(self) -> str:
        return "attune-help"

    @property
    def version(self) -> str:
        return self._version
