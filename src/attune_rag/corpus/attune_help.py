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
from importlib import import_module
from importlib.resources import as_file, files
from pathlib import Path

from .base import RetrievalEntry
from .directory import DirectoryCorpus

_OVERRIDES_PATH = Path(__file__).parent / "summaries_override.json"


class AttuneHelpCorpus:
    """Loads the bundled ``attune_help`` templates as a corpus."""

    def __init__(self) -> None:
        try:
            attune_help = import_module("attune_help")
        except ImportError as exc:
            raise RuntimeError(
                "AttuneHelpCorpus requires the [attune-help] extra. "
                "Install with: pip install 'attune-rag[attune-help]'"
            ) from exc

        templates = files("attune_help").joinpath("templates")
        with as_file(templates) as templates_path:
            root = Path(templates_path)
        if not root.is_dir():
            raise RuntimeError(
                f"attune_help templates directory not found at {root}. "
                "The attune-help package layout may have changed."
            )

        self._version = getattr(attune_help, "__version__", "unknown")
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
            root=root,
            summaries_file="summaries_by_path.json",
            extra_summaries=overrides,
        )

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
