"""AttuneHelpCorpus — thin adapter over the bundled attune-help templates.

Requires the ``attune-rag[attune-help]`` optional extra.

v0.1.0 note on sidecars
-----------------------

``attune-help`` ships ``summaries.json`` and
``cross_links.json`` beside its templates, but the shapes
do not match ``DirectoryCorpus``'s expectations:

- ``summaries.json`` is keyed by feature name
  (``"security-audit"``) rather than template path
  (``"concepts/tool-security-audit.md"``).
- ``cross_links.json`` uses a nested
  ``{version, stats, links, tag_index, workflow_map}``
  layout keyed by short IDs (``"com-auth-strategies"``).

v0.1.0 loads templates without sidecars. A schema adapter
that surfaces attune-help's richer metadata as
``RetrievalEntry.summary`` / ``related`` / ``metadata`` is
a v0.2.0 concern (see spec Open Questions).
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from importlib.resources import as_file, files
from pathlib import Path

from .base import RetrievalEntry
from .directory import DirectoryCorpus


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
        # v0.1.0: no sidecars (schema mismatch documented above).
        self._inner = DirectoryCorpus(root=root)

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
