"""Public alias-file loader shared between DirectoryCorpus and AttuneHelpCorpus.

Re-exported as ``attune_rag.corpus.load_aliases_from_file``. Strict
behavior at the public boundary: missing or malformed files raise
typed exceptions with the file path in the message. The bundled
``AttuneHelpCorpus`` wraps calls in a tolerant try/except to preserve
backward-compat for its own override file; user-corpus callers get
the strict semantics.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_aliases_from_file(path: Path | str) -> dict[str, list[str]]:
    """Load a path-keyed extra-aliases JSON file.

    Schema::

        {
            "rel/path/to/template.md": ["alias one", "alias two"],
            "_comment": "underscore-prefixed keys ignored",
            ...
        }

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        Mapping of relative path → list of alias strings. Suitable to
        pass straight through to ``DirectoryCorpus(extra_aliases=...)``
        or to merge with an inline ``extra_aliases`` dict.

    Raises:
        FileNotFoundError: ``path`` does not exist. The path is in the
            exception message.
        ValueError: The file is unreadable, malformed JSON, not a JSON
            object at top level, or contains values that aren't lists
            of strings. The path is in the exception message.

    Behavior:
        - Top-level keys starting with ``_`` are silently dropped (so
          files can carry inline ``_comment`` documentation).
        - Non-list values for any non-underscore key raise ``ValueError``.
        - Non-string entries within a value list raise ``ValueError``.
        - Empty lists are kept (no aliases for that path is meaningful).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"extra_aliases_file not found: {p}")
    try:
        raw_text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to read {p}: {exc}") from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top-level JSON must be an object")
    result: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"{p}: keys must be strings; got {type(key).__name__}")
        if key.startswith("_"):
            continue
        if not isinstance(value, list):
            raise ValueError(f"{p}: value for {key!r} must be a list; got {type(value).__name__}")
        for i, alias in enumerate(value):
            if not isinstance(alias, str):
                raise ValueError(f"{p}: {key!r}[{i}] must be a string; got {type(alias).__name__}")
        result[key] = list(value)
    return result
