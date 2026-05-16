"""Deprecated alias for :mod:`attune_rag.editor.autocomplete`.

See :mod:`attune_rag.editor._rename` for context. Removed in
attune-rag 0.3.0.
"""

from __future__ import annotations

from warnings import warn

from . import autocomplete as _impl

warn(
    "attune_rag.editor._autocomplete is deprecated; use "
    "attune_rag.editor.autocomplete instead. Will be removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)
