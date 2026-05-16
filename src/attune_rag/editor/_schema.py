"""Deprecated alias for :mod:`attune_rag.editor.schema`.

See :mod:`attune_rag.editor._rename` for context. Removed in
attune-rag 0.3.0.
"""

from __future__ import annotations

from warnings import warn

from . import schema as _impl

warn(
    "attune_rag.editor._schema is deprecated; use "
    "attune_rag.editor.schema instead. Will be removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)
