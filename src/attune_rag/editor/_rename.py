"""Deprecated alias for :mod:`attune_rag.editor.rename`.

The underscore-prefixed name was a historical accident — Python's
convention is that ``_foo`` modules are private, but attune-gui imports
this submodule directly. Phase 3 of the v1.0 roadmap renamed the real
implementation to :mod:`attune_rag.editor.rename`. This shim re-exports
everything for one release of backward compatibility.

Removed in attune-rag 0.3.0.
"""

from __future__ import annotations

from warnings import warn

from . import rename as _impl

warn(
    "attune_rag.editor._rename is deprecated; use "
    "attune_rag.editor.rename instead. Will be removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)
