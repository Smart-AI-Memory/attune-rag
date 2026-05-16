"""Backward-compat tests for the renamed editor submodules.

Phase 3 of the v1.0 roadmap renamed five editor submodules from
``_foo`` to ``foo`` (the underscore convention falsely signaled
"private" — attune-gui imports them directly). The five underscored
paths now exist as deprecation shims that re-export the new module
and emit ``DeprecationWarning``.

These tests verify:

1. Importing the old name emits exactly one ``DeprecationWarning``.
2. The shim transparently proxies public *and* private attributes
   (attune-gui touches ``_hunks`` via the old path).

The shims are scheduled for removal in attune-rag 0.3.0 — when
that lands, delete this file along with the shim modules.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

SHIM_PAIRS = [
    ("attune_rag.editor._rename", "attune_rag.editor.rename"),
    ("attune_rag.editor._schema", "attune_rag.editor.schema"),
    ("attune_rag.editor._lint", "attune_rag.editor.lint"),
    ("attune_rag.editor._autocomplete", "attune_rag.editor.autocomplete"),
    ("attune_rag.editor._references", "attune_rag.editor.references"),
]


def _fresh_import(name: str):
    """Drop any cached copy so ``import`` re-runs the module body."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize(("old", "new"), SHIM_PAIRS)
def test_shim_emits_deprecation_warning(old: str, new: str) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _fresh_import(old)
    dep = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1, f"expected one DeprecationWarning from {old}, got {len(dep)}"
    msg = str(dep[0].message)
    assert old in msg, f"warning should name the deprecated path: {msg!r}"
    assert new in msg, f"warning should name the replacement: {msg!r}"
    assert "0.3.0" in msg, f"warning should name the removal version: {msg!r}"


@pytest.mark.parametrize(("old", "new"), SHIM_PAIRS)
def test_shim_proxies_public_attributes(old: str, new: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_mod = _fresh_import(old)
        new_mod = importlib.import_module(new)
    # Sample a public class/function defined in the new module.
    public_names = [
        n for n in dir(new_mod) if not n.startswith("_") and getattr(new_mod, n, None) is not None
    ]
    assert public_names, f"{new} has no public attributes to proxy"
    for name in public_names:
        assert getattr(old_mod, name) is getattr(
            new_mod, name
        ), f"{old}.{name} should proxy to {new}.{name}"


def test_shim_proxies_private_hunks() -> None:
    """attune-gui touches ``_hunks`` via the old path — keep that working."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_mod = _fresh_import("attune_rag.editor._rename")
        new_mod = importlib.import_module("attune_rag.editor.rename")
    assert old_mod._hunks is new_mod._hunks
