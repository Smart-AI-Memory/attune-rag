"""Render the attune-rag dashboard HTML with an embedded snapshot."""

from __future__ import annotations

import importlib.resources as _ilr
import json
from pathlib import Path
from typing import Any

_SENTINEL_SNAPSHOT = "__ATTUNE_SNAPSHOT__"
_SENTINEL_TITLE = "__ATTUNE_TITLE__"

_SYSTEM_DIRS = frozenset({"/etc", "/sys", "/proc", "/dev", "/boot", "/sbin", "/bin", "/usr/bin"})


def _validate_output_path(out: Path) -> None:
    raw = str(out)
    if "\x00" in raw:
        raise ValueError("Output path contains a null byte.")
    # Compare against POSIX-form of the path so the system-dir check works
    # uniformly on Windows, where ``str(Path("/etc/x"))`` returns ``\etc\x``
    # and would never match the literal ``/etc/`` prefix. ``as_posix()``
    # normalizes separators to forward slashes; the system dirs guarded
    # here are POSIX-only by nature, but we still reject the same paths
    # from any platform for cross-platform parity.
    posix = out.as_posix()
    for sdir in _SYSTEM_DIRS:
        if posix == sdir or posix.startswith(sdir + "/"):
            raise ValueError(f"Output path is inside a system directory: {sdir}")
    if not out.parent.exists():
        raise ValueError(f"Parent directory does not exist: {out.parent}")


def render(
    out: Path,
    snapshot: dict[str, Any],
    title: str = "attune-rag dashboard",
) -> Path:
    """Render the dashboard template to *out* with *snapshot* embedded as JSON."""
    _validate_output_path(out)
    tmpl = (
        _ilr.files("attune_rag.dashboard")
        .joinpath("templates/dashboard.html")
        .read_text(encoding="utf-8")
    )
    html = tmpl.replace(_SENTINEL_SNAPSHOT, json.dumps(snapshot)).replace(_SENTINEL_TITLE, title)
    out.write_text(html, encoding="utf-8")
    return out
