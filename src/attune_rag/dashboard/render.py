"""Render the attune-rag dashboard HTML with an embedded snapshot."""

from __future__ import annotations

import html as _html
import importlib.resources as _ilr
import json
from pathlib import Path
from typing import Any

_SENTINEL_SNAPSHOT = "__ATTUNE_SNAPSHOT__"
_SENTINEL_TITLE = "__ATTUNE_TITLE__"

# Original denylist + their macOS-resolved equivalents. On macOS,
# ``Path("/etc/foo").resolve()`` returns ``/private/etc/foo`` because
# ``/etc`` is a symlink to ``/private/etc``. Without the ``/private``
# mirrors a user typing ``--out /etc/foo`` would be normalized through
# the symlink before this check sees it, and the original denylist
# wouldn't match. Mirrors W09.S.011's fix in ``eval/bench_prompts.py``.
_SYSTEM_DIRS = frozenset(
    {"/etc", "/sys", "/proc", "/dev", "/boot", "/sbin", "/bin", "/usr/bin"}
    | {"/private/etc", "/private/sys", "/private/dev"}
)


def _json_for_script_block(obj: Any) -> str:
    # A literal ``</script>`` inside a JSON string value would terminate
    # the surrounding ``<script>`` block in the browser. Escape ``<`` (and
    # the U+2028 / U+2029 line separators that some older JS parsers
    # stumble on) using ``\u`` escapes, which remain valid JSON.
    return json.dumps(obj).replace("<", "\\u003c").replace(" ", "\\u2028").replace(" ", "\\u2029")


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
    # ``title`` lands inside ``<title>…</title>``; HTML-escape so a value
    # like ``</title><script>…`` can't break out. ``snapshot`` lands inside
    # ``<script>window.__SNAPSHOT__ = …;</script>``; JSON-encode and then
    # neutralize the script-terminator vector via ``_json_for_script_block``.
    rendered = tmpl.replace(_SENTINEL_SNAPSHOT, _json_for_script_block(snapshot)).replace(
        _SENTINEL_TITLE, _html.escape(title, quote=True)
    )
    out.write_text(rendered, encoding="utf-8")
    return out
