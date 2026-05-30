"""Drift-guard test for vendored Claude Code session hooks.

The 7 hook files under ``.claude/hooks/`` are byte-identical copies
of attune-ai's canonical hooks. ``.canonical-sha256`` carries the
expected sha256 of each. This test fails if any vendored file's
hash diverges, or if the file set itself changes without the
manifest being refreshed.

To refresh after an upstream change: ``make sync-hooks``.

See specs/sibling-claude-hooks/ in the attune umbrella workspace.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
MANIFEST = HOOKS_DIR / ".canonical-sha256"


def _parse_manifest() -> dict[str, str]:
    """Parse `shasum -a 256` output into ``{filename: hexdigest}``."""
    out: dict[str, str] = {}
    if not MANIFEST.exists():
        return out
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        hexdigest, _, name = line.partition("  ")
        if not name:
            continue
        out[name.strip()] = hexdigest.strip()
    return out


def _on_disk_hooks() -> set[str]:
    """Python hook files actually present in `.claude/hooks/`."""
    if not HOOKS_DIR.is_dir():
        return set()
    return {
        f.name
        for f in HOOKS_DIR.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    }


def test_manifest_exists() -> None:
    """``.canonical-sha256`` must exist; ``make sync-hooks`` generates it."""
    assert MANIFEST.exists(), f"manifest missing at {MANIFEST}. Run `make sync-hooks` to generate."


def test_manifest_covers_all_on_disk_hooks() -> None:
    """Manifest must list every Python hook file on disk and vice versa."""
    manifest = _parse_manifest()
    on_disk = _on_disk_hooks()
    missing_from_manifest = on_disk - set(manifest.keys())
    extra_in_manifest = set(manifest.keys()) - on_disk
    assert not missing_from_manifest and not extra_in_manifest, (
        f"manifest/on-disk mismatch — "
        f"missing_from_manifest={sorted(missing_from_manifest)}, "
        f"extra_in_manifest={sorted(extra_in_manifest)}. "
        f"Run `make sync-hooks` to refresh."
    )


@pytest.mark.parametrize("name,canonical_hex", sorted(_parse_manifest().items()))
def test_hook_matches_canonical_hash(name: str, canonical_hex: str) -> None:
    """Each vendored hook file matches its canonical sha256."""
    f = HOOKS_DIR / name
    assert f.exists(), f"vendored hook missing: {name}"
    actual = hashlib.sha256(f.read_bytes()).hexdigest()
    assert actual == canonical_hex, (
        f"{name} drift detected. "
        f"vendored sha256={actual}, canonical={canonical_hex}. "
        f"Run `make sync-hooks` to re-copy from attune-ai canonical."
    )
