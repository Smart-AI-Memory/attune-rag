"""Behavior smoke tests for vendored Claude Code session hooks.

Each hook is a standalone Python script under ``.claude/hooks/``;
tests invoke it via subprocess (stdin = JSON payload, exit code =
contract). We assert exit-code semantics, not stdout content, so
that prose tweaks in canonical attune-ai don't break these tests
without genuine behavior drift.

Contract (per specs/sibling-claude-hooks/ design):
- ``security_guard``: exit 2 to block, 0 to allow.
- ``spec_orient``: exit 0 always (informational SessionStart hook).
- ``format_on_save``: exit 0 always (best-effort PostToolUse hook).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"


def _run_hook(
    name: str,
    payload: dict | None = None,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess:
    """Invoke a hook script with a JSON stdin payload."""
    env = os.environ.copy()
    if env_overrides is not None:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / name)],
        input=json.dumps(payload or {}),
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=timeout,
    )


# --------------------------------------------------------------------
# security_guard: PreToolUse Bash + Edit|Write — exit 2 blocks, 0 allows
# --------------------------------------------------------------------


def test_security_guard_blocks_eval_bash() -> None:
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "python -c \"eval('1+1')\""},
    }
    result = _run_hook("security_guard.py", payload)
    assert result.returncode == 2, (
        f"expected exit 2 (block), got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_security_guard_blocks_exec_bash() -> None:
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "python -c \"exec('print(1)')\""},
    }
    result = _run_hook("security_guard.py", payload)
    assert result.returncode == 2


def test_security_guard_allows_benign_bash() -> None:
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
    }
    result = _run_hook("security_guard.py", payload)
    assert result.returncode == 0, (
        f"expected exit 0 (allow), got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_security_guard_blocks_path_traversal_write() -> None:
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": "/etc/passwd", "content": "x"},
    }
    result = _run_hook("security_guard.py", payload)
    assert result.returncode == 2, (
        f"expected exit 2 (block) on /etc/passwd write, got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# --------------------------------------------------------------------
# spec_orient: SessionStart — exit 0 always
# --------------------------------------------------------------------


def test_spec_orient_exit_0_with_populated_specs(tmp_path: Path) -> None:
    """When ``docs/specs/<feature>/`` has files, hook emits + exits 0."""
    (tmp_path / ".git").mkdir()  # workspace-root marker
    spec_dir = tmp_path / "docs" / "specs" / "demo-feature"
    spec_dir.mkdir(parents=True)
    (spec_dir / "requirements.md").write_text("# Demo Feature\n\n**Status**: draft\n")
    result = _run_hook("spec_orient.py", cwd=tmp_path)
    assert result.returncode == 0, (
        f"spec_orient must exit 0 (informational). "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_spec_orient_exit_0_with_empty_workspace(tmp_path: Path) -> None:
    """When no specs exist, hook still exits 0 (informational hooks never block)."""
    (tmp_path / ".git").mkdir()
    result = _run_hook("spec_orient.py", cwd=tmp_path)
    assert result.returncode == 0


# --------------------------------------------------------------------
# format_on_save: PostToolUse Edit|Write — exit 0 when formatters absent
# --------------------------------------------------------------------


def test_format_on_save_exit_0_when_formatters_absent(tmp_path: Path) -> None:
    """Hook is best-effort; missing black/ruff is not a failure."""
    f = tmp_path / "demo.py"
    f.write_text("x = 1\n")
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(f)},
    }
    # Strip PATH so black/ruff lookups all fail.
    result = _run_hook(
        "format_on_save.py",
        payload,
        env_overrides={"PATH": "/nonexistent"},
    )
    assert result.returncode == 0, (
        f"format_on_save must exit 0 even when formatters are absent. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
