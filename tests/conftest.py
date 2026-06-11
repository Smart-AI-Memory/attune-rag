"""Suite-wide guards.

Pin the auth route to ``api`` and clear the Claude Code session
marker so running the suite *inside* a Claude Code session can't
auto-route un-mocked judge calls to real subscription calls
(mirrors attune-author's suite conftest). Tests that exercise the
routing logic set their own env via ``monkeypatch`` — in-test
patches run after this fixture and win.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _pin_api_auth_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATTUNE_RAG_AUTH_MODE", "api")
    monkeypatch.delenv("CLAUDECODE", raising=False)
