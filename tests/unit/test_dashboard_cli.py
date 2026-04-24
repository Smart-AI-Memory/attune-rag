"""Tests for the `attune-rag dashboard` CLI subcommands."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from attune_rag.cli import main

_FAKE_SNAP = {"timestamp": "2026-04-23T00:00:00Z", "retrieval": {}, "freshness": {}}


def test_dashboard_render_help():
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "render", "--help"])
    assert exc.value.code == 0


def test_dashboard_refresh_help():
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "refresh", "--help"])
    assert exc.value.code == 0


def test_dashboard_render_missing_out_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "render"])
    assert exc.value.code != 0


def test_dashboard_render_end_to_end(tmp_path):
    out = tmp_path / "test.html"
    with patch("attune_rag.dashboard.refresh.build_snapshot", return_value=_FAKE_SNAP):
        rc = main(["dashboard", "render", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    html = out.read_text()
    assert "2026-04-23" in html


def test_dashboard_render_open_flag(tmp_path):
    out = tmp_path / "test.html"
    with (
        patch("attune_rag.dashboard.refresh.build_snapshot", return_value=_FAKE_SNAP),
        patch("webbrowser.open") as mock_open,
    ):
        rc = main(["dashboard", "render", "--out", str(out), "--open"])
    assert rc == 0
    mock_open.assert_called_once()


def test_dashboard_render_no_refresh_cmd_required():
    # --refresh-cmd should no longer exist; ensure render works without it
    from attune_rag.cli import build_parser
    parser = build_parser()
    # parse_known_args so unknown flags don't raise
    ns, _ = parser.parse_known_args(["dashboard", "render", "--out", "/tmp/x.html"])
    assert not hasattr(ns, "refresh_cmd")
