"""Tests for attune_rag.dashboard.render."""
from __future__ import annotations

import json

import pytest

from attune_rag.dashboard.render import _validate_output_path, render

_SNAP = {"timestamp": "2026-04-23T00:00:00Z", "retrieval": {}, "freshness": {}}


def test_render_writes_file(tmp_path):
    out = tmp_path / "dash.html"
    result = render(out, _SNAP)
    assert result == out
    assert out.exists()


def test_snapshot_baked_in(tmp_path):
    out = tmp_path / "dash.html"
    snap = {"marker": "unique-test-value-xyz"}
    render(out, snap)
    assert "unique-test-value-xyz" in out.read_text()


def test_title_baked_in(tmp_path):
    out = tmp_path / "dash.html"
    render(out, _SNAP, title="My Custom Title")
    assert "My Custom Title" in out.read_text()


def test_sentinels_fully_replaced(tmp_path):
    out = tmp_path / "dash.html"
    render(out, _SNAP)
    html = out.read_text()
    assert "__ATTUNE_SNAPSHOT__" not in html
    assert "__ATTUNE_TITLE__" not in html


def test_snapshot_is_valid_json_in_output(tmp_path):
    out = tmp_path / "dash.html"
    snap = {"retrieval": {"precision_at_1": 0.9}, "freshness": {}}
    render(out, snap)
    html = out.read_text()
    # The baked snapshot must appear as valid JSON in the script block
    start = html.find("window.__SNAPSHOT__ = ") + len("window.__SNAPSHOT__ = ")
    end = html.find(";", start)
    embedded = json.loads(html[start:end])
    assert embedded["retrieval"]["precision_at_1"] == 0.9


def test_rejects_etc_path():
    with pytest.raises(ValueError, match="system directory"):
        _validate_output_path(__import__("pathlib").Path("/etc/dashboard.html"))


def test_rejects_proc_path():
    with pytest.raises(ValueError, match="system directory"):
        _validate_output_path(__import__("pathlib").Path("/proc/dashboard.html"))


def test_rejects_dev_path():
    with pytest.raises(ValueError, match="system directory"):
        _validate_output_path(__import__("pathlib").Path("/dev/null"))


def test_rejects_null_byte():
    with pytest.raises(ValueError, match="null byte"):
        _validate_output_path(__import__("pathlib").Path("/tmp/dash\x00board.html"))


def test_rejects_nonexistent_parent(tmp_path):
    with pytest.raises(ValueError, match="Parent directory does not exist"):
        _validate_output_path(tmp_path / "no_such_dir" / "dash.html")
