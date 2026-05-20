"""Tests for attune_rag.dashboard.render."""

from __future__ import annotations

import json

import pytest

from attune_rag.dashboard.render import (
    _json_for_script_block,
    _validate_output_path,
    render,
)

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


# ── XSS hardening ──


def test_snapshot_script_terminator_escaped(tmp_path):
    out = tmp_path / "dash.html"
    render(out, {"k": "</script><img onerror=alert(1)>"})
    html = out.read_text()
    # The script-block sentinel landed inside <script>…</script>;
    # a literal "</script>" in the embedded JSON would break out.
    assert "</script><img" not in html
    assert "\\u003c/script>" in html


def test_snapshot_line_separators_escaped(tmp_path):
    out = tmp_path / "dash.html"
    render(out, {"k": "a b c"})
    html = out.read_text()
    # Older JS parsers stumble on raw U+2028/U+2029 inside JSON strings
    # that live in a <script> block. Escaped form keeps the JSON valid.
    assert " " not in html.split("window.__SNAPSHOT__")[1].split(";")[0]
    assert " " not in html.split("window.__SNAPSHOT__")[1].split(";")[0]
    assert "\\u2028" in html
    assert "\\u2029" in html


def test_title_html_escaped(tmp_path):
    out = tmp_path / "dash.html"
    render(out, _SNAP, title="</title><script>alert('xss')</script>")
    html = out.read_text()
    # Title lands inside <title>…</title>; the closing tag must not survive verbatim.
    assert "</title><script>" not in html
    assert "&lt;/title&gt;&lt;script&gt;" in html


def test_title_ampersand_escaped(tmp_path):
    out = tmp_path / "dash.html"
    render(out, _SNAP, title="A & B")
    html = out.read_text()
    assert "<title>A &amp; B</title>" in html


def test_json_for_script_block_escapes_lt():
    out = _json_for_script_block({"k": "<a>"})
    assert "<a>" not in out
    assert "\\u003c" in out


def test_json_for_script_block_remains_valid_json():
    payload = {"k": "</script>  <b>"}
    out = _json_for_script_block(payload)
    assert json.loads(out) == payload


def test_dashboard_template_has_sri_on_cdn_script():
    import importlib.resources as _ilr
    import re

    tmpl = (
        _ilr.files("attune_rag.dashboard")
        .joinpath("templates/dashboard.html")
        .read_text(encoding="utf-8")
    )
    # Every external CDN <script src=…> tag (which may span multiple lines)
    # must carry an integrity hash so a compromised CDN cannot inject code.
    for tag in re.findall(r"<script\b[^>]*src=[^>]*>", tmpl, flags=re.DOTALL):
        if "//" in tag:  # external — skip inline scripts
            assert "integrity=" in tag, f"CDN script missing SRI:\n{tag}"
            assert "crossorigin=" in tag, f"CDN script missing crossorigin:\n{tag}"
