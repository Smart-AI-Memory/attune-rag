"""Tests for ``scripts/security_scan.py`` — Phase 4 W0.10."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "security_scan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("security_scan", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["security_scan"] = module
    spec.loader.exec_module(module)
    return module


ss = _load_module()


def _write(tmp: Path, name: str, source: str) -> Path:
    p = tmp / name
    p.write_text(source, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Check 1 — dynamic code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["eval", "exec", "compile", "__import__"])
def test_dynamic_code_flagged(name: str) -> None:
    tree = ast.parse(f"x = {name}('1+1')")
    findings = ss.check_dynamic_code(tree, "fake.py")
    assert len(findings) == 1
    assert findings[0].kind == "dynamic-code"
    assert findings[0].severity == "high"
    assert name in findings[0].detail


def test_dynamic_code_clean() -> None:
    tree = ast.parse("x = sum([1, 2, 3])\nprint(x)")
    assert ss.check_dynamic_code(tree, "fake.py") == []


@pytest.mark.parametrize(
    "snippet",
    [
        "import re\nre.compile(r'\\w+')",
        "import textwrap\ntextwrap.dedent('x')",
        "class Pat:\n    def eval(self): pass\n\nPat().eval()",
        "from subprocess import compile_args\ncompile_args()",
    ],
)
def test_dynamic_code_attribute_form_not_flagged(snippet: str) -> None:
    """``re.compile()`` and friends share a NAME with the dangerous
    builtins but are different functions. Flagging them would swamp
    PR comments and train reviewers to ignore the gate."""
    tree = ast.parse(snippet)
    assert ss.check_dynamic_code(tree, "fake.py") == []


# ---------------------------------------------------------------------------
# Check 2 — path traversal
# ---------------------------------------------------------------------------


def test_path_traversal_literal_open_is_clean() -> None:
    tree = ast.parse("with open('/etc/hosts') as f: pass")
    assert ss.check_path_traversal(tree, "fake.py") == []


def test_path_traversal_variable_open_flagged() -> None:
    tree = ast.parse("def f(p):\n    return open(p).read()\n")
    findings = ss.check_path_traversal(tree, "fake.py")
    assert len(findings) == 1
    assert findings[0].kind == "path-traversal"
    assert findings[0].severity == "medium"


def test_path_traversal_f_string_with_var_flagged() -> None:
    """f-strings with embedded variables are NOT treated as literals
    because the variable portion can be user-influenced."""
    tree = ast.parse('def f(p):\n    return open(f"/tmp/{p}").read()\n')
    findings = ss.check_path_traversal(tree, "fake.py")
    assert findings, "f-string with variable should be flagged"


def test_path_traversal_pure_f_string_clean() -> None:
    """An f-string with no embedded variables (just constants) is
    literal — uncommon but possible."""
    tree = ast.parse('open(f"/etc/hosts")')
    assert ss.check_path_traversal(tree, "fake.py") == []


def test_path_constructor_with_var_flagged() -> None:
    tree = ast.parse("from pathlib import Path\ndef f(p):\n    return Path(p)\n")
    findings = ss.check_path_traversal(tree, "fake.py")
    assert findings


def test_path_open_no_args_ignored() -> None:
    """Pathological case: open() with no positional args — don't crash."""
    tree = ast.parse("open()")  # would error at runtime, but we shouldn't crash
    findings = ss.check_path_traversal(tree, "fake.py")
    assert findings == []


# ---------------------------------------------------------------------------
# Check 3 — secrets
# ---------------------------------------------------------------------------


def test_secret_anthropic_key_flagged() -> None:
    src = 'API_KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123"'
    findings = ss.check_secrets(src, "fake.py")
    assert any("Anthropic" in f.detail for f in findings)


def test_secret_github_classic_pat_flagged() -> None:
    src = 'GH = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"'
    findings = ss.check_secrets(src, "fake.py")
    assert any("GitHub classic PAT" in f.detail for f in findings)


def test_secret_aws_key_flagged() -> None:
    src = 'AWS = "AKIAIOSFODNN7EXAMPLE"'
    findings = ss.check_secrets(src, "fake.py")
    assert any("AWS" in f.detail for f in findings)


def test_secret_adjacent_keyword_flagged() -> None:
    src = 'api_key = "abcd1234efgh5678ijkl9012"'
    findings = ss.check_secrets(src, "fake.py")
    assert any("api_key" in f.detail or "api-key" in f.detail for f in findings)


def test_secret_placeholder_skipped() -> None:
    """Common placeholder values shouldn't fire."""
    src = (
        'api_key = "your_api_key_here"\n'
        'token = "xxxxxxxxxxxxxxxxxxx"\n'
        'secret = "<your-key>"\n'
    )
    findings = ss.check_secrets(src, "fake.py")
    assert findings == []


def test_secret_comment_line_skipped() -> None:
    """An adjacent-keyword match inside a comment is not a real leak."""
    src = '# api_key = "abcd1234efgh5678ijkl9012"  # example in docs'
    findings = ss.check_secrets(src, "fake.py")
    assert findings == []


def test_secret_detail_redacts_value() -> None:
    """Findings must not leak the actual secret into PR comments."""
    src = 'API_KEY = "sk-ant-api03-supersecretvaluethatshouldberedacted"'
    findings = ss.check_secrets(src, "fake.py")
    assert findings
    for f in findings:
        assert "supersecret" not in f.detail
        assert "•" in f.detail


def test_secret_clean_source() -> None:
    src = "import os\nkey = os.environ['ANTHROPIC_API_KEY']\n"
    assert ss.check_secrets(src, "fake.py") == []


def test_secret_line_number_correct() -> None:
    src = '\n\n\nAPI_KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123"\n'
    findings = ss.check_secrets(src, "fake.py")
    assert findings[0].line == 4


# ---------------------------------------------------------------------------
# Check 4 — deserialization / shell=True
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "import pickle\npickle.load(open('x'))",
        "import pickle\npickle.loads(b'x')",
        "import marshal\nmarshal.load(open('x'))",
        "import marshal\nmarshal.loads(b'x')",
    ],
)
def test_deserialization_flagged(snippet: str) -> None:
    tree = ast.parse(snippet)
    findings = ss.check_deserialization(tree, "fake.py")
    assert findings
    assert findings[0].kind == "deserialization"
    assert findings[0].severity == "high"


def test_yaml_load_without_loader_flagged() -> None:
    tree = ast.parse("import yaml\nyaml.load(text)")
    findings = ss.check_deserialization(tree, "fake.py")
    assert any("yaml.load()" in f.detail for f in findings)


def test_yaml_load_with_safe_loader_clean() -> None:
    tree = ast.parse("import yaml\nyaml.load(text, Loader=yaml.SafeLoader)")
    findings = ss.check_deserialization(tree, "fake.py")
    assert findings == []


def test_yaml_safe_load_clean() -> None:
    tree = ast.parse("import yaml\nyaml.safe_load(text)")
    assert ss.check_deserialization(tree, "fake.py") == []


def test_subprocess_shell_true_flagged() -> None:
    tree = ast.parse("import subprocess\nsubprocess.run('ls', shell=True)")
    findings = ss.check_deserialization(tree, "fake.py")
    assert any("shell=True" in f.detail for f in findings)


def test_subprocess_no_shell_clean() -> None:
    tree = ast.parse("import subprocess\nsubprocess.run(['ls', '-l'])")
    assert ss.check_deserialization(tree, "fake.py") == []


def test_subprocess_shell_false_clean() -> None:
    tree = ast.parse("import subprocess\nsubprocess.run('ls', shell=False)")
    assert ss.check_deserialization(tree, "fake.py") == []


# ---------------------------------------------------------------------------
# scan_file + scan_paths
# ---------------------------------------------------------------------------


def test_scan_file_clean(tmp_path: Path) -> None:
    p = _write(tmp_path, "clean.py", "x = 1\nprint(x)\n")
    assert ss.scan_file(p, repo_root=tmp_path) == []


def test_scan_file_multiple_findings(tmp_path: Path) -> None:
    src = (
        "import pickle\n"
        'API_KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123"\n'
        "eval('1+1')\n"
        "pickle.loads(blob)\n"
    )
    p = _write(tmp_path, "multi.py", src)
    findings = ss.scan_file(p, repo_root=tmp_path)
    kinds = {f.kind for f in findings}
    assert kinds == {"secret", "dynamic-code", "deserialization"}


def test_scan_file_skips_unreadable(tmp_path: Path) -> None:
    """A path that doesn't exist returns no findings (scan_file is
    forgiving; the CLI is where missing-path becomes exit 2)."""
    findings = ss.scan_file(tmp_path / "missing.py", repo_root=tmp_path)
    assert findings == []


def test_scan_file_recovers_secrets_when_parse_fails(tmp_path: Path) -> None:
    """Even if the AST parse fails, the regex-based secret scan still
    runs — that's the most security-critical of the four checks."""
    src = (
        'API_KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123"\n'
        "this is not valid python at all\n"
    )
    p = _write(tmp_path, "broken.py", src)
    findings = ss.scan_file(p, repo_root=tmp_path)
    assert any(f.kind == "secret" for f in findings)


def test_scan_file_uses_repo_relative_label(tmp_path: Path) -> None:
    sub = tmp_path / "src" / "attune_rag"
    sub.mkdir(parents=True)
    p = sub / "thing.py"
    p.write_text("eval('1+1')\n")
    findings = ss.scan_file(p, repo_root=tmp_path)
    assert findings[0].file == "src/attune_rag/thing.py"


def test_scan_paths_skips_non_python(tmp_path: Path) -> None:
    py = _write(tmp_path, "real.py", "eval('1+1')")
    md = _write(tmp_path, "readme.md", "eval('1+1')")  # not Python
    findings = ss.scan_paths([py, md], repo_root=tmp_path)
    assert all(f.file.endswith(".py") for f in findings)


# ---------------------------------------------------------------------------
# filter_by_severity
# ---------------------------------------------------------------------------


def test_filter_by_severity_high_threshold() -> None:
    fs = [
        ss.Finding(kind="x", severity="low", file="a.py", line=1, detail=""),
        ss.Finding(kind="x", severity="medium", file="a.py", line=2, detail=""),
        ss.Finding(kind="x", severity="high", file="a.py", line=3, detail=""),
    ]
    filtered = ss.filter_by_severity(fs, "high")
    assert [f.severity for f in filtered] == ["high"]


def test_filter_by_severity_medium_threshold() -> None:
    fs = [
        ss.Finding(kind="x", severity="low", file="a.py", line=1, detail=""),
        ss.Finding(kind="x", severity="medium", file="a.py", line=2, detail=""),
        ss.Finding(kind="x", severity="high", file="a.py", line=3, detail=""),
    ]
    filtered = ss.filter_by_severity(fs, "medium")
    assert {f.severity for f in filtered} == {"medium", "high"}


# ---------------------------------------------------------------------------
# render_comment
# ---------------------------------------------------------------------------


def test_render_comment_clean_says_no_findings() -> None:
    body = ss.render_comment([], threshold="high")
    assert "No findings" in body
    assert ss.COMMENT_MARKER in body


def test_render_comment_marker_present_top_and_bottom() -> None:
    body = ss.render_comment([], threshold="high")
    assert body.count(ss.COMMENT_MARKER) == 2


def test_render_comment_with_findings_groups_by_kind() -> None:
    fs = [
        ss.Finding(kind="secret", severity="high", file="a.py", line=1, detail="leak"),
        ss.Finding(kind="dynamic-code", severity="high", file="b.py", line=2, detail="eval"),
    ]
    body = ss.render_comment(fs, threshold="high")
    assert "Hardcoded secrets" in body
    assert "Dynamic code execution" in body
    assert "leak" in body
    assert "eval" in body


def test_render_comment_is_deterministic() -> None:
    fs = [
        ss.Finding(kind="secret", severity="high", file="b.py", line=2, detail="x"),
        ss.Finding(kind="secret", severity="high", file="a.py", line=1, detail="y"),
    ]
    first = ss.render_comment(fs, threshold="high")
    second = ss.render_comment(list(reversed(fs)), threshold="high")
    assert first == second


def test_render_comment_info_only_below_threshold() -> None:
    """Findings below the threshold are still listed, but flagged as
    non-blocking informational."""
    fs = [ss.Finding(kind="path-traversal", severity="medium", file="a.py", line=1, detail="x")]
    body = ss.render_comment(fs, threshold="high")
    assert "informational" in body or "below severity" in body


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_exit_0_on_clean(tmp_path: Path) -> None:
    p = _write(tmp_path, "clean.py", "x = 1\n")
    rc = ss.main(["--paths", str(p), "--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_exit_1_on_high_severity(tmp_path: Path) -> None:
    p = _write(tmp_path, "bad.py", "eval('1+1')\n")
    rc = ss.main(["--paths", str(p), "--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_exit_0_when_only_medium_at_high_threshold(tmp_path: Path) -> None:
    """A medium-severity finding (path-traversal) doesn't fail the
    gate by default — the threshold is high. Reported but not blocking."""
    p = _write(tmp_path, "path.py", "def f(x):\n    return open(x).read()\n")
    rc = ss.main(["--paths", str(p), "--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_exit_1_when_medium_threshold(tmp_path: Path) -> None:
    """Lowering the threshold to medium turns path-traversal into a block."""
    p = _write(tmp_path, "path.py", "def f(x):\n    return open(x).read()\n")
    rc = ss.main(
        [
            "--paths",
            str(p),
            "--repo-root",
            str(tmp_path),
            "--severity-threshold",
            "medium",
        ]
    )
    assert rc == 1


def test_main_exit_2_when_path_missing(tmp_path: Path) -> None:
    rc = ss.main(["--paths", str(tmp_path / "missing.py")])
    assert rc == 2


def test_main_writes_comment_when_requested(tmp_path: Path) -> None:
    p = _write(tmp_path, "bad.py", "eval('1+1')\n")
    out = tmp_path / "comment.md"
    rc = ss.main(
        [
            "--paths",
            str(p),
            "--repo-root",
            str(tmp_path),
            "--comment-out",
            str(out),
        ]
    )
    assert rc == 1
    body = out.read_text(encoding="utf-8")
    assert "Dynamic code execution" in body
    assert ss.COMMENT_MARKER in body


def test_main_writes_comment_on_clean_run(tmp_path: Path) -> None:
    """Clean runs still produce a comment (so the gate posts visible
    'no findings' confirmation) — keeps maintainers confident the
    scan actually ran."""
    p = _write(tmp_path, "clean.py", "x = 1\n")
    out = tmp_path / "comment.md"
    rc = ss.main(
        [
            "--paths",
            str(p),
            "--repo-root",
            str(tmp_path),
            "--comment-out",
            str(out),
        ]
    )
    assert rc == 0
    assert "No findings" in out.read_text(encoding="utf-8")


def test_main_writes_json_when_requested(tmp_path: Path) -> None:
    p = _write(tmp_path, "bad.py", "eval('1+1')\n")
    json_out = tmp_path / "findings.json"
    rc = ss.main(
        [
            "--paths",
            str(p),
            "--repo-root",
            str(tmp_path),
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 1
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert any(f["kind"] == "dynamic-code" for f in data)
