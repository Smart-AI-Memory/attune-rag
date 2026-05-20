"""Per-PR security scan — stdlib checks for the four classic patterns.

Phase 4 W0.10 deliverable. Run by ``.github/workflows/security-scan.yml``
on PRs touching sensitive subtrees:

- ``src/attune_rag/editor/**`` — rename + lint walk user paths.
- ``src/attune_rag/providers/**`` — handles API tokens, may log requests.
- ``src/attune_rag/dashboard/**`` — emits snapshots; ensure no secret leakage.
- ``src/attune_rag/cli.py`` — user-facing argv parsing.

Four check classes, all conservative — **false positives are OK,
false negatives are not** (per design.md §3). attune-ai's deeper
analysis (W0.9) is the secondary sweep on findings.

1. **dynamic-code** — ``eval`` / ``exec`` / ``compile`` /
   ``__import__`` calls. Almost always a code-injection vector when
   the argument is dynamic.
2. **path-traversal** — file operations with non-literal path
   arguments that could be user-controlled. Conservative: flags any
   ``open()``/``Path()`` where the path comes from a variable rather
   than a literal. Maintainer triages.
3. **secret** — hardcoded credentials in source. Regex on raw text
   for common token shapes (Anthropic ``sk-ant-*``, OpenAI ``sk-*``,
   GitHub PATs ``ghp_*`` / ``github_pat_*``, AWS ``AKIA*``) and
   long high-entropy literals adjacent to ``api_key`` /
   ``token`` / ``password`` / ``secret`` keywords.
4. **deserialization** — ``pickle.load(s)`` / ``marshal.load(s)`` /
   ``yaml.load`` without explicit ``Loader=SafeLoader`` /
   ``subprocess.*(..., shell=True)``. Each is a known RCE vector
   when fed untrusted input.

Exit codes mirror the rest of the freeze-gate toolchain:

- 0 — no findings at or above the severity threshold (default ``high``).
- 1 — at least one finding at or above threshold.
- 2 — validation error (file unreadable, etc.).

Pure stdlib. Safe in any CI image.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMMENT_MARKER = "<!-- attune-rag-security-scan -->"

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class Finding:
    kind: str  # "dynamic-code" | "path-traversal" | "secret" | "deserialization"
    severity: str  # "high" | "medium" | "low"
    file: str  # repo-relative path
    line: int
    detail: str  # one-line description

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Check 1 — dynamic code execution
# ---------------------------------------------------------------------------


_DYNAMIC_CALL_NAMES = frozenset({"eval", "exec", "compile", "__import__"})


def check_dynamic_code(tree: ast.AST, file: str) -> list[Finding]:
    """Flag bare ``eval`` / ``exec`` / ``compile`` / ``__import__`` calls.

    Only bare ``Name``-form calls are flagged; attribute-form calls
    like ``re.compile(...)`` or ``subprocess.compile_args(...)`` are
    NOT the dangerous builtins — flagging them would swamp PR comments
    with false positives (re.compile is on nearly every Python file)
    and train maintainers to ignore the gate.
    """
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Only the bare builtin form is dangerous. re.compile,
        # textwrap.compile_etc., obj.eval — all different functions.
        if not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in _DYNAMIC_CALL_NAMES:
            findings.append(
                Finding(
                    kind="dynamic-code",
                    severity="high",
                    file=file,
                    line=getattr(node, "lineno", 0),
                    detail=f"call to {name}() — dynamic code execution",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Check 2 — path-traversal (conservative)
# ---------------------------------------------------------------------------


_FILE_OPEN_NAMES = frozenset({"open"})
# Calls on pathlib.Path() with a non-literal first arg are flagged.
_PATH_CTOR_NAMES = frozenset({"Path"})


def check_path_traversal(tree: ast.AST, file: str) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in _FILE_OPEN_NAMES or name in _PATH_CTOR_NAMES:
            if not node.args:
                continue
            arg0 = node.args[0]
            if _is_literal_path(arg0):
                continue
            # Non-literal first arg — could be user-controlled. Flag
            # as medium so maintainer triages but doesn't auto-fail.
            findings.append(
                Finding(
                    kind="path-traversal",
                    severity="medium",
                    file=file,
                    line=getattr(node, "lineno", 0),
                    detail=(
                        f"{name}() called with non-literal path arg; "
                        "confirm the value is sanitized if user-influenced"
                    ),
                )
            )
    return findings


def _is_literal_path(node: ast.AST) -> bool:
    """A literal path is a string Constant or a join/format of Constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        # f"...{x}..." can still inject. Conservative: not literal.
        return all(isinstance(v, ast.Constant) for v in node.values)
    if isinstance(node, ast.Attribute):
        # e.g. ``Path.cwd``. Conservative: not interesting.
        return True
    return False


# ---------------------------------------------------------------------------
# Check 3 — hardcoded secrets (regex on raw source)
# ---------------------------------------------------------------------------


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI API key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("GitHub classic PAT", re.compile(r"ghp_[A-Za-z0-9]{36}\b")),
    ("GitHub fine-grained PAT", re.compile(r"github_pat_[A-Za-z0-9_]{60,}")),
    ("AWS access key id", re.compile(r"AKIA[A-Z0-9]{16}\b")),
    ("Slack token", re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}")),
)

# Adjacent-keyword secret: a long literal next to a known sensitive
# keyword. Catches custom secrets the explicit patterns miss.
_SECRET_ADJACENT_RE = re.compile(
    r"""(?ix)
    \b(api[_-]?key|secret(?:[_-]?key)?|token|password|passwd|auth|bearer)\b
    \s*[:=]\s*
    ["']([^"']{16,})["']
    """,
)


def check_secrets(source: str, file: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()

    for name, pat in _SECRET_PATTERNS:
        for m in pat.finditer(source):
            line_no = source.count("\n", 0, m.start()) + 1
            value = m.group(0)
            findings.append(
                Finding(
                    kind="secret",
                    severity="high",
                    file=file,
                    line=line_no,
                    detail=f"{name} pattern match: {_redact(value)}",
                )
            )

    for m in _SECRET_ADJACENT_RE.finditer(source):
        keyword = m.group(1)
        value = m.group(2)
        # Skip obvious placeholders.
        if _is_placeholder(value):
            continue
        line_no = source.count("\n", 0, m.start()) + 1
        # Skip if the line is a comment example or a test fixture.
        line_text = lines[line_no - 1] if 0 <= line_no - 1 < len(lines) else ""
        if _is_comment_or_example(line_text):
            continue
        findings.append(
            Finding(
                kind="secret",
                severity="high",
                file=file,
                line=line_no,
                detail=f"possible hardcoded {keyword}: {_redact(value)}",
            )
        )

    return findings


def _redact(value: str) -> str:
    """Show only the first 4 chars; rest as •••."""
    if len(value) <= 4:
        return "•••"
    return value[:4] + "•" * min(len(value) - 4, 8)


_PLACEHOLDER_VALUES = frozenset(
    {
        "your_api_key_here",
        "your-api-key-here",
        "your_token_here",
        "your-token-here",
        "your_secret_here",
        "your-secret-here",
        "example",
        "placeholder",
        "xxxxxxxxxxxxxxxx",
        "<your-key>",
        "<api_key>",
        "redacted",
        "<redacted>",
    }
)


def _is_placeholder(value: str) -> bool:
    norm = value.strip().lower()
    if norm in _PLACEHOLDER_VALUES:
        return True
    # All same character → likely a placeholder ("xxxx..." / "0000...").
    if len(set(norm)) <= 2 and len(norm) >= 8:
        return True
    return False


def _is_comment_or_example(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return True
    # docstring lines (single quote, triple-quoted prefixes)
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    return False


# ---------------------------------------------------------------------------
# Check 4 — unsafe deserialization / shell=True
# ---------------------------------------------------------------------------


# fully-qualified call names that are unsafe by default
_UNSAFE_DESERIALIZE = frozenset(
    {
        "pickle.load",
        "pickle.loads",
        "marshal.load",
        "marshal.loads",
    }
)


def check_deserialization(tree: ast.AST, file: str) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified = _qualified_call_name(node.func)
        if qualified in _UNSAFE_DESERIALIZE:
            findings.append(
                Finding(
                    kind="deserialization",
                    severity="high",
                    file=file,
                    line=getattr(node, "lineno", 0),
                    detail=f"{qualified}() — RCE if input is untrusted",
                )
            )
            continue

        # yaml.load without Loader=Safe* is the classic CVE.
        if qualified == "yaml.load" and not _yaml_load_is_safe(node):
            findings.append(
                Finding(
                    kind="deserialization",
                    severity="high",
                    file=file,
                    line=getattr(node, "lineno", 0),
                    detail=(
                        "yaml.load() without Loader=SafeLoader — RCE if "
                        "input is untrusted; use yaml.safe_load() instead"
                    ),
                )
            )
            continue

        # subprocess.* with shell=True
        if qualified in {
            "subprocess.run",
            "subprocess.call",
            "subprocess.Popen",
        } and _has_shell_true(node):
            findings.append(
                Finding(
                    kind="deserialization",
                    severity="medium",
                    file=file,
                    line=getattr(node, "lineno", 0),
                    detail=(
                        f"{qualified}(..., shell=True) — command injection "
                        "if any arg is user-controlled"
                    ),
                )
            )
    return findings


def _yaml_load_is_safe(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "Loader":
            value_repr = ast.unparse(kw.value)
            if "Safe" in value_repr or "safe" in value_repr:
                return True
    return False


def _has_shell_true(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
            if kw.value.value is True:
                return True
    return False


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _call_name(node: ast.AST) -> str | None:
    """Return the leaf name of a Call's func node, or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _qualified_call_name(node: ast.AST) -> str | None:
    """Return ``module.attr`` for ``module.attr(...)`` or ``module.sub.attr``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


# ---------------------------------------------------------------------------
# File-level scan + aggregation
# ---------------------------------------------------------------------------


def scan_file(path: Path, *, repo_root: Path | None = None) -> list[Finding]:
    """Scan one Python file. Non-Python or unreadable files return ``[]``."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    # Always emit POSIX-style file labels. Findings end up in PR
    # markdown comments and JSON sidecar files — both contexts
    # expect forward slashes regardless of which OS produced them.
    # ``str(PathWindows(...))`` would otherwise leak backslashes
    # into output that downstream tooling can't grep cleanly.
    if repo_root is not None:
        try:
            rel = path.resolve().relative_to(repo_root.resolve())
            file_label = rel.as_posix()
        except ValueError:
            file_label = path.as_posix()
    else:
        file_label = path.as_posix()

    findings: list[Finding] = []
    # Secret scanning runs on the raw source — no AST parse needed
    # and works on broken Python too.
    findings.extend(check_secrets(source, file_label))

    try:
        tree = ast.parse(source, filename=file_label)
    except SyntaxError:
        return findings  # secrets-only on parse-failed files

    findings.extend(check_dynamic_code(tree, file_label))
    findings.extend(check_path_traversal(tree, file_label))
    findings.extend(check_deserialization(tree, file_label))
    return findings


def scan_paths(paths: list[Path], *, repo_root: Path | None = None) -> list[Finding]:
    """Scan a list of files. Skips non-Python paths silently."""
    out: list[Finding] = []
    for p in paths:
        if p.suffix != ".py":
            continue
        out.extend(scan_file(p, repo_root=repo_root))
    return out


def filter_by_severity(findings: list[Finding], threshold: str) -> list[Finding]:
    """Return only findings at or above ``threshold``."""
    min_rank = SEVERITY_ORDER[threshold]
    return [f for f in findings if SEVERITY_ORDER.get(f.severity, 0) >= min_rank]


# ---------------------------------------------------------------------------
# Output: markdown PR comment
# ---------------------------------------------------------------------------


def render_comment(findings: list[Finding], *, threshold: str) -> str:
    """Render a deterministic markdown PR comment.

    Findings are grouped by kind, then sorted by (severity desc, file,
    line) within each group. The comment carries a stable marker so the
    workflow can edit-in-place.
    """
    lines: list[str] = [COMMENT_MARKER, "## Security scan", ""]

    if not findings:
        lines.append("✅ No findings at or above severity `" + threshold + "`.")
        lines.append("")
        lines.append(COMMENT_MARKER)
        return "\n".join(lines) + "\n"

    blocking = filter_by_severity(findings, threshold)
    info_only = [f for f in findings if f not in blocking]

    if blocking:
        lines.append(
            f"⚠️ {len(blocking)} finding(s) at or above severity `{threshold}` "
            "— maintainer review required."
        )
    else:
        lines.append(
            f"ℹ️ {len(info_only)} informational finding(s) below severity "
            f"`{threshold}` — non-blocking; triage during W0.11."
        )
    lines.append("")

    by_kind: dict[str, list[Finding]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)

    kind_titles = {
        "dynamic-code": "Dynamic code execution",
        "path-traversal": "Path traversal",
        "secret": "Hardcoded secrets",
        "deserialization": "Unsafe deserialization",
    }

    for kind in ("secret", "dynamic-code", "deserialization", "path-traversal"):
        items = by_kind.get(kind)
        if not items:
            continue
        lines.append(f"### {kind_titles[kind]}")
        lines.append("")
        lines.append("| Severity | File | Line | Detail |")
        lines.append("|---|---|---:|---|")
        items_sorted = sorted(
            items, key=lambda x: (-SEVERITY_ORDER.get(x.severity, 0), x.file, x.line)
        )
        for f in items_sorted:
            lines.append(f"| `{f.severity}` | `{f.file}` | {f.line} | {f.detail} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Findings are produced by the conservative stdlib scan in "
        "`scripts/security_scan.py`. False positives are expected — "
        "the deeper attune-ai `/security-audit` pass (Phase 4 W0.9) "
        "is the secondary sweep on real signals._"
    )
    lines.append("")
    lines.append(COMMENT_MARKER)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="security_scan",
        description=(
            "Per-PR security scan. Checks dynamic-code, path-traversal, "
            "hardcoded secrets, and unsafe deserialization. Exits 0 "
            "(clean at threshold), 1 (findings at/above threshold), "
            "or 2 (validation error)."
        ),
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        required=True,
        help="Files to scan. Non-Python paths are skipped.",
    )
    parser.add_argument(
        "--severity-threshold",
        choices=["low", "medium", "high"],
        default="high",
        help=(
            "Findings at or above this severity drive exit code 1; "
            "lower-severity findings are still reported but don't "
            "fail the gate."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root for portable file labels (default: cwd).",
    )
    parser.add_argument(
        "--comment-out",
        type=Path,
        default=None,
        help="Optional path to write a markdown PR-comment body.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write findings as JSON (for tooling).",
    )
    args = parser.parse_args(argv)

    # Validate input files exist (the workflow passes git-diff output;
    # a stale list would silently scan nothing and report green — bad).
    missing = [p for p in args.paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: path does not exist: {p}", file=sys.stderr)
        return 2

    findings = scan_paths(args.paths, repo_root=args.repo_root)

    if args.comment_out is not None:
        args.comment_out.parent.mkdir(parents=True, exist_ok=True)
        args.comment_out.write_text(
            render_comment(findings, threshold=args.severity_threshold),
            encoding="utf-8",
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([f.to_dict() for f in findings], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    blocking = filter_by_severity(findings, args.severity_threshold)
    for f in findings:
        marker = "FAIL" if f in blocking else "info"
        print(
            f"{marker} {f.kind} ({f.severity}) {f.file}:{f.line}: {f.detail}",
            file=sys.stderr,
        )

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
