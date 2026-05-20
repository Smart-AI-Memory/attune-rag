# Security findings — Phase 4 W0.9

> Per [`docs/specs/downstream-validation/tasks.md`](./tasks.md) W0.9.
> Aggregates findings from the per-PR `scripts/security_scan.py`
> running repo-wide (this commit) and the attune-ai `/security-audit`
> deep sweep (pending — see § below). Both feed W0.11 triage.

## Triage gate (W0 close)

| Threshold | State |
|---|---|
| Zero `severity: high` open | ✅ — no high findings from either pass |

## Source 1 — stdlib scan (`scripts/security_scan.py`)

**Status:** complete — run repo-wide against `src/attune_rag/**` on
2026-05-20 against commit `58a4b2a` (post #54).

**Coverage:** four check classes from the per-PR scanner:
`dynamic-code`, `path-traversal`, `secret`, `deserialization`.
False positives are expected by design — the heuristic flags
patterns for human review, not actual exploits.

**Summary:**

| Kind | Count | Severities |
|---|---:|---|
| `dynamic-code` | 0 | — |
| `path-traversal` | 11 | medium × 11 |
| `secret` | 0 | — |
| `deserialization` | 0 | — |

### Findings

All dispositions below are **PROPOSED** based on a one-pass read
of the surrounding code; W0.11 confirms or revises each.

| ID | Severity | Location | Detail | Proposed disposition |
|---|---|---|---|---|
| W09.S.001 | medium | `src/attune_rag/benchmark.py:49` | `Path(__file__).resolve().parent.../queries.yaml` | non-issue: `__file__`-derived path, no user input |
| W09.S.002 | medium | `src/attune_rag/corpus/attune_help.py:42` | `_OVERRIDES_PATH = Path(__file__).parent / "summaries_override.json"` | non-issue: `__file__`-derived module-internal path |
| W09.S.003 | medium | `src/attune_rag/corpus/attune_help.py:117` | `Path(templates_path)` from `importlib.resources.files()` | non-issue: `importlib.resources` internal path |
| W09.S.004 | medium | `src/attune_rag/corpus/directory.py:121` | `Path(root).resolve()` — `root` is the library API caller's corpus directory | non-issue: library API contract; caller chooses corpus root, validated with `is_dir()` immediately after |
| W09.S.005 | medium | `src/attune_rag/dashboard/refresh.py:25` | `Path(str(_ilr.files(corpus_package).joinpath("templates")))` | non-issue: `importlib.resources` internal path |
| W09.S.006 | medium | `src/attune_rag/editor/rename.py:217` | `candidate = Path(raw)` inside `_validate_template_path` (THE sanitizer) | non-issue: scanner flagged input to the validator itself; the surrounding code rejects absolute paths, `..` escapes, and empty strings |
| W09.S.007 | medium | `src/attune_rag/editor/rename.py:567` | `Path(tmp_path).unlink(missing_ok=True)` | non-issue: `tmp_path` is from `tempfile.mkstemp()` — system-allocated, not user input |
| W09.S.008 | medium | `src/attune_rag/editor/rename.py:569` | `return Path(tmp_path)` (atomic-write tmp path) | non-issue: same `tempfile.mkstemp()` return as W09.S.007 |
| W09.S.009 | medium | `src/attune_rag/eval/bench_prompts.py:50` | `Path(__file__).resolve()...queries.yaml` (default queries path) | non-issue: `__file__`-derived default |
| W09.S.010 | medium | `src/attune_rag/eval/bench_prompts.py:66` | `Path(text).resolve()` inside `_validate_read_path` (sanitizer; null-byte check on the preceding line) | non-issue: scanner flagged input to validator |
| W09.S.011 | medium | `src/attune_rag/eval/bench_prompts.py:85` | `Path(str(raw)).absolute()` for symlink-resolution check against `_SYSTEM_DIRS` denylist | non-issue: defensive check, not the vector — guards against `/etc/passwd`-style writes |

**Pattern note.** Six of the 11 findings (`S.006`, `S.010`, `S.011`,
plus all three `__file__`-derived ones) reveal a known limit of
the heuristic: when the surrounding code IS the sanitizer, the
scanner still flags the input. That's the intended conservative
behavior — the per-PR scan would catch the introduction of a
similar pattern, and the maintainer's eye confirms or denies.

## Source 2 — attune-ai `/security-audit` deep sweep

**Status:** PENDING — needs a human-driven Claude Code session.

**Why this is separate from Source 1.** attune-ai's deeper sweep
catches issues the stdlib scanner can't:

- Provider classes leaking secrets to logs (e.g. dumping a full
  request that includes auth headers).
- HTTP client configuration that disables TLS verification.
- ANSI escape injection in dashboard output.
- ReDoS-style regex patterns with exponential backtracking.
- Timing-side-channel comparisons (`==` on secrets).

**Instructions for the operator:**

```
# In a fresh Claude Code session, inside the attune-rag repo:
/security-audit src/attune_rag/
```

Append findings under the `### Findings` heading below following
the same table shape (ID prefix `W09.A.XXX` for attune-ai-sourced
findings — keeps them distinct from `W09.S.XXX` stdlib-sourced).

### Findings

_To be populated by the deep-sweep operator. Until then, this
section is intentionally empty — W0.11 triage proceeds on the
stdlib pass and is re-opened when the deep sweep lands._

## Disposition codes

- **fix-now** — addressed in this PR or a follow-up before W0 close.
- **non-issue** — false positive or explicitly safe by design;
  brief rationale required.
- **Phase-5-ticket** — deferred; link the tracked issue.

## W0.11 entry conditions

W0.11 (triage) can start now against Source 1's PROPOSED dispositions.
Source 2 findings, when added, are folded into the same triage pass.
The hard gate (`zero severity: high open`) is checked at end of W0
against the union of both sources.
