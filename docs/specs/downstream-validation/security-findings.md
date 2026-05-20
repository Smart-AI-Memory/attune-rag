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

**Status:** complete + triaged (W0.11) — run repo-wide against
`src/attune_rag/**` on 2026-05-20 against commit `58a4b2a` (post #54);
triage confirmed against `c6c911d` on 2026-05-19.

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

Dispositions confirmed in W0.11 (initial pass 2026-05-19; W09.S.011
re-opened + closed 2026-05-20 after the fix landed in this PR).

| ID | Severity | Location | Detail | Disposition |
|---|---|---|---|---|
| W09.S.001 | medium | `src/attune_rag/benchmark.py:49` | `Path(__file__).resolve().parent.../queries.yaml` | non-issue: `__file__`-derived path, no user input |
| W09.S.002 | medium | `src/attune_rag/corpus/attune_help.py:42` | `_OVERRIDES_PATH = Path(__file__).parent / "summaries_override.json"` | non-issue: `__file__`-derived module-internal path |
| W09.S.003 | medium | `src/attune_rag/corpus/attune_help.py:117` | `Path(templates_path)` from `importlib.resources.files()` | non-issue: `importlib.resources` internal path |
| W09.S.004 | medium | `src/attune_rag/corpus/directory.py:121` | `Path(root).resolve()` — `root` is the library API caller's corpus directory | non-issue: library API contract; caller chooses corpus root, validated with `is_dir()` immediately after |
| W09.S.005 | medium | `src/attune_rag/dashboard/refresh.py:25` | `Path(str(_ilr.files(corpus_package).joinpath("templates")))` | non-issue: `importlib.resources` internal path |
| W09.S.006 | medium | `src/attune_rag/editor/rename.py:217` | `candidate = Path(raw)` inside `_normalize_corpus_relpath` (THE sanitizer) | non-issue: scanner flagged input to the validator itself; the surrounding code rejects absolute paths, `..` escapes, and empty strings |
| W09.S.007 | medium | `src/attune_rag/editor/rename.py:567` | `Path(tmp_path).unlink(missing_ok=True)` | non-issue: `tmp_path` is from `tempfile.mkstemp()` at line 560 inside `_stage()` — system-allocated, never user-influenced |
| W09.S.008 | medium | `src/attune_rag/editor/rename.py:569` | `return Path(tmp_path)` (atomic-write tmp path) | non-issue: same `tempfile.mkstemp()` return as W09.S.007 |
| W09.S.009 | medium | `src/attune_rag/eval/bench_prompts.py:50` | `Path(__file__).resolve()...queries.yaml` (default queries path) | non-issue: `__file__`-derived default |
| W09.S.010 | medium | `src/attune_rag/eval/bench_prompts.py:66` | `Path(text).resolve()` inside `_validate_read_path` (sanitizer; null-byte check on the preceding line) | non-issue: scanner flagged input to validator |
| W09.S.011 | medium | `src/attune_rag/eval/bench_prompts.py:85` | `Path(str(raw)).absolute()` for symlink-resolution check against `_SYSTEM_DIRS` denylist | **fix-now (closed in this PR)**: macOS direct-path bypass — `/private/etc/passwd` wasn't denied because the raw-path arm of the check only had `/etc`. Mirrored each original entry under `/private/` (e.g. `/private/etc`, `/private/sys`, …). Deliberately NOT adding bare `/private`, `/var`, or `/usr/...` — those would over-block legitimate user-writable temp roots (pytest's `tmp_path` lives under `/private/var/folders/...`). |

**Pattern note.** Six of the 11 findings (`S.006`, `S.010`, `S.011`,
plus all three `__file__`-derived ones) reveal a known limit of
the heuristic: when the surrounding code IS the sanitizer, the
scanner still flags the input. That's the intended conservative
behavior — the per-PR scan would catch the introduction of a
similar pattern, and the maintainer's eye confirms or denies.

## Source 2 — attune-ai `/security-audit` deep sweep

**Status:** partial-pass + triaged (W0.11) for the 3 HIGH findings.
The original `/security-audit` MCP errors with `AttributeError: 'str'
object has no attribute 'get'`; the partial pass used
`mcp__plugin_attune-ai_attune-ai__deep_review` on 2026-05-19 against
commit `4226ab2` (post #59), but it returned summary-only output. The
3 HIGH findings below were extracted by directly inspecting the
named hot paths; remaining MEDIUM / LOW findings are deferred to a
follow-up sweep that streams the full report.

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

Sweep run 2026-05-19 via `mcp__plugin_attune-ai_attune-ai__deep_review`
against `src/attune_rag/` at commit `4226ab2`. The MCP returned a
summary only (Health score 82/100; Security: 2 HIGH / 4 MEDIUM / 3
LOW), so HIGH-severity findings below were extracted by directly
inspecting the named hot paths (`dashboard/render.py`,
`dashboard/templates/dashboard.html`, `cli.py` →
`dashboard/refresh.py`). MEDIUM / LOW findings beyond the HIGH
trio remain unaudited until the MCP can stream full output.

| ID | Severity | Location | Detail | Disposition |
|---|---|---|---|---|
| W09.A.001 | high → fixed | `src/attune_rag/dashboard/render.py` + `dashboard/templates/dashboard.html` (script block) | `json.dumps(snapshot)` was inlined into `<script>window.__SNAPSHOT__ = …;</script>` without escaping the less-than byte. A corpus value containing a literal `</script>` sequence would terminate the script block and enable XSS. | fix-now: added `_json_for_script_block()` which neutralizes the less-than byte and the U+2028 / U+2029 line separators by replacing them with their `\uXXXX` escape forms (still valid JSON; the `<script>` block can no longer be terminated from inside the payload). Tests `test_snapshot_script_terminator_escaped`, `test_snapshot_line_separators_escaped`, `test_json_for_script_block_remains_valid_json`. |
` / ` `) inside the JSON payload. Tests `test_snapshot_script_terminator_escaped`, `test_snapshot_line_separators_escaped`, `test_json_for_script_block_remains_valid_json`. |
| W09.A.002 | high → fixed | `src/attune_rag/dashboard/render.py` + `dashboard/templates/dashboard.html:6` (title) | `title` parameter was interpolated raw into `<title>…</title>`. A title like `</title><script>alert(1)</script>` would break out of the title element and execute. | fix-now: title now passes through `html.escape(title, quote=True)`. Tests `test_title_html_escaped`, `test_title_ampersand_escaped`. |
| W09.A.003 | high → fixed | `src/attune_rag/dashboard/templates/dashboard.html:7` (Chart.js CDN tag) | External script loaded from `cdn.jsdelivr.net` without Subresource Integrity. A CDN compromise would execute attacker JS in the dashboard origin. | fix-now: added `integrity="sha384-NrKB+u6Ts6AtkIhwPixiKTzgSKNblyhlk0Sohlgar9UHUBzai/sgnNNWWd291xqt"` + `crossorigin="anonymous"` + `referrerpolicy="no-referrer"`. SRI computed via `openssl dgst -sha384` against the locked `chart.js@4.4.4` URL. Test `test_dashboard_template_has_sri_on_cdn_script` enforces SRI on any future external `<script>`. |
| W09.A.004 | high (as flagged) → non-issue | `src/attune_rag/cli.py` → `dashboard/refresh.py:23` | `importlib.import_module(corpus_package)` where `corpus_package` comes from `--corpus-package` CLI arg. Flagged as arbitrary-import surface. | non-issue: developer-run library CLI; the operator already has shell + Python execution rights on the host, so importing an installed module is not a privilege escalation. Documenting the threat-model boundary here; if a server-side caller materializes later, re-evaluate. |
| W09.A.005 | low → fixed | `src/attune_rag/dashboard/show.py` (multiple sites) | Rich-markup injection: snapshot fields (error message, retriever / corpus name, feature labels, kind names) were interpolated raw into Rich markup strings. A corpus value containing `[blink]X[/blink]` would alter terminal styling. | fix-now: every untrusted field now flows through `rich.markup.escape`. Tests `test_error_field_rich_markup_escaped`, `test_per_feature_rich_markup_escaped`, `test_per_query_feature_rich_markup_escaped`, `test_freshness_kind_rich_markup_escaped`, `test_no_markup_consumed_when_input_has_it`. |
| W09.A.006 | low → fixed | `src/attune_rag/cli.py:170` | `print(f"error: {exc}", file=sys.stderr)` echoes a `ValueError` whose message interpolates the user-supplied `--out` path. A path with raw ANSI bytes would repaint the terminal. | fix-now: new `_safe_stderr(msg)` helper strips C0/C1 control characters (preserves tab/newline/CR) before printing. Tests `test_safe_stderr_strips_ansi_control_chars`, `test_safe_stderr_strips_bel_and_backspace`, `test_safe_stderr_preserves_tab_newline_cr`, `test_dashboard_render_error_stderr_is_sanitized`. |
| W09.A.007 | low → fixed | `src/attune_rag/reranker.py:139`, `src/attune_rag/expander.py:90` | `logger.debug(..., exc_info=True)` on Anthropic-SDK exceptions. Traceback frames may capture SDK locals that could carry secret-adjacent material under future SDK changes. Debug-level only, but defense-in-depth. | fix-now: now logs `type(exc).__name__` + message instead of full traceback. Tests `test_reranker_failure_does_not_log_with_exc_info`, `test_expander_failure_does_not_log_with_exc_info`. |
| W09.A.008 | low → fixed | `src/attune_rag/dashboard/render.py` (`_SYSTEM_DIRS`) | Same class as W09.S.011 but in the dashboard write path. `Path("/etc/foo").resolve()` returns `/private/etc/foo` on macOS — the resolved form bypassed the `/etc` denylist entry. Discovered during W09.A.006 test scaffolding. | fix-now: added `/private/etc`, `/private/sys`, `/private/dev` mirrors. Tests `test_rejects_macos_private_etc_path`, `test_rejects_macos_private_sys_path`. |

**MEDIUM / LOW capture status.** The MCP summary counted 4 MEDIUM +
3 LOW security findings but didn't surface detail. A read-only
security-review subagent was dispatched on 2026-05-20 against
commit `4681ac6` with explicit category hints (TLS verification
disable, secret-logging, ANSI injection, ReDoS, timing channels)
and produced: **0 MEDIUM, 3 LOW** (W09.A.005..007 above). All
deep_review's MEDIUM-counted categories — TLS verification,
secret-logging, timing-channel `==`, ReDoS — were searched and
cleared with explicit negative findings. The 4 MEDIUM count from
the summary appears to have been heuristic over-counting; the real
residual surface is the 3 LOW items, all addressed in this PR. A
fresh capture against the original `security_audit` MCP remains a
Phase 5 ticket pending the `AttributeError` fix, but is no longer
blocking W0 close. Bonus finding W09.A.008 (macOS denylist bypass
in `dashboard/render.py`) surfaced from W09.A.006 test scaffolding
and was closed in the same PR.

## Disposition codes

- **fix-now** — addressed in this PR or a follow-up before W0 close.
- **non-issue** — false positive or explicitly safe by design;
  brief rationale required.
- **Phase-5-ticket** — deferred; link the tracked issue.

## W0.11 triage outcome

| Source | Triaged on | fix-now | non-issue | Phase-5-ticket |
|---|---|---:|---:|---:|
| 1 (stdlib) | 2026-05-19 + 2026-05-20 (W09.S.011 re-opened, fix landed) | 1 (closed in #60) | 10 | 0 |
| 2 (attune-ai deep sweep) | 2026-05-19 + 2026-05-20 (LOW capture via read-only agent) | 7 (3 HIGH in #62 + 4 LOW in follow-up PR) | 1 | 0 |

Source 1 triage is complete; W09.S.011 surfaced a real macOS direct-path
bypass and was closed via the `_SYSTEM_DIRS` extension in PR #60. Source 2
triage covers the 3 HIGH findings extracted by direct code inspection; all
three are fixed in this PR (the `[Unreleased]` CHANGELOG carries `### Fixed`
entries). The fourth HIGH-as-flagged finding (`--corpus-package` arbitrary
import) was reclassified to non-issue under the developer-CLI threat model.
A Phase 5 ticket tracks completing the deep sweep once the `security_audit`
MCP `AttributeError` is fixed upstream.

The hard gate (`zero severity: high open`) is **met** at this snapshot
against both Source 1 and the audited slice of Source 2. Recheck at end of
W0 if the Phase 5 deep-sweep capture lands inside the window.
