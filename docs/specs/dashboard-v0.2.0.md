# Spec: Shipped Dashboard Template (attune-rag v0.2.0)

Status: shipped in 0.1.6 — implementation diverged from locked
decisions; see "Implementation Note" below. Original spec body
preserved as historical design context.
Target version: 0.2.0 (originally planned; feature landed early in 0.1.6)
Owner: Patrick
Created: 2026-04-22
Reconciled: 2026-04-24

---

## Objective

Ship a reusable Cowork-compatible HTML dashboard as part of
attune-rag itself, so any attune-rag user can render a live
benchmark/freshness dashboard with a single CLI call. The
dashboard file is self-contained, embeds its own refresh
command, and works in any Cowork session that has access to
the installed attune-rag package.

Today the only copy lives at
`attune-ai/scripts/attune_rag_dashboard_refresh.py` and is
coupled to the attune-ai repo layout. v0.2.0 promotes it into
attune-rag as first-class functionality.

---

## Scoping Decisions (Locked)

The following was decided during the planning Q&A. These are
not open questions — they define v0.2.0's shape.

1. Entry point: CLI only. Exposed as
   `attune-rag dashboard {render,refresh}`. No Python API is
   documented or committed as public; `attune_rag.dashboard`
   is internal.
2. Refresh strategy: the refresh command is baked into the
   rendered HTML at render time, via a `--refresh-cmd` flag.
   The dashboard invokes it through `window.cowork.callMcpTool`
   exactly as the existing attune-ai artifact does.
3. Release scope: full refactor into a new `dashboard`
   subpackage with golden tests and this spec doc. Not a hack
   shipped in the scripts directory.
4. Configurability: only the corpus package name is
   parameterizable (`--corpus-package`, default `attune_help`).
   Queries.yaml path, k, trials, and thresholds stay as
   sensible defaults inside the refresh implementation.

---

## Implementation Note (2026-04-24)

The shipped implementation diverged from the locked scoping
decisions above. Treat this section as the source of truth for
what's actually in the package; the rest of this document
captures the original design intent.

**What shipped (as of 0.1.6):**

- `dashboard render` bakes the snapshot itself at render time
  rather than embedding a `--refresh-cmd` that the browser
  invokes via MCP. The rendered HTML is fully self-contained
  with its snapshot payload; no live refresh from the browser.
- Two sentinels in the template: `__ATTUNE_SNAPSHOT__` and
  `__ATTUNE_TITLE__` (the spec's `__REFRESH_CMD__` /
  `__CORPUS_PACKAGE__` model was dropped).
- `render(out, snapshot, title) -> Path` signature — takes a
  prebuilt snapshot dict, not a refresh command string.
- `dashboard render` gained an optional `--open` flag (opens
  the file in the default browser). The spec listed this as an
  open question; answered yes.
- A third subcommand — `dashboard show` — was added for a Rich
  terminal dashboard (not in original scope). It calls
  `build_snapshot` directly and renders to the console with
  Rich tables.
- `dashboard refresh` still exists and still emits one JSON
  object to stdout (the Snapshot Shape contract is preserved).

**Why the divergence:**

The MCP-refresh model required a live Cowork session to be
useful. Baking the snapshot at render time makes the HTML
portable (shareable, attachable to tickets, viewable offline)
at the cost of staleness. Re-running `dashboard render`
regenerates the file — simpler UX than an MCP round-trip.

The path-validation, security, and snapshot-shape requirements
from the original spec all carried through unchanged.

---

## Non-Goals

- No React / framework dependency. Plain HTML + vanilla JS +
  Chart.js UMD from CDN, same as the existing artifact.
- No server. The dashboard is a static file that hits Cowork's
  MCP bridge directly from the browser.
- No multi-corpus comparison view. One corpus per dashboard.
- No persistence beyond what Cowork already provides for its
  artifacts. Snapshots are ephemeral per refresh.
- No auth / access control — inherits whatever the Cowork
  session has.

---

## Module Layout

```
src/attune_rag/dashboard/
├── __init__.py          # empty; this is internal
├── render.py            # template load + refresh_cmd bake
├── refresh.py           # snapshot generation (moved from attune-ai)
└── templates/
    └── dashboard.html   # self-contained template, loaded via
                         # importlib.resources
```

Rationale: matches attune-rag's existing subpackage style
(`corpus/`, `providers/`, `eval/`) and keeps the template
shipped as package data so no filesystem lookup is needed at
install time.

---

## CLI Surface

Added as a new subcommand in `src/attune_rag/cli.py` alongside
`query`, `corpus-info`, `providers`.

### `attune-rag dashboard render`

Writes a self-contained HTML file with the refresh command
already embedded.

```
attune-rag dashboard render \
    --out ~/attune-rag-dashboard.html \
    --refresh-cmd "uv run attune-rag dashboard refresh" \
    [--corpus-package attune_help] \
    [--title "attune-rag dashboard"]
```

Required flags:

- `--out PATH` — destination file. Validated against path
  traversal and system directories (per attune repo
  standards; reuse `_validate_file_path` or port it).
- `--refresh-cmd CMD` — command string the dashboard will
  execute via `window.cowork.callMcpTool`'s bash bridge on
  every refresh click / page load.

Optional flags:

- `--corpus-package NAME` — defaults to `attune_help`.
  Passed through to `dashboard refresh` inside the baked
  command if the user doesn't override it themselves.
- `--title TEXT` — defaults to `attune-rag dashboard`.

Exit codes: `0` success, `2` validation error (bad path,
missing refresh-cmd).

### `attune-rag dashboard refresh`

Emits a single JSON snapshot to stdout. Shape matches the
existing attune-ai refresh script output (see Snapshot Shape
below). This is what the dashboard invokes on every reload.

```
attune-rag dashboard refresh [--corpus-package attune_help]
```

Stdout contract: exactly one JSON object, preceded by nothing
useful. Callers must parse from the first `{` to skip any
structlog console output (existing lesson in attune-ai's
CLAUDE.md — structlog writes to stdout by default). The
rendered dashboard already handles this; keeping the
convention means the existing JS parser in the template
doesn't need to change.

Exit codes: `0` success with a complete snapshot, `1` partial
snapshot (e.g. queries.yaml not found) still emitted but with
per-section errors, `2` unrecoverable.

---

## Refresh Command Embedding

At render time, `--refresh-cmd` is JSON-escaped and substituted
into a single placeholder in the template:

```html
<script>
  window.__REFRESH_CMD__ = {{ refresh_cmd | tojson }};
  window.__CORPUS_PACKAGE__ = {{ corpus_package | tojson }};
</script>
```

The template uses a minimal substitution — either Jinja2 (already
in the venv, see jinja2 3.1.6) or a plain `str.replace` of two
sentinel tokens. Decision deferred to implementation; Jinja2 is
acceptable but not required. If Jinja2 is used, it stays a dev
dependency — the rendered output is static HTML with no runtime
template engine.

Why embed rather than serve: the dashboard is opened as a local
file from whatever path `--out` writes to; there's no server
context to read flags from. Baking is the only way to make a
standalone HTML file remember its own data source.

---

## Snapshot Shape

Preserved from the current refresh script so the JS in the
template doesn't need to change:

```json
{
  "timestamp": "2026-04-22T14:03:11Z",
  "retrieval": {
    "retriever": "cascade",
    "corpus": "attune_help",
    "precision_at_1": 0.82,
    "recall_at_k": 0.94,
    "mean_latency_ms": 12.3,
    "max_latency_ms": 47.1,
    "total_queries": 40,
    "k": 3,
    "per_difficulty": {"easy": {...}, "medium": {...}, "hard": {...}},
    "per_feature": {"security-audit": {...}, ...},
    "per_query": [{...}, ...]
  },
  "freshness": {
    "attune_help_version": "0.7.1",
    "summaries_by_path_keys": 287,
    "kinds": ["concept", "task", "reference", ...],
    "kind_totals": {"concept": 26, "task": 26, ...},
    "features": ["security-audit", ...],
    "per_feature": {"security-audit": {"kind_counts": {...}}, ...}
  }
}
```

Any field the consumer can't render is rendered as "n/a" by
the template — forward compatible with new sections.

---

## Implementation Milestones

Iterative per the Spec-Driven workflow. Each milestone ends
with a green test suite and a usable artifact.

### M1: Move refresh logic into the package

- Copy `attune_rag_dashboard_refresh.py` logic into
  `src/attune_rag/dashboard/refresh.py` as a proper module
  with typed functions and docstrings.
- Add `attune_rag dashboard refresh` CLI subcommand that
  wires to it.
- Unit tests: `tests/unit/test_dashboard_refresh.py` —
  snapshot shape contract, corpus-package override, error
  paths when queries.yaml is missing.
- Smoke test: `uv run attune-rag dashboard refresh | head -c
  500` prints valid JSON.

### M2: Ship the template as package data

- Extract the existing artifact HTML (from the attune-ai
  Cowork artifact) into
  `src/attune_rag/dashboard/templates/dashboard.html` with
  two sentinel tokens.
- Add `[tool.hatch.build.targets.wheel]` or equivalent
  `package-data` config so the template ships in the wheel.
- Sanity test: `importlib.resources.files("attune_rag.dashboard")
  .joinpath("templates/dashboard.html").read_text()` works
  post-install.

### M3: Render command

- Implement `src/attune_rag/dashboard/render.py::render(out,
  refresh_cmd, corpus_package, title) -> Path`.
- Add `attune-rag dashboard render` CLI subcommand.
- Path validation: reject null bytes, system directories,
  and paths outside the user's home unless `--out` is an
  explicit absolute path they provided.
- Golden test: `tests/unit/test_dashboard_render.py`
  renders to tmp_path, reads the file back, and asserts both
  sentinel values substituted correctly. Uses a small
  deterministic refresh-cmd string and `corpus_package=foo`.
- No network, no subprocess — rendering is pure string ops.

### M4: Version bump + changelog + README

- Bump `pyproject.toml` to `0.2.0`.
- Add a CHANGELOG entry under "0.2.0 — Dashboard".
- Update README with a three-line usage example.
- Tag and publish per the existing release workflow.

### M5 (post-merge): retire the attune-ai-side script

- Delete `attune-ai/scripts/attune_rag_dashboard_refresh.py`.
- Update the attune-ai Cowork artifact to invoke
  `uv run attune-rag dashboard refresh` instead of the local
  script. No behavior change — same JSON contract.
- Document the deprecation in attune-ai's lessons log.

---

## Test Plan

All tests live under `tests/unit/` and must pass on the
existing CI matrix (Python 3.10–3.13, Ubuntu + macOS).

1. `test_dashboard_refresh.py`
   - Happy path: queries.yaml present → full snapshot with
     all sections populated.
   - Missing queries.yaml: exit code 1, JSON still emitted
     with `retrieval.error` populated.
   - Corpus package override: `--corpus-package foo` flows
     through to the freshness section's corpus lookup.
   - Stdout contract: first `{` marks the JSON start.

2. `test_dashboard_render.py`
   - Renders to `tmp_path`, verifies file exists.
   - Verifies both `window.__REFRESH_CMD__` and
     `window.__CORPUS_PACKAGE__` are set to the passed
     values (JSON-escaped).
   - Rejects a path inside `/etc`, `/sys`, `/proc`, `/dev`.
   - Rejects a path containing a null byte.
   - Rejects a path whose parent doesn't exist (with a
     clear error message, not a raw OSError).

3. `test_dashboard_cli.py`
   - `attune-rag dashboard render --help` exits 0.
   - `attune-rag dashboard refresh --help` exits 0.
   - `attune-rag dashboard render` without `--out` exits 2.
   - End-to-end: render to tmp, then grep for the baked
     refresh command.

No integration tests against real Cowork — out of scope.

---

## Dependencies

No new required dependencies. Jinja2 is already in the venv
and may be used for rendering; if adopted, it becomes a core
runtime dep. A simpler `str.replace` of two sentinel tokens is
acceptable and preferred if Jinja2 is only used here (avoids
adding a runtime dep for one substitution). Implementation
decision made during M3; spec allows either.

Chart.js stays CDN-loaded inside the template — no Python
dep. SRI hash from the existing artifact is preserved.

---

## Security

Path-traversal and null-byte checks on `--out` following the
pattern documented in
`attune-ai/.claude/rules/attune/coding-standards-index.md`.
The refresh command is baked into a static file under the
user's control, so shell-injection concerns are the user's —
but the dashboard must not `eval()` or string-concatenate it
at page load. It must be passed as a single argument to the
MCP bash tool exactly as written.

The rendered HTML runs in the Cowork sandbox; no privilege
elevation beyond what Cowork already permits for artifacts.

---

## Migration Notes

For users on 0.1.x with their own dashboard copies:

- No breaking API changes. `attune_rag.retrieval`,
  `attune_rag.pipeline`, and the CLI `query` / `corpus-info`
  / `providers` subcommands are untouched.
- The attune-ai dashboard artifact keeps working during the
  transition. Post-M5 it switches to invoking the installed
  CLI rather than a local script.
- Users who forked the old refresh script can either keep
  their fork (it still works — private functions in
  `attune_rag.benchmark` are stable for 0.2.x) or migrate to
  `attune-rag dashboard refresh` for zero maintenance.

---

## Open Questions

1. Should the render step support a `--open` flag that
   auto-opens the file in the user's browser? Nice-to-have,
   not in v0.2.0 scope. Revisit if users ask.
2. Do we want a deterministic snapshot mode (fixed seed) for
   reproducible demo screenshots? Skip unless a second use
   case materializes.

---

## References

- Existing artifact script:
  `attune-ai/scripts/attune_rag_dashboard_refresh.py`
- Cowork artifact API: `window.cowork.callMcpTool`, eager
  on-mount refresh pattern
- attune repo coding standards:
  `attune-ai/.claude/rules/attune/coding-standards-index.md`
- Spec-driven workflow preference:
  `attune-ai/CLAUDE.md`
