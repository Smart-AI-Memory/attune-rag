# Spec: Shipped Dashboard Template (attune-rag)

## Phase 2: Design

**Status**: complete

### Drift from original plan (read first)

The shipped 0.1.6 implementation diverged from the locked scoping decisions in the original plan. **The shipped behavior is the source of truth**; the rest of this design.md captures the original design intent for historical context.

**What shipped (as of 0.1.6):**

- `dashboard render` **bakes the snapshot itself at render time** rather than embedding a `--refresh-cmd` that the browser invokes via MCP. The rendered HTML is fully self-contained with its snapshot payload; **no live refresh from the browser**.
- Two sentinels in the template: `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__` (the spec's `__REFRESH_CMD__` / `__CORPUS_PACKAGE__` model was dropped).
- `render(out, snapshot, title) -> Path` signature — takes a prebuilt snapshot dict, not a refresh command string.
- `dashboard render` gained an optional `--open` flag (opens the file in the default browser). Original spec listed this as an open question; answered yes.
- A third subcommand — `dashboard show` — was added for a Rich terminal dashboard (not in original scope). Calls `build_snapshot` directly and renders to the console with Rich tables.
- `dashboard refresh` still exists and still emits one JSON object to stdout (Snapshot Shape contract preserved).

**Why the divergence:**

The MCP-refresh model required a live Cowork session to be useful. **Baking the snapshot at render time makes the HTML portable** (shareable, attachable to tickets, viewable offline) at the cost of staleness. Re-running `dashboard render` regenerates the file — simpler UX than an MCP round-trip.

The path-validation, security, and snapshot-shape requirements from the original spec all carried through unchanged.

### Architecture

#### Module layout

```
src/attune_rag/dashboard/
├── __init__.py          # empty; this is internal
├── render.py            # template load + sentinel substitution
├── refresh.py           # snapshot generation (moved from attune-ai)
├── show.py              # Rich terminal dashboard (added in shipped impl)
└── templates/
    └── dashboard.html   # self-contained template, loaded via
                         # importlib.resources
```

Rationale: matches attune-rag's existing subpackage style (`corpus/`, `providers/`, `eval/`) and keeps the template shipped as package data so no filesystem lookup is needed at install time.

### API changes

CLI-only (no Python API committed as public). Three subcommands added to `src/attune_rag/cli.py` alongside `query`, `corpus-info`, `providers`.

#### `attune-rag dashboard render` (shipped)

Writes a self-contained HTML file with the snapshot already baked in.

```
attune-rag dashboard render \
    --out ~/attune-rag-dashboard.html \
    [--corpus-package attune_help] \
    [--title "attune-rag dashboard"] \
    [--open]
```

Required flags:
- `--out PATH` — destination file. Validated against path traversal and system directories.

Optional flags:
- `--corpus-package NAME` — defaults to `attune_help`. Determines which corpus is benchmarked.
- `--title TEXT` — defaults to `attune-rag dashboard`.
- `--open` — open the rendered file in the default browser after writing.

Exit codes: `0` success, `2` validation error (bad path, missing required flag).

> **Original-plan signature** (preserved here for historical record): `--refresh-cmd CMD` was required and baked into the HTML for browser-side MCP invocation. Dropped in shipped impl in favor of baked-snapshot model.

#### `attune-rag dashboard refresh` (shipped)

Emits a single JSON snapshot to stdout. Shape matches the existing attune-ai refresh script output (see Snapshot Shape below).

```
attune-rag dashboard refresh [--corpus-package attune_help]
```

Stdout contract: exactly one JSON object, preceded by nothing useful. Callers parse from the first `{` to skip any structlog console output (existing lesson; the rendered dashboard already handles this).

Exit codes: `0` success with complete snapshot, `1` partial snapshot (e.g. `queries.yaml` not found) still emitted with per-section errors, `2` unrecoverable.

#### `attune-rag dashboard show` (shipped — added beyond original scope)

Rich terminal dashboard. Calls `build_snapshot` directly and renders to the console with Rich tables. No HTML, no file output.

### Data model changes

#### Snapshot shape (preserved verbatim from existing attune-ai script)

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

Any field the consumer can't render is rendered as "n/a" by the template — forward-compatible with new sections.

### UI/UX

The dashboard is a static HTML file. Key UX properties:

- **Self-contained:** opens in any browser; no server required.
- **Portable:** shareable via attachment / ticket / email; viewable offline.
- **Refresh model (shipped):** baked at render time. Re-run `attune-rag dashboard render` to regenerate. **No live refresh button** in the rendered HTML.
- **Chart.js loaded from CDN** with the existing artifact's SRI hash preserved.
- **Title customisable** via `--title`.
- **`--open` convenience flag** auto-opens the file in the user's default browser.

### Cross-layer impact

- **attune-rag** (primary): new `dashboard` subpackage, new CLI subcommands, template shipped as package data, version bump, changelog entry, README example.
- **attune-ai** (M5 only): delete `scripts/attune_rag_dashboard_refresh.py`; update Cowork artifact to invoke `uv run attune-rag dashboard refresh` instead of the local script. No JSON-shape change — the artifact JS is unchanged.

No impact on attune-gui, attune-help, or attune-author.

### Tradeoffs & alternatives

#### Refresh strategy

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| Bake snapshot at render time (shipped) | Portable; offline-viewable; simplest UX; no MCP coupling | Stale until next render; no live updates | **Yes (shipped 0.1.6)** |
| `--refresh-cmd` embedded; browser invokes via MCP (original plan) | Live updates inside Cowork | Requires live Cowork session to be useful; opaque shell-injection surface; not portable | No (deferred / dropped) |
| Server-backed dashboard | Always live | Daemon to run; auth concerns; out of scope | No |

#### Sentinel substitution mechanism

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| Plain `str.replace` of sentinel tokens | Zero new runtime deps; obvious; testable | Fragile if template grows complex | **Yes (shipped)** |
| Jinja2 rendering | Familiar; battle-tested | Adds a runtime dep for one substitution | No |

#### Configurability

Only the corpus package name is parameterizable (`--corpus-package`, default `attune_help`). Queries.yaml path, k, trials, and thresholds stay as sensible defaults inside the refresh implementation. Adding more knobs is explicitly deferred.

### Refresh command embedding (original-plan section, NOT shipped)

> Preserved for historical record. The shipped impl uses snapshot baking instead.

At render time, `--refresh-cmd` would have been JSON-escaped and substituted into a single placeholder in the template:

```html
<script>
  window.__REFRESH_CMD__ = {{ refresh_cmd | tojson }};
  window.__CORPUS_PACKAGE__ = {{ corpus_package | tojson }};
</script>
```

Why embed rather than serve: the dashboard is opened as a local file from whatever path `--out` writes to; there's no server context to read flags from. Baking is the only way to make a standalone HTML file remember its own data source. (This rationale also justifies the snapshot-baking approach that ultimately shipped.)

### Security

Path-traversal and null-byte checks on `--out` following the pattern documented in `attune-ai/.claude/rules/attune/coding-standards-index.md`. The refresh command (in the original plan) would have been baked into a static file under the user's control, so shell-injection concerns are the user's — but the dashboard must not `eval()` or string-concatenate it at page load. It must be passed as a single argument to the MCP bash tool exactly as written.

The rendered HTML runs in the Cowork sandbox (when used inside Cowork); no privilege elevation beyond what Cowork already permits for artifacts.

### Migration notes

For users on 0.1.x with their own dashboard copies:

- **No breaking API changes.** `attune_rag.retrieval`, `attune_rag.pipeline`, and the CLI `query` / `corpus-info` / `providers` subcommands are untouched.
- **The attune-ai dashboard artifact keeps working** during the transition. Post-M5 it switches to invoking the installed CLI rather than a local script.
- **Forks of the old refresh script** can either keep their fork (it still works — private functions in `attune_rag.benchmark` are stable for 0.2.x) or migrate to `attune-rag dashboard refresh` for zero maintenance.

### References

- Existing artifact script (pre-M5): `attune-ai/scripts/attune_rag_dashboard_refresh.py`
- Cowork artifact API: `window.cowork.callMcpTool`, eager on-mount refresh pattern
- attune repo coding standards: `attune-ai/.claude/rules/attune/coding-standards-index.md`
- Spec-driven workflow preference: `attune-ai/CLAUDE.md`
