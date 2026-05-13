# Spec: Shipped Dashboard Template (attune-rag)

## Phase 3: Tasks

**Status**: complete

### Implementation order

Iterative per the Spec-Driven workflow. Each milestone ends with a green test suite and a usable artifact.

| # | Task | Layer | Status | Notes |
|---|------|-------|--------|-------|
| M1.1 | Copy `attune_rag_dashboard_refresh.py` logic into `src/attune_rag/dashboard/refresh.py` as a proper module with typed functions and docstrings. | attune-rag | done | M1 — Move refresh logic into the package. |
| M1.2 | Add `attune-rag dashboard refresh` CLI subcommand wired to `dashboard/refresh.py`. | attune-rag | done | |
| M1.3 | Unit tests in `tests/unit/test_dashboard_refresh.py` — snapshot shape contract, corpus-package override, error paths when `queries.yaml` is missing. | attune-rag | done | |
| M1.4 | Smoke test: `uv run attune-rag dashboard refresh \| head -c 500` prints valid JSON. | attune-rag | done | |
| M2.1 | Extract the existing artifact HTML (from the attune-ai Cowork artifact) into `src/attune_rag/dashboard/templates/dashboard.html` with sentinel tokens. | attune-rag | done | M2 — Ship the template as package data. **Drift:** sentinels became `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__` instead of the spec's `__REFRESH_CMD__` / `__CORPUS_PACKAGE__`. |
| M2.2 | Add `[tool.hatch.build.targets.wheel]` (or equivalent) `package-data` config so the template ships in the wheel. | attune-rag | done | |
| M2.3 | Sanity test: `importlib.resources.files("attune_rag.dashboard").joinpath("templates/dashboard.html").read_text()` works post-install. | attune-rag | done | |
| M3.1 | Implement `src/attune_rag/dashboard/render.py::render(out, snapshot, title) -> Path`. | attune-rag | done | M3 — Render command. **Drift:** signature takes a prebuilt `snapshot` dict, not a `refresh_cmd` string. |
| M3.2 | Add `attune-rag dashboard render` CLI subcommand. | attune-rag | done | |
| M3.3 | Path validation: reject null bytes, system directories, and paths outside the user's home unless `--out` is an explicit absolute path the user provided. | attune-rag | done | |
| M3.4 | Add optional `--open` flag (auto-open in default browser). | attune-rag | done | **Drift:** original spec's open question answered yes; flag added. |
| M3.5 | Golden test in `tests/unit/test_dashboard_render.py` — render to `tmp_path`, read back, assert sentinel substitution; deterministic title + small fixture snapshot. | attune-rag | done | No network, no subprocess — rendering is pure string ops. |
| M3.6 | (Bonus, not in original spec) Add `attune-rag dashboard show` CLI subcommand — Rich terminal dashboard. Calls `build_snapshot` directly and renders to console. | attune-rag | done | **Drift:** added beyond original scope; no objection — independent of HTML pipeline. |
| M4.1 | Bump `pyproject.toml` to target version. | attune-rag | done | **Drift:** feature shipped in **0.1.6**, not 0.2.0 as originally planned. |
| M4.2 | Add a CHANGELOG entry under the appropriate version. | attune-rag | done | |
| M4.3 | Update README with a three-line usage example. | attune-rag | done | |
| M4.4 | Tag and publish per the existing release workflow. | attune-rag | done | |
| M5.1 | Delete `attune-ai/scripts/attune_rag_dashboard_refresh.py`. | attune-ai | done | M5 — Retire the attune-ai-side script (post-merge). |
| M5.2 | Update the attune-ai Cowork artifact to invoke `uv run attune-rag dashboard refresh` instead of the local script. No behavior change — same JSON contract. | attune-ai | done | |
| M5.3 | Document the deprecation in attune-ai's lessons log. | attune-ai | done | |

### Dependencies

```
M1 → M2 → M3 → M4 → M5
```

Each milestone is a self-contained release-able unit:

- **M1** ships a working `dashboard refresh` CLI but no rendered HTML.
- **M2** ships the template as package data; not yet exposed via CLI.
- **M3** ships `dashboard render` end-to-end.
- **M4** is the version bump + release.
- **M5** is post-release cleanup; can land in a follow-up release of attune-ai.

### Testing strategy

All tests live under `tests/unit/` and must pass on the existing CI matrix (Python 3.10–3.13, Ubuntu + macOS).

#### `test_dashboard_refresh.py`

- Happy path: `queries.yaml` present → full snapshot with all sections populated.
- Missing `queries.yaml`: exit code 1, JSON still emitted with `retrieval.error` populated.
- Corpus package override: `--corpus-package foo` flows through to the freshness section's corpus lookup.
- Stdout contract: first `{` marks the JSON start.

#### `test_dashboard_render.py`

- Renders to `tmp_path`, verifies file exists.
- Verifies the snapshot sentinel and the title sentinel are substituted to the passed values (JSON-escaped).
- Rejects a path inside `/etc`, `/sys`, `/proc`, `/dev`.
- Rejects a path containing a null byte.
- Rejects a path whose parent doesn't exist (with a clear error message, not a raw `OSError`).

#### `test_dashboard_cli.py`

- `attune-rag dashboard render --help` exits 0.
- `attune-rag dashboard refresh --help` exits 0.
- `attune-rag dashboard render` without `--out` exits 2.
- End-to-end: render to tmp, then grep for the baked snapshot fields.

No integration tests against real Cowork — out of scope.

### Dependencies (build/runtime)

No new required dependencies.

- **Jinja2:** considered; **not adopted** for substitution (plain `str.replace` chosen instead — avoids adding a runtime dep for one substitution).
- **Chart.js:** stays CDN-loaded inside the template — no Python dep. SRI hash from the existing artifact is preserved.

### Rollback plan

The dashboard subpackage is additive — no changes to `attune_rag.retrieval`, `attune_rag.pipeline`, or the existing `query` / `corpus-info` / `providers` CLI subcommands. Rollback strategy:

- **If the dashboard module has a bug:** `git revert` the introducing commit; CLI loses `dashboard` subcommands but everything else is untouched.
- **If M5 breaks the attune-ai Cowork artifact:** restore `attune-ai/scripts/attune_rag_dashboard_refresh.py` from git; the artifact's `--refresh-cmd` flips back to the local script.
- **Full revert:** version bump in M4 is the only release-visible change pre-M5; bump back and yank the wheel if pre-distribution release; otherwise publish a patch release with the dashboard module removed.

---

## Phase 4: Implementation

**Status**: complete

### Completion checklist

- [x] All milestones (M1–M5) marked done
- [x] Tests pass on Python 3.10–3.13 × Ubuntu + macOS
- [x] Path-validation tests cover system dirs, null bytes, missing parent
- [x] Snapshot shape contract preserved (existing artifact JS unchanged)
- [x] Released in attune-rag **0.1.6** (originally targeted 0.2.0; landed early)
- [x] CHANGELOG + README updated
- [x] M5: `attune_rag_dashboard_refresh.py` removed from attune-ai; Cowork artifact switched to installed CLI
- [x] Bonus: `attune-rag dashboard show` (Rich terminal dashboard) shipped in same release

### Drift summary (vs. original locked plan)

1. **Refresh model:** snapshot baked at render time, not `--refresh-cmd` invoked via MCP.
2. **Template sentinels:** `__ATTUNE_SNAPSHOT__` + `__ATTUNE_TITLE__`, not `__REFRESH_CMD__` + `__CORPUS_PACKAGE__`.
3. **`render()` signature:** `(out, snapshot, title) -> Path` — takes a dict, not a command string.
4. **`--open` flag:** added (was an open question in the original spec).
5. **`dashboard show` subcommand:** added beyond original scope.
6. **Release version:** shipped in **0.1.6**, not 0.2.0.
