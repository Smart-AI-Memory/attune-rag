# Spec: Public API Surface Freeze for attune-rag 0.2.0

## Phase 3: Tasks

**Status**: **all in-scope work complete 2026-05-16.** M1–M4 shipped in attune-rag 0.1.18 (PR #36) + Phase 2 in 0.1.19 (PR #38). M5.1 + M5.2 merged in attune-gui via PR #36 (commit `af8d3fc`, 2026-05-16T11:46:31Z), bundled with the live-marker / CI guard from PR #32. M5.3 (contract test against published 0.1.19) verified green 2026-05-16 — 32/32 attune-rag-touching tests pass. **The formal 0.2.0 SemVer freeze is queued in [`docs/specs/api-v0.2.0-cut/`](../api-v0.2.0-cut/) (W4.4 successor spec, tied to Phase 4 burn-in). The classifier flip `3 - Alpha` → `5 - Production/Stable` is queued in [`docs/specs/v1.0.0-release/`](../v1.0.0-release/) (Phase 5) — *not* part of 0.2.0.**

### Implementation order

Five milestones. M1 → M2 → M3 in attune-rag must all land before the
0.2.0 tag. M4 is the version cut. M5 is the downstream cleanup in
attune-gui, post-release.

| # | Task | Layer | Status | Notes |
|---|------|-------|--------|-------|
| M1.1 | Add `AttuneHelpCorpus` to the `__all__` in [src/attune_rag/__init__.py](src/attune_rag/__init__.py) and re-export it from the root. | attune-rag | **done** | The one symbol addition allowed by this freeze. Driven by the attune-help audit. The module is safe to import eagerly — only the `from_attune_help()` classmethod pulls in the optional extra, so no lazy import was needed. |
| M1.2 | Add `AttuneHelpCorpus` to the `__all__` in [src/attune_rag/corpus/__init__.py](src/attune_rag/corpus/__init__.py). | attune-rag | **done** | Mirrors M1.1 at the subpackage level. |
| M1.3 | Confirm no other PUBLIC `__all__` lists need changes. Diff each PUBLIC module's existing `__all__` against design.md's tables. | attune-rag | **done** | Audited during spec scoping — no other gaps. |
| M2.1 | `git mv src/attune_rag/editor/_rename.py src/attune_rag/editor/rename.py`. | attune-rag | **done** | M2 — editor submodule renames. |
| M2.2 | Same for `_schema.py`, `_lint.py`, `_autocomplete.py`, `_references.py`. | attune-rag | **done** | |
| M2.3 | Add a shim file at each old path (`_rename.py` etc.) that re-exports from the new name and emits a `DeprecationWarning`. Pattern in design.md. | attune-rag | **done** | Used module-level `__getattr__` (PEP 562) instead of `from .new import *` so private names like `_hunks` proxy too — attune-gui touches those. |
| M2.4 | Update [src/attune_rag/editor/__init__.py](src/attune_rag/editor/__init__.py) to import from the new non-underscore paths. | attune-rag | **done** | Public `__all__` is unchanged — only the relative imports move. |
| M2.5 | Update any internal callers within attune-rag that import from `_rename` etc. by qualified path. | attune-rag | **done** | Two intra-`editor` callers (`lint.py`, `rename.py`) plus `tests/unit/test_editor_schema.py:57`. |
| M2.6 | Tests: cover that the shim module imports succeed AND emit `DeprecationWarning`. Add to `tests/unit/test_editor_rename_shims.py`. | attune-rag | **done** | 11 tests covering warning text/version/public-attr proxy/`_hunks` private-attr proxy. |
| M3.1 | Create `tests/unit/test_api_surface.py` per the design.md sketch. | attune-rag | **done** | M3 — surface lock. 17 tests: parametrized `__all__` lock for 4 PUBLIC modules + qualified-path import for 10 PUBLIC submodules + version shape + 2 qualified-import explicit checks. |
| M3.2 | The `EXPECTED_*` frozensets in the test must match design.md's tables byte-for-byte. | attune-rag | **done** | Source of truth. |
| M3.3 | Confirm the test fails when a PUBLIC symbol is removed (manual sanity: delete one entry from `__all__`, run test, expect red). | attune-rag | **done** | Verified via monkey-patch — surface test reports "added/removed" diff cleanly. |
| M4.1 | Write `docs/POLICY.md` per design.md's outline (six sections). | attune-rag | **done** | M4 — policy + docs. Six sections: what's covered, SemVer commitment, retire procedure, add procedure, underscore convention, resources. |
| M4.2 | Add "Public API" section to [README.md](README.md) listing the PUBLIC symbols and submodules, with a link to `docs/POLICY.md`. | attune-rag | **done** | Inserted before "Status". Groups symbols by topic (Pipeline, Corpus, Retrieval, Provenance, Prompting, Hybrid retrieval) + enumerates PUBLIC submodules + notes the 0.3.0 shim removal. |
| M4.3 | Add a CHANGELOG entry. | attune-rag | **done** | Drafted under `[Unreleased]` with Added/Changed/Deprecated sections. Note explicitly says 0.2.0 ships only after Phase 2 lands. |
| M4.4 | Bump `pyproject.toml` and `__version__` to `0.1.18` (retargeted from 0.2.0 — the work is backward-compatible groundwork, not the formal freeze). Classifier remains `Development Status :: 3 - Alpha`. The 0.2.0 SemVer freeze is queued in [`api-v0.2.0-cut/`](../api-v0.2.0-cut/); the classifier flip to `5 - Production/Stable` is queued in [`v1.0.0-release/`](../v1.0.0-release/) — two separate cuts. | attune-rag | **done** | Both `pyproject.toml` and `src/attune_rag/__init__.py` set to 0.1.18. |
| M4.5 | Tag and publish per the existing release workflow (`/attune-release-check` then `gh release create`). | attune-rag | **done** | 0.1.18 tagged + published 2026-05-16T11:23:47Z via PR #36 (commit `6d95ee1`). 0.1.19 followed in PR #38 + PR #39 (Phase 2 close-out + CHANGELOG precision fix) and published 2026-05-16T12:02:48Z. Both releases passed the `attune-release-check` skill and the manual `pypi` environment approval gate. |
| M5.1 | In attune-gui: replace `attune_rag.editor._rename` imports with `attune_rag.editor.rename`. Same for `_schema`. | attune-gui | **done** | Landed in attune-gui on branch `feature/attune-rag-0.2-editor-rename` (commit `5bf35ec`). 5 files: 2 route handlers, 2 tests, 1 docstring. Note: the commit message refers to "attune-rag 0.2.0" but the renames actually ship in 0.1.18 — technical capability unchanged. |
| M5.2 | In attune-gui: remove the unpublished-module guard in `sidecar/attune_gui/_editor_dep.py` once the floor version of attune-rag is bumped to `>=0.1.18` (retargeted from 0.2.0). | attune-gui | **done** | Merged in attune-gui PR #36 (commit `af8d3fc`, 2026-05-16T11:46:31Z). The 503 guard + `test_editor_dep.py` were removed; floor pinned at `attune-rag>=0.1.18,<0.2` in attune-gui's `pyproject.toml`. PR #36 also bundled M5.1's non-underscore import migration. |
| M5.3 | Run attune-gui's full test suite (including the contract tests in `sidecar/tests/test_contract_attune_rag.py`) against attune-rag 0.1.19. | attune-gui | **done** | 2026-05-16. After upgrading attune-gui's venv to `attune-rag==0.1.19` (`uv pip install --upgrade`), `pytest -q` gives **32/32 attune-rag-touching tests pass**: `test_contract_attune_rag` (5 tests), `test_editor_template` (19), `test_services_rag_pipeline` (8). Full-suite count: 489 pass / 3 fail / 3 deselected / 1 xfailed. The 3 failures (`test_cowork_specs` ×2, `test_cowork_templates::test_templates_staleness_thresholds`) reproduce on attune-gui's `main` with the M5 branch stashed — pre-existing, unrelated to attune-rag. |

### Dependencies

```
M1 (additions to __all__) ─┐
M2 (editor renames + shims)┼─→ M3 (surface test) ─→ M4 (policy + version + ship) ─→ M5 (attune-gui cleanup)
                           ┘
```

**Phase ordering:** Phase 2 (release-quality baseline) must merge
before M4.4. Phase 3 (this spec) can otherwise run concurrently with
Phase 2 per Decision 3 in the v1.0 roadmap. M1–M3 do not change the
public surface in a way that affects Phase 2's testing scope —
adding `AttuneHelpCorpus` to `__all__` and renaming private modules
are both backward-compatible.

### Definition of done

- [ ] `tests/unit/test_api_surface.py` exists, is green, and fails when
      any PUBLIC symbol is removed (verified once locally per M3.3).
- [ ] `docs/POLICY.md` exists and is linked from README.
- [ ] `README.md` has a "Public API" section enumerating the PUBLIC
      surface.
- [ ] All five `attune_rag.editor._*` private modules exist as
      deprecation shims (real implementations at the non-underscore
      paths).
- [ ] `AttuneHelpCorpus` is in both root and `attune_rag.corpus`
      `__all__` lists.
- [x] `pyproject.toml` is at version `0.1.18` (re-targeted from
      `0.2.0` — the work is additive groundwork, not the formal
      freeze; bumped again to `0.1.19` in PR #38). The formal
      0.2.0 SemVer freeze is queued in
      [`api-v0.2.0-cut/`](../api-v0.2.0-cut/); the classifier flip
      `3 - Alpha` → `5 - Production/Stable` is queued in
      [`v1.0.0-release/`](../v1.0.0-release/) — two separate cuts.
- [x] CHANGELOG has an "Added", "Changed", and "Deprecated" section
      for 0.1.18 (and a follow-up `[0.1.19]` section for Phase 2).
- [x] attune-gui's contract test (`test_contract_attune_rag.py`)
      passes against the published 0.1.19 wheel (M5.3, verified
      2026-05-16; 32/32 attune-rag-touching tests green).
- [x] attune-gui's `_editor_dep.py` 503 guard is removed (M5.2,
      commit `b5f4d3b` on `feature/attune-rag-0.2-editor-rename`;
      branch unmerged pending M5.3).

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| Renaming `_rename.py` → `rename.py` causes a stale `.pyc` to mask an import error in CI. | Ensure CI clears `__pycache__` before pytest runs (existing pre-commit handles this). |
| The `DeprecationWarning` from the editor shims is silenced by pytest's default warning filter and the deprecation slips past CI. | Add `-W error::DeprecationWarning::attune_rag.editor` to the test invocation for the shim test only. |
| attune-help's `test_template_prefixes.py` mirrors `template_schema.json` — schema drift between 0.1.x and 0.2.0 silently breaks it. | M4.3 CHANGELOG explicitly calls out the schema file as PUBLIC. Confirm no schema-shape changes are part of the 0.2.0 cut (separate from this spec). |
| Phase 2 reshapes a module that this spec freezes. | Re-run the audit (`grep` for cross-repo imports) before tagging 0.2.0. The snapshot test will also flag any drift from the design.md tables. |
| Downstream forks of attune-rag depend on a symbol that the audit missed. | The freeze is *additive* to what the audit found. If a real consumer materializes, promotion to PUBLIC goes through the deprecation policy in M4.1. |

### Out of scope (deferred)

- **Signature locking.** The 0.2.0 freeze is symbol-level only. Locking
  call signatures across releases is a Phase 5 candidate.
- **`py.typed` marker.** Track separately; would change downstream
  type-checking ergonomics in a way that needs its own validation.
- **Promoting `eval/` to PUBLIC.** Re-evaluate at Phase 5 when the
  classifier flips. The harness needs more soak time.
- **Promoting `dashboard/` to PUBLIC.** Locked OUT by dashboard-v0.2.0
  spec. Revisit only if a real programmatic consumer materializes.
- **Promoting `benchmark.py` to PUBLIC.** CLI / `python -m` is the
  contract; no programmatic API surface today.

### Follow-ups (post-0.1.18)

- **Worktree-local `.venv` ambiguity.** Three Python environments
  are currently in play for this worktree: the main repo's
  `/Users/patrickroebuck/attune-rag/.venv`, an auto-created
  worktree-local `.venv` (made when `uv run` was first invoked
  here), and the global `~/.pyenv/versions/3.10.11/site-packages/`.
  `uv run` picks the project-local `.venv` and ignores activated
  VIRTUAL_ENV, but the resolution path is non-obvious. Cleanup:
  `rm -rf .claude/worktrees/admiring-feynman-51ae7d/.venv` and let
  `uv sync --extra dev --extra author` re-create it from
  `pyproject.toml` cleanly. Defer until after the 0.1.18 polished
  regen lands so we don't lose attune-author 0.13.0 mid-flight.
- **Yanked `attune-help==0.10.0` pin** — being addressed in the
  same commit as the polished regen (pyproject floor bumped to
  `>=0.10.1`).
- **Dashboard preview-pane render mode (post-freeze 0.2.0).** Add a
  `--open=preview|browser|none` flag to `attune-rag dashboard render`.
  Default behavior unchanged (file-only output). When `CLAUDECODE=1`
  is set and `--open=preview` is selected (or auto-detected), surface
  the rendered HTML via the Claude Code preview MCP so it appears in
  the Cowork preview pane instead of opening a separate browser
  window. File-output stays canonical — preview is an additive UX
  layer, not a replacement, so CI / cron / headless contexts are
  unaffected. Blocked during Phase 4 freeze: a new CLI flag fails the
  W0.1 enforcer's `### Added` check and resets the cadence clock.
  Land in the 0.2.0 cut as `### Added`. Note: distinct from the
  attune-gui Cowork live-dashboard surface — this is the static
  snapshot dashboard from `dashboard/render.py`, embedded in the
  preview pane for in-conversation review.
- **`KeywordRetriever.MIN_ALIAS_OVERLAP` knob — shipped early
  under freeze-override in 0.1.22 (was originally deferred).**
  Default `2` lands as an `### Added` bullet in the
  `[Unreleased]` block with `[Override-rationale]` documented
  inline; cadence clock not reset (`Security`-scoped exception
  pattern). The 0.2.0 cut should still re-affirm the knob in its
  public-surface inventory and add it to the snapshot test in
  `tests/unit/test_api_surface.py` as a deliberate class
  attribute (not a new `__all__` entry). Full trace:
  [`docs/specs/selection-criteria-robustness/proposal.md`](../selection-criteria-robustness/proposal.md).
- **Embedding co-signal for genuine semantic ties (workstream B).**
  gq-020 was the documented "needs embeddings" case before
  `MIN_ALIAS_OVERLAP` shipped; it now passes top-1 lexically.
  Embedding work moves from "necessary fix for 0.1.x" to
  "0.2.0+ semantic floor lift". Re-scope before opening.
