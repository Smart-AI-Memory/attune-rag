# Spec: attune-rag 0.2.0 cut — Tasks

> **Status: scaffolding — not executable; promotes via `/spec` scoping pass after W4.2.**

This file is a **scaffold**, not a runnable task list. A `tasks.md`
in a scoping spec is **not executable** — see user memory
`feedback_spec_scoping_vs_approved`. The `/spec` pass that runs after
Phase 4 W4.2 promotes this into an approved spec with concrete scripts,
acceptance criteria, and dependency arrows.

## Phase 3: Tasks

**Status:** scaffolding. No work in this spec is approved to start.

### Implementation order (sketch)

Three milestones, mirroring the shape of
[`api-v0.2-public-surface/tasks.md`](../api-v0.2-public-surface/tasks.md)
but at half the granularity since the substantive design work happened
in Phase 3 already.

| # | Task | Layer | Status | Notes (to be filled by scoping) |
|---|------|-------|--------|---------------------------------|
| **M1 — Pre-cut audit** | | | | |
| M1.1 | Re-run `tests/unit/test_api_surface.py` against current `main`; verify zero drift from the frozen surface tables in [`api-v0.2-public-surface/design.md`](../api-v0.2-public-surface/design.md). | attune-rag | scaffold | unscoped |
| M1.2 | Re-grep attune-gui (gating downstream per ROADMAP Decision 2) for `attune_rag.*` imports; verify every consumed symbol is in the frozen `__all__`. | attune-rag | scaffold | unscoped |
| M1.3 | Refresh README "Public API" section if any wording is stale; refresh [`docs/POLICY.md`](../../POLICY.md) to replace "**formal SemVer commitments take effect with 0.2.0** (gated on Phase 2…)" with "took effect at 0.2.0 on `<date>`". | attune-rag | scaffold | unscoped |
| **M2 — Cut & release** | | | | |
| M2.1 | Bump `pyproject.toml` version `0.1.z → 0.2.0`. Classifier stays at `Development Status :: 3 - Alpha`. | attune-rag | scaffold | unscoped |
| M2.2 | Sync `src/attune_rag/__init__.py` `__version__` to `0.2.0`. | attune-rag | scaffold | unscoped |
| M2.3 | Roll CHANGELOG `[Unreleased]` into `[0.2.0] — <date>`. Section ordering per existing CHANGELOG convention. No `### Added` entries (would fail the W0.1 freeze enforcer on the cut PR itself). | attune-rag | scaffold | unscoped |
| M2.4 | Tag + PyPI publish via the existing release workflow (`/attune-release-check` skill + manual `pypi` environment approval). | attune-rag | scaffold | unscoped |
| **M3 — Post-cut verification** | | | | |
| M3.1 | Verify `pip install attune-rag==0.2.0` from PyPI in a clean venv; smoke-test the public surface import-by-import. | attune-rag | scaffold | unscoped |
| M3.2 | In attune-gui: bump the floor pin from `attune-rag>=0.1.18,<0.2` to `attune-rag>=0.2.0,<0.3`; run the contract test suite (`test_contract_attune_rag.py`) against the published 0.2.0 wheel. | attune-gui | scaffold | unscoped |
| M3.3 | Start the 7-day no-hotfix watch. Any hotfix during the window blocks Phase 5 (v1.0.0) entry per [ROADMAP-v1.md Phase 5 Gate](../ROADMAP-v1.md#phase-5--100-release). | attune-rag | scaffold | unscoped |

### Dependencies (sketch)

```
M1 (pre-cut audit) ─→ M2 (cut & release) ─→ M3 (post-cut verification)
```

Plus the upstream gate: **Phase 4 W4.2–W4.3 complete** is a hard
precondition for M1. See [`requirements.md`](requirements.md) entry
gates.

### Definition of done (sketch)

To be finalized during `/spec` scoping. Initial bullets:

- [ ] 0.2.0 on PyPI.
- [ ] `tests/unit/test_api_surface.py` green against the published wheel.
- [ ] attune-gui pinned at `attune-rag>=0.2.0,<0.3`; its contract test
      passes against the published wheel.
- [ ] CHANGELOG `[0.2.0]` section dated; `[Unreleased]` reset to empty.
- [ ] `pyproject.toml` Development Status classifier still
      `3 - Alpha` (Phase 5 owns the flip).
- [ ] [`docs/POLICY.md`](../../POLICY.md) updated to past-tense on the
      "takes effect at 0.2.0" clauses.
- [ ] 7-day no-hotfix watch started; tracked in `exit-summary.md`
      (Phase 4 W4.3) or a new file at this spec's root.

### Risks & mitigations (sketch)

To be expanded during scoping. Initial candidates listed in
[`design.md`](design.md#risks--mitigations-sketch).

### Out of scope (deferred)

- **Production/Stable classifier flip.** Phase 5 (v1.0.0).
- **Adding any new public symbol.** Goes through deprecation policy at
  0.3.0 or later.
- **`py.typed` marker.** Tracked as a 0.3.0 candidate per Phase 3
  follow-ups in [`api-v0.2-public-surface/tasks.md`](../api-v0.2-public-surface/tasks.md).
- **Signature locking.** Phase 5 candidate per Phase 4 out-of-scope.
- **Re-auditing the surface.** Audit was Phase 3 work; this spec
  ratifies the verdict, it does not re-litigate it.

### Follow-ups (post-0.2.0)

To be filled during scoping. Expected entries:

- Hand off to Phase 5 / v1.0.0 spec once the 7-day no-hotfix watch
  clears.
- Reconcile any `[Unreleased]` items that accumulated during the W4
  freeze and waited on the cut.
