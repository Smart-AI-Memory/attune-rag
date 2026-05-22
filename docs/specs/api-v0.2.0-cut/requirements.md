# Spec: attune-rag 0.2.0 cut — Requirements

> **Status:** **scoped 2026-05-22.** Requirements below are the
> non-negotiables the implementation in [`tasks.md`](tasks.md) must
> satisfy; the entry-gate checklist is mechanically verified by
> M0.1-M0.5 in `tasks.md`.

This spec ratifies the public surface frozen at the symbol level in
Phase 3 ([`api-v0.2-public-surface`](../api-v0.2-public-surface/))
into a SemVer-level 0.2.0 cut. It does **not** add or change any
public symbols.

## Phase 1: Requirements

**Status:** scaffolding — to be filled in by the `/spec` scoping pass
after Phase 4 W4.2 fires.

### Entry gates (inherited from Phase 4)

All gates below must be **green at the moment this spec is activated**.
They are deliverables of the Phase 4 burn-in, not work this spec
performs. If any gate is red, the freeze extends and this spec stays
in scaffolding.

- [ ] **Cadence-clean.** All four `cadence-week-{1,2,3,4}.md` reports
      under [`docs/specs/downstream-validation/`](../downstream-validation/)
      show **zero `### Added`** entries under `[Unreleased]` for four
      consecutive weeks. (W4.2 hard gate.)
- [ ] **Perf baseline holds.** `perf-thresholds.json` is locked at
      **N = 50 runs, σ = 3.0** with `include_llm: true`
      (per PR [#77](https://github.com/Smart-AI-Memory/attune-rag/pull/77)).
      No perf-gate red on `main` during the four-week soak.
- [ ] **Downstream gate stayed green.** The attune-gui downstream gate
      (`.github/workflows/downstream-attune-gui.yml`) was **blocking**
      from W3.2 through W4.4 and reported green for every PR landing
      on `main` during that window.
- [ ] **Security findings clean.** `docs/specs/downstream-validation/security-findings.md`
      has **zero open `severity: high`** items at W4 close. Any open
      findings are `severity: medium` or below with a rationale or a
      Phase 5 ticket.
- [ ] **Exit summary written.** `docs/specs/downstream-validation/exit-summary.md`
      (Phase 4 W4.3 deliverable) exists, summarizes the perf trend +
      security disposition + downstream-green record, and **explicitly
      recommends cutting 0.2.0**.

### Requirements (to be expanded during scoping)

#### Functional

- [ ] The 0.2.0 release ratifies the **exact** public surface locked by
      [`tests/unit/test_api_surface.py`](../../tests/unit/test_api_surface.py)
      at the moment of the cut. No symbol additions, no removals.
- [ ] [`docs/POLICY.md`](../../POLICY.md) commitments take effect:
      "PUBLIC symbols are not removed within the same minor version
      (e.g. anything PUBLIC in 0.2.0 stays through every 0.2.z)" is now
      binding rather than honor-system.
- [ ] `pyproject.toml` version bumps from the latest 0.1.z to `0.2.0`.
- [ ] `pyproject.toml` Development Status classifier **stays at**
      `3 - Alpha`. The flip to Production/Stable is Phase 5's job.
- [ ] `src/attune_rag/__init__.py` `__version__` matches `pyproject.toml`.
- [ ] CHANGELOG rolls the `[Unreleased]` block into a `[0.2.0]`
      section with the date of the cut.

#### Process

- [ ] No `### Added` entries land in CHANGELOG between cut-day and the
      end of the post-release 7-day hotfix watch (see Phase 5 entry
      gate).
- [ ] Tag and PyPI publish use the existing release workflow
      (`/attune-release-check` skill + manual `pypi` environment
      approval).

### Out of scope

- **Production/Stable classifier flip.** Phase 5 (v1.0.0 spec).
- **Adding new public symbols.** Any new symbol enters the surface
  via the deprecation policy in [`docs/POLICY.md`](../../POLICY.md)
  starting at 0.3.0; no additions during the cut itself.
- **`py.typed` marker.** Tracked as a 0.3.0 candidate per Phase 3
  follow-ups in
  [`api-v0.2-public-surface/tasks.md`](../api-v0.2-public-surface/tasks.md).
- **Signature locking.** Phase 5 candidate.

### Open questions (closed at scoping 2026-05-22)

All eight scoping decisions are recorded in
[`tasks.md`](tasks.md#scoping-decisions-confirmed-2026-05-22). The
spec was authored decision-complete; no questions remain open. The
M0 entry-gate verification step in `tasks.md` is the only addition
the scoping pass made beyond confirming the existing design.
