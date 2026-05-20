# Spec: attune-rag 1.0.0 release

## Phase 1: Requirements

> **Status: scaffolding — not yet scoped; activates after the 0.2.0
> cut closes cleanly + the 7-day post-1.0.0 no-hotfix gate is
> achievable (per [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5 gate).**

- **Owner:** Patrick
- **Target version:** 1.0.0
- **Shape parent:** [api-v0.2-public-surface/requirements.md](../api-v0.2-public-surface/requirements.md)

### Problem statement

`attune-rag` will, by the time Phase 4 closes, have a
documented-and-frozen public API (0.2.0), a perf baseline that has
held across one freeze, and a downstream consumer (attune-gui) that
has survived a minor bump. What it will *not* have is a SemVer-level
stability claim. The package is still classified
[`Development Status :: 3 - Alpha`](../../../pyproject.toml) and
[`docs/POLICY.md`](../../POLICY.md) only governs 0.x removals.

Downstream maintainers cannot pin against attune-rag with the
confidence that a major version implies until the package itself
says so. Phase 5 is the small, mostly-paperwork phase that makes
the claim official: classifier flip, support-window doc, 1.x
deprecation cycle, the cut.

### Entry gates (inherited)

Every entry gate is owned upstream — this spec only checks them
before scoping begins. Reproduced from
[README.md](README.md#inherited-entry-gates):

- [ ] **Phase 4 W4.3 exit-summary exists and recommends the cut.**
      File: `docs/specs/downstream-validation/exit-summary.md`.
      The recommendation is the Phase-4 owner's call; Phase 5
      reads, it does not re-litigate.
- [ ] **0.2.0 cut spec executed and shipped.** The W4.4 successor
      spec [`docs/specs/api-v0.2.0-cut/`](../api-v0.2.0-cut/) (already
      scaffolded — see
      [#83](https://github.com/Smart-AI-Memory/attune-rag/pull/83))
      has been scoped via `/spec`, all milestones completed,
      `attune-rag==0.2.0` published to PyPI. The 0.2.0 cut spec
      explicitly *does not* flip the classifier — that gate stays
      this spec's problem.
- [ ] **0.2.0 has been on PyPI for at least N days with zero
      hotfixes** — N is a placeholder. Pin N during the formal
      scoping pass (`/spec` on this directory). Candidate range:
      14–30 days. Rationale: long enough that a latent regression
      would have surfaced via the weekly downstream-validation
      cycle; short enough that the 1.0.0 cut doesn't drift
      indefinitely.
- [ ] **attune-gui pinned to 0.2.x** and reports clean across one
      full weekly downstream-validation cycle on the new pin
      (cycle definition lives in
      [downstream-validation/design.md](../downstream-validation/design.md)).
- [ ] **Perf-thresholds baseline holds.** No regression past the
      `perf-thresholds.json` σ-band recorded in
      [downstream-validation/perf-baseline.md](../downstream-validation/perf-baseline.md)
      across the 0.2.0 release window.
- [ ] **`security-findings.md` has zero open severity-high items.**
      File: `docs/specs/downstream-validation/security-findings.md`.
- [ ] **`docs/POLICY.md` updated for 1.x.** This is the actual
      Phase-5 doc work (M2.1 in [tasks.md](tasks.md)) — not an
      external prerequisite. Listed here so the gate-check is
      complete in one place.

### Scope

**In scope (Phase 5 only):**

- Final pre-release audit pass: `/security-audit`, `/deep-review`,
  coverage check on the locked public surface.
- Documentation updates:
  - `docs/POLICY.md` — append a "Support window" section and a
    "1.x deprecation cycle" section on top of the existing 0.2.x
    policy.
  - `README.md` — headline update (no longer "alpha"), Public-API
    section unchanged.
  - `CHANGELOG.md` — `[1.0.0]` roll-up (Phase 4 burn-in summary,
    classifier flip, policy update).
- Source / metadata changes:
  - `pyproject.toml` classifier: `3 - Alpha` → `5 - Production/Stable`.
  - `pyproject.toml` + `src/attune_rag/__init__.py` version:
    `0.2.x` → `1.0.0`.
- Release mechanics: tag `v1.0.0`, PyPI publish (via the standard
  `attune-release-check` skill flow), GitHub release notes.
- Post-release watch: seven-day no-hotfix gate. Any hotfix release
  in that window restarts the seven-day clock (per the Phase 5
  gate in [ROADMAP-v1.md](../ROADMAP-v1.md)).

**Out of scope (Non-Goals):**

- **New public symbols.** The surface is what 0.2.0 ratified;
  Phase 5 does not expand it. New surface lands as 1.0.x or 1.1.0
  *after* the cut, under the policy this spec extends.
- **Eval / perf re-baseline.** Inherited from Phases 1 and 4.
- **Signature-level locking.** Symbol-level lock test from 0.2.0
  is the contract. Signature locking remains a 1.x follow-on if
  ever taken up.
- **`py.typed` marker.** Carried over from 0.2.0 backlog; revisit
  as a 1.0.x or 1.1.0 candidate, not as part of the cut.
- **Phase-5 backlog grooming as a deliverable.** Triage happens at
  scoping time (see [design.md](design.md) "Backlog disposition");
  individual items either fold into `tasks.md` here, promote to
  their own spec under `docs/specs/`, or close as won't-do.

### Disposition of in-progress Phase-5 backlog items

[docs/specs/phase-5-backlog/items.md](../phase-5-backlog/items.md)
exists (scaffolded in attune-rag
[PR #82](https://github.com/Smart-AI-Memory/attune-rag/pull/82))
with 11 items across quality (Q1–Q4), perf (P1–P4), test-audit
(T1–T3), and methodology (M1).

**M1 has already promoted** to its own spec at
[docs/specs/perf-baseline-multi-run/](../perf-baseline-multi-run/)
([PR #86](https://github.com/Smart-AI-Memory/attune-rag/pull/86))
— it is a Phase 5 deliverable in its own right and not subject to
this spec's triage. The remaining 10 items (Q1–Q4, P1–P4, T1–T3)
are triaged at Phase-5 scoping time into one of three buckets:

1. **Fold into [tasks.md](tasks.md).** Small, on-the-critical-path,
   no spec needed.
2. **Promote to its own spec** under `docs/specs/`. Large enough
   to want its own scoping pass.
3. **Won't-do.** Close with a note in `phase-5-backlog/items.md`.

This is process, not a deliverable — call it out here so it isn't
mistaken for a Phase-5 work item.

### User stories

1. *As a downstream maintainer*, I want attune-rag to declare a
   support window so I can plan my pin-update cadence and know how
   long my chosen minor will receive security fixes.
2. *As an attune-rag contributor*, I want a written 1.x deprecation
   policy so I know the procedure for retiring a PUBLIC symbol
   without surprising downstreams *under SemVer-major rules*
   (the 0.x procedure is documented; 1.x is stricter).
3. *As an attune-gui developer*, I want to be able to pin
   `attune-rag>=1.0,<2.0` and trust that the surface I depend on
   will not move within 1.x except after a one-minor deprecation
   warning.
4. *As a release manager (Patrick)*, I want the cut to be small and
   mostly paperwork — the heavy lifting (eval gate, surface freeze,
   downstream validation) all happened in earlier phases.

### Edge cases & open questions

Resolved during the `/spec` scoping pass. Listed here as
placeholders so they aren't forgotten.

| Question / Edge case | Placeholder resolution |
|---|---|
| What value of N for "0.2.0 soak before cut"? | Pin during scoping. Candidate: 14–30 days. |
| What length of support window per minor? | Pin during scoping. Candidate: latest minor receives security fixes for the duration of the next minor's life, plus N months. |
| How many minors of deprecation warning before removal in 1.x? | Pin during scoping. Candidate: at least one full minor between warning and removal (matches POLICY.md's 0.x→1.x stricter step). |
| What happens if a P0 / security hotfix fires during the seven-day post-release watch? | Per [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5 gate: the seven-day clock restarts. |
| What happens to Phase-5 backlog items not folded in? | Promote to own spec or close as won't-do — see "Disposition" section. |
| Should the cut bundle any 0.2.x → 1.0.0 surface tidy-ups (alphabetise `__all__`, etc.)? | Decide during scoping. The cut is the rare moment when surface churn is cheap; small tidy-ups may bundle if they are *purely* cosmetic and the snapshot test is updated atomically. |
| What is the `1.0.0` `### Added` section in CHANGELOG, given the freeze policy? | Treat 1.0.0 as a release-of-the-burn-in: `### Added` lists the *declaration* changes (classifier flip, support-window policy); `### Changed` lists policy-level changes (deprecation cycle tightens). Code-level changes shipped earlier. |

### Affected layers

- [x] **attune-rag** — `pyproject.toml`, `src/attune_rag/__init__.py`,
      `docs/POLICY.md`, `README.md`, `CHANGELOG.md`, tag + PyPI publish.
- [ ] **attune-gui** — no code change required. The 0.2.x pin
      established in Phase 4 is what gets exercised across the cut.
- [ ] **attune-help** — no code change required.
- [ ] **attune-author** — no code change required.
