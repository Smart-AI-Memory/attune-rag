# Spec: Downstream Validation (Phase 4 of v1.0 Roadmap)

**Status**: approved (2026-05-19)

- **Owner:** Patrick
- **Created:** 2026-05-16
- **Depends on:** Phase 3 complete (api-v0.2-public-surface — done)
- **Successor of:** 0.2.0 formal cut (deferred from api-v0.2-public-surface spec
  per its "all in-scope work complete" status note)
- **Target outcome:** four consecutive weeks of feature-freeze + downstream-green
  data → unblocks the formal 0.2.0 cut

> Phase 4 of the v1.0 roadmap (see
> [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md)). This is **process-heavy, not
> code-heavy**: most of the work is *not adding things* while measuring how
> the system behaves. The active work is the gate machinery and the audit
> passes; the wait is the soak time itself.

---

## Phase 1: Requirements

**Status**: approved (2026-05-19)

### Problem statement

The 0.1.18 + 0.1.19 releases shipped substantial groundwork toward the formal
0.2.0 API freeze: explicit `__all__` per public module, snapshot test, editor
submodule renames, faithfulness `--thinking` default decided. The public
surface is *documented* and *snapshot-tested*, but **not yet contractually
frozen** — that's the line 0.2.0 crosses.

Before crossing it, we need evidence that the current surface holds up under
real use:

- attune-gui (the gating downstream per Decision 2 in the roadmap) must
  consume 0.1.x cleanly through a `0.2.x → 0.3.x` bump without breakage.
- Performance characteristics of retrieval + reranker must be measured and
  documented so the next minor's perf changes have a baseline to compare
  against.
- Security audit must be clean — no live `eval`/`exec`, no path-traversal
  vectors in the editor rename + lint paths, no secret leakage in
  `dashboard refresh` snapshots, no provider-adapter token-handling bugs.
- The change cadence has to demonstrate that the API is stable enough to
  freeze: four consecutive weekly CHANGELOG entries with only `Fixed` /
  `Changed` / `Security` and **no `Added`**.

Today, none of those gates have mechanical enforcement. The change-cadence
gate in particular relies on the maintainer remembering to look at the
CHANGELOG — automated CI doesn't refuse a PR that adds an `Added` entry
during the freeze.

### Scope

**In scope:**

- **Feature-freeze enforcement** — a CI check that fails any PR that adds an
  `### Added` entry under `[Unreleased]` or under a `[0.2.0]` block.
  Exception path: maintainer-only override label `freeze-override` for
  documented edge cases (security additions, vendored-dep upgrades that
  change the surface inadvertently).
- **Perf baseline** — a reproducible benchmark of retrieval + reranker hot
  paths producing `docs/specs/downstream-validation/perf-baseline.md` +
  `perf-thresholds.json`, following the same `mean − 2σ` strategy Phase 1
  locked for quality metrics. Wired into CI as an advisory check first,
  promoted to gating in week 3 once the floor is stable.
- **Security audit** — `/security-audit` (attune-ai) repo-wide pass at the
  start of Phase 4 + per-PR scans for changes touching `editor/`,
  `providers/`, `dashboard/`, or `cli.py`. Findings tracked in
  `docs/specs/downstream-validation/security-findings.md`.
- **Downstream gate** — attune-gui's full test suite runs on every
  attune-rag PR via a reusable workflow call (manual or automatic). The PR
  description names attune-gui's HEAD SHA used for the validation run so
  failures can be reproduced.
- **Cadence dashboard** — a simple `scripts/changelog_cadence.py` that walks
  the last N CHANGELOG entries and reports the Added/Fixed/Changed/Security
  breakdown. Run weekly; results go to `docs/specs/downstream-validation/
  cadence-week-N.md`.

**Out of scope (Non-Goals):**

- The 0.2.0 release itself. Phase 4 unblocks it; the cut is a successor
  spec.
- The `Production/Stable` classifier flip (Phase 5).
- New optional extras or backends — feature freeze.
- API surface additions. The surface is locked from week 1.
- Coverage targets beyond what the existing Phase 1 thresholds already gate.
  (Phase 4's `/smart-test` work raises coverage toward 90% but explicitly
  doesn't *add* public symbols — only tests for existing ones.)

### Downstream validation criteria

Per Decision 2 in the v1.0 roadmap, **attune-gui is the gating downstream**.
attune-gui is "green" for Phase 4 purposes when:

| Check | Threshold |
|---|---|
| `pytest sidecar/tests` against published attune-rag | 100% pass on the editor + RAG paths; cowork failures are OK if they predate Phase 4 and have a tracking issue |
| Editor contract tests (`test_contract_attune_rag.py`, `test_editor_*`) | 100% pass |
| The `feature/attune-rag-0.2-editor-rename` branch's M5.2 commit verified | already done 2026-05-16; confirms M5.2 → published 0.1.19 path |
| Bumping attune-gui's `attune-rag` pin from `0.1.x` to a hypothetical `0.3.0` (simulated by removing the `<0.2` ceiling against a 0.2.x dev build) | no `ImportError`, no `DeprecationWarning` re-emergence, route handlers still return same shapes |

### User stories

1. *As an attune-rag maintainer*, I want CI to refuse PRs that add public
   surface during the freeze — so the four-week cadence gate doesn't rely on
   my vigilance.
2. *As an attune-rag user planning an upgrade*, I want a documented perf
   baseline I can compare my own benchmark runs against — so I can tell if
   retrieval got slower in my workload, not just whether the maintainer
   thinks it's fast enough.
3. *As an attune-gui maintainer (also Patrick)*, I want attune-rag's CI to
   tell me when an attune-rag change breaks attune-gui's tests — without
   me running them by hand after every attune-rag release.
4. *As a security-aware downstream*, I want a `SECURITY.md` audit summary
   covering what attune-rag does and doesn't do with secrets, untrusted
   input, and provider tokens.

### Edge cases & open questions

| Question / Edge case | Resolution |
|---|---|
| What if a security fix requires adding a public symbol during freeze? | Allow via `freeze-override` PR label + a paragraph in CHANGELOG `[Security]` explaining the addition. Document in `freeze-overrides.md` for audit trail. |
| What counts as "Added"? Is a new CLI subcommand surface? A new optional extra? | Yes to both. Anything that changes what downstreams can rely on. New private helpers, internal refactors, and `### Fixed` for existing public behavior do NOT count. |
| Perf threshold strategy: same `mean − 2σ`? | Yes — Decision 1 logic carries. Re-measure if hardware or Python version changes mid-Phase. |
| What hardware/runner profile for perf? | GitHub Actions `ubuntu-latest`, same as the existing benchmark.yml. Document the runner SKU in `perf-baseline.md` so re-measures are comparable. |
| How does attune-gui's CI know which attune-rag SHA to test? | attune-rag's reusable workflow accepts `attune_rag_ref` input. Default: HEAD of the calling PR. attune-gui's workflow installs from that SHA via `pip install git+...@<sha>`. |
| What if attune-gui's main branch has unrelated failing tests during Phase 4? | The downstream gate uses attune-gui's `feature/attune-rag-0.2-editor-rename` branch as the validation target (it's known-green for the M5 paths) until the cowork-test failures and the test/pass-2-llm-discipline branch are reconciled. Switch to `main` once those resolve. |
| Phase 4 estimate is 4 weeks. Calendar dates? | Start: 2026-05-23 (week after this spec lands). End: 2026-06-20. Adjusted by however long the spec review + machinery setup takes (likely +1 week). |
| If a regression surfaces in week 3, does the four-week clock reset? | Yes for the cadence gate. The freeze starts fresh once the regression is fixed. Documented in `cadence-week-N.md` entries so the reset is visible. |

### Affected layers

- [x] attune-rag — adds CI workflows, scripts, the perf baseline + security
      findings docs, and the cadence script.
- [x] attune-gui — gains a reusable-workflow integration so attune-rag's PRs
      can trigger attune-gui's test suite against the in-flight SHA.
- [ ] attune-help — none directly. May benefit indirectly from any
      retrieval perf improvements found during burn-in.
- [ ] attune-author — none directly.
