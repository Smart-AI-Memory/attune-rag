# Spec: attune-rag 1.0.0 release

## Phase 3: Tasks

**Status:** parked (2026-07-13) — scaffolding + narrative prep only
(PRs #87 #89 #92 #111 #145 #146, 2026-05-20→22); v1.0.0 NOT cut
(latest tag v0.8.0); no spec-dir activity since 2026-05-22; freeze
through 2026-07-27; remaining: `/spec` scoping pass, then M1 audit →
M2 docs → M3 cut → M4 watch. Until then the table below is a
placeholder skeleton, *not* a work-tracker.

- **Shape parent:** [api-v0.2-public-surface/tasks.md](../api-v0.2-public-surface/tasks.md)

### Implementation order (skeleton)

Four milestones. M1 is read-only (audit), M2 is paperwork (docs +
policy), M3 is the cut, M4 is the post-release watch.

```
M1 (pre-release audit) → M2 (docs roll) → M3 (cut & release) → M4 (7-day no-hotfix watch)
```

### Tasks

| # | Task | Layer | Notes |
|---|------|-------|-------|
| **M1** | **Pre-release audit** — must complete before M2 starts. | | |
| M1.1 | Re-run `/security-audit` repo-wide on the head commit. Triage any new findings into `docs/specs/downstream-validation/security-findings.md` (or successor file). Zero open severity-high is an entry gate. | attune-rag | Reuses Phase-4 finding-disposition format. |
| M1.2 | Re-run `/deep-review` on the locked public surface. Findings either fold into M2 docs work or open as 1.0.x backlog items. | attune-rag | One mid-Phase, one at end — same cadence as Phase 4. |
| M1.3 | Final test coverage check on the PUBLIC surface (target carried over from Phase 4 — `~90 %` on public modules per [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 4). Anything below the bar opens as a 1.0.x test-gap item. | attune-rag | Uses `attune-ai:smart-test` to find gaps; does not auto-add tests to the cut. |
| **M2** | **Docs roll** — runs after M1 closes, lands as one PR. | | |
| M2.1 | Update [docs/POLICY.md](../../POLICY.md) — append "Support window" section (length pinned at scoping; see [requirements.md](requirements.md)). | attune-rag | Sketch in [design.md](design.md) §"Support window". |
| M2.2 | Update [docs/POLICY.md](../../POLICY.md) — append "Deprecation under 1.x" section. The existing §3 (0.x procedure) stays for historical context. | attune-rag | Sketch in [design.md](design.md) §"1.x deprecation cycle". |
| M2.3 | Update [README.md](../../../README.md) — drop "alpha" framing from the headline; link to the new support-window section. Public-API section is unchanged. | attune-rag | One-line headline edit + one cross-link. |
| M2.4 | Roll [CHANGELOG.md](../../../CHANGELOG.md) — add `[1.0.0]` entry summarizing the Phase-4 burn-in outcome, classifier flip, and policy updates. `### Added` for the *declarations*; `### Changed` for the deprecation-cycle tightening. | attune-rag | No code-level `### Added`/`### Changed` here — those shipped in 0.2.x. |
| **M3** | **Cut & release** — must run as one PR + tag pair. | | |
| M3.1 | Flip the classifier in [pyproject.toml](../../../pyproject.toml): `Development Status :: 3 - Alpha` → `Development Status :: 5 - Production/Stable`. `4 - Beta` is intentionally skipped — Phase 4 is the Beta. | attune-rag | One-line edit. |
| M3.2 | Bump version in both [pyproject.toml](../../../pyproject.toml) and [src/attune_rag/\_\_init\_\_.py](../../../src/attune_rag/__init__.py) from `0.2.x` → `1.0.0`. | attune-rag | Verified by the `attune-release-check` skill before tag. |
| M3.3 | Tag `v1.0.0` and publish to PyPI via the standard `attune-release-check` → `gh release create` flow. | attune-rag | Same release path as 0.2.x — no bespoke tooling. |
| M3.4 | Write the GitHub release notes — Phase-4 burn-in summary + link to the new POLICY.md sections + statement that the public surface is unchanged from 0.2.x. | attune-rag | Pulls from CHANGELOG entry (M2.4). |
| **M4** | **Post-release watch** — passive monitoring, 7+ days. | | |
| M4.1 | Start the seven-day no-hotfix clock at `1.0.0` publish time. | attune-rag | Manual check-in — no automation needed. |
| M4.2 | Treat any `1.0.z` hotfix in the post-release window as **evidence the cut wasn't quite right**, not as a failure of the gate. Log the root cause in the Phase-5 retrospective (`exit-summary.md`): what would have caught this in M1's audit? Ship hotfixes on their actual urgency, then read the retrospective to decide whether the 1.0 claim itself needs walking back (rare) or just whether M1's audit needs strengthening (typical). The seven-day no-hotfix window is a *signal-strength threshold*, not a license to delay real fixes. | attune-rag | Earlier framing ("any hotfix restarts the clock") silently incentivized delaying fixes to preserve a clean window — backwards. [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5 gate text should be updated to match in a follow-up. |
| M4.3 | Once seven consecutive days have passed with no hotfix, close Phase 5. Open `docs/specs/post-1.0.0-watch/` (or fold into a 1.1.0 spec) only if there are outstanding items from M1.2 / M1.3. | attune-rag | Phase-5 retrospective lives in the close-out commit message and/or a short `exit-summary.md` under this spec dir. |

### Dependencies

- M1 depends on Phase-4 W4.3 `exit-summary.md` existing and
  recommending the cut.
- M1 depends on 0.2.0 having been on PyPI for ≥ N days (N pinned
  at scoping) with zero hotfixes.
- M2 depends on M1 closing clean (no audit blockers).
- M3 depends on M2 landing as one PR (so the docs ship in the
  same commit that flips the classifier).
- M4 depends on M3 publishing successfully (PyPI + GitHub
  release).

See [requirements.md](requirements.md) §"Entry gates (inherited)"
for the upstream gates this whole spec depends on.

### Definition of done (placeholder)

All checkboxes pinned during the scoping pass — listed here so the
shape is visible:

- [ ] M1.1–M1.3 audit findings either disposed of or opened as
      1.0.x backlog (none block the cut).
- [ ] [docs/POLICY.md](../../POLICY.md) has "Support window" and
      "Deprecation under 1.x" sections.
- [ ] [README.md](../../../README.md) headline no longer says
      "alpha".
- [ ] [CHANGELOG.md](../../../CHANGELOG.md) has a `[1.0.0]` entry.
- [ ] [pyproject.toml](../../../pyproject.toml) classifier is
      `Development Status :: 5 - Production/Stable`.
- [ ] [pyproject.toml](../../../pyproject.toml) and
      `src/attune_rag/__init__.py` both at `1.0.0`.
- [ ] `attune-rag==1.0.0` on PyPI; GitHub release published.
- [ ] Seven consecutive days post-`1.0.0` publish with no hotfix
      release.

### Risks & mitigations (placeholder)

Pinned during scoping. Listed here so the shape is visible:

| Risk | Mitigation sketch |
|---|---|
| 0.2.0 ships latent regression that surfaces inside the N-day soak. | The soak is *the gate*. Reset the soak clock, ship the fix as 0.2.z, re-soak. The 1.0.0 cut waits. |
| A Phase-5 backlog item turns out to be a 1.0.0-cut blocker mid-scoping. | Either fold into M2 (if it's a doc/policy fix) or pause Phase 5 and promote the item to its own spec. The cut waits. |
| Seven-day no-hotfix clock keeps restarting (i.e. 1.0.0 keeps needing fixes). | Same signal as Phase 4 not having actually closed. Roll the classifier back to `3 - Alpha` in a 1.0.z patch, re-open Phase 4 for another burn-in cycle. Painful but recoverable. |
| Cosmetic surface tidy-ups folded into M3 break the snapshot test. | Atomic commit: surface change + `EXPECTED_*` constants update + CHANGELOG line in the same PR, gated by CI on the snapshot test. |
| `attune-release-check` skill rejects the cut for a stale `__version__` or dirty tree. | The skill is the gate — fix and re-run. Not a risk in the "blocks the cut" sense, only in the "adds an iteration" sense. |

### Scoping-time considerations (flagged 2026-05-22)

Items the scoping pass should resolve — not yet executable, but the
shape is visible enough to capture now:

- **Release narrative — methodology framing.** Scoping should decide
  whether M2 grows a docs sub-task (or a new milestone slots between
  M2 and M3) for the *external* release announcement. M3.4 covers
  the GitHub release notes (factual, terse); the narrative piece is
  separate — leading with "we measured our reranker and shipped the
  measurement, not the opinion." Bundled-corpus numbers are the
  proof; user-corpus-onboarding is the CTA. Highest-leverage release
  artifact that isn't code. Owner / channel (README hero, blog,
  Show HN) pinned at scoping. Originating discussion:
  perf-baseline-multi-run M3 wrap, 2026-05-22.

### Out of scope (deferred)

See [requirements.md](requirements.md) §"Out of scope (Non-Goals)"
for the full list. Highlights:

- **New PUBLIC symbols.** Land as 1.0.x or 1.1.0 after the cut.
- **Signature-level locking.** 1.x follow-on at most.
- **`py.typed` marker.** 1.0.x or 1.1.0 candidate.
- **Eval / perf re-baseline.** Inherited from Phases 1 + 4.

### Follow-ups (post-1.0.0)

Pinned during scoping; expected candidates:

- **1.0.1 hotfix slot** held open by default during the seven-day
  watch — no work pre-planned, capacity reserved.
- **1.0.x backlog opens** for any audit findings from M1.2 / M1.3
  that weren't blockers but warrant follow-up.
- **Phase-5 backlog items not folded in** either get their own
  spec or close as won't-do (see [design.md](design.md)
  §"Backlog disposition").
