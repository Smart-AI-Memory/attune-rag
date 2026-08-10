# Spec: attune-rag 1.0.0 release

## Phase 1: Requirements

> **Status: approved 2026-08-09** — scoped and ratified by Patrick in
> the same session, including D7 = Option A. Scoping decisions are
> locked in [decisions.md](decisions.md); read that first. It also
> records the three stale premises this file carried until 2026-08-09
> and how each was corrected.

- **Owner:** Patrick
- **Target version:** 1.0.0
- **Cut parent:** 0.9.x (per [decisions.md](decisions.md) D1 — **not**
  0.2.x, which this file assumed until the scoping pass)
- **Shape parent:** [api-v0.2-public-surface/requirements.md](../archive/api-v0.2-public-surface/requirements.md)

### Problem statement

`attune-rag` has a documented-and-frozen public API, a perf baseline
methodology that held across a freeze, and downstream consumers that
have survived repeated minor bumps. What it does *not* have is a
SemVer-level stability claim. The package is classified
[`Development Status :: 4 - Beta`](../../../pyproject.toml) and
[`docs/POLICY.md`](../../POLICY.md) only governs 0.x removals.

> **Corrected 2026-08-09.** This paragraph previously said the package
> was classified `3 - Alpha`. It never was — `git log -S'Development
> Status' -- pyproject.toml` returns one commit, the initial scaffold,
> at `4 - Beta`. See [decisions.md](decisions.md) D2; the release copy
> must not claim to be leaving alpha.

Downstream maintainers cannot pin against attune-rag with the
confidence that a major version implies until the package itself
says so. Phase 5 is the small, mostly-paperwork phase that makes
the claim official: classifier flip, support-window doc, 1.x
deprecation cycle, the cut.

### Entry gates — audited 2026-08-09

Audited during the scoping pass against the tree and PyPI, not against
status headers. Six of seven inherited gates are closed; the seventh is
this spec's own doc work. **One gate the scaffolding never had — consumer
pins (M0) — is open and blocks the cut.**

| Gate | Verdict | Evidence |
|---|---|---|
| Phase 4 W4.3 exit-summary exists and recommends the cut | ✅ | `downstream-validation/exit-summary.md:48` "Recommendation"; cut authorized ahead of nominal calendar (W4 −26 d via W4.2 override) |
| Predecessor cut executed and shipped | ✅ | 0.2.0 executed 2026-05-25; **0.3.0–0.9.0 have since shipped.** Re-read per [decisions.md](decisions.md) D1 — the parent is 0.9.x |
| Predecessor on PyPI ≥ N days, zero hotfixes | ✅ | N pinned at **14** (D4). 0.9.0 shipped 2026-07-18 (PR [#199](https://github.com/Smart-AI-Memory/attune-rag/pull/199)) — 22 days, no 0.9.1 |
| Downstream consumer clean on the current pin | ✅ | attune-gui green on `>=0.1.22,<1.0`; attune-ai re-validated against 0.9.0 (`pyproject.toml:78`) |
| Perf-thresholds baseline holds | ⚠️ **methodology ✅, numbers stale** | `perf-baseline.md` is methodology v2 at σ=2.0 — but measured 2026-05-22 on 0.1.x code. **M1 re-measures.** See D8 |
| `security-findings.md` zero open severity-high | ✅ | "Zero `severity: high` open"; W09.A.001–003 all `high → fixed` with tests |
| `docs/POLICY.md` updated for 1.x | ⬜ this spec's M2.1 | not an external prerequisite |
| **Consumer pins admit 1.0.0** | ❌ **BLOCKING — new** | gui `<1.0`, attune-ai `<0.10`, attune-author `<0.9`. Nothing resolves 1.0.0 today. See D3 / **M0** |

### Scope

**In scope (Phase 5 only):**

- Final pre-release audit pass: `/security-audit`, `/deep-review`,
  coverage check on the locked public surface.
- Documentation updates:
  - `docs/POLICY.md` — append a "Support window" section and a
    "1.x deprecation cycle" section on top of the existing 0.2.x
    policy.
  - `README.md` — headline update to a **Beta → Stable** promotion
    (D2 — *not* "no longer alpha"; it never was), Public-API section
    unchanged. Closes as a no-op if the headline never claimed a
    maturity level.
  - `CHANGELOG.md` — `[1.0.0]` roll-up (burn-in summary, classifier
    flip, policy update).
- **Consumer pin widening (M0) — added 2026-08-09 (D3).** attune-gui,
  attune-ai, and attune-author all cap attune-rag below 1.0.0. Widening
  them to `<2.0` is a prerequisite, not a follow-up: without it the cut
  publishes a release nothing can install.
- Source / metadata changes:
  - `pyproject.toml` classifier: `4 - Beta` → `5 - Production/Stable`.
  - `pyproject.toml` + `src/attune_rag/__init__.py` version:
    `0.9.x` → `1.0.0`.
  - Two ratified cosmetic tidy-ups folded in (D9): Q4 `__all__`
    ordering, Q3 local-variable rename.
- Release mechanics: tag `v1.0.0`, PyPI publish (via the standard
  `attune-release-check` skill flow), GitHub release notes.
- Post-release watch: seven-day no-hotfix gate, **read per
  [tasks.md](tasks.md) M4.2** — ship hotfixes on their real urgency and
  log the root cause. The older "any hotfix restarts the clock" wording
  in [ROADMAP-v1.md](../ROADMAP-v1.md) is superseded (D10); it
  incentivized delaying fixes to protect a clean window.

**Out of scope (Non-Goals):**

- **New public symbols.** The surface is what 0.9.x carries; Phase 5
  does not expand it. New surface lands as 1.0.x or 1.1.0 *after* the
  cut, under the policy this spec extends.
  > ~~One explicit, recorded exception — D7 = Option A~~ **Void
  > 2026-08-10:** the D7 gap turned out not to exist —
  > `DirectoryCorpus` has carried `extra_aliases_file` since PR #130
  > (user-corpus-onboarding M2). The Non-Goal stands unamended: 1.0.0
  > adds **no** public symbols and the 1.0.0 ≡ 0.9.x surface identity
  > holds cleanly. See decisions.md D7 void note.
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

> **✅ Triage completed 2026-08-09** — [decisions.md](decisions.md) D9.
> Outcome: **Q3 + Q4** fold into the cut (M3.5); **Q1 + T2** and
> **T1 + T3** promote to two new specs (scaffolded during Phase 5, not
> executed); **P1–P4** and **Q2** land as 1.0.x. Nothing in the backlog
> blocks the cut. Nothing was closed won't-do.

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

**All resolved 2026-08-09.** Full reasoning in
[decisions.md](decisions.md); this table is the index.

| Question / Edge case | Resolution |
|---|---|
| What value of N for the pre-cut soak? | **14 days** (D4). Already satisfied — 0.9.0 has 22 days with zero hotfixes. |
| What length of support window per minor? | **6 months** past the next minor, or 6 from its own release, whichever is longer (D5). Bug fixes: latest minor only. POLICY.md must carry the scope-honesty paragraph verbatim. |
| How many minors of deprecation warning before removal in 1.x? | Warning in `1.M.0`; removal **only at 2.0.0** (D6). Ratified as sketched. |
| What happens if a P0 / security hotfix fires during the seven-day watch? | **tasks.md M4.2 governs, not ROADMAP-v1.md.** Ship on real urgency; log root cause in the retrospective. The window is a signal-strength threshold, not a reason to delay fixes. ROADMAP's older "clock restarts" wording is corrected per D10. |
| What happens to Phase-5 backlog items not folded in? | Triaged (D9): Q3+Q4 fold into the cut; Q1+T2 and T1+T3 promote to two new specs; P1–P4 and Q2 land as 1.0.x. |
| Should the cut bundle surface tidy-ups (alphabetise `__all__`, etc.)? | **Yes, exactly two** — Q4 (`__all__` ordering) and Q3 (local variable rename). Both cosmetic, snapshot test updated in the same commit. |
| What is the `1.0.0` `### Added` section, given the freeze policy? | Unchanged from the scaffolding: `### Added` lists *declarations* (classifier flip, support-window policy); `### Changed` lists policy tightening. Code-level changes shipped in 0.2.x–0.9.x. |
| **Do consumers admit 1.0.0?** *(new — not in the scaffolding)* | **No. Blocking.** All four cap below 1.0. M0 widens them to `<2.0` against 0.9.x first, using the surface-lock test as evidence (D3). |
| **Is the `user-corpus-onboarding` framing earned?** *(new)* | **Yes, fully — all three deliverables shipped** (2026-08-10 correction): guide ✅, harness ✅, and the `DirectoryCorpus` override ✅ (`extra_aliases_file`, PR #130). The D7 "gap" was a scoping premise error; M0.5 void. |

### Affected layers

**Corrected 2026-08-09.** The scaffolding marked every consumer "no code
change required." That is wrong for three of them — see D3.

- [x] **attune-rag** — `pyproject.toml` (version + classifier),
      `src/attune_rag/__init__.py`, `docs/POLICY.md`, `README.md`,
      `CHANGELOG.md`, `tests/unit/test_api_surface.py` (Q4 snapshot),
      tag + PyPI publish.
- [x] **attune-gui** — **pin change required.** `<1.0` → `<2.0`
      (M0.3). Without it the cut is unusable downstream.
- [x] **attune-ai** — **pin change required.** `<0.10` → `<2.0` at three
      sites (M0.2), with the re-validation its own comment demands.
- [x] **attune-author** — **pin change required, and already broken.**
      `>=0.8.0,<0.9` excludes the published 0.9.0 (M0.1).
- [ ] **attune-help** — no attune-rag dependency; genuinely unaffected.
