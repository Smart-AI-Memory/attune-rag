# Spec: attune-rag 1.0.0 release — scoping decisions

> **Status: scoped 2026-08-09.** This file is the output of the formal
> `/spec` scoping pass that [requirements.md](requirements.md),
> [design.md](design.md), and [tasks.md](tasks.md) each defer to. The
> decisions below are binding on the cut PR; the three scaffolding files
> are corrected in the same commit to match.
>
> **Approved 2026-08-09** — Patrick ratified the decisions including
> D7 = Option A (same session, at retro). All ten decisions are locked;
> execution is unblocked.

- **Owner:** Patrick
- **Scoped:** 2026-08-09
- **Scaffolded:** 2026-05-20

---

## 0. Why this scoping pass had to start with a reconciliation

The scaffolding was written on 2026-05-20, when the package was at
0.1.22 and the next planned cut was 0.2.0. **Three of the spec's load-bearing
premises are no longer true**, and one of them was never true. A scoping
pass that pinned `N` and moved on would have produced a spec that could
not execute.

| # | Scaffolding says | Reality (verified 2026-08-09) | Evidence |
|---|---|---|---|
| 1 | Next cut is 0.2.0; 1.0.0 follows it | **0.9.0 is published.** The 0.2.0 cut executed 2026-05-25; 0.3.0–0.9.0 all shipped since | `origin/main` `pyproject.toml:7` = `0.9.0`; PyPI = 0.9.0; PR [#199](https://github.com/Smart-AI-Memory/attune-rag/pull/199) |
| 2 | Classifier flip is `3 - Alpha` → `5 - Production/Stable`; "`4 - Beta` is intentionally skipped" | **The package has never been Alpha.** It has been `4 - Beta` since the initial scaffold commit | `git log -S'Development Status' -- pyproject.toml` returns exactly one commit (`c3c4bb9`, initial scaffold), value `4 - Beta` |
| 3 | (silent) | **Every downstream consumer caps attune-rag below 1.0.** Publishing 1.0.0 today would resolve for nobody | see D3 |

Premise 3 is the one that matters most: it is not a stale fact, it is a
**gap**. The scaffolding never considered it, and it is a hard blocker on
the cut.

---

## D1 — 1.0.0 cuts from the 0.9.x line, not 0.2.x

**Decision.** The cut's parent is the latest 0.9.x, not 0.2.0. Every
reference to "0.2.0" as the soak subject, the pin target, or the
predecessor version is rewritten to "0.9.x".

**Consequence for the entry gates.** The inherited gate *"0.2.0 has been
on PyPI for at least N days with zero hotfixes"* is re-read as *"the
latest 0.9.x has been on PyPI for at least N days with zero hotfixes."*
See D4 — it is already satisfied.

**What does not change.** The surface-freeze argument. 1.0.0 adds no
public symbols and removes none (requirements.md "Out of scope" stands),
so 1.0.0's surface is 0.9.x's surface. That identity is what makes D3's
pin-widening safe by construction.

---

## D2 — The classifier flip is `4 - Beta` → `5 - Production/Stable`

**Decision.** M3.1 flips `Development Status :: 4 - Beta` →
`Development Status :: 5 - Production/Stable`. The diff is still one
line; what changes is everything written *around* it.

**Retire the "Beta is intentionally skipped" rationale.**
[design.md](design.md) §"Classifier flip" argues that jumping 3 → 5
avoids "a second metadata churn one cycle later." That argument is void:
the package sat at Beta the whole time, so there was never a skip. The
paragraph is deleted rather than reworded — there is no decision left to
justify.

**Downstream copy must not claim to leave "alpha".** This is the part
with teeth:

- `README.md` (M2.3) — the task text says *"drop 'alpha' framing from
  the headline."* Check what the headline actually says first; if it
  never claimed alpha, M2.3 is a no-op and should be closed as such
  rather than inventing an edit.
- `CHANGELOG.md` (M2.4) — the `[1.0.0]` entry describes a Beta → Stable
  promotion. An "out of alpha" narrative would be a false claim in the
  permanent record.
- GitHub release notes (M3.4) — same.

**Cost of having missed this.** Low, caught pre-execution. Worth noting
as a class though: the fact was wrong in *both* requirements.md and
design.md, and each cited the other's framing. Two documents agreeing is
not corroboration when one was copied from the other; only
`git log -S` against the actual file settled it.

---

## D3 — Consumer pin-widening is a prerequisite milestone (new: M0)

**The gap.** All four downstream consumers cap attune-rag below 1.0.0:

| Consumer | Current pin | Resolves 1.0.0? |
|---|---|---|
| attune-gui | `attune-rag>=0.1.22,<1.0` (`pyproject.toml:32`) | ✗ |
| attune-ai | `attune-rag>=0.1.5,<0.10` (three sites: lines 78, 222, 406) | ✗ |
| attune-author | `attune-rag>=0.8.0,<0.9` (`[rag]` extra, line 54) | ✗ |

Publishing 1.0.0 into this state produces a release that no consumer can
install. The scaffolding's Affected-layers table marks attune-gui,
attune-help, and attune-author as *"no code change required"* — that is
wrong for gui, author, and attune-ai.

**Pre-existing drift found in passing:** attune-author's `<0.9` cap
already excludes the published 0.9.0. attune-author has been unable to
resolve the current attune-rag since 2026-07-18 and nothing surfaced it.
Fix it in the same milestone.

**Decision.** Add **M0 — consumer pin widening**, sequenced *before* the
cut, not after.

**Resolving the chicken-and-egg.** Consumers cannot test against 1.0.0
before it exists, and the cut should not ship into pins that exclude it.
The surface-lock snapshot test breaks the cycle: because 1.0.0 ≡ 0.9.x
surface (D1), a consumer that passes against 0.9.x passes against 1.0.0.
So M0 widens pins to `<2.0` against **0.9.x**, with
`tests/unit/test_api_surface.py` as the evidence that the widening is
safe. No release candidate, no TestPyPI staging.

**Per-repo, per workspace deploy order** (root `CLAUDE.md`: rag →
consumers; one layer per commit — four separate PRs, not one):

| # | Repo | Change |
|---|---|---|
| M0.1 | attune-author | `>=0.8.0,<0.9` → `>=0.9.0,<2.0`; re-validate against 0.9.0 (this repairs existing drift, independent of the cut) |
| M0.2 | attune-ai | `<0.10` → `<2.0` at all three sites; the line-78 comment demands explicit re-validation before lifting the cap — honor it, don't just edit the bound |
| M0.3 | attune-gui | `<1.0` → `<2.0`; full rag + editor contract suite green |
| M0.4 | attune-help | confirmed no attune-rag dependency — no-op, recorded for completeness |

**M0 gates the cut.** M3.3 (tag + publish) does not run until M0.1–M0.3
are merged and green.

---

## D4 — N (pre-cut soak) = 14 days. Already satisfied.

**Decision.** N = 14 days on the latest 0.9.x with zero hotfixes.

**Rationale.** The scaffolding's candidate range was 14–30. 14 is chosen,
not split, because the soak's stated job is *"long enough that a latent
regression would have surfaced via the weekly downstream-validation
cycle"* — that is two full cycles at 14 days. Beyond that the gate stops
buying information and starts buying delay, which the requirements
explicitly warn against ("short enough that the 1.0.0 cut doesn't drift
indefinitely").

**Status: satisfied.** 0.9.0 merged via PR
[#199](https://github.com/Smart-AI-Memory/attune-rag/pull/199) on
2026-07-18 — 22 days as of 2026-08-09 — with no 0.9.1 published. The
soak gate is closed before the phase starts.

**Confirm at cut time,** don't re-derive from this file: the count runs
from the PyPI *upload* timestamp, not the PR merge date. They are within
hours of each other here, so 22 ≫ 14 either way, but the check is cheap.

---

## D5 — Support window: security fixes for 6 months past the next minor

**Decision.** Pin N = **6 months** in the design.md sketch:

> Each minor release of 1.x receives security fixes for **6 months**
> after the *next* minor release ships, or **6 months** from its own
> release date, whichever is longer. Bug fixes are only guaranteed for
> the latest minor.

**Rationale.** design.md is explicit that this number is a labor budget
for a sole developer, not an SLA inferred from observed usage. 6 months
is the midpoint of the stated 3–12 range and survives the only test that
matters here: *could Patrick actually backport a security fix to a
6-month-old minor without it becoming the week's work?* At the current
release cadence (0.8.0 → 0.9.0 in 9 days) a 6-month window can span many
minors — that is the labor risk, and it is the reason not to reach for 12.

**Ship the scope-honesty paragraph verbatim.** design.md §"Support
window" requires the POLICY.md section to reproduce the paragraph
explaining that attune-rag's only external consumer is also maintained by
the same author. M2.1 lands the numbers *and* that paragraph; a support
window published without it overstates what the claim is worth.

**Revisit trigger, written into POLICY.md:** if a consumer outside
Smart-AI-Memory pins attune-rag, re-open D5 before the next major.

---

## D6 — 1.x deprecation cycle: ratified as sketched

**Decision.** Adopt [design.md](design.md) §"1.x deprecation cycle"
unchanged. Warning lands in `1.M.0`; removal cannot happen before
`2.0.0`; CHANGELOG `Deprecated` entry with a spec link; `EXPECTED_*`
constants in `tests/unit/test_api_surface.py` cleared in the removal PR.

No open question remained here — the scaffolding listed it as
"pin during scoping" but had already specified the mechanism completely.
Recorded as ratified so it is not re-litigated.

**One correction.** The sketch's closing line says the five
`attune_rag.editor._*` shims "are scheduled for removal in 0.3.0 under
the 0.x policy." We are at 0.9.0 — verify at M1 whether that removal
happened. If the shims are still present at 1.0.0 they inherit the far
stricter 1.x policy and cannot be removed until 2.0.0. That is a real
consequence of the version drift in §0 and is the single most expensive
thing on this page if missed.

---

## D7 — `user-corpus-onboarding`: one gap remains, and it is surface

**This is the decision that sizes the phase. It needs Patrick's ruling.**

[design.md](design.md) §"Phase 5 scope" ratifies the strategic framing —
v1.0.0 claims *"deterministic retrieval framework for your own markdown
corpus, with the attune-help corpus as the bundled exemplar"* — and calls
`user-corpus-onboarding` **"load-bearing for the v1.0.0 framing,"** not
optional.

**Audit against the code (2026-08-09):**

| Deliverable | State |
|---|---|
| "Your own corpus" guide | ✅ shipped — `docs/USER_CORPUS_GUIDE.md`, 939 lines |
| Quality harness | ✅ shipped — `scripts/measure_corpus.py` |
| First-class `aliases_override.json` for **`DirectoryCorpus`** | ❌ **not shipped** — the override path exists only on `AttuneHelpCorpus` (`corpus/attune_help.py:44`); `corpus/directory.py` has no override support at all |

The spec header still reads `scoped 2026-05-22`, which is why this looked
unstarted; two of three deliverables are in fact done.

**Why the gap is awkward.** `DirectoryCorpus` is the class a user points
at *their own* corpus. The override mechanism — the thing that lets them
correct retrieval on their own content — works only for the bundled
exemplar. That is precisely inverted relative to the framing the cut is
supposed to earn.

**And it is public surface**, so the freeze makes the timing binary:

- **Option A — land it pre-cut (recommended).** Adds one milestone
  (M0.5, parallel with M0). Preserves the ratified framing. Cost:
  extends the phase; it is genuinely new public surface entering under a
  spec whose Non-Goals say "no new public symbols" — so this decision is
  an explicit, recorded amendment to that Non-Goal, not an oversight.
- **Option B — descope to 1.1.0.** Cheaper and strictly freeze-clean, but
  then the README and release notes must *not* claim the framework
  framing at 1.0.0, and design.md's framing decision needs an explicit
  amendment saying so. Shipping the framing without the mechanism is the
  one outcome to avoid.

**Recommendation: A.** The framing decision is already ratified and the
release copy is built on it. But B is coherent and cheaper, and the
choice is a scope call, not a technical one — hence the approval gate.

> **RESOLVED 2026-08-09 — Patrick ruled Option A.** The `DirectoryCorpus`
> `aliases_override.json` support lands pre-cut as **M0.5**, sequenced
> before M0.1–M0.3 so consumers validate once against a 0.9.x that
> already carries the symbol. This is the recorded amendment to the
> "no new public symbols" Non-Goal in [requirements.md](requirements.md).
> The steelman for B (a stability cut is the wrong moment for unburned-in
> surface) was put to Patrick explicitly at the same session's retro and
> A was reaffirmed. Consequence accepted: M0.5's symbol enters 1.0.0
> without Phase-4 burn-in; the M4 seven-day watch is its burn-in.

> **VOIDED 2026-08-10 — the gap this decision resolves does not exist.**
> On starting M0.5, the implementation read of
> [`corpus/directory.py`](../../../src/attune_rag/corpus/directory.py)
> found the deliverable already shipped: `DirectoryCorpus` has carried an
> `extra_aliases_file: Path | str | None` kwarg with strict
> `load_aliases_from_file` semantics since PR
> [#130](https://github.com/Smart-AI-Memory/attune-rag/pull/130) — the
> `user-corpus-onboarding` spec's own **M2 milestone** (tasks M2.1–M2.5:
> public helper in `corpus.__all__`, kwarg, dedicated test file
> `tests/unit/test_directory_corpus_extra_aliases_file.py`, surface
> snapshot updated, `AttuneHelpCorpus` refactored onto the helper). The
> guide documents it (`USER_CORPUS_GUIDE.md` §3, "one-liner override
> loading").
>
> **The premise error was mine and it is exactly the class this scoping
> pass exists to catch:** the D7 audit grepped for the literal string
> `aliases_override` in `directory.py` — the shipped parameter is named
> `extra_aliases_file`, so the grep returned nothing and I concluded
> "no override support at all." A feature-shaped grep (or reading the
> constructor signature) would have found it. Symmetric with §0's
> lesson: verify against the live tree means *reading* the tree, not
> string-matching one spelling of a concept.
>
> **Consequences:**
> - **M0.5 is void** — nothing to build. All three `user-corpus-onboarding`
>   deliverables (guide, harness, override) are shipped; the v1.0.0
>   framing is fully earned as-is.
> - **The Non-Goal amendment is void** — 1.0.0 adds no public symbols
>   after all. The surface identity (1.0.0 ≡ 0.9.x) holds cleanly,
>   which simplifies M0: consumers validate against 0.9.x exactly as
>   published, one ordering, no re-validation caveat.
> - Patrick's A-over-B ruling stands recorded as the scope preference
>   it expressed (ship the override before the cut) — satisfied
>   trivially, since it already shipped pre-cut.
> - The `user-corpus-onboarding` spec header still reads
>   `scoped 2026-05-22` with its work complete — same drift class D8
>   flagged for `perf-baseline-multi-run`. Both go to the next
>   status sweep.
> - **Next action is M0.1** (attune-author pin repair).

---

## D8 — `perf-baseline-multi-run` is already satisfied

**Decision.** Mark the perf-stability precondition ✅ and drop it from the
critical path.

**Evidence.** `docs/specs/downstream-validation/perf-baseline.md` reads
`Methodology version: 2`, `Sigma: 2.0`, `Invocations: 5`,
`Runs per invocation: 20`, measured 2026-05-22 at commit `6fbe6d7`, with
`inter_run_stdev` recorded per metric. That is exactly what
[ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5 asks for — *"restores σ=2.0 with
a defensible inter-run noise model"* — and what
`perf-baseline-multi-run` R1 requires.

**Caveat, and it is not small.** That baseline is from 2026-05-22, at
0.1.x. The retrieval hot path has changed materially since — PR
[#194](https://github.com/Smart-AI-Memory/attune-rag/pull/194) moved
`ALIASES_WEIGHT` 1.5 → 2.0 and changed preview construction. **M1 re-runs
the multi-run baseline against the cut commit.** The *methodology* is
settled (that was the deliverable); the *numbers* are stale and a 1.0.0
perf claim resting on an 11-week-old measurement of different code is not
defensible.

The `perf-baseline-multi-run` spec header still says `scoped 2026-05-22`
despite its output being in the tree — flag for the next status sweep.

---

## D9 — Phase-5 backlog triage (10 items)

Discharges the triage the scaffolding defers to scoping time
([requirements.md](requirements.md) §"Disposition",
[design.md](design.md) §"Backlog disposition"). Source:
[phase-5-backlog/items.md](../phase-5-backlog/items.md). M1 was already
promoted and is not re-triaged.

| Item | Bucket | Reasoning |
|---|---|---|
| **Q4** — alphabetise `__all__` (`providers/__init__.py:59`) | **Fold into the cut** | The one item that is *cheaper now than ever again*. Ordering-only, no symbols added or removed, snapshot test updated in the same commit. requirements.md explicitly contemplates this ("purely cosmetic … updated atomically"). |
| **Q3** — `validator` → `keyword` (`editor/schema.py:90`) | **Fold into the cut** | Local variable name, not public surface. Deferred only because the module was freeze-locked; the cut PR is the moment that lock lifts. |
| **Q1** + **T2** — `_RollbackState` extraction + disk-fault simulation | **Promote → own spec** | items.md already says to land them together. ~90 LOC across three rollback layers on the highest-regression-surface module. Not a cut blocker; a refactor of that size inside a release PR is exactly what the burn-in was meant to prevent. |
| **T1** + **T3** — VCR cassette + `cached_prefix` contract test | **Promote → own spec** | Both need a live SDK call to record. That intersects `testing-conventions.md` (the `live` marker, the CI `ANTHROPIC_API_KEY` guard, the $20/mo cap) — a policy decision, not a test-writing task. |
| **P1–P4** — retrieval hot-path micro-opts | **1.0.x backlog, one perf-only PR** | items.md defers them until the inter-run baseline exists; per D8 the methodology now does. All behaviour-equivalent, all `S`. Land post-cut so the perf PR is measured against the fresh M1 baseline rather than muddying the cut's own numbers. |
| **Q2** — lift three `_iter_entries` helpers | **1.0.x backlog, blocked-check first** | items.md blocks this on the `editor/_*.py` shim removal. Per D6 that removal's status is unverified at 0.9.0. Resolve D6's shim question first; Q2 follows or stays blocked on it. |

**Net effect on the cut:** two cosmetic items fold in (Q3, Q4); nothing
else touches it. Two new specs get scaffolded (not executed) during
Phase 5; five items land as 1.0.x.

---

## D10 — `ROADMAP-v1.md` is stale and gets corrected in this commit

`ROADMAP-v1.md` reads `Last updated 2026-05-20`, `Current version 0.1.22`,
`Current phase: Phase 4`. Everything in §0 applies to it too.

**Decision.** Update the header table (current version → 0.9.0, last
updated → 2026-08-09) and the Phase 5 stanza (classifier `4 - Beta` → `5`,
D8's ✅, the M0 prerequisite) in the same commit as this file. A roadmap
that disagrees with the spec it points at is how §0 happened in the first
place.

**Also fix** the Phase 5 gate text on hotfix handling. tasks.md M4.2
already supersedes it — *"any hotfix restarts the clock"* silently
incentivizes delaying real fixes to protect a clean window, which is
backwards — but ROADMAP-v1.md still carries the old wording, and
[requirements.md](requirements.md) "Edge cases" cites ROADMAP as
authoritative for it. Make M4.2 the single source.

---

## Decisions summary

| # | Decision | Status |
|---|---|---|
| D1 | Cut parent is 0.9.x, not 0.2.x | locked |
| D2 | Classifier flip is `4 - Beta` → `5`; no "leaving alpha" copy | locked |
| D3 | **M0 consumer pin-widening gates the cut** (4 repos, 4 PRs) | locked |
| D4 | N = 14 days — already satisfied (22 days, zero hotfixes) | locked, ✅ |
| D5 | Support window = 6 months, with the scope-honesty paragraph | locked |
| D6 | 1.x deprecation cycle ratified as sketched; **verify shim removal** | locked, 1 open check |
| D7 | `user-corpus-onboarding` — Option A ratified 2026-08-09; **VOIDED 2026-08-10: the gap was fiction** — `extra_aliases_file` shipped in PR #130 (M2 of that spec). M0.5 void; no Non-Goal exception; surface identity holds | closed |
| D8 | perf methodology ✅; **numbers must be re-measured at M1** | locked |
| D9 | Backlog triage — Q3/Q4 fold in, 2 specs promoted, 5 to 1.0.x | locked |
| D10 | ROADMAP-v1.md corrected in this commit | locked |

## Open items carried into execution

1. ~~**D7 ruling**~~ — resolved Option A (2026-08-09), then **VOIDED
   2026-08-10**: the deliverable already shipped (PR #130). M0.5 void;
   first real task is M0.1.
2. **D6 shim check** — are the five `attune_rag.editor._*` shims still
   present at 0.9.0? If yes they inherit 1.x rules and are stuck until
   2.0.0. Answer at M1, before the docs roll.
3. **D8 re-measure** — multi-run perf baseline against the cut commit.
   Gates the release-notes perf claim, not the tag.
