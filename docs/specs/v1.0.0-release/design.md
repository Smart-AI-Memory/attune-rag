# Spec: attune-rag 1.0.0 release

## Phase 2: Design

> **Status: scaffolding — not yet scoped; activates after the 0.2.0
> cut closes cleanly + the 7-day post-1.0.0 no-hotfix gate is
> achievable (per [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5 gate).**

- **Shape parent:** [api-v0.2-public-surface/design.md](../api-v0.2-public-surface/design.md)

### What 1.0.0 means vs. 0.2.0

| Dimension | 0.2.0 | 1.0.0 |
|---|---|---|
| Public surface | Documented, snapshot-tested, frozen. | **Same surface.** No additions. |
| SemVer commitment | 0.x semantics: no removals within a minor; removals OK at next minor with a prior deprecation warning ([POLICY.md §2](../../POLICY.md#2-semver-commitment)). | 1.x semantics: removals only at major bumps, after at least one full minor with a `DeprecationWarning` at the symbol's call site. |
| Classifier | `Development Status :: 3 - Alpha` | `Development Status :: 5 - Production/Stable` |
| Support window | Not documented. | Documented in `POLICY.md` (length pinned at scoping; see [requirements.md](requirements.md)). |
| Burn-in evidence | Pre-Phase-4. | Phase 4 complete + N-day post-0.2.0 soak. |

The cut is a **claim**, not new code. The work that earns the
claim happened in Phases 1–4. Phase 5 codifies it.

### Classifier flip

One-line change in [pyproject.toml](../../../pyproject.toml):

```diff
 classifiers = [
-    "Development Status :: 3 - Alpha",
+    "Development Status :: 5 - Production/Stable",
     "Intended Audience :: Developers",
```

`Development Status :: 4 - Beta` is intentionally skipped. The
Phase 4 burn-in is the substance of the Beta step; once it
passes, jumping straight to Stable matches what the package
*means* and avoids a second metadata churn one cycle later.

### Support window

Policy lands as a new section in [docs/POLICY.md](../../POLICY.md).
Sketch (numbers pinned at scoping):

> ### 7. Support window
>
> Each minor release of 1.x receives security fixes for **N
> months** after the *next* minor release ships, or **N months**
> from its own release date, whichever is longer. Bug fixes are
> only guaranteed for the latest minor.
>
> Example (illustrative, with N=6): 1.0.x receives security
> fixes through `release_date(1.1.0) + 6 months`. When 1.1.0
> ships, 1.0.x users have 6 months to upgrade before security
> support ends.

Rationale for *what kind* of policy this is:

- **Latest-minor-only for bug fixes** matches the pattern most
  Python libraries follow and keeps maintenance bounded for a
  sole-developer project.
- **Security fixes for the previous minor for N months** is the
  smallest concession that gives downstreams a real upgrade
  window without forcing every consumer to track every minor.
- **N is a knob, not a constant.** Pin during scoping. The
  current candidate range (3–12 months) maps to "how often is
  Patrick willing to backport a security fix" — a labor budget,
  not a technical constraint.

### 1.x deprecation cycle

Policy lands as a new section in [docs/POLICY.md](../../POLICY.md)
that supersedes the 0.x deprecation procedure from §3 (which
remains documented for historical context). Sketch:

> ### 8. Deprecation under 1.x
>
> Removing a PUBLIC symbol from 1.x:
>
> 1. **Land a `DeprecationWarning`** at the symbol's call site,
>    naming the deprecated path, the replacement, and the major
>    version in which removal will occur.
> 2. **CHANGELOG entry under "Deprecated"** with a link to the
>    spec/issue that motivated the removal.
> 3. **Ship at least one full minor release** with the warning
>    live before removing. "One full minor" means: the warning
>    appears in 1.M.0 and the removal cannot happen before 2.0.0.
> 4. **Removal lands at the major bump** (2.0.0), with the
>    `EXPECTED_*` constants in
>    [`tests/unit/test_api_surface.py`](../../../tests/unit/test_api_surface.py)
>    and any shim cleared in the same PR.
>
> The five `attune_rag.editor._*` underscore-shims introduced in
> 0.2.0 are scheduled for removal in 0.3.0 under the 0.x policy
> (§3) — they do not move to the 1.x policy.

The difference from 0.x is one of *strictness*: in 0.x a
removal can happen at the next minor after a deprecation
warning; in 1.x it has to wait for the next major. Same shape,
longer clock.

### Backlog disposition

Triage [phase-5-backlog/items.md](../phase-5-backlog/items.md)
during scoping.

**Already promoted (no triage needed):** M1 — multi-run perf-baseline
methodology landed as its own spec at
[docs/specs/perf-baseline-multi-run/](../perf-baseline-multi-run/)
([PR #86](https://github.com/Smart-AI-Memory/attune-rag/pull/86)).
It is a Phase 5 deliverable that ships *outside* this cut spec; its
implementation phase modifies `scripts/measure_perf_baseline.py` +
`.github/workflows/perf.yml` (gate plumbing, not public surface),
which is why it could not land during the freeze. Treat it as
parallel work; the 1.0.0 cut does not block on its completion.

The remaining 10 items (Q1–Q4, P1–P4, T1–T3) triage into three
buckets:

- **Fold into [tasks.md](tasks.md).** Candidates: cosmetic surface
  tidy-ups (Q3, Q4 — alphabetise `__all__`, jsonschema attribute
  rename) — only if they can ship atomically with the cut and the
  snapshot test updates in the same commit.
- **Promote to own spec.** Sizeable items that warrant their own
  scoping pass. Example pattern: M1 itself, already promoted via
  [PR #86](https://github.com/Smart-AI-Memory/attune-rag/pull/86)
  — anything Q1–T3 of similar scope (e.g. Q1's `_RollbackState`
  helper extraction in `editor/rename.py`) would follow the same
  path.
- **Won't-do.** Anything that fails the cost/benefit at scoping
  time. Close with a note in
  [phase-5-backlog/items.md](../phase-5-backlog/items.md) so the
  decision is preserved.

The triage itself is a scoping-time activity, not a 1.0.0-cut
deliverable. The cut waits for nothing in the backlog.

### Release mechanics

Reuses the existing `attune-release-check` skill (`/release-prep`
wraps it) — see [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5
attune-ai workflows. The skill enforces:

- `__version__` matches the tag about to be cut.
- Working tree clean.
- CI green on the head commit.
- CHANGELOG has an entry for the version about to ship.
- Version not already on PyPI.

No bespoke release tooling is introduced for 1.0.0 — the
release flow is what 0.2.x already uses.

### Post-release watch

Seven-day no-hotfix gate. The mechanism:

- After tag + publish, the seven-day clock starts.
- If any hotfix release (`1.0.1+`) fires inside that window,
  the seven-day clock restarts at the hotfix's publish date.
- The gate "closes" when no hotfix has fired for seven
  consecutive days post-publish.

This is the gate that ratifies the 1.0.0 claim. Until it
closes, the classifier change is provisional — a high-rate of
hotfix firings would be evidence the cut was premature.

### What this design intentionally does *not* cover

- **A `1.0.0-rc.N` release candidate flow.** Phase 4 *is* the
  RC. Adding a formal RC step on top of a 4-week burn-in is
  process for its own sake.
- **A press release / external announcement.** Out of scope —
  the README headline update is the only externally-visible
  announcement.
- **API additions of any kind.** See [requirements.md](requirements.md)
  "Out of scope".
