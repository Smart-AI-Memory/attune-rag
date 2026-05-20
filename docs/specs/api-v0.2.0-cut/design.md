# Spec: attune-rag 0.2.0 cut — Design

> **Status: scaffolding — not yet scoped; activates after Phase 4 W4.2.**

This document sketches the shape of the 0.2.0 cut. It is **not** a
detailed design — the `/spec` scoping pass after W4.2 fills in the
acceptance criteria, scripts, and CHANGELOG copy.

## Phase 2: Design

### SemVer interpretation for this cut

0.2.0 is a **minor bump from the 0.1.x line** with a single contractual
change: the public surface frozen at the symbol level in Phase 3 is
now ratified at the SemVer level. Concretely:

| Aspect | Before (0.1.x) | At 0.2.0 |
|---|---|---|
| Names in `__all__` (root + PUBLIC submodules) | Documented and lock-tested; can still change in a minor bump if the spec is updated in the same release. | **Frozen.** No removals within 0.2.x; removals at the next minor (0.3.0) require a prior 0.2.z `DeprecationWarning`. |
| Importability of PUBLIC submodules by qualified path (`attune_rag.editor.rename`, `attune_rag.corpus.attune_help`, etc.) | Lock-tested. | **Frozen.** Same removal rules. |
| `attune_rag.__version__` presence + PEP-440 shape | Asserted by lock test. | Same; value bumps to `0.2.0`. |
| `src/attune_rag/editor/template_schema.json` shape | Backward-compat-only changes enforced by W0.1 freeze enforcer. | Same; the contract is now SemVer-binding rather than honor-system. |
| Internal modules (`cli.py`, `benchmark.py`, `eval/`, `dashboard/`, `providers/{claude,gemini}.py`, anything not in `__all__`) | INTERNAL; may move/rename/disappear. | Unchanged — still INTERNAL. |

**What 0.2.0 is not:**

- Not a Production/Stable claim. The Development Status classifier
  in `pyproject.toml` **stays at `3 - Alpha`** through 0.2.x. The
  Alpha → Production/Stable flip is Phase 5's job (v1.0.0 spec).
- Not a re-audit of the surface. The audit ran in Phase 3
  ([`api-v0.2-public-surface/requirements.md`](../api-v0.2-public-surface/requirements.md))
  and is locked by [`tests/unit/test_api_surface.py`](../../tests/unit/test_api_surface.py).
  0.2.0 ratifies the audit's verdict unchanged.
- Not a place to land new public symbols. Any symbol promotion goes
  through the deprecation policy starting at 0.3.0.

### Public surface map (reference)

The frozen surface — to be re-verified by M1 before the cut, not
re-designed here — is exactly what is locked today by
[`tests/unit/test_api_surface.py`](../../tests/unit/test_api_surface.py)
and documented in [`docs/POLICY.md` §1](../../POLICY.md#1-whats-covered).
For convenience the modules whose `__all__` is part of the contract are:

| Module | `__all__` source | Notes |
|---|---|---|
| [`attune_rag`](../../../src/attune_rag/__init__.py) | root re-export hub | The list locked in Phase 3 (M1.1: `AttuneHelpCorpus` added; no other changes since). |
| [`attune_rag.corpus`](../../../src/attune_rag/corpus/__init__.py) | sub-package | Includes `AttuneHelpCorpus` per Phase 3 M1.2. |
| [`attune_rag.providers`](../../../src/attune_rag/providers/__init__.py) | sub-package | `LLMProvider`, `get_provider`, `list_available`. |
| [`attune_rag.editor`](../../../src/attune_rag/editor/__init__.py) | sub-package | Plus qualified-path PUBLIC submodules `rename`, `schema`, `lint`, `autocomplete`, `references`. |
| [`attune_rag.corpus.attune_help`](../../../src/attune_rag/corpus/attune_help.py) | qualified PUBLIC module | `AttuneHelpCorpus`. |
| [`attune_rag.corpus.help_adapter`](../../../src/attune_rag/corpus/help_adapter.py) | qualified PUBLIC module | `HelpCorpusAdapter` Protocol. |

**Resource contract:**
[`src/attune_rag/editor/template_schema.json`](../../../src/attune_rag/editor/template_schema.json)
— path, file format, and backward-compatible JSON-schema evolution
(enforced by the W0.1 freeze enforcer).

**Deprecation shims still load-bearing through 0.2.x:** the five
`attune_rag.editor._{rename,schema,lint,autocomplete,references}.py`
PEP 562 shims from Phase 3 M2.3 emit `DeprecationWarning` and continue
to work through every 0.2.z; they may be removed at 0.3.0 per
[`docs/POLICY.md`](../../POLICY.md).

### Classifier flip (where it does NOT happen)

The `Development Status :: 3 - Alpha` classifier in
[`pyproject.toml`](../../../pyproject.toml) **stays at Alpha** through
0.2.x. Rationale:

- **0.2.0 = SemVer freeze.** The contract is "no PUBLIC removals
  within the minor version."
- **Phase 5 = stability claim.** Production/Stable is an external
  posture (downstreams can rely on us for production use); it requires
  Phase 4 burn-in + a seven-day post-release no-hotfix window (see
  [ROADMAP-v1.md Phase 5 Gate](../ROADMAP-v1.md#phase-5--100-release))
  + an explicit roadmap deliverable. None of those are coupled to the
  SemVer minor bump.

Conflating the two would force every later minor bump to re-prove
production-readiness, which is not how SemVer works.

### Deprecation policy (inheritance, not new)

The deprecation policy is already documented at
[`docs/POLICY.md`](../../POLICY.md) (Phase 3 deliverable M4.1). The
0.2.0 cut does not rewrite it — it activates the binding clauses that
the policy itself defers to 0.2.0:

- §2 "**0.2.x onward.** No PUBLIC symbols are removed within the same
  minor version…" — binding from this cut forward.
- §3 retire procedure (`DeprecationWarning` in a prior 0.2.z before a
  removal at 0.3.0) — applies to any post-cut removal proposal.
- §4 add procedure — applies to any 0.2.z that introduces a new
  PUBLIC symbol.

`docs/POLICY.md` may need a minor wording change at the cut to
replace "**formal SemVer commitments take effect with 0.2.0** (gated
on Phase 4…)" with "took effect at 0.2.0 on `<date>`". That copy edit
is the only POLICY.md touch in scope for this spec.

### Risks & mitigations (sketch)

To be expanded during scoping. Initial candidates:

| Risk | Mitigation sketch |
|---|---|
| A surface drift slipped past the lock test during the freeze (e.g. a `### Changed` PR quietly removed a symbol). | M1 pre-cut audit re-runs the lock test against the current `main` and diffs `__all__` against the frozen design.md tables. |
| attune-gui consumes a symbol we didn't catch in the original audit. | M1 re-runs the cross-repo grep from [`api-v0.2-public-surface/requirements.md`](../api-v0.2-public-surface/requirements.md#downstream-import-audit) against the current attune-gui main. |
| The 0.2.0 wheel installs cleanly but breaks attune-gui's pin syntax (`>=0.1.x,<0.2`). | M3 post-cut verification pins attune-gui to `>=0.2.0,<0.3` and runs its full suite. |
| A hotfix is required within the 7-day window → Phase 5 entry gate resets. | Acceptable. Phase 5 has its own no-hotfix-7-days clock; the 0.2.0 cut is not contractually obligated to be hotfix-free, only to honor SemVer thereafter. |
