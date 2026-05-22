# Spec: user-corpus-onboarding — requirements

> **Status:** **scoped 2026-05-22.** Invariants below are the
> non-negotiables the implementation in [`tasks.md`](tasks.md) must
> satisfy. The seven open questions at the bottom of this file are
> all decided — see
> [`tasks.md` scoping decisions table](tasks.md#scoping-decisions-locked-2026-05-22).
> Invariants are inherited from the v1.0.0 framing decision and the
> alias-expansion sweep's strict-dominance discipline.

## Entry gates (inherited from upstream phases)

This spec activates when **all of** the following hold:

1. **Phase 4 W4.2 green** — four consecutive `cadence-week-{1,2,3,4}.md`
   reports show zero `### Added` under `[Unreleased]`. The freeze
   discipline that produced the 0.2.0 cut is what makes v1.0.0
   credible; the user-corpus-onboarding work *follows* that, not
   parallel to it.
2. **0.2.0 cut shipped to PyPI.** Per
   [`api-v0.2.0-cut`](../api-v0.2.0-cut/) M2. The new public surface
   this spec introduces (`extra_aliases_file` kwarg, the harness
   module) lands at v1.0.0, not 0.2.x — so the 0.2.0 SemVer commitment
   has to be in place first.
3. **0.2.0 7-day no-hotfix watch closed clean.** Per
   [`api-v0.2.0-cut`](../api-v0.2.0-cut/) M3.3. Any hotfix in the
   window extends the watch; no Phase 5 work starts until the watch
   closes clean.
4. **Phase 5 opens for `/spec` scoping.** Per
   [ROADMAP-v1.md Phase 5](../ROADMAP-v1.md#phase-5--100-release).

The scaffold itself (these four files plus README) does not require
any of the above — scaffold is docs-only and freeze-compliant.
Implementation is what gates on the above.

## Functional requirements

### R1: Quality harness reproduces the bundled-baseline numbers

When run against the bundled `attune-help` corpus and
[`tests/golden/queries.yaml`](../../../tests/golden/queries.yaml) +
[`tests/golden/queries_paraphrased.yaml`](../../../tests/golden/queries_paraphrased.yaml),
the harness must produce **byte-identical** aggregate numbers to
the current baseline-1.md and the current `test_golden.py` reports.

This is the regression-net property: the bundled corpus is the
ground truth for the harness's own correctness.

Specifically:
- Baseline P@1: 1.00 (40/40)
- Baseline R@3: 1.00 (40/40)
- Paraphrased P@1: 0.9125 (73/80) ± 0
- Paraphrased R@3: 1.00 (80/80) ± 0

### R2: Harness CLI is CI-suitable

- Exit code 0 on watermark-pass, non-zero on watermark-fail.
- All output to stdout; errors to stderr.
- Deterministic output ordering (queries enumerated in YAML order,
  not hash order).
- No network calls, no LLM calls (unless the user opts into rerank
  via the same pipeline-construction path attune-rag uses elsewhere).
- Reproducible across re-runs on the same inputs.

### R3: Harness Python API surface is `__all__`-listed

The harness's primary entry points (`measure`, `MeasureResult`,
likely a couple of helpers) appear in `attune_rag.__all__` or in
the subpackage's `__all__` so the API surface is enumerable.

This follows the existing API-surface discipline from
[`api-v0.2-public-surface`](../api-v0.2-public-surface/). The new
surface is `### Added` and must not land before the 0.2.0 cut.

### R4: `extra_aliases_file` kwarg semantics match the existing
`extra_aliases` kwarg

- File path is `Path | str | None`.
- Missing file → `FileNotFoundError` (with the resolved path in the
  exception message).
- Malformed JSON → `ValueError` (with the file path).
- Underscore-prefixed keys silently filtered (mirrors
  `AttuneHelpCorpus`).
- Schema validation: each value `Iterable[str]`, non-string values
  rejected with a typed error.
- File + inline `extra_aliases` both supported; inline wins on
  per-path collision.

The semantics must be **identical** to what `AttuneHelpCorpus`
does internally (which already works against the alias-expansion
sweep's regression suite); the refactor is moving that logic out
to be reusable, not redesigning it.

### R5: The guide is structurally consistent with the existing docs

Same voice as [`docs/POLICY.md`](../../POLICY.md). Same cross-link
discipline (every claim that depends on a measurement links to its
artifact). Same status-banner pattern if it grows sub-pages.

## Quality requirements

### Q1: Strict-dominance held across all implementation PRs

Same discipline as the alias-expansion sweep: every implementation
PR runs the full baseline diagnostic before merge. R@3 and P@1 on
the bundled corpus must not move by even ±0.01 unless the PR
explicitly documents and justifies the move.

The harness *itself* makes this trivial — running it against the
bundled corpus is `python -m attune_rag.measure_corpus
--corpus-path .help/templates --queries tests/golden/queries.yaml`.

### Q2: The implementation passes the existing test suite

All 848+ unit + golden tests pass against any PR. No test
disabled, no skip added, no xfail without an explicit `[xfail-rationale]`
block in the PR body (mirroring the freeze-override pattern).

### Q3: New public surface stays within the v1.0.0 cut budget

The 0.2.0 cut froze the symbol surface. v1.0.0 ratifies it. The new
symbols introduced by this spec — `attune_rag.measure_corpus.measure`,
`MeasureResult`, the `extra_aliases_file` kwarg, possibly
`load_aliases_from_file` — must be enumerated in the v1.0.0 cut
PR's CHANGELOG `### Added` section and included in the
[`test_api_surface.py`](../../../tests/unit/test_api_surface.py)
`EXPECTED_*` constants.

The total surface budget for this spec is **≤ 5 new public symbols**.
If the design grows past that, the scoping pass re-evaluates.

### Q4: Documentation density matches the alias-expansion-sweep
artifact

The guide is sized to match the substance shipped. The
alias-expansion-sweep documented its 13-PR arc in ~400 lines across
README + design + tasks; the user-corpus guide should be at least
that detailed for the bundled-corpus authoring discipline (which is
the heart of what users need to learn).

## Out-of-scope requirements (deferred)

These are not requirements for the v1.0.0 ship of this spec; listed
so they don't slip in unbudgeted:

- **No multi-corpus reporting.** One corpus per harness run.
- **No web UI.** CLI + Python API only.
- **No LLM-driven alias authoring.** Manual authoring discipline;
  LLM-assisted authoring is a separate future spec.
- **No `summaries_override.json`-file kwarg.** Aliases are the
  load-bearing lever; summaries can follow in v1.1.0 if usage
  justifies.
- **No watermark for mean reranker faithfulness.** Faithfulness
  watermarking lives in the existing
  [`release-quality-baseline`](../release-quality-baseline/) spec;
  this harness reports the number but doesn't gate on it for user
  corpora.

## Open questions for scoping

(Each gets a decision at the `/spec` pass; recorded here as the
unresolved set the scoping pass inherits.)

1. **Module name.** `attune_rag.measure_corpus` vs
   `attune_rag.harness` vs `attune_rag.quality` — pick one.
2. **CLI entry point.** `python -m attune_rag.measure_corpus` vs a
   `console_scripts` entry (`attune-rag-measure`) — both? One?
3. **Default watermark value.** 0.85 R@3 (matches the bundled
   floor) vs 0.80 (more permissive default for new corpora) vs no
   default (require explicit flag).
4. **Guide size in M3.** Full 1500-line guide vs leaner 600-line
   "v1" with worked example deferred to v1.1.0. Calendar-pressure
   dependent.
5. **`load_aliases_from_file` helper.** Ship as a public function
   alongside the kwarg, or just the kwarg? Helper adds one symbol
   to the budget but is genuinely useful for advanced users.
6. **Symmetry with `summaries`.** Add `extra_summaries_file=` in
   the same scope? Decision deferred but worth deciding rather than
   silently saying "no."
7. **Rerank-by-default in the harness.** The bundled baseline
   includes the reranker (it's part of the pipeline default);
   user-corpus runs may or may not want it on by default. Default
   should probably mirror the pipeline default, with a `--no-rerank`
   flag for ablation.

## Definition of done (sketch)

To be finalized at scoping. Initial bullets:

- [ ] Harness CLI + Python API ship, listed in `__all__`.
- [ ] `extra_aliases_file=` kwarg ships on `DirectoryCorpus`.
- [ ] Guide exists at `docs/USER_CORPUS_GUIDE.md`.
- [ ] All cross-links from `README.md`, `docs/POLICY.md`,
      `DirectoryCorpus` docstring, and harness `--help` resolve.
- [ ] Harness reproduces bundled-baseline numbers byte-identically
      (R1).
- [ ] All unit + golden tests pass (Q2).
- [ ] New public symbols enumerated in the v1.0.0 cut CHANGELOG
      `### Added` and `test_api_surface.py` (Q3).
- [ ] Symbol budget ≤ 5 (Q3).
- [ ] Strict-dominance held across all implementation PRs (Q1).
