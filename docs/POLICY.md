# attune-rag Public API and Deprecation Policy

This policy governs the public surface of the `attune-rag` package.
The surface is **documented and snapshot-tested as of 0.1.18**;
**formal SemVer commitments take effect with 0.2.0** (gated on
Phase 4 of the v1.0 roadmap — downstream-validation burn-in; the
cut spec is [docs/specs/api-v0.2.0-cut/](specs/api-v0.2.0-cut/)).
Between those two markers the freeze is honor-system: the lock
test in `tests/unit/test_api_surface.py` catches accidental drift,
but downstreams should treat 0.1.x as still-evolving and pin tightly.

The spec that introduced this policy is
[docs/specs/api-v0.2-public-surface/](specs/api-v0.2-public-surface/).

## 1. What's covered

The public surface — the set of names and paths attune-rag promises to
keep stable across minor versions within the same major — is exactly:

1. **Names in `attune_rag.__all__`** — the root re-export hub. The
   current list is locked by
   [`tests/unit/test_api_surface.py`](../tests/unit/test_api_surface.py)
   and enumerated in README's [Public API](../README.md#public-api)
   section.
2. **Names in the `__all__` of each PUBLIC submodule**:
   - `attune_rag.corpus`
   - `attune_rag.providers`
   - `attune_rag.editor`
3. **Importability of each PUBLIC submodule by qualified path**:
   - `attune_rag.corpus.attune_help`
   - `attune_rag.corpus.help_adapter`
   - `attune_rag.editor.rename`
   - `attune_rag.editor.schema`
   - `attune_rag.editor.lint`
   - `attune_rag.editor.autocomplete`
   - `attune_rag.editor.references`
4. **The package attribute `attune_rag.__version__`** — its presence
   and PEP-440 shape, not its specific value.
5. **The data file `src/attune_rag/editor/template_schema.json`** —
   path, file format, and backward-compatible JSON schema evolution.

Anything *not* in the list above is INTERNAL and may move, rename, or
disappear in any release without notice — even names that happen to be
importable today.

## 2. SemVer commitment

The project follows [Semantic Versioning](https://semver.org/) with the
following commitments:

- **0.1.x (pre-freeze).** The surface is documented but not
  contractually frozen. The lock test gates accidental drift in PRs,
  but a minor bump may still introduce changes if the spec is
  updated in the same release.
- **0.2.x onward.** No PUBLIC symbols are removed within the same
  minor version (e.g. anything PUBLIC in 0.2.0 stays through every
  0.2.z). PUBLIC symbols *may* be removed at the next minor bump
  (0.3.0) provided they were marked deprecated in a prior 0.2.z.
- **1.x onward (future).** PUBLIC symbols are only removed at major
  version bumps, and only after at least one minor release shipped
  with a `DeprecationWarning` at the symbol's call site.

The classifier in `pyproject.toml` (`Development Status`) tracks this
commitment level. The 1.0 release flips it to
`Production/Stable` — see the v1.0 roadmap (Phase 5).

## 3. Procedure for retiring a PUBLIC symbol

When a PUBLIC symbol needs to go:

1. **Open a PR adding a `DeprecationWarning` at the call site.** The
   warning must name (a) the deprecated path, (b) the replacement
   (or "no replacement; downstream should pin and migrate manually"),
   and (c) the release in which it will be removed.
2. **Land a CHANGELOG entry under "Deprecated"** linking the spec/
   reasoning if non-obvious.
3. **Ship at least one minor release with the warning live** before
   removing. The window gives downstream maintainers time to adapt.
4. **Remove in the next major release** (or the next minor for 0.x).
   Update the `EXPECTED_*` constants in
   [`tests/unit/test_api_surface.py`](../tests/unit/test_api_surface.py)
   and any deprecation shim in the same PR.

The five `attune_rag.editor._*` shim modules introduced in 0.2.0
follow this procedure — they are scheduled for removal in 0.3.0.

## 4. Procedure for adding a PUBLIC symbol

Promoting an INTERNAL symbol (or introducing a brand-new one) to the
PUBLIC surface:

1. **Land the symbol in source** and add it to the relevant `__all__`.
2. **Update the `EXPECTED_*` constants** in
   [`tests/unit/test_api_surface.py`](../tests/unit/test_api_surface.py).
   The test will fail loudly without this — that's the gate.
3. **Add a CHANGELOG entry under "Added"** for the next release.
4. **Add an entry to the README's
   [Public API](../README.md#public-api) section** in the same PR.

Symbols are only PUBLIC once all four steps land together. A symbol
in source `__all__` but not in the surface test is INTERNAL by
construction — the test is the source of truth.

## 5. Underscore convention

Module names starting with `_` are **never** part of the public
surface, even if a downstream imports them today. The five-module
rename in 0.2.0 (`attune_rag.editor._rename` → `attune_rag.editor.rename`,
plus four siblings) is the one-time correction of a historical
accident — the deprecation shims at the underscore paths exist for
backward compatibility only and are removed in 0.3.0.

New submodules that need to be PUBLIC must use non-underscore names
from the start. New submodules that need to be INTERNAL use the
underscore prefix and stay out of `__all__`.

## 6. Resources

The data file `src/attune_rag/editor/template_schema.json` is part of
the contract because attune-help and other downstream tools read or
mirror it.

- **Additions** (new optional fields, new permitted enum values) are
  backward compatible and ship under "Added".
- **Tightenings** (removing a permitted enum value, making a field
  required, tightening a regex) are breaking. They follow the same
  deprecation timeline as code: announce in CHANGELOG "Deprecated",
  ship at least one minor release with the looser shape still
  honored, then tighten.
- **File-path changes** (renaming or moving the schema file) are
  treated identically to renaming a PUBLIC module: introduce a
  symlink or copy, deprecate the old path, remove at the next major.

## 7. Behavioral commitments

§1–§6 govern the package's *symbol-level* surface — what names
exist and how they evolve. This section governs **behavioral**
stability: what `attune-rag` commits to *doing* across a minor
series, beyond the names staying put.

### 7.1 What we commit to

**Retrieval-quality non-regression on the bundled corpus.** The
baseline numbers locked in
[docs/specs/release-quality-baseline/baseline-1.md](specs/release-quality-baseline/baseline-1.md)
are *floors*, not point-in-time observations:

- Baseline `precision_at_1`: 1.00 (40 / 40)
- Baseline `recall_at_3`: 1.00 (40 / 40)

These hold across every 1.x release. A PR cannot land that drops
either metric on the bundled `attune-help` corpus + locked
[`tests/golden/queries.yaml`](../tests/golden/queries.yaml). The
strict-dominance discipline used during the alias-expansion sweep
(see [docs/specs/alias-expansion-sweep/](specs/alias-expansion-sweep/))
is the contractual floor; every internal change runs the full
baseline diagnostic before merge.

Enforcement: [`tests/golden/test_golden.py`](../tests/golden/test_golden.py)
on every PR, plus the watermark guard for the paraphrased set
(R@3 floor at 85%, set in 0.1.23).

### 7.2 What we do *not* commit to

Selection-criteria internals evolve freely within a minor series
and ship as CHANGELOG `### Changed` — not `### Added`, not breaking:

- **Exact document ordering** beyond the P@1 / R@3 floors. Two
  candidates at adjacent scores may swap rank-2 vs rank-3 between
  releases.
- **Score values** — retrieval scores, faithfulness numbers, rerank
  deltas may drift across releases as the scoring pipeline evolves.
- **Alias dictionary contents** —
  [`aliases_override.json`](../src/attune_rag/corpus/aliases_override.json)
  and per-template frontmatter aliases may grow, shrink, or change.
- **Tokenizer behavior and stem rules** — `_tokenize()`,
  `_MIN_STEM_LEN`, stemmer choice are all internal.
- **Reranker prompt wording** and the `candidate_multiplier`
  default (currently `3`).

These are the levers we tune to *hold* the §7.1 floors, not the
floors themselves.

### 7.3 Faithfulness — tracked, not committed

Mean faithfulness on the bundled corpus is measured in CI on
retrieval-touching PRs (per
[docs/specs/release-quality-baseline/](specs/release-quality-baseline/)),
currently at `mean − σ·stdev` = 0.9698 over N=20 runs. **We do
not commit to a faithfulness floor at the POLICY level for 1.x.**
The metric depends on a Claude judge call (network + API key +
model availability), and its composition depends on the reranker's
default posture (see
[reranker-evaluation D5](specs/reranker-evaluation/)) — pinning a
faithfulness floor before D5's verdict locks the reranker default
would commit to a number whose definition is in flight.

The CI gate is a *development quality signal*, not a downstream
SemVer contract. 1.1.0+ may revisit this as a conditional
commitment once D5 lands and telemetry
(per [docs/specs/telemetry/](specs/telemetry/)) produces
real-distribution data to back the floor.

### 7.4 Provenance

This section codifies as POLICY what the alias-expansion sweep
(0.1.23) and the
[release-quality-baseline](specs/release-quality-baseline/) spec
(Phase 1 of v1.0) already practiced: every retrieval change ships
with a baseline-diagnostic check; baseline numbers are not allowed
to move. §7 makes that internal discipline a downstream-facing
commitment.

## See also

- [docs/specs/api-v0.2-public-surface/requirements.md](specs/api-v0.2-public-surface/requirements.md)
  — the audit and module-by-module verdict.
- [docs/specs/api-v0.2-public-surface/design.md](specs/api-v0.2-public-surface/design.md)
  — the proposed `__all__` per PUBLIC module with rationale.
- [tests/unit/test_api_surface.py](../tests/unit/test_api_surface.py)
  — the executable contract.
- [docs/specs/release-quality-baseline/baseline-1.md](specs/release-quality-baseline/baseline-1.md)
  — the locked baseline numbers §7.1 commits to.
- [docs/specs/reranker-evaluation/](specs/reranker-evaluation/)
  — the D5 diagnostic that resolves the reranker-default decision
  §7.3 forward-references.
