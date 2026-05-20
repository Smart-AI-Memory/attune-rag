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

## See also

- [docs/specs/api-v0.2-public-surface/requirements.md](specs/api-v0.2-public-surface/requirements.md)
  — the audit and module-by-module verdict.
- [docs/specs/api-v0.2-public-surface/design.md](specs/api-v0.2-public-surface/design.md)
  — the proposed `__all__` per PUBLIC module with rationale.
- [tests/unit/test_api_surface.py](../tests/unit/test_api_surface.py)
  — the executable contract.
