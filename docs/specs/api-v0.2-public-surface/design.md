# Spec: Public API Surface Freeze for attune-rag 0.2.0

## Phase 2: Design

**Status**: 0.1.18 candidate (M1–M3 + M4.1–M4.4 + M5.1 realized; framing softened from "freeze" to "groundwork toward the 0.2.0 freeze")

### Surface principle

A symbol is PUBLIC only if it is named in the `__all__` of either:

1. the top-level [`attune_rag/__init__.py`](src/attune_rag/__init__.py), or
2. a sub-package `__init__.py` that is itself listed below.

Any other name — even one importable with `from attune_rag.foo import
bar` — is INTERNAL and may move, rename, or vanish in any release.

A PUBLIC submodule is a submodule whose qualified path is part of the
contract. There are exactly **four** PUBLIC submodules for 0.2.0:

- `attune_rag.editor`
- `attune_rag.corpus` (and its individually-listed children below)
- `attune_rag.providers`
- (the root package is implicitly the fifth.)

`attune_rag.dashboard`, `attune_rag.eval`, `attune_rag.benchmark`, and
`attune_rag.cli` are not PUBLIC submodules.

### Proposed `__all__` per PUBLIC module

The frozen surface for 0.2.0. Symbol names are exactly what already
exists in source. Rationale columns explain *why* this symbol earns a
slot — the test fails if any of these disappear.

#### `attune_rag` (root)

| Symbol | Rationale |
|---|---|
| `RagPipeline` | The pipeline orchestrator. Constructor for every downstream consumer. |
| `RagResult` | Return value of `RagPipeline.run`. Downstreams inspect `.text`, `.citation`, etc. |
| `CitationRecord` | Document-level provenance record. Part of `RagResult.citation`. |
| `CitedSource` | One cited document. Part of `CitationRecord.hits`. |
| `ClaimCitation` | Claim-level provenance from the Anthropic Citations API. Part of `RagResult.claim_citations`. |
| `format_citations_markdown` | Render helper for `CitationRecord` → markdown. Documented in the README. |
| `format_claim_citations_markdown` | Render helper for `ClaimCitation` lists → markdown. |
| `CorpusProtocol` | Pluggable corpus interface. Documented extension point. |
| `RetrievalEntry` | One row from a corpus. Part of the `CorpusProtocol` contract. |
| `DirectoryCorpus` | Default corpus implementation. Top consumer (attune-gui, attune-help). |
| `KeywordRetriever` | Default retriever. Constructor used by every downstream and by tests in attune-help. |
| `RetrievalHit` | One ranked hit. Part of the retriever contract. |
| `RetrieverProtocol` | Pluggable retriever interface. Documented extension point. |
| `build_augmented_prompt` | Prompt-template helper. Used by `RagPipeline` and exposed for downstreams that build prompts directly. |
| `PROMPT_VARIANTS` | Tuple of available prompt-variant identifiers. Stable enum-like surface for variant selection. |
| `QueryExpander` | LLM query-expansion adapter. Imported by attune-gui. |
| `LLMReranker` | LLM rerank adapter. Re-exported for parity with `QueryExpander`. |
| **NEW:** `AttuneHelpCorpus` | attune-help imports this via the qualified `attune_rag.corpus.attune_help.AttuneHelpCorpus`. Re-exporting at root makes the constraint visible in the snapshot test. The qualified import path continues to work — no call-site change required. |

> The root-level `__version__` attribute is also PUBLIC (downstream
> probes for it) but is not a `__all__` member by convention. The
> surface test asserts its presence and string-shape (PEP 440) but not
> its value.

#### `attune_rag.corpus`

| Symbol | Rationale |
|---|---|
| `CorpusProtocol` | Re-exported here for `from attune_rag.corpus import ...`. |
| `RetrievalEntry` | Same. |
| `DirectoryCorpus` | Same. |
| `AliasInfo` | Template-editor alias metadata. Used by `attune_rag.editor`. |
| `DuplicateAliasError` | Raised by corpus loaders on duplicate-alias detection. |
| **NEW:** `AttuneHelpCorpus` | Currently documented in the corpus `__init__.py` docstring but not in `__all__`. Phase 3 promotes it because attune-help depends on this import path. |

#### `attune_rag.corpus.attune_help`

A single-class submodule. PUBLIC because attune-help imports
`AttuneHelpCorpus` via this fully-qualified path.

| Symbol | Rationale |
|---|---|
| `AttuneHelpCorpus` | The thin adapter over the bundled attune-help templates. Exposed both here and re-exported at the root. |

#### `attune_rag.corpus.help_adapter`

A single-Protocol submodule. PUBLIC because attune-help references
this Protocol by qualified path in its adapter docstrings.

| Symbol | Rationale |
|---|---|
| `HelpCorpusAdapter` | `runtime_checkable` Protocol implemented by attune-help. Frozen to keep the inversion (rag never imports help) intact. |

#### `attune_rag.providers`

| Symbol | Rationale |
|---|---|
| `LLMProvider` | The provider Protocol. Documented extension point. |
| `get_provider` | Factory by name. Used by `cli.py` and by user-facing scripts in attune-help. |
| `list_available` | Feature-detection: which provider SDKs are importable. Used to gate UI affordances. |

`ClaudeProvider` and `GeminiProvider` are intentionally not in `__all__` —
they are reached only via `get_provider(...)`. If a downstream needs
to subclass them, file a follow-on; promotion goes through the
deprecation policy.

#### `attune_rag.editor`

The full member list comes from the existing
[`editor/__init__.py`](src/attune_rag/editor/__init__.py) `__all__` —
20 symbols. Frozen verbatim:

```python
__all__ = [
    "Diagnostic", "FileEdit", "FileMove", "Hunk",
    "Reference", "ReferenceContext", "ReferenceKind",
    "RenameCollisionError", "RenameError", "RenamePlan",
    "SchemaError", "Severity",
    "apply_rename", "autocomplete_aliases", "autocomplete_tags",
    "find_references", "lint_template", "load_schema",
    "parse_frontmatter", "plan_rename", "validate_frontmatter",
]
```

Rationale: every symbol above is consumed by attune-gui's template
editor. Each one earns its slot by being part of one of the editor's
five subsystems (lint, autocomplete, references, rename, schema).
Adding new symbols requires going through the deprecation policy in
reverse (announce → land in minor version).

#### `attune_rag.editor.{rename,schema,lint,autocomplete,references}`

These five submodules are the renamed-from-underscore files. Each is
PUBLIC at the *submodule* level — i.e. `attune_rag.editor.rename` is
importable, and the snapshot test asserts it imports. Their internal
symbols are not part of the `__all__` lock (the parent
`attune_rag.editor` re-exports the public names).

The underscore-prefixed shims (`_rename`, `_schema`, etc.) are
**INTERNAL** and explicitly excluded from the snapshot test by name,
even though they continue to exist for 0.2.x as backward-compat
trampolines. The deprecation policy documents their removal in 0.3.0.

### Snapshot test design

Lands in `tests/unit/test_api_surface.py` during the implementation
phase. The shape:

```python
import importlib

EXPECTED_ROOT_ALL = frozenset({...})
EXPECTED_CORPUS_ALL = frozenset({...})
EXPECTED_PROVIDERS_ALL = frozenset({...})
EXPECTED_EDITOR_ALL = frozenset({...})

PUBLIC_SUBMODULES = (
    "attune_rag.corpus",
    "attune_rag.corpus.attune_help",
    "attune_rag.corpus.help_adapter",
    "attune_rag.providers",
    "attune_rag.editor",
    "attune_rag.editor.rename",
    "attune_rag.editor.schema",
    "attune_rag.editor.lint",
    "attune_rag.editor.autocomplete",
    "attune_rag.editor.references",
)

def test_root_all_is_frozen():
    import attune_rag
    assert frozenset(attune_rag.__all__) == EXPECTED_ROOT_ALL

def test_public_submodules_import():
    for path in PUBLIC_SUBMODULES:
        importlib.import_module(path)

def test_root_has_version():
    import attune_rag, re
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-.]\w+)?", attune_rag.__version__)
```

Three checks: `__all__` membership per PUBLIC module, importability of
every PUBLIC submodule, and `__version__` presence + PEP-440 shape.

Failure mode: any drift forces a deliberate update to the
`EXPECTED_*` constants — which is a review surface that catches
accidental removals and forces a deprecation-policy decision on
intentional ones.

### Deprecation policy (`docs/POLICY.md` outline)

The policy doc lands in implementation. Outline:

1. **What's covered** — every symbol in this design.md's `__all__`
   tables, plus the four PUBLIC submodules, plus the resource file
   `attune_rag/editor/template_schema.json`, plus `__version__`.
2. **SemVer commitment** — for 0.x: no removals of PUBLIC symbols
   within the same minor version. For 1.x onward: removals only at
   major versions, after at least one minor-version deprecation
   warning.
3. **Procedure for retiring a PUBLIC symbol:**
   1. Open a PR adding a `DeprecationWarning` at the symbol's call
      site, naming the replacement (or "no replacement; downstream
      should pin and migrate manually").
   2. Land a CHANGELOG entry under "Deprecated".
   3. Ship at least one minor release with the warning live.
   4. Remove in the next major release. Update the snapshot test
      `EXPECTED_*` constants in the same PR.
4. **Procedure for adding a PUBLIC symbol:**
   1. Land the symbol in source, add it to the relevant `__all__`.
   2. Update the snapshot test `EXPECTED_*` constants.
   3. Document in CHANGELOG under "Added".
   4. README "Public API" section gets an entry.
5. **Underscore convention** — module names starting with `_` are
   *never* part of the public surface, even if a downstream imports
   them today. The editor submodule renames in this spec are the
   one-time correction of a historical accident.
6. **Resources** — the JSON schema file
   `attune_rag/editor/template_schema.json` is part of the contract.
   Backwards-incompatible schema changes (removing required fields,
   tightening enums) follow the same deprecation timeline as code.

### Editor submodule rename — backward-compat plan

For each pair `(_old, new)` in `{(_rename, rename), (_schema, schema),
(_lint, lint), (_autocomplete, autocomplete), (_references,
references)}`:

1. Rename the file in-place (`git mv src/attune_rag/editor/_rename.py
   src/attune_rag/editor/rename.py`).
2. Create a new shim file at the old path:
   ```python
   # src/attune_rag/editor/_rename.py
   """Deprecated alias — use ``attune_rag.editor.rename`` instead.

   Removed in 0.3.0.
   """
   from warnings import warn
   from . import rename as _rename_real

   warn(
       "attune_rag.editor._rename is deprecated; use "
       "attune_rag.editor.rename instead. Will be removed in 0.3.0.",
       DeprecationWarning,
       stacklevel=2,
   )

   # Re-export everything the old module had.
   from .rename import *  # noqa: F401,F403
   ```
3. Update the parent `editor/__init__.py` to import from the new
   non-underscore paths.
4. Cross-repo: file a tracking issue/PR in attune-gui to update its
   imports. Land that PR after 0.2.0 ships.

### Note: roadmap doc not yet committed

The original Phase 3 brief references `docs/specs/ROADMAP-v1.md` and
`docs/specs/release-quality-baseline/`. Neither exists in this
branch yet. This spec is written to mirror the `dashboard-v0.2.0`
shape (requirements.md / design.md / tasks.md), which is the only
completed spec on disk. When the roadmap lands, the "Status" headers
in requirements.md / design.md / tasks.md should be updated to
reference it directly.
