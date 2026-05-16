# Spec: Public API Surface Freeze for attune-rag 0.2.0

**Status**: 0.1.18 candidate ready (M1–M3 + M4.1–M4.3 + M5.1 done; M4.4 retargeted from 0.2.0 to 0.1.18; M4.5 ready to fire)

- **Owner:** Patrick
- **Created:** 2026-05-16
- **Target version:** 0.2.0 (does not ship until Phase 2 lands)

> Phase 3 of the v1.0 roadmap. The roadmap doc
> (`docs/specs/ROADMAP-v1.md`) is not yet committed on this branch —
> this spec is the working artifact for Phase 3 and stands alone until
> the roadmap lands. Phase 2 (release-quality baseline) must land
> before 0.2.0 ships.
>
> **Re-targeting decision (2026-05-16):** the work originally scoped
> as 0.2.0 is shipping as **0.1.18** because every change is
> backward-compatible (additive re-export, deprecation shims for
> renamed modules, internal surface lock test, documentation).
> Calling it "0.2.0 API freeze" required the Phase 2 release-quality
> baseline to land first; calling it "0.1.18 groundwork toward the
> 0.2.0 freeze" doesn't. The formal SemVer freeze + classifier flip
> stay queued for 0.2.0/Phase 5.
>
> **M1–M3, M4.1–M4.4, and M5.1** have landed. In attune-rag:
> `AttuneHelpCorpus` re-exported at root; five editor submodules
> renamed with deprecation shims at the old paths;
> [tests/unit/test_api_surface.py](tests/unit/test_api_surface.py)
> locks the surface for every PUBLIC module; [docs/POLICY.md](docs/POLICY.md)
> documents the deprecation policy (taking effect formally in 0.2.0);
> README has a "Public API" section; CHANGELOG has a `[0.1.18]`
> entry; `pyproject.toml` and `__version__` bumped. In attune-gui:
> the route handlers and tests are migrated off the underscore editor
> paths (branch `feature/attune-rag-0.2-editor-rename`).
>
> **Remaining:** M4.5 (tag + publish 0.1.18), M5.2 (delete attune-gui's
> `_editor_dep.py` 503 guard, gated on 0.1.18 published), M5.3 (run
> attune-gui's tests against published 0.1.18).

---

## Phase 1: Requirements

**Status**: complete (this document)

### Problem statement

attune-rag is still classified `Development Status :: 3 - Alpha`
([pyproject.toml](pyproject.toml)). The package has shipped 17 patch
releases and is consumed by three downstream projects (attune-gui,
attune-help, attune-author). Today there is no committed boundary
between "this is the public API we will keep stable" and "this is
internal and may change". Symptoms:

- The package re-exports a curated set from
  [src/attune_rag/__init__.py](src/attune_rag/__init__.py) but the
  invariants of that list are not tested. Anyone can quietly drop a
  name during a refactor.
- Submodules (`attune_rag.editor`, `attune_rag.corpus.attune_help`,
  `attune_rag.corpus.help_adapter`) that downstreams already import
  by qualified path are not documented as public anywhere — they're
  load-bearing but undeclared.
- The `attune_rag.editor` submodule uses underscore-prefixed module
  names (`_rename`, `_schema`) that attune-gui imports directly,
  contradicting Python's normal "underscore == private" convention.
- README has no "Public API" section. There's no deprecation policy.

0.2.0 is the natural cut-line for this. We freeze a documented surface,
add a snapshot test that fails on accidental surface change, and write
the policy that governs future changes.

### Scope

**In scope:**

- A verdict — PUBLIC / INTERNAL / MIXED — for every module under
  `src/attune_rag/`, traced back to actual downstream usage.
- A proposed `__all__` for every PUBLIC module, with one-line
  rationale per symbol (lands in `design.md`).
- A snapshot test under `tests/unit/test_api_surface.py` that locks
  the surface (lands in Phase 3 implementation, not this session).
- A deprecation policy under `docs/POLICY.md` covering: how a symbol
  moves from PUBLIC → deprecated → removed, the SemVer commitment,
  and the minimum support window (lands in Phase 3 implementation).
- A "Public API" section in `README.md` (lands in Phase 3
  implementation).
- A rename pass on `attune_rag.editor`: `_rename` → `rename`,
  `_schema` → `schema`, etc. — underscore-prefixed submodules are
  not legal members of a public surface.

**Out of scope (Non-Goals):**

- Writing `__all__` declarations in source files. Phase 3 spec only.
- Writing the snapshot test. Phase 3 implementation.
- Flipping the classifier to `Production/Stable`. That's Phase 5.
- Shipping 0.2.0. That's gated on Phase 2.
- Adding new public symbols. Freeze ratifies what already exists;
  it does not expand surface area.
- Changing call signatures of any frozen symbol.
- Type-stub publication (`py.typed`) — track as a follow-on.

### Downstream import audit

The point of the freeze is to honor what downstreams already depend on.
Audited via:

```bash
find /Users/patrickroebuck/{attune-gui,attune-help,attune-author} \
  -name '*.py' -not -path '*/.claude/*' \
  | xargs grep -hE 'from attune_rag\b|import attune_rag\b|attune_rag\.'
```

**attune-gui (gating downstream per Decision 2):**

| Symbol / path | Used in | Notes |
|---|---|---|
| `attune_rag.RagPipeline` | `sidecar/attune_gui/services/rag_pipeline.py`, `tests/test_rag_workspace.py`, `tests/test_routes_rag.py` | Top-level re-export. Stable. |
| `attune_rag.DirectoryCorpus` | same | Top-level re-export. Stable. |
| `attune_rag.QueryExpander` | `tests/test_routes_rag.py` | Top-level re-export. Stable. |
| `attune_rag.editor` (submodule import) | `sidecar/attune_gui/_editor_dep.py` | Loaded via `importlib.import_module("attune_rag.editor.<name>")`. Currently guarded with a 503 because attune-gui knows the submodule is "unpublished". Phase 3 removes that condition by formally publishing it. |
| `attune_rag.editor._rename` | `tests/test_editor_ws.py`, `routes/editor_template.py` (lazy) | **Private-name access on a downstream's "private" name.** Phase 3 renames to `attune_rag.editor.rename` and the downstream switches. |
| `attune_rag.editor._schema` | `tests/test_editor_dep.py` | Same — rename to `attune_rag.editor.schema`. |

**attune-help:**

| Symbol / path | Used in | Notes |
|---|---|---|
| `attune_rag.{DirectoryCorpus, KeywordRetriever, RagPipeline}` | `scripts/benchmark_all_fixtures.py`, `scripts/show_low_p1_misses.py`, `scripts/repolish_low_p1.py` | Top-level re-exports. |
| `attune_rag.corpus.attune_help.AttuneHelpCorpus` | `src/attune_help/adapters/rag.py` | Qualified submodule import. Must be PUBLIC. |
| `attune_rag.corpus.help_adapter.HelpCorpusAdapter` (Protocol) | documented in `src/attune_help/adapters/__init__.py`; attune-help implements it | Qualified submodule import. Must be PUBLIC. |
| `src/attune_rag/editor/template_schema.json` (resource file) | mirrored by `attune-help/tests/test_template_prefixes.py` | PUBLIC resource — schema file format is a contract. |

**attune-author:**

| Symbol / path | Used in | Notes |
|---|---|---|
| `import attune_rag` (probe) | `tests/test_rag_hook.py` | Top-level package importable — trivially PUBLIC. |
| `attune_rag.RagPipeline` (patched) | `tests/conftest.py` | Top-level re-export. |

**Constraint summary** — anything Phase 3 calls PUBLIC must cover, at
minimum, this set:

- Top-level: `RagPipeline`, `RagResult`, `DirectoryCorpus`,
  `KeywordRetriever`, `QueryExpander`
- Submodule: `attune_rag.corpus.attune_help.AttuneHelpCorpus`
- Submodule: `attune_rag.corpus.help_adapter.HelpCorpusAdapter`
- Submodule (after rename): `attune_rag.editor` plus its non-underscore
  members
- Resource: `attune_rag/editor/template_schema.json`

### Module-by-module verdict

Every Python module under `src/attune_rag/`. PUBLIC = frozen surface,
covered by the snapshot test, governed by the deprecation policy.
INTERNAL = may change in any release without notice. MIXED = the
module contains both; the design phase splits the public surface
from the rest via `__all__`.

| Module | Verdict | Justification |
|---|---|---|
| [`__init__.py`](src/attune_rag/__init__.py) | PUBLIC | The root re-export hub. Already has `__all__`. Phase 3 freezes the current list verbatim — no additions, no removals. |
| [`pipeline.py`](src/attune_rag/pipeline.py) | PUBLIC | `RagPipeline`, `RagResult`. Top-level re-export; load-bearing for all three downstreams. |
| [`provenance.py`](src/attune_rag/provenance.py) | PUBLIC | `CitationRecord`, `CitedSource`, `ClaimCitation`, `format_citations_markdown`, `format_claim_citations_markdown`. Re-exported at root. `build_citation_record` is the constructor used by `pipeline.py` — INTERNAL. |
| [`retrieval.py`](src/attune_rag/retrieval.py) | PUBLIC | `KeywordRetriever`, `RetrievalHit`, `RetrieverProtocol`. Re-exported at root. `_stem`, `_tokenize` — INTERNAL (private by convention). |
| [`prompts.py`](src/attune_rag/prompts.py) | MIXED | `build_augmented_prompt`, `PROMPT_VARIANTS` PUBLIC (re-exported at root). `join_context`, `join_context_numbered`, `_join` INTERNAL — not in root `__all__`, used only by `pipeline.py` and `eval/`. |
| [`expander.py`](src/attune_rag/expander.py) | PUBLIC | `QueryExpander`. Re-exported at root. Already has a local `__all__`. |
| [`reranker.py`](src/attune_rag/reranker.py) | PUBLIC | `LLMReranker`. Re-exported at root. |
| [`corpus/__init__.py`](src/attune_rag/corpus/__init__.py) | PUBLIC | Already exports `CorpusProtocol`, `RetrievalEntry`, `DirectoryCorpus`, `AliasInfo`, `DuplicateAliasError`. Used qualified by attune-help. |
| [`corpus/base.py`](src/attune_rag/corpus/base.py) | PUBLIC | Source of `CorpusProtocol`, `RetrievalEntry`, `AliasInfo`, `DuplicateAliasError`. |
| [`corpus/directory.py`](src/attune_rag/corpus/directory.py) | PUBLIC | Source of `DirectoryCorpus`. |
| [`corpus/attune_help.py`](src/attune_rag/corpus/attune_help.py) | PUBLIC | Source of `AttuneHelpCorpus`. attune-help imports `attune_rag.corpus.attune_help.AttuneHelpCorpus` qualified — must be PUBLIC. **Action:** add `AttuneHelpCorpus` to the root `__all__` in Phase 3 (currently only documented in the docstring). |
| [`corpus/help_adapter.py`](src/attune_rag/corpus/help_adapter.py) | PUBLIC | `HelpCorpusAdapter` Protocol. attune-help references it by qualified path. |
| [`providers/__init__.py`](src/attune_rag/providers/__init__.py) | PUBLIC | `LLMProvider`, `get_provider`, `list_available`. Has explicit `__all__`. Used by `cli.py` and by callers wanting provider feature detection. |
| [`providers/base.py`](src/attune_rag/providers/base.py) | PUBLIC | Source of `LLMProvider` Protocol. |
| [`providers/claude.py`](src/attune_rag/providers/claude.py) | INTERNAL | Instantiated only via `get_provider("claude", ...)`. Lazy-imported. |
| [`providers/gemini.py`](src/attune_rag/providers/gemini.py) | INTERNAL | Same as claude.py. |
| [`editor/__init__.py`](src/attune_rag/editor/__init__.py) | PUBLIC | attune-gui consumes the whole submodule. Already has a thorough `__all__`. Phase 3 ratifies it. |
| [`editor/_rename.py`](src/attune_rag/editor/_rename.py) | PUBLIC (rename → `rename.py`) | attune-gui imports `attune_rag.editor._rename` directly. Phase 3 renames the file to `rename.py`, lands a one-release re-export alias under the old name, and updates attune-gui in lockstep. |
| [`editor/_schema.py`](src/attune_rag/editor/_schema.py) | PUBLIC (rename → `schema.py`) | Same as `_rename.py`. |
| [`editor/_lint.py`](src/attune_rag/editor/_lint.py) | PUBLIC (rename → `lint.py`) | Reachable via `editor.__init__` (`lint_template`, `Diagnostic`, `Severity`). Renamed for consistency. |
| [`editor/_autocomplete.py`](src/attune_rag/editor/_autocomplete.py) | PUBLIC (rename → `autocomplete.py`) | Same — reachable via `editor.__init__`. |
| [`editor/_references.py`](src/attune_rag/editor/_references.py) | PUBLIC (rename → `references.py`) | Same. |
| [`editor/template_schema.json`](src/attune_rag/editor/template_schema.json) | PUBLIC RESOURCE | attune-help reads/mirrors this file. The file path and JSON-schema shape are both contract. |
| [`cli.py`](src/attune_rag/cli.py) | INTERNAL | Reachable via `attune-rag` console script. No programmatic import contract. |
| [`benchmark.py`](src/attune_rag/benchmark.py) | INTERNAL | `python -m attune_rag.benchmark` only. No programmatic import contract. |
| [`eval/__init__.py`](src/attune_rag/eval/__init__.py) | INTERNAL | `FaithfulnessJudge` is an internal experiment harness. Not imported by attune-gui / attune-help / attune-author. May graduate to PUBLIC in a future release once the surface stabilizes. |
| [`eval/faithfulness.py`](src/attune_rag/eval/faithfulness.py) | INTERNAL | Same. |
| [`eval/bench_prompts.py`](src/attune_rag/eval/bench_prompts.py) | INTERNAL | Experiment script. |
| [`dashboard/__init__.py`](src/attune_rag/dashboard/__init__.py) | INTERNAL | Locked in dashboard-v0.2.0 spec: "No public Python API. CLI is the only documented entry point." |
| [`dashboard/render.py`](src/attune_rag/dashboard/render.py) | INTERNAL | Same. |
| [`dashboard/refresh.py`](src/attune_rag/dashboard/refresh.py) | INTERNAL | Same. |
| [`dashboard/show.py`](src/attune_rag/dashboard/show.py) | INTERNAL | Same. |

### User stories

1. *As a downstream maintainer (attune-gui, attune-help, attune-author)*,
   I want a single, documented list of "what attune-rag promises to keep
   stable across minor versions" — so I can pin against the package
   without grep'ing the source.
2. *As an attune-rag maintainer*, I want a snapshot test that fails when
   the public surface changes — so accidental removals are caught in CI,
   not by a downstream breaking on update.
3. *As an attune-rag contributor*, I want a written deprecation policy —
   so I know the procedure for retiring a public symbol without
   surprising downstreams.
4. *As an attune-gui developer*, I want `attune_rag.editor.rename` (and
   friends) to be importable without the friendly-guard 503 — so the
   template editor stops being "experimental dependency" and becomes
   first-class.

### Edge cases & open questions

| Question / Edge case | Resolution |
|---|---|
| `_rename` / `_schema` / `_lint` / `_autocomplete` / `_references` rename — backward compat? | Ship a one-release shim: keep the underscore-prefixed module names as thin `from .new_name import *` re-exports for 0.2.x. Drop them in 0.3.0. Documented in the deprecation policy. |
| Should `AttuneHelpCorpus` be re-exported at the top level? | Yes. attune-help imports it qualified today, but adding it to the root makes the constraint visible in the surface test. No call-site change required. |
| What about `build_citation_record`? It's used in `pipeline.py` and reads like a public constructor. | INTERNAL for 0.2.0. We don't promote symbols during a freeze. If a downstream materializes a need, promote in 0.3.0 under the policy. |
| `RetrieverProtocol` — should this be PUBLIC even though no current downstream subclasses it? | Yes. It's the documented extension point for custom retrievers. It's already in the root `__all__`. Freeze ratifies the documented intent. |
| `eval/` — INTERNAL but the module docstring describes it as a "harness". Risk of premature freeze? | Keep INTERNAL. Re-evaluate in Phase 5 once the harness stabilizes. The classifier flip is the natural moment. |
| Snapshot test: lock symbols only, or also signatures? | Symbols only for 0.2.0 — `__all__` membership per PUBLIC module. Signature locking is a Phase 5 follow-on (needs `inspect.signature` introspection and a careful policy on what counts as a "compatible" change). |
| What if Phase 2 reshapes a module we'd freeze? | Re-run the audit before tagging 0.2.0. Phase 3 spec is non-binding until the snapshot is committed; the audit is the contract. |
| Do we need a `py.typed` marker? | Out of scope for Phase 3. Track as 0.3.0 candidate. |
| Where does the deprecation policy live? | `docs/POLICY.md`. Linked from README's new "Public API" section. |

### Affected layers

- [x] attune-rag — adds `__all__` (no new ones beyond what's needed; existing lists are ratified), `tests/unit/test_api_surface.py`, `docs/POLICY.md`, README "Public API" section, editor submodule renames with one-release shims.
- [x] attune-gui — updates `attune_rag.editor._rename` / `_schema` imports to the new names; removes the unpublished-module guard in `_editor_dep.py` once 0.2.0 ships. Tracked as a follow-up PR in the attune-gui repo, gated on 0.2.0 release.
- [ ] attune-help — no source change. Audit confirms its current imports are covered by the proposed PUBLIC surface.
- [ ] attune-author — no source change. Same.
