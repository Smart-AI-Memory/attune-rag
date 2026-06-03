# Tasks: alias-overlap-remediation

> **Status:** approved 2026-06-03 — executable. See
> [README](README.md). Tasks T1–T3 are the attune-rag engine PR;
> T4–T5 are downstream consumer-note PRs in their own repos.

## T1 — Build-time single-token-alias warning (attune-rag)

**Files:** `src/attune_rag/corpus/directory.py` (detection in
`__init__`), reuse `_tokenize` from `attune_rag.retrieval`.

- Read effective floor from `KeywordRetriever.MIN_ALIAS_OVERLAP`;
  short-circuit if `<= 1`.
- After the `extra_aliases` merge, count `alias-degraded` entries
  per [design.md](design.md) §2.
- Emit one `logging.warning` when the §3 threshold is crossed.
- Add `warn_alias_overlap: bool = True` constructor flag (or
  module-level `_WARN_ALIAS_OVERLAP_DEFAULT` if the surface test
  pins signatures — verify first per D2).
- **Verify:** `test_api_surface.py` snapshot unchanged.

## T2 — Tests (attune-rag)

**Files:** `tests/unit/test_corpus_alias_warning.py` (new).

- single-token-only corpus → exactly one warning (caplog),
  message names count + floor.
- multi-token corpus → silent.
- `MIN_ALIAS_OVERLAP = 1` (subclass) → silent.
- `warn_alias_overlap=False` → silent.
- **Strict-dominance:** bundled golden snapshot byte-identical +
  bundled corpus silent (proves R4 + that the tuned corpus does
  not trip the warning).

## T3 — Guide update (attune-rag)

**Files:** `docs/USER_CORPUS_GUIDE.md` §4.2.

- Describe the warning: when it fires, the logger name to silence
  it, and the `warn_alias_overlap=False` flag.
- Reiterate the subclass override as the interim ergonomic path
  and note the v1.0.0 kwarg is backlogged.

## T4 — Consumer note: attune-ai (separate repo PR)

**Files:** attune-ai `CHANGELOG.md` / release notes.

- One-line note: default `RagPipeline()` consumers
  (`memory/personal.py`, `workflows/rag_code_gen.py`,
  `mcp/workflow_handlers.py`) run over un-tuned corpora; if
  personal-memory recall quality looks off, see attune-rag
  `USER_CORPUS_GUIDE` §4.2.
- Non-blocking on the engine PR.

## T5 — Consumer note: attune-author (separate repo PR)

**Files:** attune-author `CHANGELOG.md` / release notes.

- One-line note: `rag_hook.py` +
  `orchestration/commands/rag.py` build pipelines over arbitrary
  project-path corpora; same §4.2 pointer.
- Non-blocking on the engine PR.

## Sequencing

T1 → T2 (same PR; tests gate the change). T3 in the same PR (doc
ships with behavior). T4/T5 are independent follow-ups that can
land any time after T1 merges.

## v1.0.0 backlog (out of scope here)

- Public `min_alias_overlap=` kwarg threaded
  `RagPipeline -> KeywordRetriever`, scoped against the surface
  budget in the v1.0.0 pass (per D1).
