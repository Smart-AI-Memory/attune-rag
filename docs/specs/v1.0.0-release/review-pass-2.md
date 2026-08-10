# M1.2 second deep-review pass — during the M4 watch

**Run:** 2026-08-10 (day 0 of the watch), ruled at the post-release
retro. Complements the delta-scoped first pass (PR #204); together
they satisfy the "one mid-phase, one at end" cadence.

## Scope actually covered

**Full reads (line-by-line):**
- `editor/rename.py` apply path — the three-phase rollback
  (`_apply_moves` → `_stage_edits` → `_commit_staged_edits`): layering
  verified correct (each phase's except unwinds everything prior and
  re-raises); move+edit interaction safe (staging reads post-move
  content, so restore-then-unmove sequences correctly);
  `_ensure_parents`/`_undo_created_dirs` ordering verified
  (deepest-first rmdir).
- `pipeline.py` (whole module) — confidence formula's `or 1.0` guard
  vs the `min_score=0` abstention-off case, reranker over-fetch
  (`k × candidate_multiplier` then `[:k]`), the four-way native-
  citations behavior matrix, cache-split threshold. No defects.
- (First pass, PR #204:) `retrieval.py`, `expander.py`,
  `providers/claude.py`, `corpus/directory.py`, `editor/schema.py`.

**Risk-pattern sweep** (remaining public modules — `corpus/base`,
`providers/base`, `providers/gemini`, `cli`, `editor/{lint,references,
autocomplete}`, `provenance`, `reranker`, `embedding`, `transformer`,
`measure_corpus`): mutable default args (0 hits), unannotated broad
excepts (0 — the one hit is `_stage`'s cleanup-and-reraise), path ops
on user input (1 — `cli.py:236`, an output destination the user names
themselves), provider error-path shape (clean: ImportError→RuntimeError
with install hint, NotImplementedError default).

## Findings

**No blocking defects.** Three observations, none release-affecting:

| # | Observation | Disposition |
|---|---|---|
| 1 | `_ALIASES_BLOCK_RE.sub(count=1)` can strip a body-level `aliases:` line in a template with NO frontmatter aliases block — preview-only, sub-scoring impact | 1.0.x backlog (from pass 1) |
| 2 | `KeywordRetriever.__init__` shadows documented class constant `MIN_SCORE` via instance attribute | stylistic; leave (from pass 1) |
| 3 | `rename._refresh_corpus` is a no-op on wrapper corpora (`AttuneHelpCorpus` has no `_loaded`) — benign because `_corpus_root` rejects wrappers before any mutation; worth an inline comment if the root resolution ever learns to unwrap | 1.0.x backlog (comment-only) |

## Verdict

The public surface is consistent with the Production/Stable claim.
The known residual risks remain the two promoted specs from the D9
triage (Q1+T2 rollback refactor + disk-fault simulation; T1+T3 live
SDK cassettes) — neither is new information.
