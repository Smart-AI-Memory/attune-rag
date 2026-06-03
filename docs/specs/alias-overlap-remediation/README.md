# Spec: alias-overlap-remediation (attune-rag)

> **Status:** **approved 2026-06-03** — scope ratified with
> Patrick. Decision D1: ship a build-time single-token-alias
> **warning** (zero public-surface cost) + consumer-impact docs;
> **defer** the public `min_alias_overlap` kwarg to v1.0.0
> scoping. The subclass override path remains the documented
> escape hatch in the interim.

## Purpose

Remediate the silent retrieval regression introduced by
[`KeywordRetriever.MIN_ALIAS_OVERLAP = 2`](../../../src/attune_rag/retrieval.py)
(shipped in 0.1.22) for **user-supplied corpora that have not been
alias-tuned**.

The default `MIN_ALIAS_OVERLAP = 2` requires at least two distinct
query tokens to overlap an entry's alias-token union before any
alias hit is credited. The bundled `AttuneHelpCorpus` was
backfilled with 180+ multi-token aliases (the 0.1.23
alias-expansion sweep) specifically to satisfy this floor. **User
corpora authored with single-token aliases (`["security", "api",
"mcp"]`) lose all alias signal under the default — silently.**

This is the regression the Phase 0.1 family-plan diagnosis
flagged (window 2026-05-21 to present). It is a design choice
with a thinly-broadcast mitigation, not a code bug. The harm is
the **silence**: an un-tuned corpus degrades with no diagnostic.

## What this spec does NOT do

- **No retrieval behavior change.** The warning is pure
  observability; it does not touch `_score_entry` or any ranking
  logic. The UX-regression guard is therefore trivially satisfied
  — the change cannot regress retrieval because it does not alter
  scoring.
- **No default flip.** `MIN_ALIAS_OVERLAP` stays `2`. Flipping to
  `1` would regress the bundled corpus that was tuned for `2`.
- **No public-surface growth.** No new symbol against the
  5-symbol v1.0.0 budget (4/5 used). See
  [decisions.md](decisions.md) D1.

## Documents

- [requirements.md](requirements.md) — what must be true when done
- [design.md](design.md) — warning mechanism + detection logic
- [decisions.md](decisions.md) — ratified scope decision (D1)
- [tasks.md](tasks.md) — executable task breakdown
- [risks.md](risks.md) — risk register

## Cross-repo footprint

- **attune-rag** (this repo): the build-time warning + tests +
  `USER_CORPUS_GUIDE` update.
- **attune-ai**: consumer-impact note — `personal.py`,
  `rag_code_gen.py`, `workflow_handlers.py` construct the default
  `RagPipeline()` over un-tuned corpora.
- **attune-author**: consumer-impact note — `rag_hook.py`,
  `orchestration/commands/rag.py` construct pipelines over
  arbitrary project-path corpora.

The consumer-doc tasks ship as separate PRs in their own repos;
they are tracked here for completeness but do not block the
attune-rag engine fix.
