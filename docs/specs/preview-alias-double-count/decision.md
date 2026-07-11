# Preview alias double-count — decision record

**Status:** implemented (fast-tracked measurement-driven fix; this doc is the
spec-of-record). Follow-up to the alias-expansion-sweep promotion
([attune-help#9](https://github.com/Smart-AI-Memory/attune-help/issues/9),
attune-rag #193).

## Problem

`KeywordRetriever`'s content signal tokenizes
`entry.content[:CONTENT_PREVIEW_CHARS]` (500 chars) — and `entry.content` is
the raw template file, frontmatter included. Two consequences surfaced when
the sweep aliases moved into attune-help frontmatter:

1. **Displacement.** A large `aliases:` block fills the preview window and
   pushes the actual body out of the content signal. On the 650-query
   attune-help fixture benchmark this cost 1.7pp overall P@1
   (75.1% → 73.4%), leaving ~0.4pp headroom over the 73% CI gate.
2. **Double-counting.** Alias tokens in the preview earn content-weight
   credit *in addition to* the dedicated aliases field — and without the
   `MIN_ALIAS_OVERLAP=2` gating that field applies. This was invisible while
   the sweep aliases lived in `aliases_override.json` (the override path
   never touched content), and became load-bearing the moment they were
   promoted.

## Variants measured

All rows: attune-help 0.13.0 corpus, attune-rag @ #193.

| variant | golden base P@1/R@3 | golden para P@1/R@3 | help fixtures P@1/R@3 | release gate |
|---|---|---|---|---|
| no change | 1.0000 / 1.0000 | 0.9250 / 1.0000 | 73.4% / 83.2% | pass (fixture margin 0.4pp) |
| strip whole frontmatter | 0.9500 / 1.0000 | 0.8750 / 0.9875 | 74.8% / — | **FAIL** (P@1 0.95 < 0.975) |
| strip aliases block only | 0.9750 / 1.0000 | 0.8875 / 0.9875 | 75.5% / 83.8% | pass (golden at exact floor) |
| strip aliases + `ALIASES_WEIGHT` 2.0 | **1.0000 / 1.0000** | 0.9000 / 0.9875 | **75.1% / 83.7%** | **pass** |
| strip aliases + `ALIASES_WEIGHT` 2.5 | 1.0000 / 1.0000 | 0.9000 / 0.9875 | 74.5% / 83.7% | pass |

Key readings:

- Whole-frontmatter strip fails: `name:`/`tags:` tokens are real preview
  signal (tags are scored nowhere else).
- Aliases-only strip regresses exactly the alias-targeted golden queries
  ("publish to PyPI", "readme lies", "before I ship") — the double-count was
  silently functioning as extra alias weight.
- `ALIASES_WEIGHT` 1.5 → 2.0 replaces that accidental weight with an explicit
  one; 2.5 overshoots (fixture false positives).

## Decision

`_ALIASES_BLOCK_RE` drops the `aliases:` block (inline flow or block style)
from the preview slice, and `ALIASES_WEIGHT` moves 1.5 → 2.0. Net effect vs
before the fix: golden gates unchanged at full margin, help fixture P@1 back
to its pre-promotion 75.1% (2.1pp gate margin), and template frontmatter
size can no longer distort content ranking — future alias additions affect
only the aliases field, gated by `MIN_ALIAS_OVERLAP`.

Golden paraphrased P@1 reads 0.9250 → 0.9000 (2 of 80 queries; both
XFAIL-tolerated, R@3 floor 70% vs measured 98.75%): those two rode the
ungated single-token alias leak the fix removes.

## Rollback

Revert the commit. Both knobs are class attributes; a subclass can also
restore pre-fix behavior locally (`ALIASES_WEIGHT = 1.5` and overriding
`_entry_field_tokens`). Golden report + `_R1_REFERENCE` + pinned test
numbers must move with any revert (they were regenerated in the same
commit — the diff is the contract).
