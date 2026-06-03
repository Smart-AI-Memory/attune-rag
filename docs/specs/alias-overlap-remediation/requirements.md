# Requirements: alias-overlap-remediation

> **Status:** approved 2026-06-03. See [README](README.md).

## Problem

`MIN_ALIAS_OVERLAP = 2` (default since 0.1.22) silently zeroes
alias contribution for any corpus entry whose aliases are
single-token. Users who author single-word aliases — the obvious,
intuitive shape — get degraded retrieval with **no signal** that
the knob is the cause. The mitigation (`USER_CORPUS_GUIDE` §4.2,
subclass + `MIN_ALIAS_OVERLAP = 1`) exists but is undiscoverable
unless the user already suspects the knob.

## Functional requirements

- **R1 — Build-time detection.** When a corpus is constructed with
  the effective `MIN_ALIAS_OVERLAP >= 2`, detect entries whose
  alias set tokenizes to single-token aliases only (no entry has a
  multi-token alias that could ever satisfy the floor).
- **R2 — One-shot warning.** Emit a single `logging.warning` at
  corpus-build time when the count of affected entries crosses a
  threshold (see [design.md](design.md) §3). The message names the
  count, the active floor, and points at the override path. It
  fires **once per corpus construction**, not per entry.
- **R3 — Opt-out.** The warning is suppressible via standard
  Python logging configuration (logger name documented) and via an
  explicit `warn_alias_overlap=False` constructor flag — a private
  behavior toggle, **not** a public-surface symbol (see D1).
- **R4 — No behavior change.** Retrieval scoring, ranking, and all
  golden/perf numbers are byte-identical pre/post. The warning is
  observability only.
- **R5 — Consumer docs.** `USER_CORPUS_GUIDE` §4.2 is updated to
  describe the warning and the subclass override. Downstream
  consumers (attune-ai, attune-author) get a one-line release-note
  pointer (tracked, separate PRs).

## Non-functional requirements

- **N1 — Zero public surface.** No new entry in any `__all__`; the
  `test_api_surface.py` snapshot is unchanged. Preserves the last
  v1.0.0 budget slot.
- **N2 — Freeze-legal.** Internal logging + a private kwarg; no
  public symbol. Ships under the post-0.2.0 cadence without
  resetting any clock.
- **N3 — Cheap detection.** The single-token scan reuses the
  existing `_tokenize` and runs once at build, O(total aliases).
  No per-query cost.

## Done when

- The warning fires on an un-tuned corpus and is silent on the
  bundled `AttuneHelpCorpus` (which is multi-token-tuned).
- `warn_alias_overlap=False` and logger-level config both suppress
  it.
- Golden + perf snapshots unchanged (proves R4).
- `USER_CORPUS_GUIDE` §4.2 updated.
- attune-ai + attune-author consumer-note PRs opened (may merge
  after the engine PR).
