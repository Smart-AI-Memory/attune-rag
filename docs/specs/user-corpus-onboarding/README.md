# Spec: user-corpus-onboarding (attune-rag)

> **Status:** scaffolding — not executable; promotes to scoped via
> `/spec` pass when Phase 5 opens. No `tasks.md` until then.
> **Workstream:** Phase 5 of the [v1.0 roadmap](../ROADMAP-v1.md#phase-5--100-release).
> **Activation gate:** Phase 4 W4.2 (cadence-clean) + 0.2.0 cut +
> 7-day no-hotfix watch + Phase 5 opens.
> **Freeze posture:** spec scaffold is docs-only and freeze-compliant.
> Implementation (harness code + `DirectoryCorpus` override-from-file
> kwarg) is `### Added` and cannot land before the 0.2.0 cut.

## Purpose

Ship the **"this works for your corpus"** story for v1.0.0.

attune-rag's current public surface (`DirectoryCorpus`,
`KeywordRetriever`, `RagPipeline`, etc.) lets a user point at their
own markdown directory and get retrieval out. But there's no
documented path for *measuring quality* on that corpus, no first-class
override mechanism for arbitrary corpora (it's currently second-class
inside `AttuneHelpCorpus`), and no guide for the authoring discipline
that produced the bundled corpus's 100% / 100% / 100% baseline and
paraphrased R@3 = 100%.

This spec ships the missing layer: a **quality harness**, a
**first-class override-from-file mechanism**, and a **"your own
corpus" guide**. Together they earn the v1.0.0 Production/Stable
claim for the [framework framing](../v1.0.0-release/design.md#phase-5-scope-decided-2026-05-21)
— not just for the bundled `AttuneHelpCorpus` exemplar.

## Why v1.0.0, not v1.1.0

The [Phase 5 scope decision](../v1.0.0-release/design.md#phase-5-scope-decided-2026-05-21)
committed v1.0.0 to the framework framing — *"deterministic retrieval
framework for your own markdown corpus, with attune-help as the
bundled exemplar."* Under that framing:

- **Shipping v1.0.0 without the quality harness is inconsistent.**
  Calling the package "Production/Stable" while users can't measure
  quality on their own corpus is a framing mismatch — the whole
  point of "Production/Stable" is that downstream users can rely on
  it, and reliance presupposes measurement.
- **Shipping v1.0.0 without the first-class override mechanism is
  asymmetric.** `AttuneHelpCorpus` users have a clean override path
  (`aliases_override.json` loaded at construction); arbitrary
  `DirectoryCorpus` users have to write their own JSON-loading code.
  At v1.0.0 SemVer-binding time, that asymmetry becomes a documented
  bug, not a known-rough-edge.
- **The guide is what makes the framework discoverable.** The
  authoring discipline that produced the 100% baseline (frontmatter
  aliases, override patterns, `MIN_ALIAS_OVERLAP` knob, stemmer
  gotchas like `bites → bit`) is currently implicit — readable only
  by reverse-engineering the alias-expansion-sweep PRs. v1.0.0 makes
  it explicit.

## What ships

### 1. Quality harness (`### Added`)

A script + Python API that takes:

- A corpus directory (any `DirectoryCorpus`-compatible markdown root)
- A baseline queries file (same shape as
  `tests/golden/queries.yaml`)
- An optional paraphrased queries file (same shape as
  `tests/golden/queries_paraphrased.yaml`)

…and produces:

- Per-query top-k retrieval results
- Aggregate P@1, R@3, per-difficulty breakdown
- A markdown report mirroring `docs/specs/release-quality-baseline/baseline-1.md`'s shape

The harness reuses the existing
[`scripts/measure_baseline_variance.py`](../../../scripts/measure_baseline_variance.py)
machinery where possible — it's not net-new measurement code, it's
generalization of the existing per-corpus measurement code.

### 2. First-class override-from-file for `DirectoryCorpus` (`### Added`)

Today: `DirectoryCorpus(root)` plus `extra_aliases` kwarg as a
Python dict. Convenient only for callers who already load the JSON
themselves.

After: `DirectoryCorpus(root, extra_aliases_file="path/to/aliases.json")`
or similar. Mirrors the convenience that `AttuneHelpCorpus` already
has internally. Five lines of code; large discoverability win.

Design decision deferred to scoping: exact kwarg name, schema of
the file (mirror `summaries_override.json` shape?), error semantics
on malformed file.

### 3. "Your own corpus" guide (docs)

A new top-level document — likely
[`docs/USER_CORPUS_GUIDE.md`](../../USER_CORPUS_GUIDE.md) — that
walks through:

- **Corpus structure.** Directory layout, file naming, frontmatter
  schema (link to `editor/template_schema.json`).
- **Frontmatter aliases.** What they are, how to author them,
  multi-token alias intent (the `MIN_ALIAS_OVERLAP = 2` consequence).
- **The override file pattern.** When to use it vs frontmatter; the
  override-then-promote workflow.
- **The `MIN_ALIAS_OVERLAP` knob.** Why default `2`, when to flip
  to `1`, what the trade-off measurement looks like.
- **Stemmer gotchas.** The `bites → bit` discovery from the
  alias-expansion sweep; how to use the `_tokenize()` helper to
  validate alias candidates before authoring.
- **Quality measurement.** Pointer to the harness; the
  baseline-paraphrased structure; how to author your own
  paraphrased set; how to interpret the watermark.

Reuses the alias-expansion-sweep's lessons (`feedback_alias_stem_validation.md`,
the strict-dominance discipline) as documentation source.

## What's *not* in scope

- **Embedding retriever for arbitrary corpora.** Tracked by the
  [`embedding-retriever`](../embedding-retriever/#scope-of-the-defer)
  spec — scope-specifically deferred there. If the harness shows
  user corpora consistently fail to close their paraphrased gaps
  via frontmatter + override (i.e. the discipline we shipped for
  attune-help doesn't transfer), that spec's revival case re-opens.
  This spec assumes the discipline transfers; if measurement
  contradicts that assumption, the embedding-retriever spec is the
  follow-up.
- **A web UI for the harness.** CLI + Python API only. Web UI is
  attune-gui's territory, not attune-rag's. If attune-gui wants to
  surface the harness in its dashboard, it consumes the Python API.
- **Automatic alias authoring (LLM-driven).** The harness measures;
  the user authors. Inferring aliases from a corpus + query set is
  interesting future work but lives in a separate spec — needs its
  own evaluation against the strict-dominance discipline before it
  can ship.
- **A multi-corpus story.** The harness measures one corpus at a
  time. If users want to compare two corpora, they run it twice.
  Multi-corpus dashboards are a v1.1.0+ feature if they're a
  feature at all.
- **The `QueryExpander` re-framing PR.** Tracked separately — see
  the v1.0.0 roadmap planning item #2. The QueryExpander is one
  authoring lever this guide will *mention*, but its re-framing
  ships as its own PR ahead of the guide so the guide can cite the
  refreshed docstring/README.

## Layout

- [`design.md`](design.md) — proposed mechanism for each of the
  three pieces above, with candidate options and alternatives
  considered. Resolved at the `/spec` pass.
- [`requirements.md`](requirements.md) — invariants the
  implementation must satisfy (strict-dominance, freeze-window
  compatibility shape, public-surface size budget).
- [`risks.md`](risks.md) — scope creep, harness-maintenance burden,
  the "if user corpora don't transfer the discipline" failure mode.

Promoted to executable when the `/spec` pass adds:

- [`tasks.md`](tasks.md) — M1 harness, M2 override-from-file kwarg,
  M3 guide, M4 documentation polish. Currently scaffold.

## Provenance

Scoped during the architectural conversation on 2026-05-21 that
locked the [Phase 5 scope decision](../v1.0.0-release/design.md#phase-5-scope-decided-2026-05-21).
The work shape was identified as "the largest single architectural
item for v1.0.0 — load-bearing for the framework framing."

The harness shape draws on the existing
[`scripts/measure_baseline_variance.py`](../../../scripts/measure_baseline_variance.py)
and the
[`docs/specs/release-quality-baseline/baseline-1.md`](../release-quality-baseline/baseline-1.md)
report format. The override-from-file mechanism is the generalization
of the [`aliases_override.json`](../../../src/attune_rag/corpus/aliases_override.json)
pattern shipped in 0.1.23 (per
[`docs/specs/alias-expansion-sweep/`](../alias-expansion-sweep/)).
The guide synthesizes the discipline that produced the sweep — see
[`feedback_alias_stem_validation.md`](../../../.claude/projects/-Users-patrickroebuck-attune-rag/memory/feedback_alias_stem_validation.md)
and the M12 near-regression lesson in the sweep's project memory.
