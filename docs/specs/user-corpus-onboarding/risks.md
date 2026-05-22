# Spec: user-corpus-onboarding — risks

> **Status:** scaffolding — not executable. Risk register, not a
> mitigation plan. Mitigations are implementation-time work.

## 1. The discipline doesn't transfer

**Risk:** The alias-expansion sweep's discipline (multi-token
aliases, override mechanism, MIN_ALIAS_OVERLAP=2) produced 100%
paraphrased R@3 on the **attune-help corpus**. That corpus is a
specific shape — short markdown files (200-500 chars typical), with
hand-curated frontmatter, a small taxonomy (concepts/, tasks/,
quickstarts/, references/), tens of templates per category. **The
discipline may not transfer cleanly to arbitrary user corpora.**

Failure modes:

- **Long-form corpora** (per-file >5KB, e.g. RFC documents,
  research papers). Token frequency distributions look very
  different; alias-overlap becomes too easy to satisfy on noise.
- **Very large corpora** (>1000 files). Cross-template alias
  collisions become statistically inevitable; the `MIN_ALIAS_OVERLAP`
  knob may need to be higher.
- **Low-curation corpora** (e.g. auto-generated docs). The
  frontmatter-alias path assumes someone is *willing to author
  aliases*. If that's not realistic for the user, the harness will
  measure poor quality and the override path will be too much work
  to close.

**Posture:** **document the assumption, measure the failure mode.**

The user-corpus guide states explicitly that the discipline was
validated on the attune-help corpus (short, curated, small) and
points out the corpus shapes most likely to need different treatment.
The harness *measures* whether the discipline worked — if a user
runs the harness and sees paraphrased R@3 < 0.70 after a reasonable
authoring pass, that's evidence for them that this corpus shape may
need the embedding-retriever path (see
[`embedding-retriever`](../embedding-retriever/#scope-of-the-defer)
scope-specific defer).

This risk **does not block v1.0.0**. It reframes v1.0.0's claim:
"production-stable for corpora where the documented discipline
applies; the harness tells you whether it applies to yours." That's
honest framing for the framework framing.

**Trigger for re-scoping:** if telemetry-style observations from
early v1.0.0 users show >50% of user-corpora failing to hit
paraphrased R@3 ≥ 0.85 after onboarding, revisit the embedding-retriever
spec's revival case. (Telemetry is v1.1.0+ per Phase 5 scope; this
data point becomes available later, not at v1.0.0 ship.)

## 2. Harness-maintenance burden

**Risk:** The harness is one more script with its own surface area
and own bug class. Once shipped publicly, breaking its output format
or behavior is a SemVer event under the v1.x deprecation policy. The
implementation must hit a maintenance budget similar to
`scripts/measure_perf_baseline.py` (single file, single owner,
bounded scope).

**Posture:** accept; bound by simplicity.

- Single module (`attune_rag/measure_corpus.py`); one CLI entry
  point; one Python API entry point.
- Output format is markdown + JSON. Both are versioned
  (`report_version: 1` in the JSON) so format changes don't break
  downstream parsers.
- The bundled-corpus reproduction property (R1 in requirements) is
  the regression net for the harness's own correctness — any
  refactor that moves the bundled numbers fails its own test.
- No web service, no daemon, no plugin system. The harness runs and
  exits.

If maintenance burden grows past one-file scope, that's evidence we
over-scoped; cut features (e.g. the JSON output, the watermark flag,
the per-difficulty breakdown) until it fits.

## 3. The override-from-file kwarg is misused

**Risk:** The kwarg makes `aliases_override.json` a public-facing
pattern. Users may:

- Author overrides that break their own corpus's baseline (e.g.
  add an alias that collides across templates).
- Ship overrides without measuring the effect (skip the
  strict-dominance discipline).
- Mistake the override mechanism for the *only* customization path
  and never author frontmatter aliases (the strategic-vs-tactical
  distinction gets lost).

**Posture:** mitigate via documentation + cross-template error.

- The cross-template `DuplicateAliasError` (which the existing
  `DirectoryCorpus._build` already raises) catches the
  collision-across-templates case at construction time. Loud
  failure, not silent miss.
- The guide explicitly documents the override-then-promote
  workflow: overrides are tactical, frontmatter is strategic. The
  override mechanism in attune-rag's own corpus is cited as an
  example of *temporary* use ahead of upstream promotion in
  attune-help.
- The harness reports a "your overrides" section listing what was
  loaded from the file, so the user can see what's actually
  affecting their measurements.

The guide section on the override file pattern is *prescriptive*
about discipline: "if you add an override and don't re-run the
harness, you've shipped an untested behavior change."

## 4. Calendar slip from harness scope creep

**Risk:** "A quality harness that works for arbitrary corpora" is a
seductive scope. Adding "compare two corpora side-by-side", "track
quality over time across commits", "auto-tune MIN_ALIAS_OVERLAP",
"suggest aliases from miss-set" — all genuinely useful, all out of
scope for v1.0.0.

**Posture:** scope hard at the spec pass.

- The spec's "What's not in scope" section in
  [README.md](README.md) is the contract. The `/spec` pass that
  promotes this to executable does not loosen those bounds.
- M1 ships the minimum harness: single-corpus, single-run, two
  output formats (md + json), one watermark flag. Period.
- Additions ship as separate specs in v1.1.0+. Each addition
  re-runs the v1.0.0 framework-framing test ("does this make the
  framing more or less consistent?").

**Trigger for re-scoping:** if M1 estimate at scoping exceeds 3
weeks, cut features until it fits.

## 5. The harness becomes a substitute for the alias-expansion sweep
discipline

**Risk:** Users (or future-maintainers) treat the harness as the
*answer* — "run the harness, hit the watermark, ship it" — instead
of as a *measurement tool* embedded in a discipline.

The actual discipline is:
1. Author frontmatter aliases (or overrides) following the
   multi-token guidance.
2. Run the harness.
3. Inspect the miss-set.
4. Author more aliases (or upstream-promote existing ones).
5. Re-run.
6. Strict-dominance check: aggregate must not regress while you
   improve the miss-set.

If users only do (1) and (2), they get a number; they don't get
quality. The harness alone is not the win.

**Posture:** mitigate via guide framing + harness output.

- The guide section on "Quality measurement" frames the harness as
  the third step in a discipline, not the only step.
- The harness output's footer points back to the guide and lists
  the discipline steps explicitly ("if your R@3 is below your
  watermark, here are the next moves...").
- The bundled `attune-help` corpus is cited as a worked example:
  "we hit 100% paraphrased R@3 by running this loop 13 times — see
  `docs/specs/alias-expansion-sweep/`."

This is a documentation risk, not a code risk. Mitigation is
discipline-of-framing in the writing.

## 6. The kwarg's symmetry with `summaries` becomes a follow-up wart

**Risk:** Per the design decision to defer `extra_summaries_file=`,
the API has asymmetric customization paths — aliases get a clean
file kwarg, summaries get only the inline dict kwarg. A user who
wants both customizations has to use a hybrid pattern.

**Posture:** accept for v1.0.0; track as v1.1.0 candidate.

The asymmetry is real but small. Most user-corpora that need
custom summaries also need custom aliases, and the inline dict
kwarg works for both — it's only the *file-loading convenience*
that's asymmetric. The guide can show both patterns side-by-side.

If v1.1.0 demand justifies it, `extra_summaries_file=` ships as a
non-breaking addition. The asymmetry doesn't get *worse* over time;
it just sits there until cleaned up.

## 7. The reranker-evaluation diagnostic (D5) and the harness happen
in different timeframes

**Risk:** D5 (the `reranker-evaluation` spec, item #1 from the
architecture plan) measures reranker effectiveness on the bundled
corpus and produces a recommendation. The harness, shipped under
this spec, includes reranker output by default (mirroring the
pipeline default).

If D5's verdict is "the reranker doesn't help meaningfully on the
bundled corpus" or "the reranker costs latency without lifting
metrics", the harness's default behavior may be inconsistent with
D5's recommendation. Two surfaces saying different things.

**Posture:** sequence D5 before this spec's M1.

D5 is a Phase 4 W2 deliverable (per the v1.0.0 architectural plan);
this spec's implementation runs in Phase 5. D5 lands first, its
verdict is incorporated into the harness's default and into the
guide's discussion of when to use the reranker. The two artifacts
ship consistent recommendations.

If D5's verdict surfaces *after* M1 scoping is locked, the harness
default may need a follow-up PR to align. Mitigation: the scoping
pass for this spec (which runs *after* Phase 4 W2) inherits D5's
verdict and bakes it into M1.

---

None of the above blocks scaffolding the spec or the `/spec` pass
that promotes it. The point of the register is that each risk is
*named* so the implementation milestones can design against it.
