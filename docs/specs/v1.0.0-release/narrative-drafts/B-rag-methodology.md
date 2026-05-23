# Post B — *"How to Actually Measure a RAG Pipeline (And Why Most Benchmarks Lie)"*

> **Status:** v1 full draft, 2026-05-22. Not yet polished for
> publication. Sections are written through; charts and code blocks
> are described, not embedded. Numbers are real, pulled from
> `docs/specs/downstream-validation/perf-thresholds.json` at the
> locked v2 baseline commit `6fbe6d7`.

- **Role in trilogy:** credibility deep-dive. Earns the right to
  Post A's claim.
- **Target length:** 2000–2500 words. Current draft: ~2300.
- **Hero asset:** the two-runs-one-pipeline overlay — same config,
  same data, different "winners." Built from
  `tests/perf/baseline.py` rolled twice with a fixed seed swap.
- **Channels (proposed):** Blog + cross-link from Post A. Aim for
  inbound links from anyone writing about RAG eval over the next
  18 months.

---

## One-line pitch

Single-run benchmarks don't measure anything. Here's what the
methodology has to look like before the numbers mean what you
think they mean.

---

## 1. The pitch most RAG benchmarks make

Open any RAG framework's README and you'll find something like this:

> *We ran our pipeline on Dataset X. Recall@5 is 0.78. End-to-end
> latency is 240 ms.*

It looks like a measurement. It's not.

That number was produced by a single run. Maybe two — one for the
README, one for the screenshot. There's no variance reported. There's
no LLM-judge in the loop, so the "recall" is whatever the retriever
returned, not whether the generated answer was faithful. There's no
held-out eval set, so the pipeline was tuned on the same corpus it's
being measured on. And there's no statistical threshold — if a future
PR makes recall 0.76, is that a regression or a coin flip?

Each one of those gaps is enough on its own to make the number
meaningless. Most published RAG benchmarks have all four.

This post is what we did instead, on `attune-rag`, when we got
serious about measurement. It's the methodology that produced our
own v1.0.0 release numbers — and that produced the finding (in
[Post A](A-rerank-measurement.md)) that our reranker is neutral on
a well-aliased corpus. If you don't trust that finding, you
shouldn't — until you've read this post. After you have, the only
honest move is to either accept the receipts or re-run them.

---

## 2. What changes when you measure variance

Here's the gut-punch.

Take any single RAG pipeline. Don't change anything — same code,
same config, same corpus, same query set. Run it twice. Plot the
mean latency of each retriever component side by side.

> *Hero chart 1: two stacked horizontal bars per component
> ("keyword retriever", "reranker", "pipeline end-to-end"), each
> showing Run 1 mean and Run 2 mean. For the reranker, the bars
> visibly differ by 30–40%. For the keyword retriever, they're
> indistinguishable.*

That gap on the reranker bar isn't because anything broke. It's
because LLM-bound components have wide intrinsic variance —
network latency to the model endpoint, queueing, model-side
scheduling jitter, sometimes outright cold-start delays. A single
run captures one sample of a wide distribution and presents it as
"the" performance number.

If you publish that number, and a future PR's number lands inside
the same distribution, you have no way to know whether it's a
regression. You'll false-alarm on every PR that happens to draw a
slow sample, and you'll silently accept actual regressions that
happen to draw a fast sample. The numbers are noise, presented as
signal.

What you need is the *shape* of the distribution — not just where
the most recent sample landed.

---

## 3. The four levers that earn trust

There are exactly four levers that turn a single-run benchmark into
something a reasonable engineer can act on. None are exotic. All
four are non-optional.

**Lever 1: Runs ≥ 20.**
A single run gives you one sample. Three runs gives you a rough
mean and no usable variance estimate. Ten runs gets you in the
ballpark. Twenty runs is where the inter-run standard deviation
stabilises for the kind of distributions RAG components produce —
heavy-tailed on the LLM-bound side, narrow on the in-process side.
We use 20 runs per invocation, repeated across 5 invocations, for
a total of 100 measurements per metric.

**Lever 2: `include_llm = True`.**
The most contentious one. Yes, it costs API calls. Yes, it makes
the CI suite slower. But the largest variance in any RAG pipeline
lives in the LLM hop, and if you measure without it you're
optimising the cheap component while the expensive component
silently regresses. We made `include_llm=True` the default for
baseline runs. The cost is real; the alternative is not measuring
the thing that matters.

**Lever 3: Sigma ≥ 2.0 thresholds.**
The gate threshold for "is this a regression" is `mean + sigma ×
inter_run_stdev`. `sigma=1` is a coin flip — half your real
regressions slip through. `sigma=3` is a façade — we used it
briefly as a stop-gap when our threshold model was wrong (the
"#75 sigma=3 hack", rolled back in #142). Sigma=2.0 is the value
that survives contact with real CI noise *if your variance model
is honest* (which is what Lever 4 enforces).

**Lever 4: A locked baseline file.**
The baseline is a JSON file checked into the repo at a specific
commit. It records the mean, both flavours of standard deviation
(more on that below), the threshold derived from them, and the
environment fingerprint of the run that produced it. Every future
PR's perf check reads this file and compares its own measurement
against the locked numbers. The file moves only when you do a
deliberate re-baseline — never silently, never per-PR.

Take any one of these levers out and the methodology collapses to
"a number we ran once."

---

## 4. The v2 schema — show the receipts

Here's an entry from our locked baseline file
(`docs/specs/downstream-validation/perf-thresholds.json`,
methodology version 2, commit `6fbe6d7`):

```jsonc
"llm_reranker_rerank.wall": {
  "mean":              0.211075,   // seconds, end-to-end wall time
  "stdev":             0.062108,   // inter-run stdev, kept for back-compat readers
  "threshold":         0.335291,   // mean + 2.0 * inter_run_stdev
  "inter_run_stdev":   0.062108,   // stdev of the K=5 invocation means
  "intra_run_stdev":   0.272660,   // averaged within-invocation stdev
  "invocations":       5,
  "runs_per_invocation": 20
}
```

A few things to notice.

**Two flavours of stdev.** `intra_run_stdev` is the variance
inside a single CI invocation — twenty consecutive measurements
on the same runner. `inter_run_stdev` is the variance across
invocations — five different runners (or five different time
windows on the same runner), each contributing one mean. They
measure structurally different sources of noise. `intra_run` tells
you whether one CI runner was misbehaving. `inter_run` tells you
the variance you'll actually face in production CI.

The threshold uses `inter_run_stdev`. That's the right choice
because the threshold is gating *future CI runs against past CI
runs*, which is an inter-run comparison by construction. Using
`intra_run_stdev` would underestimate the noise and produce a
false-positive storm.

**The `stdev` alias.** We kept `stdev` as a key in the v2 schema
even though it's redundant with `inter_run_stdev`. That's a
deliberate back-compat affordance for downstream readers — our
own `format_perf_delta.py` and `check_thresholds.py` consume
`mean / stdev / threshold` and keep working unchanged. Schema
evolutions that quietly break readers are how you train a team to
stop trusting the baseline file. Don't do it.

**The variance contrast.** Look at two metrics side by side:

| Metric | Mean | inter_run_stdev | Threshold (σ=2.0) |
|---|---|---|---|
| `keyword_retriever_retrieve.wall` | 0.0054 s | 0.000097 s | 0.0056 s |
| `llm_reranker_rerank.wall` | 0.211 s | 0.062 s | 0.335 s |

The keyword retriever's inter-run stdev is 1.8% of its mean. The
reranker's is 29%. Two orders of magnitude difference in noise
shape — measured at the same commit, on the same runner, on the
same corpus. Any methodology that treats both numbers with the
same precision is wrong about both.

> *Hero chart 2: log-scale dot plot. X-axis: components. Y-axis:
> mean ± 2σ band. The keyword retriever's band is a thin slit; the
> reranker's is a fat envelope.*

---

## 5. What this catches that single-run doesn't — a live example

The case for sigma=2.0 over sigma=3.0 isn't theoretical. We made the
wrong call once and walked it back this week.

In late April we shipped a perf-threshold update that bumped sigma
from 2.0 to 3.0. The reasoning at the time was that 2.0 was firing
too many false positives in CI. That was true — but the *cause*
wasn't sigma. The cause was that our variance model was still
charging the gate with `intra_run_stdev` (the narrow one) instead
of `inter_run_stdev` (the realistic one). Sigma=3.0 was a fudge
factor papering over a structural bug.

We fixed the variance model in the v2 schema. Once the gate used
`inter_run_stdev`, sigma=2.0 became the right value again. We
rolled the constant back in [#142](https://github.com/Smart-AI-Memory/attune-rag/pull/142),
which is the commit that locked the numbers you're reading above.

The reason that story is in this post: *single-run benchmarks
can't produce this kind of self-correction.* If you don't have a
variance model, there's nothing to fix. If you don't ship the
variance model in the public baseline, your community can't audit
it. Our threshold history is a thing readers can read; our
methodology version field is in the JSON; the PR rolling sigma
back has a written rationale. That's the difference between a
benchmark you can argue with and a benchmark you have to accept.

---

## 6. The reproducibility claim

If you cannot reproduce the numbers, the methodology doesn't
matter. So here's what we ship.

- **The locked baseline file** at the exact path
  `docs/specs/downstream-validation/perf-thresholds.json`, with
  the `commit` field naming the commit that produced it and the
  `environment` field naming the runner.
- **The script that produces it** at `scripts/regenerate_perf_thresholds.py`
  (or wherever the M2 aggregator landed — pin at fleshing-out
  time). Deterministic seed; explicit `--invocations` and
  `--runs-per-invocation` flags; identical defaults to the locked
  file.
- **Per-PR delta-checks** as advisory comments. Every PR's CI run
  re-measures and posts the deltas against the locked baseline.
  No silent re-baselines — drift triggers a comment, and a real
  re-baseline is its own PR with its own diff.
- **The methodology spec** at
  `docs/specs/perf-baseline-multi-run/` — design.md is the
  long-form reasoning; tasks.md is the milestone history; the
  whole thing is read-the-source documentation.

Anyone can re-run our baseline on their own hardware. We expect
the absolute numbers to drift — Linux runners on Azure x86_64 will
not match a developer's MacBook M-series, and that's fine — but
the *shape* of the variance should hold. If yours doesn't, that's
a finding, and we'd like to hear about it.

This is the contract: we publish enough that you can disagree with
us, and we precommit to listening when you do.

---

## 7. Why this matters for the field

There's a temptation in eval-land to dismiss the methodology
question as "the boring part." It isn't. It's the *only* part that
distinguishes a published number from a marketing claim.

The current state of RAG benchmarking is closer to marketing than
to measurement. Vendor leaderboards rank pipelines on single runs.
Recall and faithfulness are conflated into a single recall score.
LLM hops are silently excluded "for stability." Statistical
significance is never mentioned. Whatever number wins this week
gets posted to Twitter; next week's number replaces it without
comment.

The fix isn't a new benchmark. The fix is making methodology a
first-class part of how each tool reports its own numbers. Four
levers. Twenty runs. A locked file. Variance you can argue with.
That's the whole pitch.

If you ship a RAG framework, the most respectful thing you can do
for your users is publish numbers they can trust. That starts with
admitting how much variance lives inside the numbers you're
publishing right now — and refusing to publish further numbers
until you can quote the variance alongside the mean.

We did this because we had to: our own reranker measurement (see
[Post A](A-rerank-measurement.md)) gave us a finding that
contradicted our priors. We didn't trust it until the methodology
was bulletproof. Once it was, the finding held. The same will be
true of yours.

---

## Call to action

- **Re-run our baseline.** Pull the repo, run the script, check the
  variance against your hardware. If it diverges, file an issue —
  we want to hear it.
- **Audit your own.** If you're shipping a RAG framework, the
  question is: what does your `perf-thresholds.json` look like? If
  the answer is "we don't have one," that's the work.
- **Disagree publicly.** The methodology is in the open. The numbers
  are in the open. The threshold derivation is in the open. If we
  got it wrong, the receipts are there to argue with. We'd rather
  be corrected than admired.

---

## Open questions for fleshing-out

(These are the calls that need a human decision before publication —
not blockers for an internal draft.)

- Which corpus + query-set do we actually cite for the side-by-side
  in §4? Bundled-corpus is honest but small. The downstream-validation
  query set has bigger numbers but ties the post to internal infra.
- Do we want a sidebar comparing our methodology to specific
  competitor benchmarks (RAGAS, BEIR-style harnesses, LlamaIndex
  internal evals), or keep it positive-framing only? Comparison
  makes the post sharper at the cost of inviting a fight.
- Should we publish the raw 100-measurement JSON alongside the post,
  or keep that gated behind a "reproduce it yourself" CTA?
- Title — is the "(And Why Most Benchmarks Lie)" subtitle worth the
  combative tone, or do we soften to "(And What Most Benchmarks
  Miss)"?
- Channel call — blog only, or blog + a Show HN simultaneous? If
  Show HN, Post A goes first (week 0); B leads with "the methodology
  behind last week's reranker finding" (week 1). If blog-only, B can
  lead the trilogy.
- One unresolved technical claim in §5 — the wording around the
  "structural bug" in the pre-v2 variance model. Worth running by
  whoever owns perf-baseline-multi-run before publication to make
  sure the history is told accurately.
