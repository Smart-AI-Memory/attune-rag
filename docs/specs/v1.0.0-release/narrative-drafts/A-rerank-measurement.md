# Post A — *"We Built a Reranker. Then We Measured It."*

> **Status:** v1 full draft, 2026-05-22. ~1650 words. Numbers are
> real, pulled from the D5 diagnostic-1 verdict-lock at
> `docs/specs/reranker-evaluation/diagnostic-1.md`.

- **Role in trilogy:** marquee / inversion-of-expectation hook.
- **Target length:** 1200–1800 words. Current draft: ~1650.
- **Hero asset:** P@1 / R@3 table with and without rerank, baseline
  and paraphrased side by side. Built from `diagnostic-1.md`.
- **Channels (proposed):** README hero link, Show HN, r/LocalLLaMA,
  r/MachineLearning.

---

## One-line pitch

Every RAG vendor sells a reranker. We measured ours and shipped
the measurement, not the opinion.

---

## 1. The setup

We built a reranker. Standard architecture: BM25 retrieves a
top-K candidate set, an LLM-based reranker scores each candidate
against the query, the top results bubble up. Cross-encoder family,
nothing exotic. The literature is unambiguous about what should
happen next: precision goes up, recall holds, end-to-end answer
quality improves.

We expected the standard wins.

Then we actually measured it, on a corpus we'd spent two weeks
making clean, with a methodology we'd spent another two weeks
hardening (the long version of that story is [Post B](B-rag-methodology.md)).

Here's what came out.

---

## 2. The methodology, in one paragraph

K=5 top candidates. 20 runs per configuration, across 5 invocations
— 100 measurements per cell. `include_llm=true` for every run
(skipping the LLM hop optimizes the cheap component while the
expensive one silently regresses). Threshold for "this is a real
difference" set at mean + 2σ on the inter-run standard deviation,
not the intra-run one. Locked baseline file checked into the repo.
The full reasoning lives in [Post B](B-rag-methodology.md); for now
treat it as: *we measured this seriously, with receipts you can
audit.*

---

## 3. What we found

The receipts, lifted directly from
`docs/specs/reranker-evaluation/diagnostic-1.md`:

| Query set | Metric | Baseline (BM25-only) | With reranker | Direction |
|---|---|---|---|---|
| Baseline queries | P@1 | **1.000** | 0.985 | ↓ regression |
| Baseline queries | R@3 | **1.000** | 0.995 | ↓ regression |
| Paraphrased queries | R@3 | 0.9875 | 0.9825 | ↓ regression |
| Paraphrased queries — P@1 misses | Fixed by rerank at ≥4/5 stability | — | **1 of 10** | ↓ no meaningful lift |

That's not "rerank is neutral." That's worse than neutral. On the
baseline query set — the set where BM25 already gets every answer
right — adding the reranker moves the winning document off rank 1
in about 1.5% of runs. The reranker is *introducing errors that
weren't there.*

On paraphrased queries — the set where rerank theoretically has
the most room to help, because BM25 has to reach across query
vocabulary mismatch — only one of ten initial P@1 misses got
stably fixed by rerank (`gqp-031a`, 5/5 stability across runs).
The other nine didn't. And R@3 on paraphrased queries dropped
0.005, well outside our measurement noise.

Three separate regression conditions, any one of which would
trigger our pre-committed *"rerank-default-off"* rule. All three
fired together.

> *Hero chart: same data as the table above, rendered as a
> grouped bar chart. Two clusters per metric (baseline, rerank).
> The visual punch is that the BM25-only bars touch the ceiling
> on both baseline columns; the rerank bars dip below.*

---

## 4. Why

The instinct, when you see numbers like this, is to assume the
reranker is broken. It isn't. It's a perfectly competent
cross-encoder doing exactly what cross-encoders do: scoring
candidates against a query embedding and producing a re-ranking.

The issue is that on a corpus where BM25 already gets the right
answer 100% of the time on baseline queries, *the reranker has
nothing to fix and several things to break.* It can:

- Push a correctly-ranked document down because some other
  candidate happens to score slightly higher on the cross-encoder's
  query-document similarity.
- Pull a near-duplicate up over the true winner because the
  cross-encoder's training signal weights surface-form similarity
  in ways that don't always align with what the query actually
  meant.
- Add latency and variance — our reranker's inter-run standard
  deviation is 29% of its mean, two orders of magnitude noisier
  than the BM25 retriever it's "improving."

When the retriever is already at ceiling, every one of those failure
modes turns into pure regression. There's no upside left for the
reranker to capture; only downside it can introduce.

This is why most published RAG benchmarks show rerank wins.
*Their retrievers aren't at ceiling.* When BM25 hits 60% recall on
your corpus, a reranker that's 80% accurate on the candidate set
shows clear lift. When BM25 hits 100% on yours, that same reranker
shows what we measured: introduction of error.

---

## 5. The honest counterpoint

It would be cheap to stop here. The headline "rerank harms" is
viral; the truth is more useful.

There is a real class of corpus on which our reranker would help.
Specifically: corpora where the query and the documents don't
share enough vocabulary for BM25 to bridge — long-form natural
language queries against short documents with technical
terminology, or vice versa, or any of the cases where the
mismatch isn't fixable by curated aliases or query expansion.

For those corpora, the cross-encoder's ability to score across
vocabulary gaps is exactly the right tool. The published lift
numbers in the literature were measured on corpora like that;
they aren't wrong about that class.

The mistake is assuming every corpus is in that class. Most
aren't, once you take the retrieval side seriously. The
alias-expansion work that took our paraphrased R@3 from 28.75%
to 100% over a two-week sweep (the long version of *that* is
[Post C](C-bm25-broken.md)) closed exactly the vocabulary-mismatch
gap that rerankers usually paper over.

Once the retriever is doing its job, the reranker is solving a
problem you no longer have.

---

## 6. What we shipped

We didn't ship a verdict on rerankers. We shipped a measurement
methodology.

The `attune-rag` 1.0 release includes:

- A reranker, still — `LLMReranker`, fully functional, callable
  via the public API. The code didn't change; we just learned what
  it actually does.
- A `--no-rerank` flag and a sensible default — currently `off`
  for the bundled corpus, because that's what the measurement says.
- A `measure-corpus` command that runs the same diagnostic against
  *your* corpus. Same K=5, same 20 runs, same `include_llm=true`
  policy, same threshold derivation.
- The locked v2 perf baseline file, in the repo, with the
  measurement environment fingerprint and the methodology version
  pinned. Every PR's CI re-checks against it. No silent
  re-baselines.

The flag's default flips when *your* corpus's measurement says it
should. We don't know your corpus shape. We don't pretend to. We
ship the tool that tells you.

This is the framing we settled on for the v1.0.0 release: *we
ship measurement, not opinion.* Other frameworks ship rerank as
default-on because the literature says rerank wins. We shipped
rerank as default-off because *our measurement* said rerank
loses, on the bundled corpus, by all three of our pre-committed
regression criteria. Yours may say differently. We built the tool
that lets you find out.

---

## 7. Call to action

If you're running `attune-rag` on a corpus and you haven't
measured it: do that.

```
attune-rag measure-corpus --your-corpus-path ./your/corpus
```

You'll get back the same numbers we got back. If your paraphrased
R@3 stably improves with rerank, turn rerank on for your pipeline
— the flag is exposed, the default is overridable, the
measurement justifies the choice. If your paraphrased R@3 stays
flat or regresses, leave rerank off. The default is `off` because
that was the right answer for one specific corpus shape; it isn't
the right answer for every corpus shape, and the framework's job
is to help you find which shape you have.

The receipts for the numbers in this post live at
`docs/specs/reranker-evaluation/diagnostic-1.md` in the repo.
The methodology that produced them lives at
`docs/specs/perf-baseline-multi-run/`. The locked thresholds live
at `docs/specs/downstream-validation/perf-thresholds.json`. Every
claim has a path; every path is in the repo; every path is open
to argument.

We'd rather be corrected than admired. If the measurement is
wrong, the diagnostic is open for argument and the methodology
is open for argument. If the measurement is right — which we've
done everything we know how to do to make sure of — the
implication is that a lot of RAG benchmarks are silently
measuring the retriever's failures rather than the reranker's
successes.

Measure your own corpus. The framework is the tool. The
verdict isn't ours to ship.

---

## Open questions for fleshing-out

- Should §3's table render as a chart instead, or alongside?
  Visual punch matters more for the marquee than for B.
- Are we comfortable naming specific other RAG frameworks' default
  rerank settings (LangChain, LlamaIndex, Haystack) to make the
  contrast concrete? Sharpens the post; invites the fight.
- §4's "BM25 already at ceiling" explanation is the load-bearing
  argument. Worth a sidebar with a worked example of a single
  query where the cross-encoder picked the wrong winner? Adds
  ~200 words; makes the failure mode concrete.
- The `measure-corpus` CTA assumes the command is shipped at
  v1.0.0. If user-corpus-onboarding's M1 hasn't landed by
  publication time, soften the CTA to "the measurement script
  in the repo."
- Title alternatives — "We Built a Reranker. Then We Measured
  It." (current) vs "Our Reranker Hurt Our Pipeline. Here's How
  We Found Out." vs "Most RAG Benchmarks Are Measuring The Wrong
  Thing." First is honest; second is sharper; third is more
  combative.
