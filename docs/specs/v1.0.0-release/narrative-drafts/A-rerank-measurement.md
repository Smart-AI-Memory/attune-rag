# Post A — *"We Built a Reranker. Then We Measured It."*

> **Status:** skeleton draft. Not for publication. See
> [README.md](README.md) for the trilogy framing.

- **Role in trilogy:** marquee / inversion-of-expectation hook.
- **Target length:** 1200–1800 words.
- **Hero asset:** mean ± stdev chart, with/without rerank, error
  bars overlapping.
- **Channels (proposed):** README hero link, Show HN, r/LocalLLaMA,
  r/MachineLearning.

---

## One-line pitch

Every RAG vendor sells a reranker. We measured ours and shipped the
measurement, not the opinion.

## Section 1 — The setup

We built a reranker. Standard cross-encoder, standard rerank-top-K
flow, standard everything. We expected the standard wins.

## Section 2 — The methodology, in one paragraph

K=5, 20 runs per configuration, sigma=2.0 thresholds, include_llm=true,
locked v2 baseline. (One-line link to Post B for the deep dive.)

## Section 3 — What we found

Mean and stdev for keyword vs keyword+rerank on the bundled corpus.
The error bars overlap. The neutrality *is* the story.

## Section 4 — Why

Well-aliased BM25 is shockingly hard to beat on a clean corpus. The
reranker is paying compute to re-rank an already-correct top-K.

## Section 5 — The honest counterpoint

When does rerank actually help? Noisy corpora, ambiguous queries,
vocabulary mismatch between query and corpus. The framework tells
you which world you're in.

## Section 6 — What we shipped

Not a verdict on rerankers. The *measurement methodology*. `--no-rerank`
is a flag; the default is whatever your corpus measurement says it
should be.

## Section 7 — Call to action

`attune-rag measure-corpus` (the user-corpus-onboarding flow, when it
lands). Don't trust our number. Measure yours.

---

## Open questions for fleshing-out

- Exact corpus + query-set citation (which release-quality-baseline
  numbers ship in the post)?
- Chart tooling — matplotlib export vs Vega-Lite embed?
- Are we willing to name the specific cross-encoder we benchmarked,
  or keep it generic?
- House style for code blocks — copy-paste runnable, or illustrative?
