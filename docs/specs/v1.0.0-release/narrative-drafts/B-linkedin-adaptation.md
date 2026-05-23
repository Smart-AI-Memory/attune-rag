# Post B — LinkedIn adaptation

> **Status:** v1 draft, 2026-05-22. ~1050 words, paste-ready for
> LinkedIn's article editor. Shorter than the canonical version,
> founder-voice, no code blocks, no tables, one chart. Cross-link
> back to the canonical at publication time.
>
> **Companion to:** [B-rag-methodology.md](B-rag-methodology.md)
> (the full ~2300-word version that owns the link).

---

## Title

**What Most RAG Benchmarks Won't Tell You About Their Own Numbers**

(Subtitle / preview line: *I run attune-rag at Smart AI Memory. We
just locked our v1.0.0 performance baseline. Here's what most
benchmarks miss — and why we changed how we measured.*)

---

## Body

Take any RAG pipeline you've shipped. Don't change anything — same
code, same config, same corpus, same query set.

Run it twice.

Look at the latency numbers from both runs, side by side. On the
parts of the pipeline that touch an LLM, the two numbers will
differ — often by 30% or more. The retriever's numbers will be
nearly identical. The reranker's will not.

Nothing broke between runs. Nothing changed. You measured the same
thing twice and got two different answers.

> *(Chart: two stacked horizontal bars per component — "keyword
> retriever," "reranker," "end-to-end pipeline." Same config, two
> runs. The reranker bars visibly differ; the retriever bars are
> indistinguishable.)*

That's the gap most published RAG benchmarks pretend doesn't exist.

---

**Here's what's actually happening.**

LLM-bound components have wide intrinsic variance. Network latency
to the model endpoint, queueing, model-side scheduling jitter,
occasional cold-start delays — all of it makes a single
measurement one sample of a wide distribution. A single-run
benchmark captures one sample and presents it as "the" performance
number.

If you publish that number, and a future change lands inside the
same distribution, you have no way to know whether it's a
regression. You'll false-alarm on every change that draws a slow
sample. You'll silently accept actual regressions that draw a fast
sample. The numbers are noise, presented as signal.

This is the problem that broke our reranker evaluation. We spent
weeks chasing "improvements" that turned out to be us re-sampling
the same noisy distribution. The fix wasn't a better reranker —
the fix was a better methodology.

---

**Four levers. None optional.**

When we rebuilt how `attune-rag` measures its own pipeline, we
landed on four non-negotiables.

**Twenty runs per measurement.** Not three. Not five. Twenty runs
per invocation, repeated across five invocations. One hundred data
points per metric. That's where the variance stabilizes for the
kind of distributions a RAG pipeline produces.

**Include the LLM.** The most contentious lever. Yes, it costs API
calls. Yes, it slows CI down. The largest variance in any RAG
pipeline lives in the LLM hop, and if you measure without it you're
optimizing the cheap component while the expensive one silently
regresses. We made `include_llm=True` the default. The cost is
real; the alternative is not measuring the thing that matters.

**Sigma ≥ 2.0 thresholds.** The gate threshold for "is this a
regression" is *mean + sigma × inter-run-standard-deviation.*
Sigma=1 is a coin flip. Sigma=3 is a façade. Sigma=2.0 is the value
that survives contact with real CI noise *if your variance model
is honest* — which is the fourth lever.

**A locked baseline file.** The baseline is a JSON file checked
into the repo at a specific commit. It records the mean, both
flavors of standard deviation, the threshold derived from them,
and the environment fingerprint of the run that produced it. Every
future change reads this file and compares its own measurement
against the locked numbers. The file moves only when you do a
deliberate re-baseline — never silently, never per-change.

Take any one of these out and the methodology collapses to "a
number we ran once."

---

**The receipts.**

Here's what the methodology produces on a real pipeline.

For our keyword retriever, the inter-run standard deviation is
about 1.8% of its mean. The numbers are tight. The threshold sits
a hair above the mean.

For our reranker — same commit, same runner, same corpus, same
query set, measured the same day — the inter-run standard
deviation is **29% of its mean.** Two orders of magnitude wider
than the retriever's noise band.

Any benchmark that reports both numbers with the same precision is
wrong about both. The retriever's number is meaningful to the
microsecond. The reranker's number is meaningful to the tens of
milliseconds, at best.

That's not a critique of the reranker. It's a critique of any
methodology that doesn't *measure* the difference — and doesn't
ship its variance numbers alongside its means.

---

**Why this matters for the field.**

The current state of RAG benchmarking is closer to marketing than
to measurement. Vendor leaderboards rank pipelines on single runs.
LLM hops get silently excluded "for stability." Statistical
significance is never mentioned. Whatever number wins this week
gets posted; next week's number replaces it without comment.

The fix isn't a new benchmark. The fix is making methodology a
first-class part of how each tool reports its own numbers. Four
levers. Twenty runs. A locked file. Variance you can argue with.

If you ship a RAG framework, the most respectful thing you can do
for your users is publish numbers they can trust. That starts with
admitting how much variance lives inside the numbers you're
publishing right now — and refusing to publish further numbers
until you can quote the variance alongside the mean.

We did this because we had to. Our own reranker measurement gave
us a finding that contradicted our priors. We didn't trust it
until the methodology was bulletproof. Once it was, the finding
held.

---

**One question for the people building in this space.**

If you ship a RAG framework or pipeline today: **what does your
`perf-thresholds.json` look like?**

If the answer is "we don't have one," — that's the work.

---

**The full methodology, with the receipts, the v2 schema, and the
threshold history is at [LINK TO CANONICAL POST].** I'd rather be
corrected than admired. If we got it wrong, the numbers are open
for argument.

— *Patrick Roebuck, founder, Smart AI Memory*

---

## Editor notes (not for publication)

**Length:** ~1050 words. LinkedIn's article editor accepts up to
~125k characters; the sweet spot for engagement is 800–1200 words.
We're in band.

**Formatting moves used:**
- Founder-voice first-person opener (LinkedIn rewards "I" over "we").
- Gut-punch lede — no scene-setting, no "in this post we'll explore."
- One chart, described inline; built at publication time from
  `tests/perf/baseline.py` rolled twice.
- Bold paragraph leads to give the LinkedIn skimmer anchor points.
- The provocation question near the end is the LinkedIn-shaped CTA;
  it generates substantive comments without requiring readers to
  click out.
- External link to canonical sits in the last paragraph, where
  LinkedIn's algorithm penalty is smallest (post is already
  "complete" in the algorithm's view by then).

**Formatting moves avoided:**
- No JSON code blocks (render badly on LinkedIn).
- No tables (render worse).
- No nested headers (LinkedIn collapses them).
- No sigma=3.0 → 2.0 walkback history (needs the reader
  already-invested; lives in the canonical post).
- No competitor-benchmark namedrops (LinkedIn rewards positive
  framing; the canonical post has more room for sharper framing).

**Cross-post hygiene:**
- Publish the canonical first (blog or repo `/docs`).
- Wait 24–48 hours so the canonical accumulates initial discovery
  signal (HN, Reddit, organic search).
- Then publish this LinkedIn version with the canonical URL
  embedded. The order matters — LinkedIn's link-graph signal is
  reading the canonical as the authoritative source, not the
  other way around.

**Open questions for fleshing-out (LinkedIn-specific):**

- Hashtags — LinkedIn rewards 3–5 relevant hashtags. Candidates:
  `#RAG #LLM #MachineLearning #AIengineering #Benchmarking`.
  Choose at publication; don't bake in.
- Header image — LinkedIn auto-generates from the chart if embedded
  inline, but a custom header image gets more thumb-stop. Decide
  at publication.
- Comment-seeding — worth pre-asking one or two RAG-adjacent
  contacts to leave substantive first comments. LinkedIn's
  algorithm uses early-comment quality as a ranking input. Not
  manipulation; just basic distribution hygiene.
- Title test — "What Most RAG Benchmarks Won't Tell You About
  Their Own Numbers" vs "Most RAG Benchmarks Lie About Their Own
  Numbers" vs "If You Ship a RAG Framework, Read This Before Your
  Next Benchmark." First is balanced. Second is combative. Third
  is direct-to-buyer. Pick at publication.
