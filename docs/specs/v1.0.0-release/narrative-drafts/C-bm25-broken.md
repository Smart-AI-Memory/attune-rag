# Post C — *"Your BM25 Retriever Is Probably Broken (Here's How to Tell)"*

> **Status:** v1 full draft, 2026-05-22. ~1850 words. Story
> grounded in the alias-expansion-sweep arc (2026-05-21,
> PRs #94–#118). Numbers traceable to
> `docs/specs/embedding-retriever/` diagnostics D1–D4.

- **Role in trilogy:** technical deep-cut. Closes the loop by
  explaining why most people think rerank works in the first
  place.
- **Target length:** 1500–2000 words. Current draft: ~1850.
- **Hero asset:** before/after R@3 chart: paraphrased query set,
  28.75% → 100% across 11 PRs, with baseline P@1 held at 100% the
  whole time.
- **Channels (proposed):** Blog + r/LocalLLaMA + HN (if Post A
  landed). Audience: engineers running BM25 in production.

---

## One-line pitch

We spent two weeks chasing rerank improvements that turned out to
be retriever bugs in disguise. Here's what we found, and the
diagnostic patterns you can use to check your own pipeline.

---

## 1. A bug you almost certainly have

Here's a sentence that's true of most production RAG systems and
nobody talks about:

> *Your BM25 retriever silently fails on roughly 70% of the
> paraphrased versions of queries it answers correctly.*

We measured ours. The exact starting number, before we touched
anything: 28.75% recall@3 on a paraphrased version of a query set
where the original-phrasing R@3 was already 100%.

That's not a small gap. That's a *seventy-percentage-point* gap
between "the user phrased the query the way the docs are written"
and "the user phrased the query the way humans actually talk."

If you've never measured the paraphrased version of your query
set against your own retriever, that gap is in your pipeline too.
You just don't know how big.

---

## 2. Why this matters before we talk about fixes

The reason this is a *technical* post and not a *complaining* post:

When a paraphrased query fails to retrieve, the framework either
returns the wrong document or returns the right document at the
wrong rank. A reranker can fix the rank case, sometimes. A
reranker cannot fix the wrong-document case at all — by the time
the reranker runs, the right document isn't in the top-K
candidate set.

This means *the lift you measure from your reranker is bounded
above by the recall of your retriever.* If your retriever's
paraphrased R@K is 30%, your reranker can at most squeeze a few
points of P@1 improvement out of the 30% of queries that even
made it to the candidate set. The other 70% never had a chance.

So when a vendor reports "20% rerank lift on dataset X," what
they're often really reporting is *"our retriever silently fails
on 70% of dataset X, and our reranker recovers some of the 30%
that the retriever got partially right."* The number is real;
the interpretation is wrong. Fix the retriever and the rerank
lift disappears, because the work the reranker was doing was
recovery, not improvement.

That's how the same reranker that "lifts 20%" on someone else's
benchmark shows as neutral-to-harmful on ours (the long version
of which is [Post A](A-rerank-measurement.md)).

The first step toward measuring your reranker honestly is to fix
your retriever so the reranker has nothing to recover.

---

## 3. The diagnostic patterns

We spent two weeks doing this, in 13 sequential PRs (#94–#118),
on the `attune-rag` corpus. Final state: paraphrased R@3 = 100%,
baseline P@1 = 100%, zero baseline regression across the entire
sweep. Here are the patterns that did the work.

### Pattern 1: Run the paraphrased query set against your retriever

Most teams don't have a paraphrased query set. Make one.

The cheapest version is hand-edited: take twenty representative
queries from your real traffic, write three or four paraphrased
variants per query (different word order, synonyms, more natural
phrasing, dropped function words), and run the full set through
your retriever. Measure R@K with K=5 or K=10.

If your baseline R@K is high and your paraphrased R@K is low,
you have the gap. The size of the gap is the size of your
silent-failure surface.

This single diagnostic is the most valuable hour you can spend
on a RAG pipeline. We've never seen a production system pass it
without surprises.

### Pattern 2: Check the stemmer's failure modes

BM25 implementations typically apply a stemmer to both query and
document tokens. The standard English stemmer (Porter, Snowball,
the variants Lucene/Tantivy/Whoosh ship) is *not* always
predictable on inflected forms. A concrete trap from our sweep:

```
"bites" → "bit"     (stripped via -es, length check 5−2=3 passes)
"bite"  → "bite"    (no suffix, no strip)
"biting" → "bite"   (strip -ing → bit, then revert to bite)
```

These three surface forms produce three different stems. A user
typing one of them won't match documents containing the others
unless the BM25 implementation has been configured to expand
across all three, *and the alias union has been written by someone
who knows the stemmer's behavior at this level of detail.*

Most aren't. Most pipelines silently fail to match across these
inflections, and the failure shows up as low paraphrased recall.

The diagnostic pattern: take every alias candidate you're about
to add to your retriever, run it through your tokenizer + stemmer
chain, and verify the output. The function call we use looks like
this — yours will look similar:

```python
from attune_rag.corpus.tokenize import _tokenize

print(_tokenize("bites"))   # → ['bit']
print(_tokenize("bite"))    # → ['bite']
print(_tokenize("biting"))  # → ['bite']
```

If you've never done this, do it once for the top-twenty terms in
your alias file. You will find at least one surprise.

### Pattern 3: Hand-aliases over embeddings, for the cases hand-aliases reach

Our initial diagnostic (D1 in the spec dir) confirmed the
paraphrased-recall gap. The natural next step was to scope an
embedding retriever — semantic similarity, vector database, the
whole stack.

Before scoping, we ran one more diagnostic (D3): *what if we
just hand-curate aliases for the worst feature clusters?* The
answer was that hand-aliases closed the gap completely, at zero
baseline cost, in 11 PRs across one day.

The embedding-retriever spec is now permanently deferred — not
because embeddings are bad, but because they were a much larger
intervention than the problem actually needed. The diagnostic
caught it. Without the self-challenge, we'd have shipped a
multi-week embedding build that solved a problem aliases solved
in a day.

The diagnostic pattern: *before scoping a heavy infrastructure
intervention, scope a lightweight one and measure both.* The
heavy one wins less often than your instincts suggest.

### Pattern 4: Run the full baseline before *every* alias commit

The trap that almost ended our sweep (PR #105 / M12):

`doc-audit` had `"outdated documentation"` in its alias union,
which contributed `"document"` as a stemmed token. A new alias
for `"readme lies about code"` added `"code"` to the same union.
Independently, both looked fine. Together, a baseline query
*"create documentation for my code"* crossed our
`MIN_ALIAS_OVERLAP=2` threshold against doc-audit and dropped
P@1 from 1.00 to 0.985.

That regression was caught pre-commit *only* because the workflow
ran the full baseline diagnostic before every commit, not after
merge.

The diagnostic pattern: alias unions interact. Local-looking
changes have non-local effects on baseline retrieval. The only
reliable defense is to run the full baseline query set against
the retriever before every alias commit lands. The cost is a
30-second test; the cost of catching this after merge is a revert
PR and a story to tell.

### Pattern 5: Sequential PRs at deliberate pace catch what velocity hides

Across 13 PRs in a single day, the M12 near-regression and one
other cross-feature interaction both surfaced as *conversations*
during review, not as revert PRs after merge. The "needs a
follow-up to fix what I just did" rate was effectively zero.

Why this worked: each PR's full baseline diagnostic was visible
to the next PR's author (i.e. the same person, the next morning).
That visibility is what catches the interactions that velocity
hides.

The diagnostic pattern is also a workflow pattern: *during a
sweep that touches a shared resource (the alias union, in this
case), don't parallelize.* Sequential PRs with deliberate
pre-commit verification have a lower defect-introduction rate
than parallel PRs by enough margin to dominate the throughput
benefit of parallelism.

---

## 4. The receipt

Final state of the sweep, measured at the close of M13 and locked
into the diagnostic record:

| Query set | Metric | Before sweep | After sweep |
|---|---|---|---|
| Baseline | P@1 | 1.000 | **1.000** (unchanged) |
| Baseline | R@3 | 1.000 | **1.000** (unchanged) |
| Paraphrased | P@1 | 53.75% | **91.25%** |
| Paraphrased | R@3 | **28.75%** | **100%** |

Zero baseline regression. Two weeks of work. Eleven shipping PRs.
A spec (embedding-retriever) defensibly closed as permanently
deferred. A new spec dir (alias-expansion-sweep) holding the
play-by-play for future maintainers.

The reason the chart matters: *that 28.75% → 100% jump is the
recall headroom most RAG pipelines leave on the table.* Most
teams don't measure it, so they don't see it. Then they pay for
a reranker to recover a fraction of it. The reranker recovers
some. The headroom stays mostly invisible.

---

## 5. The callback

If you read [Post A](A-rerank-measurement.md) — the post where
we measured our reranker and found it neutral-to-harmful — this
sweep is the reason. By the time we measured the reranker, the
retriever was at ceiling. There was nothing left for the
reranker to recover.

When the retriever is at ceiling, the reranker measures as
neutral or worse. When the retriever leaks, the reranker
measures as miraculous, because it's quietly recovering
retrieval failures the team never measured.

Most reported rerank wins are the second case. Fix the
retriever and the rerank lift dissolves. That's not a bug in
the reranker; it's an artifact of measuring the wrong thing.

---

## 6. Call to action

Three concrete things you can do this week:

1. **Make a paraphrased query set.** Twenty real queries, three
   variants each. One afternoon of work.
2. **Run it against your retriever.** Same K, same metric you
   already use. Note the R@K delta between baseline and
   paraphrased.
3. **If the delta is large, fix the retriever before you tune
   the reranker.** Aliases are the cheap intervention. Stemmer
   audits are the cheap intervention. Embedding retrieval is the
   expensive intervention you may not need.

We did all three on `attune-rag` and ended up with a retriever at
ceiling and a reranker we now ship as default-off because the
measurement says it doesn't earn its keep. The receipts are at
`docs/specs/embedding-retriever/` (D1–D4 diagnostics) and
`docs/specs/alias-expansion-sweep/` (the play-by-play).

The framework's job is to help you find which class of corpus
you have. We built the tool. We won't pretend to know your
corpus shape. Run the diagnostic. Let the measurement decide.

---

## Open questions for fleshing-out

- §3.2's `_tokenize()` code block exposes an internal function
  name. Acceptable for a blog post, or do we wrap it in a small
  public helper before publication?
- The 7-lessons-from-memory framing was the original skeleton.
  This draft surfaces 5 patterns publicly and drops the two
  inside-baseball ones (spec-dir convention; defer-permanent
  semantics). Right call?
- §4's table is the same shape as Post A's §3 table by design
  (parallel structure across the trilogy). Worth a visual
  callout that they're parallel, or trust the reader?
- Title alternates — "Your BM25 Retriever Is Probably Broken
  (Here's How to Tell)" (current) vs "What We Found When We
  Stopped Trusting Our Retriever" vs "Two Weeks of Aliases Did
  What a Vector Database Couldn't." First is provocative; second
  is reflective; third is contrarian.
- §3.3 implicitly criticizes the impulse to reach for embedding
  retrieval. Do we want to soften? It's the strongest claim in
  the post and probably the most contestable.
- §5's "most reported rerank wins are the second case" is a
  load-bearing claim. Falsifiable with a controlled experiment
  we don't currently have receipts for. Soften, or commit and
  cite Post A as the closest thing to evidence?
