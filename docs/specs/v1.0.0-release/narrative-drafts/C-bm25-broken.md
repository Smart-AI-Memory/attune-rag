# Post C — *"Your BM25 Retriever Is Probably Broken (Here's How to Tell)"*

> **Status:** skeleton draft. Not for publication. See
> [README.md](README.md) for the trilogy framing.

- **Role in trilogy:** technical deep-cut. Closes the loop by
  explaining why most people think rerank works in the first place.
- **Target length:** 1500–2000 words.
- **Hero asset:** before/after diff showing a single alias-stem fix
  flipping recall from "rerank looks like a miracle" to "rerank is
  neutral."
- **Channels (proposed):** Blog + r/LocalLLaMA + HN if Post A landed.
- **Audience:** engineers running BM25 in production — Whoosh,
  Tantivy, Lucene, etc. Smaller audience than A/B; very high
  engagement per reader.

---

## One-line pitch

We spent weeks chasing rerank improvements that turned out to be
retriever bugs in disguise. Here's what we found.

## Section 1 — The bug class

Alias expansion + stemmer interactions silently produce no-match
results. Concrete example: `"bites"` stems to `"bit"`, the alias
union doesn't see it, the retriever returns wrong docs. Most teams
never know.

## Section 2 — Seven lessons from one sweep

From the alias-expansion-sweep arc (2026-05-21). Each lesson is:
one-line bug it caught, one-line fix, one-line test that pins it.

## Section 3 — The detection methodology

Full baseline diagnostic before alias commit. Stem candidates run
through `_tokenize()`. Alias union expressed as a set operation
tested with property tests.

## Section 4 — Why this matters for the rerank discussion

*Most "rerank improvements" reported in the wild are actually
retriever bug-fixes in disguise.* If your BM25 leaks, your reranker
looks like a miracle. Fix the leak; the reranker collapses to
neutral. (Direct callback to Post A.)

## Section 5 — Call to action

The diagnostic recipe + the test patterns + (if it's appropriate to
expose) a link to the alias-stem validation notes that live with the
project.

---

## Open questions for fleshing-out

- Are the alias-expansion-sweep lessons public-blog-grade as written,
  or do they need a stylistic pass for non-attune-rag readers?
- Do we name specific BM25 libraries (Whoosh, Tantivy, Lucene) or
  keep it generic?
- Is the "rerank improvements = retriever bug fixes" claim falsifiable
  with a controlled experiment we can show? If yes, that becomes the
  centerpiece chart.
- Risk: this post implicitly criticizes published RAG benchmarks. Do
  we want that fight, or do we soften?
