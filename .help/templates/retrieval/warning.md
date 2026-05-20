---
type: warning
name: retrieval-warning
feature: retrieval
depth: warning
generated_at: 2026-05-20T03:22:04.688838+00:00
source_hash: 808240403d72c9dd7f4996c2aba40fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Retrieval cautions

## What to watch for

`KeywordRetriever` scores `RetrievalEntry` objects against a query using token-overlap, suffix stemming, and stopword filtering. The results depend on several implicit behaviors — in the stemmer, the stopword list, and the field-weight configuration — that are easy to overlook when customizing or extending the retriever.

## Risk areas

### Short or common queries return low-quality results

`KeywordRetriever` strips a broad set of stopwords (including `how`, `do`, `is`, `can`, `with`, and others defined in `_STOPWORDS`) before scoring. A query composed mostly of stopwords — for example, `"how do i do this"` — reduces to an empty token set and scores every entry equally, returning essentially arbitrary top-k results. Check the post-filter token set when retrieval results look unexpectedly uniform.

### Stemming collapses unrelated terms

The suffix stemmer strips endings from `_STEM_SUFFIXES` (`ations`, `ing`, `ers`, `ed`, and so on) left-to-right until the stem is four characters or longer. This can collapse terms that share a suffix but are semantically unrelated — for example, `"rating"` and `"rating"` vs `"rat"` — producing false token-overlap matches. If your corpus contains short technical terms or abbreviations, verify that stemming is not merging them incorrectly.

### Field weights silently determine ranking order

`KeywordRetriever` applies separate weights to the `path`, `summary`, `content`, and `related` fields of each `RetrievalEntry`. A query token that matches a high-weight field (such as `path`) outranks many matches in a lower-weight field (such as `content`). If your corpus has sparse or missing path/summary metadata, results may rank entries with thin metadata above genuinely relevant content-rich entries. Populate all scored fields where possible.

### `RetrieverProtocol` duck-typing hides signature mismatches

Any object with a `retrieve(query, corpus, k)` method satisfies `RetrieverProtocol` — there is no runtime enforcement of the return type. A custom retriever that returns `None`, an empty list on error, or objects that lack `score` or `match_reason` fields will pass protocol checks silently and only fail when downstream code accesses those attributes on a `RetrievalHit`. Validate that your implementation returns a proper `Iterable[RetrievalHit]` with all three fields populated.

### `_STOPWORDS` and `_STEM_SUFFIXES` are private and may change

The stopword set and suffix list are module-level private constants. If your code imports or copies them directly, a future change to either constant will silently diverge from what `KeywordRetriever` uses internally, causing your preprocessing to disagree with the scorer. Treat these as implementation details and do not depend on them outside the module.

## How to avoid problems

1. **Log the post-filter token set for unexpected results.** When retrieval returns surprising rankings, inspect which tokens survive stopword removal and stemming for your query. This is the fastest way to distinguish a bad query from a bad corpus entry.

2. **Ensure `RetrievalEntry` metadata is complete.** Because `path` and `summary` carry higher weights than `content`, sparse metadata skews scoring. Populate all four scored fields (`path`, `summary`, `content`, `related`) when constructing corpus entries.

3. **Validate custom retriever output against `RetrievalHit`.** When implementing `RetrieverProtocol`, confirm that every returned object is a `RetrievalHit` with a non-`None` `entry`, a finite `score`, and a non-empty `match_reason` before returning from `retrieve`.

4. **Do not import `_STOPWORDS` or `_STEM_SUFFIXES` directly.** If you need to replicate tokenization logic, copy the values explicitly and document that they are snapshots, so a future change to the originals does not silently affect your code.

5. **Run targeted regression tests after any scorer change.** `pytest -k "retrieval"` covers the scoring and ranking path; run it before and after modifying weight values or tokenization logic to catch ranking-order regressions early.

## Source files

- `src/attune_rag/retrieval.py`

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
