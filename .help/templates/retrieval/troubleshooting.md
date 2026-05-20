---
type: troubleshooting
name: retrieval-troubleshooting
feature: retrieval
depth: troubleshooting
generated_at: 2026-05-20T03:22:04.690949+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Troubleshoot retrieval

## Overview

`KeywordRetriever` scores `RetrievalEntry` objects against a query using token-overlap scoring, stemming (via `_STEM_SUFFIXES`), and stopword filtering (via `_STOPWORDS`). It weights matches across four fields: path, summary, content, and related. Results are returned as a ranked list of `RetrievalHit` objects, each carrying an `entry`, a `score`, and a `match_reason`.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `retrieve()` returns an empty list | Whether all query tokens are stopwords (e.g., `"how do i"`) — after filtering, the effective query may be empty |
| Results are returned but scores are unexpectedly low | Whether the query terms survive stemming — confirm your term minus known suffixes (`ations`, `ing`, `ed`, etc.) still matches tokens in the corpus entries |
| The wrong entries are ranked first | Which fields the matched tokens appear in — path and summary weights may differ from content and related weights, skewing rank |
| `retrieve()` raises `AttributeError` or `TypeError` | Whether the `corpus` argument implements `CorpusProtocol` — any object passed as `corpus` must support the expected corpus interface |
| A custom retriever is silently ignored | Whether it implements `RetrieverProtocol` — the object must have a `retrieve(query, corpus, k)` method with that exact signature |
| Results are non-deterministic across identical calls | Whether the corpus or retriever holds mutable state that changes between calls |

## Diagnose the issue

Work through these steps in order — earlier steps are cheaper and resolve most issues.

### 1. Reproduce the failure with a minimal call

Strip the call to its required arguments:

```python
from attune_rag.retrieval import KeywordRetriever

retriever = KeywordRetriever()
hits = retriever.retrieve(query="your query", corpus=your_corpus, k=3)
print(hits)
```

Confirm the failure occurs here before investigating the surrounding context. If this call succeeds, the problem is in how the retriever is wired into your application, not in the retriever itself.

### 2. Check whether your query survives stopword filtering

`KeywordRetriever` strips tokens that appear in `_STOPWORDS` before scoring. If every token in your query is a stopword, no scoring occurs and `retrieve()` returns an empty list or zero-scored hits.

Verify your effective query tokens manually:

```python
_STOPWORDS = {'a', 'an', 'the', 'how', 'do', 'does', 'i', 'to', 'with',
              'for', 'is', 'are', 'of', 'in', 'on', 'at', 'and', 'or',
              'but', 'can', 'should', 'would', 'will', 'be', 'been', 'by',
              'my', 'me', 'we', 'it', 'this', 'that', 'these', 'those'}

query = "how do i use this"
tokens = [t for t in query.lower().split() if t not in _STOPWORDS]
print(tokens)  # [] — no tokens survive; retriever has nothing to match
```

Fix: include at least one non-stopword content term in the query.

### 3. Check whether your query terms survive stemming

`KeywordRetriever` strips the following suffixes before matching: `ations`, `ation`, `ators`, `ator`, `ates`, `ate`, `ings`, `ing`, `ions`, `ion`, `ies`, `ers`, `ed`, `er`, `es`, `s`.

If corpus entry tokens were indexed after stemming, your query terms must reduce to the same stem. Check that the stemmed form of your query token matches a stemmed form present in the corpus:

```python
_STEM_SUFFIXES = ('ations', 'ation', 'ators', 'ator', 'ates', 'ate',
                  'ings', 'ing', 'ions', 'ion', 'ies', 'ers', 'ed',
                  'er', 'es', 's')

def stem(word):
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

print(stem("configurations"))  # -> "configur"
print(stem("configure"))       # -> "configur"
```

If your query term and the corpus term don't reduce to the same stem, they won't match regardless of how semantically similar they are.

### 4. Inspect the `RetrievalHit` results you do receive

When results are returned but ranked incorrectly, inspect the `score` and `match_reason` fields on each hit:

```python
for hit in hits:
    print(hit.score, hit.match_reason, hit.entry)
```

This tells you which fields matched (path, summary, content, related) and the score each produced. If high-value entries are scoring low, the match may be occurring in a lower-weight field.

### 5. Verify corpus compatibility

If `retrieve()` raises `AttributeError` or `TypeError`, confirm your corpus object implements `CorpusProtocol`. Pass the corpus directly to a minimal `retrieve()` call (step 1) and check the traceback — Python will name the missing attribute or method.

### 6. Run the retrieval tests

```bash
pytest -k "retrieval" -v
```

If a test exercises the failing path, use its fixtures as a baseline for your reproduction. A failing test here indicates a regression in `src/attune_rag/retrieval.py` itself.

## Common fixes

- **Query contains only stopwords.** Rewrite the query to include at least one content term. Words like `how`, `do`, `i`, `is`, `the`, and `with` are all filtered before scoring.

- **Stemming mismatch between query and corpus.** Ensure both the query term and the corpus term reduce to the same stem under `_STEM_SUFFIXES`. For example, `"running"` → `"runn"` and `"runner"` → `"runn"` match, but `"run"` → `"run"` does not.

- **Custom retriever not recognized.** If you provide a custom retriever, confirm it has exactly this method signature:
  ```python
  def retrieve(self, query: str, corpus: CorpusProtocol, k: int = 3) -> Iterable[RetrievalHit]:
  ```
  Missing or renamed parameters break `RetrieverProtocol` compatibility.

- **Corpus object doesn't implement `CorpusProtocol`.** This is a change required outside the retrieval feature — update your corpus class to satisfy the expected interface before passing it to `retrieve()`.

- **Environment or dependency drift.** If retrieval behavior changed without a code change, run `pip show attune-rag` and compare against the last known-good version. Reinstall with `pip install attune-rag==<known-good-version>` to rule out a dependency change.

## Source files

- `src/attune_rag/retrieval.py`

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
