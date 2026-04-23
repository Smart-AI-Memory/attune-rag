---
type: task
feature: retrieval
depth: task
generated_at: 2026-04-23T03:33:29.766392+00:00
source_hash: 7143f387f3dccfded707adcfa52af1fdc50a71361e9de5a4bd466bc191c3f35b
status: generated
---

# Work with retrieval

Use retrieval when you need to find relevant help content based on user queries using keyword matching and scoring algorithms.

## Prerequisites

- Access to the project source code
- Familiarity with the files under `src/attune_rag/retrieval.py`

## Understand the retrieval architecture

The retrieval system has three main components:

1. **`RetrievalHit`** — A single search result containing an entry, relevance score, and explanation of why it matched
2. **`RetrieverProtocol`** — The interface that any retrieval implementation must follow with a `retrieve(query, corpus, k)` method
3. **`KeywordRetriever`** — A concrete implementation that scores content using token overlap with configurable weights for path, summary, content, and related fields

## Choose your modification approach

- **Extend with a new retriever class** if you need different scoring logic (semantic similarity, machine learning ranking, etc.)
- **Modify KeywordRetriever directly** if you need to adjust the existing token-overlap algorithm

## Implement your retrieval changes

1. **Create your retriever class** (if extending):
   ```python
   class MyCustomRetriever:
       def retrieve(self, query: str, corpus: CorpusProtocol, k: int = 3) -> Iterable[RetrievalHit]:
           # Your scoring logic here
           pass
   ```

2. **Return properly structured results**:
   Each `RetrievalHit` needs an entry, numerical score, and human-readable match reason.

3. **Handle the stopwords and stemming** (if using keyword-based logic):
   The module provides `_STOPWORDS` and `_STEM_SUFFIXES` constants for text preprocessing.

## Test your retrieval implementation

Run the retrieval tests to verify your changes work correctly:
```bash
pytest -k "retrieval"
```

## Verify the task worked

Your retrieval implementation is working when:
- Test queries return ranked results with reasonable scores
- The `match_reason` field explains why each result was selected
- Performance is acceptable for your target corpus size
