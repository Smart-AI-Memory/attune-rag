# Spec: transformer-retriever — design

> **Status:** scoping (2026-06-07). Candidates below; locked at the
> `/spec` pass (top of [`tasks.md`](tasks.md)).

## 1. Problem

Static embeddings (`model2vec`) and keyword retrieval plateau at ~0.50
hard P@1 on arbitrary-corpus paraphrases. A real transformer embedding
model reaches 0.69 (bge-small) / 0.92 R@3 (MiniLM) — but needs torch.
The task is to expose that capability as a clean opt-in without touching
the offline/deterministic core or any default.

## 2. Anchoring measurement

See [`README` §anchoring](README.md) and the
[`confidence-gated-retrieval` M1b table](../confidence-gated-retrieval/tasks.md).
Headline: bge-small embedding-only hard P@1 **0.69** (vs torch-free
0.50), MiniLM R@3 **0.92**; both tank attune-help (0.42–0.53), so the
config is embedding-primary for arbitrary corpora only.

## 3. Mechanism — reuse `EmbeddingRetriever`, swap the encoder

`EmbeddingRetriever(encoder=…)` already accepts any object with
`encode(list[str]) -> 2D array`. `SentenceTransformer.encode` matches it.
M1b ran with literally:

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-small-en-v1.5")
class Enc:
    def encode(self, texts): return m.encode(list(texts), show_progress_bar=False)
EmbeddingRetriever(encoder=Enc())
```

So the build is: a `[transformers]` extra, a thin adapter that lazily
constructs the `SentenceTransformer` (mirroring how `model2vec`
`StaticModel` is lazily loaded today), a default model id, and a factory
(`TransformerRetriever` or `EmbeddingRetriever(backend="transformers")`).
**No fusion change** — for arbitrary corpora this runs embedding-primary
(the retriever *is* the embedding leg).

## 4. Model choice

- **Default candidate: `BAAI/bge-small-en-v1.5`** — best measured hard
  P@1 (0.69), ~130 MB, 384-dim, strong retrieval model.
- `all-MiniLM-L6-v2` — lighter (~80 MB), best R@3 (0.92), lower P@1
  (0.58). A good "smaller" option.
- Larger (`bge-base`, etc.) untested — measure at M1 if footprint allows.
- Keep the model id configurable (as `EmbeddingRetriever.model_name`
  already is) so footprint/quality is the user's choice.

## 5. Symmetric vs asymmetric encoding (likely free upside)

M1b used **symmetric** encoding (query and passage encoded identically).
BGE-v1.5 models are trained for **asymmetric** retrieval — the *query*
should get an instruction prefix
("Represent this sentence for searching relevant passages:"). The 0.69
was achieved *without* that prefix, so asymmetric encoding may push
higher for free. But the current `EmbeddingRetriever` encodes query and
corpus through the same `encode()` path and can't distinguish them.

Design options (decided at scoping):

- Add an optional `query_prefix` / asymmetric-encode hook to
  `EmbeddingRetriever`, or
- Give `TransformerRetriever` its own asymmetric `retrieve()`.

Measure prefixed vs not at M1 — if it lifts, adopt it.

## 6. Operating-point guidance (corpus-type-dependent)

The retriever ships with documentation, not magic, on when to use it:

| corpus | recommended | why |
|---|---|---|
| bundled / keyword-tuned | keyword (default) | already 1.00; transformer would regress it |
| arbitrary, lexically-aligned queries | static hybrid `[embeddings]` | cheap, good enough |
| arbitrary, paraphrase-heavy | **`[transformers]` embedding-primary** | only path to 0.69/0.92 |

No auto-detection in v1; the tier is an explicit, documented choice.

## 7. What this spec does NOT decide

- Any default change (keyword/static stay defaults).
- Auto-selecting transformer vs static per corpus (future, needs a
  corpus-shape signal).
- Re-introducing a gate around the transformer (M1b showed gating throttles
  it; embedding-primary is the point).

## 8. Open questions for scoping

1. `TransformerRetriever` class vs `EmbeddingRetriever(backend=…)`?
2. Default model: `bge-small` vs `MiniLM` (P@1 vs R@3 vs footprint)?
3. Asymmetric query-prefix encoding — adopt if it measures higher?
4. `[transformers]` extra pinning — torch version floor, CPU-only wheel?
5. v1.1.0+ sequencing vs a v1.0.x opt-in add (no default change → could
   ship earlier; freeze decides).
