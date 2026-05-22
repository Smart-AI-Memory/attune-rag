# Perf baseline (Phase 5 — multi-run methodology v2)

- Methodology version: `2`
- Measured at: `2026-05-22T20:09:14Z`
- Commit: `6fbe6d7c4322d09fd4bc41bbf54182f3c070722e`
- Invocations: `5`
- Runs per invocation: `20`
- Sigma: `2.0` (threshold = mean + σ × inter_run_stdev)
- include_llm: `True`

## Metrics

| Metric | mean | intra_run_stdev | inter_run_stdev | threshold |
|--------|------|-----------------|-----------------|-----------|
| `directory_corpus_load.cpu` | 0.000055 | 0.000014 | 0.000002 | 0.000059 |
| `directory_corpus_load.wall` | 0.000054 | 0.000014 | 0.000002 | 0.000058 |
| `keyword_retriever_retrieve.cpu` | 0.005437 | 0.022590 | 0.000097 | 0.005631 |
| `keyword_retriever_retrieve.wall` | 0.005437 | 0.022594 | 0.000097 | 0.005632 |
| `llm_reranker_rerank.cpu` | 0.027729 | 0.114928 | 0.001368 | 0.030466 |
| `llm_reranker_rerank.wall` | 0.211075 | 0.272660 | 0.062108 | 0.335291 |
| `rag_pipeline_run.cpu` | 0.000633 | 0.000052 | 0.000080 | 0.000793 |
| `rag_pipeline_run.wall` | 0.000632 | 0.000051 | 0.000080 | 0.000793 |
