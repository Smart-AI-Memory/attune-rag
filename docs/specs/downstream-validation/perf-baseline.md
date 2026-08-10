# Perf baseline (Phase 5 — multi-run methodology v2)

- Methodology version: `2`
- Measured at: `2026-08-10T04:51:53Z`
- Commit: `04251daa6c3f50ecf38aac27e420a6c4261d0654`
- Invocations: `5`
- Runs per invocation: `20`
- Sigma: `2.0` (threshold = mean + σ × inter_run_stdev)
- include_llm: `True`

## Metrics

| Metric | mean | intra_run_stdev | inter_run_stdev | threshold |
|--------|------|-----------------|-----------------|-----------|
| `directory_corpus_load.cpu` | 0.000060 | 0.000018 | 0.000015 | 0.000089 |
| `directory_corpus_load.wall` | 0.000060 | 0.000018 | 0.000015 | 0.000089 |
| `keyword_retriever_retrieve.cpu` | 0.005573 | 0.023052 | 0.000824 | 0.007221 |
| `keyword_retriever_retrieve.wall` | 0.005573 | 0.023054 | 0.000823 | 0.007219 |
| `llm_reranker_rerank.cpu` | 0.029918 | 0.126597 | 0.003759 | 0.037435 |
| `llm_reranker_rerank.wall` | 0.096776 | 0.135357 | 0.015320 | 0.127417 |
| `rag_pipeline_run.cpu` | 0.000639 | 0.000050 | 0.000140 | 0.000919 |
| `rag_pipeline_run.wall` | 0.000638 | 0.000050 | 0.000140 | 0.000918 |
