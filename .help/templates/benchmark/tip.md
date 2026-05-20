---
type: tip
name: benchmark-tip
feature: benchmark
depth: tip
generated_at: 2026-05-20T03:30:01.598219+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Tip: Gate CI on faithfulness scores, not just precision and recall

Run `main()` with `--with-faithfulness` to include faithfulness scoring alongside retrieval metrics. Without it, a pipeline can pass precision/recall thresholds while still returning responses that contradict the retrieved documents.

**Why it sticks:** precision and recall measure *what* you retrieve; faithfulness measures *whether the answer reflects it* — they catch different failure modes.

**Tradeoff:** Faithfulness scoring adds latency and may require a separate model call. Keep it enabled in CI but consider disabling it in fast local iteration loops where retrieval quality is your only concern.

## Details

- Entry point: `main()` in `src/attune_rag/benchmark.py`, returns `0` on success (suitable for direct use in CI scripts).
- Thresholds are configurable, so tighten them incrementally rather than starting strict — a threshold that blocks every run provides no signal.
- Point `--query-file` at a domain-specific query set rather than the default one; generic queries rarely expose the retrieval gaps that matter for your use case.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
