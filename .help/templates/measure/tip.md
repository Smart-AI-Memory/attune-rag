---
type: tip
name: measure-tip
feature: measure
depth: tip
generated_at: 2026-06-10T06:09:41.198960+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Tip: working effectively with measure

## Recommendation

Call `watermark_failures()` on your `MeasureResult` before treating a run as passing — don't just eyeball the `p1` and `r3` fields directly.

**Why:** `watermark_failures()` encodes your quality gates in one place and returns a list of descriptive failure strings, so CI scripts and humans read the same verdict. Inspecting `p1` and `r3` manually scatters threshold logic across callers and drifts out of sync.

**Tradeoff:** You need to decide on `p1_floor` and `r3_floor` values upfront. If you pass neither, `watermark_failures()` returns an empty list regardless of scores — so a call with no floor arguments is a no-op, not a safety net.

```python
from attune_rag.measure_corpus import measure

result = measure(
    queries_path="queries.yaml",
    bundled=True,
)
failures = result.watermark_failures(p1_floor=0.7, r3_floor=0.85)
if failures:
    raise SystemExit(f"Quality gate failed: {failures}")
```

**Tags:** `measure`, `corpus`, `quality`, `watermark`, `p-at-1`, `r-at-3`
