---
type: tip
name: dashboard-tip
feature: dashboard
depth: tip
generated_at: 2026-06-10T06:08:44.513314+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Tip: Working effectively with the dashboard pipeline

Call `build_snapshot()` first, then pass its return value downstream — `render()` writes an HTML file and `display()` prints to the terminal. Keeping the snapshot as the handoff point between stages means you can swap or skip either output without re-running the benchmark.

**Why it sticks:** the snapshot dict is the single source of truth for a dashboard run; producing it once and routing it to multiple outputs is cheaper than invoking the full pipeline twice.

**Tradeoff:** `build_snapshot()` returns a partial result (with an error key) when `queries.yaml` is missing rather than raising. If you pass that partial snapshot straight to `render()` or `display()`, your output will be incomplete without any loud failure — so check the returned dict for an error key before continuing.

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
