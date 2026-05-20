---
type: concept
name: dashboard-concept
feature: dashboard
depth: concept
generated_at: 2026-05-20T02:45:03.105087+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard

The attune-rag dashboard is a living-docs health monitor that benchmarks a registered corpus and surfaces the results as either a packaged HTML report or a terminal summary.

## Three-stage pipeline

The dashboard runs as a sequential pipeline. Each stage has a dedicated module and its own CLI entry point under `attune-rag dashboard`:

1. **Refresh** — `build_snapshot()` runs the benchmark against the corpus and returns a snapshot dictionary. If `queries.yaml` is missing, it returns a partial snapshot with an error field rather than failing hard.
2. **Render** — `render()` writes the dashboard HTML file to a path you specify, embedding the snapshot as JSON between the `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__` sentinels in the template.
3. **Show** — `display()` pretty-prints the snapshot to the terminal using Rich, so you can inspect results without opening a browser.

Because each stage consumes the snapshot dict produced by the previous one, you can also run them independently — for example, loading a previously saved snapshot into `display()` without re-running the benchmark.

## The snapshot as shared currency

The snapshot dictionary is what flows between stages. `build_snapshot()` produces it; `render()` and `display()` consume it. This design means the HTML report and the terminal view always reflect the same data, and you can serialize the snapshot to disk to version or diff corpus health over time.

## When the dashboard matters

Use the dashboard when you need to answer "is the corpus still accurate?" after code or content changes. Because `build_snapshot()` returns a partial result on missing configuration rather than raising an exception, the dashboard stays useful even in incomplete environments — you get whatever signal is available along with a clear error indicating what is missing.

## Source layout

The four source files map directly onto the pipeline stages:

| File | Responsibility |
|------|---------------|
| `src/attune_rag/dashboard/refresh.py` | `build_snapshot()` — benchmark the corpus and emit the snapshot dict |
| `src/attune_rag/dashboard/render.py` | `render()` — write the HTML report with the snapshot embedded |
| `src/attune_rag/dashboard/show.py` | `display()` — print the snapshot to the terminal via Rich |
