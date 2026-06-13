---
type: concept
name: dashboard-concept
feature: dashboard
depth: concept
generated_at: 2026-06-10T06:08:44.480749+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard

The attune-rag dashboard is a living-docs view of a registered corpus that shows how fresh and accurate the corpus is at any point in time.

## Three-stage pipeline

The dashboard works as a linear pipeline: **refresh → render → show**. Each stage is independent, so you can stop after any step or substitute your own output path.

```
build_snapshot()  →  render()  →  display()
     ↓                  ↓             ↓
 snapshot dict      HTML file     terminal
```

1. **Refresh** — `build_snapshot(corpus_package, queries_path)` runs the benchmark against the corpus and returns a snapshot dictionary. If `queries.yaml` is missing, it returns a partial snapshot with an error field rather than raising, so downstream stages still have something to work with.

2. **Render** — `render(out, snapshot, title)` embeds the snapshot dictionary as JSON into an HTML template and writes the result to `out`. The title defaults to `'attune-rag dashboard'`. The HTML file is self-contained and portable.

3. **Show** — `display(snapshot, console)` pretty-prints the same snapshot to the terminal using Rich. You can pass your own `Console` instance, or omit it to use the default.

Each stage also exposes a `main(corpus_package)` entry point that runs that stage end-to-end from the command line.

## Snapshot as shared currency

The snapshot dictionary returned by `build_snapshot()` is the data structure that connects all three stages. `render()` and `display()` both accept it directly, which means you can build the snapshot once and send it to the HTML report, the terminal, or both — without hitting the corpus twice.

## When the dashboard matters

Use the dashboard when you want to answer "is this corpus still accurate?" without reading every template by hand. Common situations include:

- After updating source files, to see whether indexed answers have drifted from the current code.
- In CI, by calling `build_snapshot()` and inspecting the returned dict for error fields before publishing a release.
- During local authoring, by running the `show` entry point to get a quick terminal summary without generating an HTML file.

## Corpus scope

All three functions default to `corpus_package='attune_help'`. Pass a different package name to point the dashboard at a different registered corpus.
