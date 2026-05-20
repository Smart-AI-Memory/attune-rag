---
type: concept
name: dashboard-concept
feature: dashboard
depth: concept
generated_at: 2026-05-20T03:33:38.785145+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard

## Overview

The attune-rag dashboard is a living-docs view of a registered corpus that moves data through three stages — snapshot, render, and display — to give you both a shareable HTML report and an at-a-glance terminal summary.

## Three-stage pipeline

Each stage is independent and has its own entry point, so you can run them individually or chain them together.

**1. Snapshot (`refresh`)**
`build_snapshot(corpus_package)` benchmarks the corpus against your `queries.yaml` file and returns a plain Python dict. If `queries.yaml` is missing, the function still returns a partial snapshot that includes an error field rather than raising an exception. This dict is the single source of truth that the other two stages consume.

**2. Render (`render`)**
`render(out, snapshot, title)` takes the snapshot dict, serializes it as JSON, and injects it into an HTML template by replacing the sentinel strings `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__`. The result is a self-contained HTML file written to `out` — no external data files required to open it in a browser.

**3. Display (`show`)**
`display(snapshot, console)` pretty-prints the same snapshot dict to the terminal using [Rich](https://github.com/Textualize/rich). Pass your own `Console` instance to control output destination, or omit it to use the default.

## How the pieces fit together

```
queries.yaml
     │
     ▼
build_snapshot()  ──►  snapshot dict  ──►  render()   ──►  report.html
                                      │
                                      └──►  display()  ──►  terminal
```

The snapshot dict is the interface between stages. Because `render()` and `display()` both accept an arbitrary dict, you can supply a snapshot from any source — not only from `build_snapshot()`.

## Entry points

| Function | Stage | Key behaviour |
|---|---|---|
| `build_snapshot(corpus_package)` | Snapshot | Returns a dict; degrades gracefully on missing `queries.yaml` |
| `render(out, snapshot, title)` | Render | Writes a self-contained HTML file to `out` |
| `display(snapshot, console)` | Display | Pretty-prints to the terminal via Rich |

## When the dashboard matters

Use the dashboard when you need to verify that a corpus is answering its benchmark queries correctly, share a point-in-time report with teammates (use the rendered HTML), or spot freshness regressions quickly without leaving the terminal (use `display`).
