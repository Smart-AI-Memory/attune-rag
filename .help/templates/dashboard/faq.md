---
type: faq
name: dashboard-faq
feature: dashboard
depth: faq
generated_at: 2026-05-20T03:33:38.808684+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard FAQ

## What is the dashboard?

The dashboard is a living-docs tool for a registered corpus. It runs a three-stage pipeline: **refresh** benchmarks the corpus and emits a JSON snapshot, **render** packages that snapshot into a self-contained HTML report, and **show** pretty-prints the snapshot in your terminal using Rich. Each stage has its own entry point under `attune-rag dashboard`.

## When should I use it?

Use the dashboard when you want to check the health or freshness of a registered corpus — for example, to see how well your RAG pipeline is answering benchmark queries, or to share a snapshot report with teammates as HTML. If you only need to query or update the corpus itself, look at the other features listed in `.help/features.yaml`.

## Which function should I call first?

That depends on what you want to produce:

- **`build_snapshot(corpus_package, queries_path)`** — call this first to run the benchmark and get a snapshot `dict`. If `queries.yaml` is missing, it returns a partial snapshot with an error key rather than raising.
- **`render(out, snapshot, title)`** — call this after `build_snapshot()` to write a standalone HTML file to `out` with the snapshot embedded as JSON.
- **`display(snapshot, console)`** — call this instead of `render()` if you want a Rich terminal view rather than an HTML file.

Each function's signature and return type are documented in the source files listed below.

## What happens if `queries.yaml` is missing?

`build_snapshot()` returns a partial snapshot `dict` that includes an error description rather than raising an exception. Check for an error key in the returned dict before passing the snapshot to `render()` or `display()`.

## How do I debug the dashboard?

Start by running the relevant tests:

```
pytest -k "dashboard" -v
```

If the tests pass but your code still fails, add a `logger.debug` statement at the point where the snapshot is built or rendered, then re-run with logging enabled. For known failure modes, see the troubleshooting page for this feature.

## Where are the source files?

| File | Responsibility |
|---|---|
| `src/attune_rag/dashboard/__init__.py` | Package init |
| `src/attune_rag/dashboard/refresh.py` | `build_snapshot()`, `main()` |
| `src/attune_rag/dashboard/render.py` | `render()` |
| `src/attune_rag/dashboard/show.py` | `display()` |

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
