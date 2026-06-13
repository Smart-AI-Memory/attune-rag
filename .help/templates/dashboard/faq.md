---
type: faq
name: dashboard-faq
feature: dashboard
depth: faq
generated_at: 2026-06-10T06:08:44.507915+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard FAQ

## What is the dashboard?

The dashboard is a three-stage pipeline for inspecting the health of a registered corpus: **refresh** runs the benchmark and emits a snapshot dict; **render** produces a packaged HTML report from that snapshot; **show** pretty-prints the snapshot to the terminal via Rich. Each stage has its own entry point.

## What does each stage do?

- **refresh** — `build_snapshot()` queries the corpus and returns a `dict[str, Any]`. If `queries.yaml` is missing, it returns a partial snapshot with an embedded error rather than raising.
- **render** — `render()` writes an HTML file to the path you pass as `out`, with the snapshot embedded as JSON.
- **show** — `display()` prints the snapshot to the terminal. Pass a `Console` instance if you want to redirect output; omit it to use the default Rich console.

## Which function should I call first?

Start with `build_snapshot()` in `dashboard.refresh`. It produces the snapshot dict that both `render()` and `display()` consume. Everything else in the pipeline depends on its output.

## Can I use a custom queries file?

Yes. `build_snapshot()` accepts an optional `queries_path: Path | None` argument. Pass a `Path` to your own `queries.yaml` to override the default. If you omit it (or pass `None`), the function looks for the default queries bundled with the corpus package.

## What happens if queries.yaml is missing?

`build_snapshot()` returns a partial snapshot dict that includes an error description rather than raising an exception. Check the returned dict for an error key before passing it downstream to `render()` or `display()`.

## How do I generate an HTML report?

1. Call `build_snapshot()` to get the snapshot dict.
2. Pass the snapshot to `render(out, snapshot)`, where `out` is the destination `Path`.
3. `render()` returns the `Path` it wrote, so you can open or serve the file immediately.

You can also customize the page heading with the `title` argument (default: `'attune-rag dashboard'`).

## How do I view the dashboard in the terminal?

Call `display(snapshot)` from `dashboard.show`, or run the CLI entry point for the show stage. If you want output routed to a specific destination, pass a Rich `Console` instance as the second argument.

## Where are the source files?

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

## How do I debug a failure?

Run `pytest -k "dashboard" -v` first. If the tests pass but your code still fails, add a `logger.debug` call at the suspected failure point and re-run with logging enabled. For symptom-based diagnosis, see the troubleshooting page for this feature.

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
