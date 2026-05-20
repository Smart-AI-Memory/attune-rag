---
type: quickstart
name: dashboard-quickstart
feature: dashboard
depth: quickstart
generated_at: 2026-05-20T03:33:38.810936+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# dashboard

Run the attune-rag dashboard against the default corpus with one command:

```bash
attune-rag dashboard
```

This runs the full three-stage pipeline — snapshot → render → display — and exits with code `0` on success.

## How it works

The dashboard pipeline has three stages:

1. **Refresh** — `build_snapshot()` benchmarks your corpus and returns a snapshot dict. If `queries.yaml` is missing, it returns a partial snapshot with an error field instead of raising.
2. **Render** — `render(out, snapshot)` writes a self-contained HTML file to `out` with the snapshot embedded as JSON.
3. **Display** — `display(snapshot)` pretty-prints the snapshot to the terminal using Rich.

## Run the pipeline in Python

```python
from pathlib import Path
from attune_rag.dashboard.refresh import build_snapshot
from attune_rag.dashboard.render import render
from attune_rag.dashboard.show import display

snapshot = build_snapshot(corpus_package="attune_help")
render(out=Path("dashboard.html"), snapshot=snapshot)
display(snapshot)
```

You should see Rich-formatted output in your terminal and a `dashboard.html` file written to the current directory.

## Use a custom corpus or queries file

```python
from pathlib import Path
from attune_rag.dashboard.refresh import build_snapshot

snapshot = build_snapshot(
    corpus_package="my_corpus",
    queries_path=Path("path/to/queries.yaml"),
)
```

If `queries_path` does not exist, `build_snapshot()` returns a partial snapshot — check for an `"error"` key in the result before passing it downstream.

## Expected output

A successful `display()` call prints a Rich table to the terminal. A successful `render()` call produces an HTML file whose source contains the embedded snapshot where the placeholder `__ATTUNE_SNAPSHOT__` was replaced with the snapshot JSON.

```
dashboard.html  ← self-contained HTML report
<Rich table>    ← terminal summary
```

---

**Next:** Open `dashboard.html` in a browser to review the full rendered report.
