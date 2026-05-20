---
type: task
name: dashboard-task
feature: dashboard
depth: task
generated_at: 2026-05-20T02:45:03.110698+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Work with the dashboard

Use the dashboard when you need to inspect the health of a registered corpus by running the three-stage pipeline: **refresh** benchmarks the corpus and emits a snapshot JSON, **render** packages that snapshot into an HTML report, and **show** pretty-prints the snapshot to the terminal via Rich.

## Prerequisites

- Access to the project source code
- Read access to the files under `src/attune_rag/dashboard/`

## Identify which stage to work in

The pipeline has three independent stages, each with its own entry point under `attune-rag dashboard`. Choose the stage that owns the behavior you need:

| Stage | File | Key functions |
|---|---|---|
| Refresh | `src/attune_rag/dashboard/refresh.py` | `build_snapshot()`, `main()` |
| Render | `src/attune_rag/dashboard/render.py` | `render()` |
| Show | `src/attune_rag/dashboard/show.py` | `display()`, `main()` |

## Run the dashboard pipeline

1. **Run the refresh stage** to benchmark the corpus and produce a snapshot:

   ```
   attune-rag dashboard refresh
   ```

   `build_snapshot()` queries the corpus package (default: `attune_help`) and writes a snapshot dict. If `queries.yaml` is missing, it returns a partial result with an error field — check the output for that field before continuing.

2. **Run the render stage** to produce an HTML report from the snapshot:

   ```
   attune-rag dashboard render
   ```

   `render()` writes the HTML file to the path you specify via `--out` and embeds the snapshot as JSON using the `__ATTUNE_SNAPSHOT__` sentinel.

3. **Run the show stage** to inspect the snapshot in your terminal:

   ```
   attune-rag dashboard show
   ```

   `display()` pretty-prints the snapshot using Rich. Pass a `Console` instance via `--console` to redirect output.

## Modify a stage

1. **Open the file for the stage you identified** in the table above. Read the target function's docstring, parameters, and return type to confirm it owns the behavior you need.

2. **Edit the function.** Keep naming conventions, error-handling style, and logging consistent with the rest of the file.

3. **Run the dashboard tests** to catch regressions before they affect other developers:

   ```
   pytest -k "dashboard"
   ```

## Verify success

- The refresh stage exits without an `error` key in the snapshot dict.
- The render stage writes a valid HTML file to the path you specified, with the snapshot JSON embedded (search the file for `__ATTUNE_SNAPSHOT__` to confirm).
- The show stage prints a formatted snapshot table to the terminal with no traceback.
- `pytest -k "dashboard"` reports zero failures.
