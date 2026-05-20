---
type: troubleshooting
name: dashboard-troubleshooting
feature: dashboard
depth: troubleshooting
generated_at: 2026-05-20T03:33:38.806432+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Troubleshoot dashboard

## Before you start

The `attune-rag dashboard` command runs a three-stage pipeline:

1. **Refresh** — `build_snapshot()` benchmarks the corpus and emits a snapshot dict. If `queries.yaml` is missing, it returns a partial snapshot that includes an error key rather than raising.
2. **Render** — `render()` writes an HTML file to the specified output path with the snapshot embedded as JSON, replacing the `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__` sentinels in the template.
3. **Show** — `display()` pretty-prints the snapshot to the terminal using Rich.

Each stage has its own source file. Knowing which stage failed narrows the search considerably before you look at any code.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `KeyError` or missing fields in the snapshot | Whether `queries.yaml` exists and is on the path passed to `build_snapshot()`. A missing file produces a partial snapshot with an error key, not an exception. |
| HTML output contains literal `__ATTUNE_SNAPSHOT__` or `__ATTUNE_TITLE__` | The snapshot was not injected — confirm `render()` received a non-empty snapshot dict and that the template file is the one distributed with the package. |
| Terminal output is blank or garbled | `display()` was called with an empty snapshot (check the refresh stage first) or the Rich `Console` instance is misconfigured. |
| `main()` returns a non-zero exit code | Run the CLI with `--help` to confirm argument parsing, then reproduce with an explicit `corpus_package` value. |
| Intermittent missing data across runs | Stale corpus registration or a `queries.yaml` that is updated between runs — check which file path `build_snapshot()` resolves at runtime. |

## Step-by-step diagnosis

Work through these steps in order — each one is cheaper than the next.

1. **Identify the failing stage.**
   Run only the stage that shows the symptom:

   ```bash
   # Refresh only — prints the raw snapshot dict
   python -c "from attune_rag.dashboard.refresh import build_snapshot; import json; print(json.dumps(build_snapshot(), indent=2))"

   # Render only — write HTML to /tmp and inspect it
   python -c "
   from attune_rag.dashboard.refresh import build_snapshot
   from attune_rag.dashboard.render import render
   from pathlib import Path
   render(Path('/tmp/dash_debug.html'), build_snapshot())
   "

   # Show only — pretty-print the snapshot
   python -c "
   from attune_rag.dashboard.refresh import build_snapshot
   from attune_rag.dashboard.show import display
   display(build_snapshot())
   "
   ```

2. **Check the snapshot for an error key.**
   `build_snapshot()` returns a partial dict when `queries.yaml` is missing. Inspect the output from step 1 for a top-level `"error"` key before assuming a code defect:

   ```bash
   python -c "
   from attune_rag.dashboard.refresh import build_snapshot
   s = build_snapshot()
   print(s.get('error', 'no error key — snapshot looks complete'))
   "
   ```

   If an error is present, locate `queries.yaml` and pass its path explicitly:

   ```python
   build_snapshot(queries_path=Path('/your/path/to/queries.yaml'))
   ```

3. **Confirm the corpus package is registered.**
   Both `main()` functions default to `corpus_package='attune_help'`. If you are using a different package, pass it explicitly and confirm the package is importable:

   ```bash
   python -c "import attune_help"
   ```

   A `ModuleNotFoundError` here means the corpus package is not installed in the active environment.

4. **Run the dashboard tests.**
   Check which paths are already covered before modifying code:

   ```bash
   pytest -k "dashboard" -v
   ```

   A failing test that exercises your symptom gives you a reproducible baseline and fixtures you can reuse.

5. **Enable debug logging.**
   If the above steps do not surface the problem, enable `DEBUG`-level logging and re-run the failing stage:

   ```bash
   PYTHONPATH=src python -c "
   import logging; logging.basicConfig(level=logging.DEBUG)
   from attune_rag.dashboard.refresh import build_snapshot
   build_snapshot()
   "
   ```

## Common fixes

- **Missing or mislocated `queries.yaml`.**
  Pass the path explicitly to `build_snapshot()`:

  ```python
  build_snapshot(queries_path=Path('/path/to/queries.yaml'))
  ```

  Do not place `queries.yaml` under any path in `_SYSTEM_DIRS` (`/etc`, `/sys`, `/proc`, `/dev`, `/boot`, `/sbin`, `/bin`, `/usr/bin`) — these directories are intentionally excluded.

- **Sentinel strings appear in rendered HTML.**
  The template must contain the exact strings `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__`. If you have customized the template, restore those markers. Then confirm you are calling `render()` and not writing the template file directly.

- **Empty or partial terminal output from `display()`.**
  First confirm `build_snapshot()` returns a complete snapshot (no `"error"` key). If the snapshot is valid but display is still wrong, check whether a custom `Console` instance is redirecting output — omit it to let `display()` create its own:

  ```python
  display(snapshot)   # let display() create the Console
  ```

- **Environment or dependency mismatch.**
  If the dashboard worked previously without a code change, check for a dependency version change:

  ```bash
  pip show rich attune-rag
  ```

  Then compare against the versions in your lock file or `pyproject.toml`. Roll back or pin the offending package as needed. This change is outside the dashboard module itself.

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
