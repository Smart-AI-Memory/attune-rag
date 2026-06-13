---
type: troubleshooting
name: dashboard-troubleshooting
feature: dashboard
depth: troubleshooting
generated_at: 2026-06-10T06:08:44.504866+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Troubleshoot dashboard

## Before you start

The dashboard is a three-stage pipeline: **refresh → render → show**.

- `refresh` calls `build_snapshot()` to benchmark the corpus and return a snapshot dict. If `queries.yaml` is missing, it returns a partial snapshot that includes an error key rather than raising.
- `render` calls `render()` to write a self-contained HTML file with the snapshot embedded as JSON.
- `show` calls `display()` to pretty-print the snapshot in your terminal using Rich.

Each stage is independent. When something goes wrong, narrow the failure to one stage before investigating further.

## Symptom table

| If you observe | Check |
|---|---|
| `build_snapshot()` returns a partial dict | Look for an `error` key in the returned dict — this indicates `queries.yaml` was not found at the expected path. Pass an explicit `queries_path` argument to override the default location. |
| `render()` writes an HTML file but the snapshot is missing or shows a placeholder | Confirm that `_SENTINEL_SNAPSHOT` (`__ATTUNE_SNAPSHOT__`) and `_SENTINEL_TITLE` (`__ATTUNE_TITLE__`) were replaced in the output. If you see those literal strings in the HTML, the snapshot dict was empty or `None`. |
| `display()` produces no output | Check whether you passed a `console` argument pointing to a non-visible target (for example, a `Console` with `stderr=True` when you are watching stdout). Omit `console` to let `display()` create a default Rich console. |
| `main()` returns a non-zero exit code | Run the corresponding entry point directly and inspect the printed output — `main()` in `dashboard.refresh` returns `0` on success, so any other value indicates an unhandled error in `build_snapshot()`. |
| Intermittent failures across runs | Check for environment drift: the `corpus_package` argument defaults to `'attune_help'`; if the package is not installed or has been reinstalled at a different version, snapshot contents will change. |

## Diagnosis steps

Work through these in order — each step is cheaper than the one that follows it.

1. **Identify which stage failed.**
   Run each stage in isolation using its minimal signature:

   ```python
   from attune_rag.dashboard.refresh import build_snapshot
   snapshot = build_snapshot()          # uses 'attune_help' corpus by default
   print(snapshot)
   ```

   If `snapshot` contains an `error` key, the failure is in `refresh`. If `snapshot` looks correct but the HTML is wrong, the failure is in `render`. If both are correct but nothing appears in the terminal, the failure is in `show`.

2. **Check for a missing or mislocated `queries.yaml`.**
   `build_snapshot()` returns a partial snapshot (instead of raising) when `queries.yaml` cannot be found. Inspect the returned dict for an `error` key, then pass an explicit path:

   ```python
   from pathlib import Path
   from attune_rag.dashboard.refresh import build_snapshot

   snapshot = build_snapshot(queries_path=Path("/your/path/to/queries.yaml"))
   ```

3. **Verify the rendered HTML sentinels were replaced.**
   Open the output file from `render()` in a text editor and search for `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__`. Their presence means the snapshot dict was not serialized correctly before being passed to `render()`.

4. **Confirm the corpus package is importable.**
   Run `python -c "import attune_help"` (or substitute your `corpus_package` value). An `ImportError` here means the package is not installed in the active environment, which will cause `build_snapshot()` to fail silently or return partial data.

5. **Run the targeted test suite.**
   ```
   pytest -k "dashboard" -v
   ```
   A failing test that exercises the same path as your symptom will show you the expected inputs, outputs, and any fixtures that isolate the behavior.

## Common fixes

**`queries.yaml` not found**
On a pip install this is expected: the golden query sets live in the attune-rag repo checkout (`tests/golden/`), not the published wheel. Run from a clone, or pass the path explicitly to `build_snapshot()`:
```python
snapshot = build_snapshot(
    corpus_package='attune_help',
    queries_path=Path("/absolute/path/to/queries.yaml"),
)
```

**Corpus package not installed**
Install it in the current environment:
```
pip install attune-help
```
Or verify what is installed:
```
pip show attune-help
```

**HTML output contains literal sentinel strings**
Ensure you pass a fully-populated snapshot dict — not `{}` or `None` — to `render()`:
```python
from attune_rag.dashboard.refresh import build_snapshot
from attune_rag.dashboard.render import render
from pathlib import Path

snapshot = build_snapshot()
assert "error" not in snapshot, f"Snapshot is partial: {snapshot}"
render(out=Path("dashboard.html"), snapshot=snapshot)
```

**`display()` output goes to the wrong stream**
Omit the `console` argument to use the default Rich console, or construct one explicitly targeting the stream you want:
```python
from rich.console import Console
from attune_rag.dashboard.show import display

display(snapshot, console=Console(stderr=False))
```

**Dependency version mismatch**
If the dashboard worked previously but now produces unexpected output without a code change, check whether Rich or the corpus package was updated:
```
pip show rich attune-help
```
Pin to the previously working versions in your environment if needed.

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
