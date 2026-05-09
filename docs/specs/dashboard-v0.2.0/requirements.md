# Spec: Shipped Dashboard Template (attune-rag)

**Status**: complete

> Originally `docs/specs/dashboard-v0.2.0.md`. Target version: 0.2.0; **feature actually landed early in 0.1.6**. Implementation diverged from locked scoping decisions — see "Drift from original plan" below. Original design intent is preserved in this spec for historical record; the design.md section flags exactly what shipped differently.
>
> - **Owner:** Patrick
> - **Created:** 2026-04-22
> - **Reconciled:** 2026-04-24

---

## Phase 1: Requirements

**Status**: complete

### Problem statement

Today, the only working live benchmark/freshness dashboard for attune-rag lives at `attune-ai/scripts/attune_rag_dashboard_refresh.py` and is coupled to the attune-ai repo layout. Users of attune-rag who don't have the attune-ai checkout cannot render a dashboard for their own corpus.

The honest fix is to promote the dashboard into attune-rag itself as first-class functionality:

- A reusable Cowork-compatible HTML dashboard ships as part of the attune-rag package.
- Any attune-rag user can render the dashboard with a single CLI call (`attune-rag dashboard render`).
- The dashboard file is self-contained and works in any Cowork session that has access to the installed attune-rag package.
- The attune-ai-side script is retired, eliminating the duplicate copy.

### Scope

**In scope:**

- New `dashboard` subpackage in `src/attune_rag/dashboard/` (matches attune-rag's existing `corpus/`, `providers/`, `eval/` style).
- HTML template shipped as package data (loadable via `importlib.resources`).
- CLI surface: `attune-rag dashboard render` (writes a self-contained HTML file) and `attune-rag dashboard refresh` (emits one JSON snapshot to stdout).
- Snapshot shape preserved unchanged from the existing attune-ai script — JS in the template needs no changes.
- Path-traversal and null-byte validation on `--out`.
- Golden tests + unit tests for render, refresh, and CLI.
- Version bump, changelog, README usage example.
- M5 (post-merge): retire the attune-ai-side script; switch the existing artifact to invoke the installed CLI.

**Out of scope (Non-Goals):**

- React / framework dependency. Plain HTML + vanilla JS + Chart.js UMD from CDN.
- Server. The dashboard is a static file; it hits Cowork's MCP bridge directly from the browser.
- Multi-corpus comparison view. One corpus per dashboard.
- Persistence beyond what Cowork already provides for its artifacts. Snapshots are ephemeral per refresh.
- Auth / access control — inherits whatever the Cowork session has.
- Public Python API. `attune_rag.dashboard` is internal; CLI is the only documented entry point.

### User stories

1. *As an attune-rag user*, I want one command (`attune-rag dashboard render --out ~/dash.html`) that produces a self-contained HTML file I can open in any browser — so I can see retrieval and freshness metrics for my corpus without cloning attune-ai.
2. *As a developer demoing attune-rag*, I want the dashboard to be portable (shareable via attachment, viewable offline) — so I can paste it into a ticket or send it to a colleague without telling them to install Cowork.
3. *As an existing attune-ai user*, I want the existing artifact to keep working through the transition — so my muscle memory doesn't break the day v0.2.0 (or 0.1.6) lands.
4. *As an attune-rag maintainer*, I want one canonical implementation in the package rather than two copies in two repos — so future changes don't drift.

### Edge cases & open questions

| Question / Edge case | Resolution |
|---|---|
| `queries.yaml` missing for the corpus | Refresh emits a partial snapshot with `retrieval.error` populated; CLI exits 1 (still valid JSON). |
| Path-traversal attempt in `--out` (e.g. `/etc/...`) | Reject with exit code 2 and a clear error message (not a raw `OSError`). Reuses the pattern from `attune-ai/.claude/rules/attune/coding-standards-index.md`. |
| Null byte in `--out` | Reject. |
| Parent directory of `--out` doesn't exist | Reject with a clear error message. |
| structlog console output mixed with snapshot stdout | The dashboard's JS parses from the first `{` to skip noise — convention preserved from attune-ai. |
| User opens the rendered file outside Cowork | Static HTML loads, but refresh button has no MCP bridge. Dashboard degrades gracefully — last-baked snapshot still visible. |
| Forward-compatibility with future snapshot fields | Template renders unknown sections as "n/a". |
| Should `dashboard render` auto-open the file? | Open question at spec time; **answered YES in shipped impl** — added `--open` flag (see drift note in design.md). |
| Should there be a deterministic-snapshot mode for demo screenshots? | Skip unless a second use case materializes. |

### Affected layers

- [x] attune-rag — new `dashboard` subpackage; new CLI subcommand; template as package data; tests + version bump
- [x] attune-ai — M5 only: delete `scripts/attune_rag_dashboard_refresh.py`, update Cowork artifact to invoke `attune-rag dashboard refresh`
- [ ] attune-gui — none
- [ ] attune-help — none
- [ ] attune-author — none
