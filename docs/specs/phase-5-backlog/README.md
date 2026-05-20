# Phase-5 backlog

Holding area for deferred items surfaced during the Phase 4 freeze
(W2.1 deep-review, W2.2 perf audit, and the inter-run-noise gap
exposed by PRs [#75](https://github.com/Smart-AI-Memory/attune-rag/pull/75)
+ [#77](https://github.com/Smart-AI-Memory/attune-rag/pull/77)).

## Purpose

Capture work that is **out of scope during the freeze** but should
not get lost before Phase 5 opens. Items here are scaffolding —
they become formal specs (or land directly under a Phase-5 spec) once
the v1.0.0 release prep starts. See [ROADMAP-v1.md](../ROADMAP-v1.md)
Phase 5.

## Contents

- [items.md](items.md) — table of 11 deferred items (quality, perf,
  test-audit, methodology) with target files, effort, and the
  reason each was deferred during freeze.
- [dependencies.md](dependencies.md) — what unblocks Phase-5 work.

## When items move from backlog → spec

- The 4-week API freeze closes (Phase 4 W4 complete).
- 0.2.0 cut has landed.
- Phase 5 spec (`docs/specs/v1.0.0-release/`) is created.

At that point: triage `items.md` and either fold items into the
Phase 5 spec's `tasks.md`, promote large items to their own spec
directory under `docs/specs/`, or close as won't-do.

## Ownership

Patrick Roebuck (sole developer). No external review required to
move items in or out; this file is a working backlog, not a
contract.
