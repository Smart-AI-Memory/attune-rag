# Spec: Query-distribution telemetry — design

> **Status:** scoping memo — not executable. Each section below
> picks a default and lists the alternatives considered. Defaults
> are the **proposal**, not the decision; approval gate before any
> path is implemented.

## 1. Storage backend

### Proposed default: local file (JSONL on macOS/Linux, SQLite on Windows)

**macOS / Linux:** telemetry writes append to a JSONL file under
`~/.attune-rag/telemetry/queries-YYYY-MM.jsonl` (monthly rotation,
no compression at v1). One file per month keeps `ls` legible and
makes "delete the last 30 days" a single `rm` command.

**Windows:** writes go to a SQLite database at
`~/.attune-rag/telemetry/queries.sqlite` with one row per event.
SQLite's library-level locking gives cross-process safety that
`O_APPEND` doesn't guarantee on Windows. The schema is the JSONL
record's fields as columns; the read-back tool unifies both
backends. Resolved in [open-questions.md §3](open-questions.md).

```jsonc
// one line per distinct query, updated in-place by appending
// "events" that the read-back step folds into a counter.
{ "kind": "query",
  "qhash": "sha256:abcd…",       // hashed by default; see §2
  "ts_first": "2026-05-21T14:03:12Z",
  "ts_last":  "2026-05-21T17:22:04Z",
  "count":    7,
  "k":        3,                  // retrieval k at call time
  "hit_paths": [                  // top-k template paths returned
    "concepts/workflow-orchestration",
    "tasks/refactor-plan",
    "concepts/release-prep"
  ],
  "alias_union_size": 12,         // diagnostic — how many aliases fired
  "retriever_version": "0.1.6+sweep" // git describe at startup
}
```

The on-disk shape is *append-only events* (one event per query
emission); a read-back step folds them into the dedup'd shape above
when read by the analysis tooling. This avoids in-process locking
on the write path — every `attune-rag` invocation just appends.

### Why local file is the default

- **Zero new infra.** Works the moment the toggle flips. No
  Redis dependency, no network, no auth.
- **Privacy by construction.** The data physically lives on the
  user's machine. The "what happens to my data" question has one
  answer: nothing leaves your laptop unless you specifically opt
  into a future cloud-upload step.
- **Inspectable and deletable.** `cat`, `jq`, `rm`. No SDK, no
  client library, no dashboard required to verify what's recorded.
- **Survives offline.** No retry queue, no buffering, no "telemetry
  service degraded" failure mode. The append is local IO; it
  either succeeds or the process is already dying.

### Alternatives considered

**Local + opt-in Redis.** Users who already run `attune-redis` could
mirror writes to Redis for cross-machine aggregation. This is real
value for the Smart-AI-Memory team (Patrick uses Redis already), but
it's a second code path and a second privacy surface. **Decision:**
deferred to a follow-up. Local file is the v1 default; Redis becomes
an opt-in *destination* under the same opt-in toggle later, not a
parallel decision.

**Cloud upload (opt-in).** A `/rag-share-telemetry` command that
batches local logs into an aggregated submission to a hosted
endpoint (PostHog, Plausible, custom Smart-AI-Memory backend). This
is the long-term answer to "how does the maintainer see real
distribution data without each user becoming an expert reviewer of
their own JSONL." **Decision:** explicitly out of scope for this
memo. Listed as a future extension in
[open-questions.md](open-questions.md) §4. Crossing into "data
leaves the user's machine" needs its own consent surface and its
own design memo — not a footnote to this one.

## 2. Privacy posture

### Proposed default: hash by default, full text behind second toggle

The hash is `sha256(query_normalized_lowercased_stripped)`, full
digest (32 bytes / 64 hex chars). The ~16-byte saving from
truncation is negligible at JSONL scale, and the full digest
removes a downstream gotcha if cross-user aggregation ever enters
scope (see [open-questions.md §4](open-questions.md)). Resolved
in [open-questions.md §1](open-questions.md).

Default record contains:

- `qhash` — the hash above.
- `count`, `ts_first`, `ts_last` — counter metadata.
- `k`, `hit_paths`, `alias_union_size`, `retriever_version` —
  retrieval diagnostics (no user-text content).

**Crucially: hashes alone preserve almost all of the analysis value.**
We can measure distinct-query frequency, retrieval-result stability,
miss rate (queries where `hit_paths` doesn't include any
high-quality match), and template hit distribution — all without
ever recording the query text. The only thing hashes lose is the
ability to *read* the queries the user submitted, which is exactly
the property privacy posture is protecting.

### The full-text opt-in (second toggle)

For users who want to help the maintainer with alias authoring —
i.e. who actively want the maintainer to be able to read their
queries — a *second* toggle promotes `qhash` records to `qtext`
records:

```jsonc
{ "kind": "query",
  "qhash": "sha256:abcd…",
  "qtext": "how do I refactor this giant function",  // only with second opt-in
  …
}
```

Two toggles instead of one because the privacy posture of "log
hashed query metadata" and "log raw query text" are categorically
different and should have categorically different consent.

### Retention

Default: **90 days**, with monthly file rotation making expiry
trivial (delete files older than N=3 months). Configurable down to
the current month and up to "never expire". This is a per-user
local-disk question; the only constraint is that the default has to
be defensible if the user forgets they enabled it.

### Alternatives considered

**Full text by default.** Justification: storage is local-only, so
hashing is paranoid friction. **Decision:** rejected. Two reasons.
First, the cloud-upload extension is in our future even if it's not
in v1 scope — and defaulting to hashed-only means the cloud-upload
design starts from a privacy-safe place. Second, "I forgot I had
telemetry on and now my .jsonl has every query I ever asked" is a
regret the local-only argument doesn't actually prevent. Hashes
make the default safe even if the user forgets.

**Defer the call to a follow-up memo.** Justification: privacy
posture deserves its own deliberation. **Decision:** rejected for
the wrong reason. The framing is right but the timing is wrong —
deferring it to a follow-up means we ship the *config surface*
(the toggle name and semantics) without knowing what the toggle
controls. Pick the default here, in the memo where the config
surface is also being scoped.

## 3. Sampling

### Proposed default: log every query, dedup by hash

Every retrieval call emits an event. The read-back tool folds
events with the same `qhash` into a counter (`count`, `ts_last`).
No sampling, no rate limit, no random skip.

For the CLI use case this is bounded by typing speed (~100s of
queries/day for a heavy user), so storage growth is non-issue.

### Why every-query, deduped

- **Distribution data needs distribution.** A 10% sample rate would
  make the distinct-query count an estimator with sampling error,
  which is wasted complexity at this volume.
- **Dedup is free.** The append-only event log naturally collapses
  to distinct queries at read time.
- **No bias.** Sampling decisions (10% of all? 10% of *misses*?
  10% of session starts?) introduce category effects that are very
  hard to reason about. "Every query, with frequency count" is
  unambiguous.

### Alternatives considered

**100% sampling, no dedup (raw event stream).** Justification:
preserves temporal ordering and session correlation; you can answer
"do users retry after a miss?" **Decision:** rejected for v1. The
dedup'd shape can be derived from the raw stream, and the raw
stream's only added information is order — which costs roughly
N-distinct → N storage for a question (session retries) that we
don't yet know is worth asking. Reopen if a future analysis needs
it.

**Configurable sample rate, default 100%.** Justification: a knob
costs nothing if its default is unchanged. **Decision:** rejected as
premature. The knob is config surface that lives forever; we don't
yet know what valid sample rates would even be (10%? 1%? 0.1%?).
Better to ship without the knob and add it if and when a future
user has a real volume problem.

## 4. Rollout

### Proposed default: opt-in via config + one-time first-run prompt

Two surfaces:

1. **`attune.config.json`** (the existing config file, see
   `~/attune.config.json`) gains a `telemetry` block:

   ```jsonc
   {
     "telemetry": {
       "enabled": false,        // master toggle; default off
       "log_query_text": false, // §2's second toggle; default false even when enabled
       "log_hit_paths": true,   // see open-questions.md §9; flip to false for sensitive custom corpora
       "retention_days": 90
     }
   }
   ```

   Absence of the block is treated as `enabled: false`. Existing
   configs are unaffected.

2. **First-run prompt** on the first invocation of an attune-rag
   command after telemetry support ships (detected by absence of
   the `telemetry` block in config). Single y/n prompt, decline
   defaults to disabled, no nag on subsequent runs, *no* prompt for
   the full-text toggle — that one requires explicit config edit.

### Why both surfaces

- Config is the source of truth (machine-readable, version-able,
  inspectable).
- The first-run prompt is a discoverability surface, not a consent
  surface — the consent is in the config edit. The prompt exists
  so users discover the feature exists; the silent default avoids
  it being a privacy footgun even if the user dismisses the prompt.

### Why no first-run prompt for `log_query_text`

The full-text toggle is meaningfully more sensitive than the master
toggle. Prompting for it on first run lumps two different consent
decisions into one UX moment, which is the wrong shape. Better:
master toggle has a prompt; full-text toggle requires the user to
read the config and edit a `false` to a `true`. The friction is
intentional.

### Alternatives considered

**Env-var only (`ATTUNE_RAG_TELEMETRY=1`), no config.** Justification:
smallest possible mechanism for the memo. **Decision:** rejected.
Env vars don't survive shell restarts; they're invisible from the
GUI/dashboard; they're not a sustainable home for a setting that
also has `log_query_text` and `retention_days` siblings. Use env
var as an override for the config (e.g. for CI test runs that need
telemetry off regardless of user config), not as the primary
surface.

**Config-only, no first-run prompt.** Justification: most
conservative; pure opt-in with no UX surface that could be
construed as nudging. **Decision:** rejected as too quiet. The
feature is genuinely useful for the maintainer and genuinely opt-in
on the user side; making it undiscoverable means we'd never get
any data. A one-time prompt that defaults to off on decline is the
honest middle.

**Always-on with a "disable" toggle (opt-out).** Justification: this
is how most CLI tooling does telemetry. **Decision:** rejected
hard. The user explicitly named "never default-on without consent"
as a constraint. Opt-out telemetry is also the kind of decision
that lives forever — once you ship it, you can't take it back
without breaking everyone's existing dataset. Opt-in is the only
defensible default.

## 5. Strict-dominance properties

### Latency

The retrieval hot path must not block on telemetry. The write
strategy is:

- **Single `write()` call**, `O_APPEND`, no fsync, no lock. POSIX
  guarantees small `write()`s to `O_APPEND` files are atomic, so
  no cross-process locking is needed.
- **JSON encoding of the event**, computed off the hot path (the
  hash is computed once at query receipt, before retrieval starts;
  the event row is encoded *after* retrieval returns, before the
  result is rendered to the user).
- **Failure-soft.** If the telemetry write fails (disk full,
  permissions error), it logs once and disables itself for the rest
  of the session. The retrieval result is unaffected.

Budget: <1ms p99 added to the request path. Measured in the
[`perf-baseline-multi-run`](../perf-baseline-multi-run/)
methodology once that's stood up. If it can't fit in 1ms, the
async-queue extension below activates.

### Retrieval quality

Telemetry is read-only with respect to the retrieval pipeline. It
observes `(query, k, hit_paths)` after the retriever returns. There
is no telemetry-driven behavior change at runtime.

This is the property that makes telemetry strict-dominant by
construction: the only way it can regress quality is via the
latency budget (above) or via a privacy-policy-driven removal of
information already in the user's mental model — neither of which
is a retrieval-correctness regression.

### If the 1ms budget doesn't hold

Escape hatch: an in-process queue (`queue.Queue` with a bounded
size) and a background thread that drains it to disk. This adds
complexity (graceful shutdown, queue-full backpressure) and is
explicitly **not** v1's design. The 1ms budget is the simpler-is-
better path; the queue is the fallback if measurement contradicts
the simpler path.

## 6. What gets built, when

This memo proposes the **surface** — the config block, the on-disk
format, the privacy posture. The **implementation** (the emission
code path, the read-back tooling, the prompt UX) is a separate
milestone after approval.

A rough ordering when/if approval lands:

1. Reserve the `attune.config.json` `telemetry` block surface
   (config schema only; emission code returns immediately).
2. Implement the emission path with the 1ms latency budget and
   perf-gate guard.
3. Implement the first-run prompt UX. All user-facing telemetry
   strings (prompt text, error messages, analyzer table headers)
   live in a single centralized module — localization is a
   planned future-version feature per
   [open-questions.md §6](open-questions.md), and a single string
   module is the cheap v1 prep for the future i18n pass.
4. Implement the read-back / analysis tool at
   `scripts/analyze_telemetry.py`. Shape resolved in
   [open-questions.md §7](open-questions.md): `--format json|table`,
   `--query top|misses|templates`, `--limit N`, `--since DATE`.
   Three canned aggregations: top-N by frequency, miss-set
   (low-alias-union or empty hit_paths), template hit
   distribution. Unifies JSONL (macOS/Linux) and SQLite (Windows)
   via a thin loader adapter.
5. Use the read-back output to drive the next round of
   `aliases_override.json` additions, closing the evidence loop.
   Each such addition is tagged `[telemetry]` in its commit
   message so the success metric ([open-questions.md §10](open-questions.md))
   can `grep` the count.

Steps 1–3 are config + emission. Step 4 is the analysis lever.
Step 5 is the feedback loop closing — the actual point of the
exercise.

Not in this memo: a `tasks.md` milestone breakdown. That gets
added if and when this proposal is approved.
