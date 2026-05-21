# Telemetry — open questions

> **Status:** scoping memo — questions deliberately left unresolved.
> Each entry below is a decision the memo *could* have picked but
> chose to defer. Resolution gates implementation, not approval of
> the memo itself.

## 1. Is the hash collision-resistant enough at the truncation we propose? — RESOLVED 2026-05-21

**Decision:** use the full sha256 digest (32 bytes / 64 hex chars),
not truncated.

**Rationale:** the ~16-byte saving per log row is negligible at
JSONL scale, and the full digest removes the "revisit truncation
length first" gotcha at the §4 cloud-upload trapdoor. Slightly
over-engineered for the v1 local-only case; well-engineered for
any plausible v2.

**Downstream:** [design.md §2](design.md) updated to specify the
full digest. The "truncated to 32 hex chars" language is replaced.

## 2. Does the hash get a salt, and where does the salt live? — RESOLVED 2026-05-21

**Decision:** no salt. `qhash = sha256(query_normalized)`,
deterministic across installations.

**Rationale:** keeps cross-user popularity measurable *if* §4 ever
makes it relevant. The reidentification risk for distinctive long
queries is real but is mitigated upstream by §4 keeping data
local-only and by [risks.md §3](risks.md) (privacy regret) — not
by salting. A salt would also create a new privacy artifact (the
salt file) whose loss silently breaks dedup history.

**Downstream:** [design.md §2](design.md) is already consistent
(no salt is mentioned, so no edit needed).

## 3. Is the JSONL append safe under concurrent attune-rag invocations? — RESOLVED 2026-05-21

**Decision:** dual backend.

- **macOS / Linux:** JSONL + `O_APPEND`. POSIX guarantees atomic
  writes ≤ `PIPE_BUF` (≥ 4 KB on both); a telemetry row is < 1 KB.
- **Windows:** SQLite. The atomicity story on Windows is murky
  enough that JSONL isn't safe; SQLite's library-level locking is.

Both backends emit logically-identical records (the schema in
[design.md §1](design.md)). The read-back tool ([Q7](#7-whats-the-read-back-tools-ux))
has to read both formats.

**Rationale:** preserves `cat`/`jq`/`rm` inspectability on the
primary platforms (macOS is Patrick's; Linux is most of the user
base). Windows users get a working feature instead of a disabled
one, at the cost of a less-inspectable backend. Implementation
work is bounded: SQLite's emission is a one-row INSERT.

**Downstream:**
- [design.md §1](design.md) updated to specify dual backends.
- [risks.md §9](risks.md) (Windows users can't enable telemetry)
  resolved — replaced with a note that Windows uses SQLite.
- macOS concurrent-writer empirical check still worth running at
  implementation time, but the framework is decided.

## 4. Does cloud upload ever enter scope, and if so, how? — RESOLVED 2026-05-21

**Decision:** known-deferred. Cloud upload is a possible future
extension but requires its own scoping memo before any
implementation work. **No auto-migration** from the local-only
opt-in to any future cloud opt-in — they are categorically
separate consent surfaces.

**Rationale:** locks in [risks.md §4](risks.md) (the cloud-upload
trapdoor) as policy, not just a flag. The local-only opt-in this
memo proposes is permanently scoped to local-only; a future cloud
feature reuses none of its consent.

**Open considerations for the future memo (not for resolution
here):** maintainer aggregate-access need, consent model,
operational cost (PostHog / Plausible / custom backend), payload
shape, unsubscribe / data-deletion semantics.

**Downstream:** [README.md](README.md) and [design.md §1](design.md)
already state cloud upload is out of scope. No edit needed; the
resolution makes the existing language load-bearing.

## 5. What about telemetry for the *non-retrieval* surfaces? — RESOLVED 2026-05-21

**Decision:** per-repo scope. This memo is attune-rag-only.
attune-help, attune-author, and any other attune-family surface
gets its own sibling scoping memo if/when the question is asked
there.

**Rationale:** the namespacing was already designed for this
(storage at `~/.attune-rag/telemetry/`, `telemetry` config block
in the repo-local `attune.config.json`). Each repo owns its own
opt-in semantics and data lifecycle, with no cross-repo
coordination required for v1.

**Note for future readers:** if multiple repos ship telemetry
independently and a need for a shared aggregation tool emerges,
that's a *fourth* sibling memo (`docs/specs/telemetry-cross-repo/`
in whichever repo hosts it), not a retrofit to any existing
per-repo design.

## 6. Does the first-run prompt need a localization story? — RESOLVED 2026-05-21

**Decision:** English-only for v1; **localization is a planned
future-version feature**, not a maybe-someday.

**v1 shape:** the prompt string and any telemetry-related CLI
output are plain English literals — no `t(...)` wrapper. This
matches the rest of the attune-rag CLI, which has no i18n surface
today.

**Future-version commitment:** when localization lands as a
project-wide effort, the telemetry surfaces are explicitly in
scope, not a stragglers retrofit. Concretely that means at least:

- The first-run prompt string ("Enable telemetry? [y/N]" and the
  one-line explanation that precedes it).
- Any `analyze_telemetry.py` human-readable output (the
  `--format table` path; JSON output stays locale-neutral).
- Config-related error messages (e.g. "telemetry disabled: write
  failed", "log_query_text requires enabled: true").

**Implication for v1 design:** keep the telemetry strings
centralized in one module (a small `_strings.py` or equivalent)
so the future i18n pass has a single file to wrap, rather than
hunting strings scattered across the emission path, prompt code,
and analysis tool. This is essentially free at v1 build time and
removes the most expensive part of the future retrofit.

**Note for whoever opens the i18n effort:** the
[`docs/specs/telemetry/`](.) directory's own contents (this memo)
stay English. The localization scope is the *runtime user-facing
strings*, not the engineering documentation.

## 7. What's the read-back tool's UX? — RESOLVED 2026-05-21

**Decision:** single `scripts/analyze_telemetry.py` with both JSON
and human-readable table output, and three built-in canned
aggregations:

1. **Top-N by frequency** — most-common distinct queries.
2. **Miss-set** — queries with empty `hit_paths` or
   `alias_union_size` below a threshold (the alias-authoring
   targets).
3. **Template hit distribution** — `hit_paths` rolled up across
   all queries, ranked by appearance.

Flag shape: `--format json|table` (default table),
`--query top|misses|templates` (default top), `--limit N`
(default 20), `--since DATE` (default all-time).

**Privacy at analysis time:** if the user has not opted into
`log_query_text`, output is hashes — explicit and visible, not
hidden. With the second toggle, the script joins `qhash` to
`qtext` and prints text. No silent partial-text fallback.

**Backend handling:** the script loads either JSONL (macOS/Linux)
or SQLite (Windows, see [§3](#3-is-the-jsonl-append-safe-under-concurrent-attune-rag-invocations--resolved-2026-05-21))
via a thin adapter — identical CLI surface across platforms.

**Rationale:** matches the `scripts/measure_perf_baseline.py`
precedent (single script, bounded scope). Three canned
aggregations cover the alias-authoring use case directly; users
who want more can still drop to `jq` against the underlying JSONL.

**Downstream:** [design.md §6](design.md) step 4 already names the
script. Add the canned aggregations to the description there as
part of the next round of edits if the memo is approved as-is.

## 8. Does the feature ship before or after `perf-baseline-multi-run`? — RESOLVED 2026-05-21

**Decision:** telemetry implementation ships **after**
[`perf-baseline-multi-run`](../perf-baseline-multi-run/) M2 lands
and σ is back at 2.0.

**Rationale:** the 1ms latency claim is only defensible against a
gate that isn't already inflated to swallow real regressions.
Strict-dominance discipline (the same one that protected the
13-PR alias-expansion-sweep) requires the gate be honest about
inter-runner noise before adding a new latency consumer.

**Practical effect:**
- Telemetry config surface (the `attune.config.json` `telemetry`
  block, always-disabled stub) **may** still ship inside the
  v0.2.0 window if helpful to reserve the API, but the emission
  code does not.
- The full emission + read-back implementation is Phase 5+ work,
  gated on the perf-baseline methodology rolling forward.

**Downstream:** [README.md](README.md) already says "not a v0.2.0
blocker". This decision makes the sequence explicit:
`perf-baseline-multi-run M2` → telemetry emission.

## 9. Is `hit_paths` privacy-equivalent to `qhash`? — RESOLVED 2026-05-21

**Decision:** add a `log_hit_paths` config knob in **v1**, default
`true`.

**Rationale:** the sensitive-custom-corpus case is rare but not
hypothetical, and the escape hatch is cheap to ship now. Deferring
to v2 would mean users with private corpora can't opt into
telemetry safely until then — a needless friction for a knob that's
one boolean.

**Config:**
```jsonc
{
  "telemetry": {
    "enabled": false,
    "log_query_text": false,
    "log_hit_paths": true,    // NEW — set false for sensitive custom corpora
    "retention_days": 90
  }
}
```

When `log_hit_paths: false`, the emitted record omits the
`hit_paths` field entirely. `alias_union_size` is still recorded
(it's a scalar diagnostic that doesn't leak template names).

**Downstream:**
- [design.md §4](design.md) config block updated to include
  `log_hit_paths`.
- [risks.md §5](risks.md) (hit_paths semantic leak) shifts from
  "accept for v1" to "mitigated via config knob".

## 10. How do we know when telemetry has paid for itself? — RESOLVED 2026-05-21

**Decision:** the success metric is a pair, both measurable from
artifacts already on disk:

1. **N (count of telemetry-driven alias additions).** Number of
   entries in `aliases_override.json` whose commit message or
   adjacent comment cites a telemetry-observed query as
   justification. Tracked by tagging those commits / comments
   with a `[telemetry]` marker so a `grep` can count them.

2. **R@3 trajectory of telemetry-driven additions.** Paraphrased
   R@3 against the regression suite, measured after each
   telemetry-driven addition. Compared to the R@3 of
   judgment-driven additions over the same window.

**Pass criteria:** telemetry pays for itself when **(1) is
non-zero AND (2) is comparable or better than judgment-driven
additions over the same period.**

**Fail mode worth naming:** if (1) is non-zero but (2) is worse,
the telemetry-driven path is producing aliases that overfit to
observed-query phrasing at the cost of generalization. That's a
real failure and one this metric would catch.

**Not in the metric:** time-from-miss-to-fix (cleaner signal but
needs a separate "miss report" log to compare against; can be
added later if N gets large enough to want a velocity story).

**Sequencing:** this metric is operationalized when the read-back
tool ships (Phase 5+). Before that, there's no
telemetry-driven additions to count.

**Downstream:** [design.md §6](design.md) step 5 ("Use the
read-back output to drive the next round of `aliases_override.json`
additions") is now bounded by this metric — additions need the
`[telemetry]` tag for the count to be tractable.

---

## Status

**All 10 questions resolved on 2026-05-21** in a walkthrough with
Patrick. The resolutions are captured inline above; downstream
edits to [design.md](design.md) and [risks.md](risks.md) reflect
the decisions.

The memo's posture going forward:

- **Surface is locked.** The on-disk schema, config block, hash
  shape, dual-backend storage, prompt UX, and success metric are
  all fixed at the level of resolution above.
- **Implementation gate is open.** Next step is a `requirements.md`
  + `tasks.md` if and when Patrick wants the build to proceed,
  which is sequenced **after** [`perf-baseline-multi-run`](../perf-baseline-multi-run/)
  M2 per [§8](#8-does-the-feature-ship-before-or-after-perf-baseline-multi-run--resolved-2026-05-21).
- **Cloud upload is permanently a separate memo.** Per [§4](#4-does-cloud-upload-ever-enter-scope-and-if-so-how--resolved-2026-05-21),
  no auto-migration; a future cloud feature reuses none of this
  consent.
