# Telemetry — risks

> **Status:** scoping memo — risk register, not a mitigation plan.
> Each risk below has a posture (preserve / accept / mitigate) at
> memo time. Real mitigations are implementation-time work.

## 1. Latency regression on the retrieval hot path

**Risk:** A naive synchronous JSON-encode-and-write on every
retrieval call adds measurable latency to the user-visible
response. At the current locked perf baseline (~0.0006s for
`rag_pipeline_run.cpu`), a 1ms write *doubles* the measurement.

**Posture:** preserve strict-dominance. The
[alias-expansion-sweep](../alias-expansion-sweep/) shipped 13 PRs
without a single baseline regression; telemetry must hold the same
bar.

**Mitigation surface (implementation-time):**

- 1ms budget enforced by the perf gate.
- JSON encode + write happens *after* the retrieval result is
  returned to the user-rendering layer, not before.
- Failure-soft: a write error disables telemetry for the rest of
  the session, never propagates to the user.
- Escape hatch: if 1ms can't hold, switch to a bounded `queue.Queue`
  + drain thread. Documented in [design.md §5](design.md) as the
  fallback shape.

**Dependency:** the [perf-baseline-multi-run](../perf-baseline-multi-run/)
methodology being live and σ back at 2.0. If telemetry ships
against a σ=3.0 baseline, the gate is wide enough to swallow a real
regression. See [open-questions.md §8](open-questions.md).

## 2. Retrieval-quality regression via behavior change

**Risk:** A clever-but-wrong implementation reads the
telemetry-driven popularity ordering and uses it to *modify*
retrieval behavior at runtime (e.g. boost frequently-queried
templates). This makes telemetry a retrieval-correctness surface
instead of an observation surface, which means a bug in telemetry
becomes a bug in retrieval.

**Posture:** mitigate by construction. **Telemetry is read-only
with respect to the retrieval pipeline at runtime.** The feedback
loop is offline:

```
runtime: query → retrieve → render → log
offline: read logs → (human) → author aliases_override.json
runtime (next release): retrieve uses the new aliases
```

No telemetry data influences a live retrieval call. Period. This
is the property that keeps telemetry strict-dominant on retrieval
quality.

If a future feature wants runtime telemetry-driven behavior (e.g. a
"personalized alias union"), it gets its own spec and its own
strict-dominance argument. This memo does not enable it.

## 3. Privacy regret — irreversible disclosures

**Risk:** A user enables telemetry, forgets, and accumulates 18
months of `qhash` + (eventually) `qtext` records. Then their
laptop is compromised, or the data is shared inadvertently, or the
user changes their mind about the privacy posture.

**Posture:** accept, mitigate at config level.

**Why "accept":** by definition, a tool that logs anything has this
risk. The mitigation is consent at enable time + ability to delete
at any time. Both are present:

- Consent: the first-run prompt + the `enabled: false` default.
- Deletion: `rm ~/.attune-rag/telemetry/queries-*.jsonl` is the
  full uninstall. Documented in the prompt and the README.
- Default retention: 90 days, with monthly file rotation making
  "older than N months" trivially deletable.

**Why "irreversible disclosures" matter more than the volumetric
risk:** the difference between "you logged 1k queries" and "you
logged 1M queries" doesn't change the regret much; the difference
between "you logged hashes" and "you logged plaintext" changes it
massively. Hence the two-toggle structure in [design.md §2](design.md).

## 4. Privacy regret — the cloud-upload trapdoor

**Risk:** Local-only telemetry feels safe; users opt in. Later, a
cloud-upload feature is added. Users who opted into local-only are
auto-migrated, or the consent UI is ambiguous, and queries that
were intended to stay local end up uploaded.

**Posture:** mitigate by explicit non-extension.

- Cloud upload is explicitly **not** in v1 scope ([design.md §1](design.md)).
- The local-only opt-in is for *local-only* telemetry. Any future
  cloud feature is a **separate** opt-in. No auto-migration.
- [open-questions.md §4](open-questions.md) records this so future
  readers don't accidentally treat cloud upload as a minor
  extension.

This is the single most important risk in this memo. Cloud upload
being "obviously the next step" is exactly the kind of seam where
consent gets compressed. Calling it out explicitly is how the
mitigation works.

## 5. Privacy regret — `hit_paths` semantic leak — MITIGATED 2026-05-21

**Risk:** A user's hashed queries gain semantic content via their
correlated `hit_paths`, particularly if the user has customized
their corpus to include sensitive private templates (e.g. a
`concepts/internal-secret-project` template).

**Resolution:** [open-questions.md §9](open-questions.md) chose to
ship the escape hatch in v1: a `log_hit_paths` config knob
(default `true`). Users with sensitive custom corpora set it to
`false` and `hit_paths` is omitted from emitted records.

**Mitigation surface (now v1, not v2):**

- README documents "telemetry assumes a non-sensitive corpus,
  flip `log_hit_paths` to `false` if yours isn't".
- `log_hit_paths: true|false` config knob in the v1 `telemetry`
  block.
- `alias_union_size` (scalar diagnostic) is still recorded when
  `log_hit_paths: false`; only the template path list is
  suppressed.

## 6. Scope creep — telemetry becomes "the analytics product"

**Risk:** Telemetry starts as "log queries to inform alias
authoring", grows to "log queries + retrieval timing + faithfulness
scores + …", and ends up needing its own backend, dashboard, retention
policy, GDPR/CCPA stance, etc.

**Posture:** mitigate by spec discipline.

- This memo scopes *one* signal: distinct-query distribution with
  retrieval-result diagnostics.
- Additions get their own scoping memos. The on-disk schema in
  [design.md §1](design.md) is versioned (`retriever_version`
  field; could add a `schema_version`) so additions don't break
  readers.
- The implementation steps in [design.md §6](design.md) end at
  "close the alias-authoring loop". Past that, additional signals
  require new approval, not just an implementation PR.

## 7. Maintainer-burden — analysis tooling becomes a tax

**Risk:** The read-back tool (`scripts/analyze_telemetry.py`) needs
ongoing maintenance; users send reports of it not handling their
log format; the maintainer ends up debugging telemetry instead of
shipping retrieval improvements.

**Posture:** accept; bound by simplicity.

- The read-back tool is a single `scripts/analyze_telemetry.py`,
  not a service. Maintenance cost is roughly equivalent to
  `scripts/measure_perf_baseline.py` today.
- The on-disk format is JSONL — readable by `jq`, `pandas`, or 20
  lines of Python. If the maintenance burden grows, users can drop
  to raw `jq` queries instead of waiting on tooling updates.

## 8. The synthetic paraphrase set decays as the live distribution
diverges

**Risk:** Telemetry data reveals that real queries look nothing
like the synthetic paraphrase set. Now we have an 80-query
regression suite that's protecting against a phantom while real
queries are unprotected.

**Posture:** mitigate, but only when data exists.

- Until telemetry data exists, the synthetic suite is the best
  signal we have. Keep it.
- Once telemetry data exists and shows the live distribution, **the
  synthetic suite gains real-query siblings** authored from the
  most-frequent telemetry-observed queries. The watermark moves
  with the data, not with our intuition.
- Worst case: the synthetic suite is retired in favor of a
  telemetry-derived suite. That's a future spec, not this one.

This is the same shape as [open-questions.md §10](open-questions.md) —
we don't yet have a way to know if telemetry is paying for itself.
Once we do, suite composition becomes a downstream decision.

## 9. Windows users can't enable telemetry — MITIGATED 2026-05-21

**Original risk:** macOS + Linux concurrent-append semantics are OK;
Windows isn't. If we ship "v1 disables on Windows", a class of
users gets a worse experience.

**Resolution:** [open-questions.md §3](open-questions.md) chose a
dual-backend approach — JSONL on macOS/Linux, SQLite on Windows.
Windows users get a working telemetry feature; the read-back tool
unifies both formats.

**Residual risk:** the implementation has two storage paths to
maintain. Mitigated by keeping the SQLite schema as a direct
mirror of the JSONL record fields, and by the read-back tool
loading either backend through a thin adapter.

---

None of these risks block memo approval. The point of the register
is that they're *named* — so if the memo is approved and
implementation begins, the implementation milestones know what they
have to design against.
