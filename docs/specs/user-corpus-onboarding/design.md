# Spec: user-corpus-onboarding — design

> **Status:** **scoped 2026-05-22.** All seven open questions are
> decided in [`tasks.md`](tasks.md#scoping-decisions-locked-2026-05-22).
> Sections below remain as the design narrative — the recommended
> options are the locked decisions; alternatives kept as historical
> context for future revisions.

## 1. Quality harness

### 1.1 Shape

Proposed: **CLI + Python API, sharing one implementation.**

CLI entry point (proposed name; pinned at scoping):

```bash
python -m attune_rag.measure_corpus \
  --corpus-path /path/to/markdown/root \
  --queries /path/to/queries.yaml \
  --paraphrased /path/to/queries_paraphrased.yaml \
  --output report.md \
  --watermark-r3 0.85
```

Python API (proposed shape):

```python
from attune_rag.measure_corpus import measure, MeasureResult

result: MeasureResult = measure(
    corpus_path="...",
    queries_path="...",
    paraphrased_path="...",   # optional
    watermark_r3=0.85,        # optional
)
print(result.report_markdown())   # the same text the CLI prints
print(result.p1, result.r3)       # programmatic access
```

The CLI is what users run; the Python API is what attune-gui and
other downstream consumers call. Both must produce **byte-identical**
markdown reports — the CLI is a thin wrapper around the API.

### 1.2 Report shape

The report mirrors
[`docs/specs/release-quality-baseline/baseline-1.md`](../release-quality-baseline/baseline-1.md):

- Header: corpus identifier, query-set identifier, harness version,
  retriever-version, timestamp.
- Aggregate table: P@1, R@3, mean reranker faithfulness (if rerank
  enabled), per-difficulty breakdown if `difficulty:` is present in
  the queries YAML.
- Per-query table: query, expected top-1 path, actual top-1 path,
  rank-of-expected, hit-in-top-3 boolean.
- Optional residuals section: queries where rank-of-expected > 3
  (the alias-authoring targets).
- Footer: how to interpret the numbers, link to the user-corpus
  guide.

The report is **the same shape** as the bundled baseline-1.md so
users can compare their corpus's numbers against ours in
apples-to-apples.

### 1.3 Watermark behavior

`--watermark-r3 0.85` (or whatever default the scoping pass picks)
makes the CLI return non-zero exit if aggregate R@3 falls below the
floor. This is what makes it CI-suitable for downstream users — they
can wire it into their own corpus repo's CI without writing assertion
code.

`--watermark-p1` available for the strict case; default off because
P@1 is more brittle than R@3.

### 1.4 What the harness does *not* do

- **Not a query authoring tool.** Users write their own
  `queries.yaml` and `queries_paraphrased.yaml`. The guide
  documents the schema.
- **Not a retriever evaluator.** It measures *the* `RagPipeline`
  default configuration. Comparing two retrievers is out of scope;
  if needed, users can run the harness twice with different
  pipeline configs.
- **Not a faithfulness evaluator.** Mean faithfulness is reported
  if the pipeline includes a reranker (so the harness output is
  comparable to the bundled baseline), but the harness doesn't
  invoke the LLM-as-judge faithfulness scorer — that lives in
  `eval/faithfulness.py` and is opt-in.

### 1.5 Alternatives considered

**Alternative A: Reuse `scripts/measure_baseline_variance.py` directly.**
Just document how to point it at a user corpus. Cost: that script
was authored for noise-floor measurement (N-run variance), not for
single-pass quality measurement. Forcing users through it adds N=20
runs of overhead per evaluation when they just want one number.
**Decision:** reject for the primary path. The harness is a sibling
script (`measure_corpus.py` is to single-pass quality what
`measure_baseline_variance.py` is to noise-floor measurement). The
two share format and report shape; the harness reuses the report-rendering
machinery.

**Alternative B: A `pytest` plugin.** Users mark tests with
`@attune_rag.golden_query("...")` and the plugin runs them. Cost:
locks users into pytest. Cost: harder to use as a CLI. **Decision:**
reject. The pytest path is the *internal* pattern (see
`tests/golden/test_golden.py`); it's available to users who want it
but isn't the primary surface.

**Alternative C: JSON output only, leave markdown to the user.**
Cost: every user re-implements report-rendering. Defeats the
"same shape as our baseline" property that makes the harness
discoverable. **Decision:** reject. JSON is the *machine* output
(`--format json` flag); markdown is the default human output.

## 2. First-class override-from-file for `DirectoryCorpus`

### 2.1 Current state

```python
class DirectoryCorpus:
    def __init__(
        self,
        root: Path,
        *,
        extra_aliases: dict[str, Iterable[str]] | None = None,
        # …
    ): ...
```

Callers wanting the override path must:

```python
import json
data = json.loads(Path("aliases_override.json").read_text())
data = {k: v for k, v in data.items() if not k.startswith("_")}
corpus = DirectoryCorpus(root, extra_aliases=data)
```

`AttuneHelpCorpus` already does this internally; arbitrary callers
have to reimplement it.

### 2.2 Proposed shape

Add a kwarg:

```python
class DirectoryCorpus:
    def __init__(
        self,
        root: Path,
        *,
        extra_aliases: dict[str, Iterable[str]] | None = None,
        extra_aliases_file: Path | str | None = None,  # NEW
        # …
    ): ...
```

Semantics:

- `extra_aliases_file=None` (default) — current behavior, no change.
- `extra_aliases_file="path.json"` — load + parse + filter
  underscore-prefixed keys + merge with `extra_aliases` (file +
  inline both supported; inline wins on collision for ergonomics).
- Missing file → `FileNotFoundError` at construction.
- Malformed JSON → `ValueError` with the file path in the message.
- Schema validation: same as the current `extra_aliases` (each value
  is `Iterable[str]`; non-string values rejected).

### 2.3 Schema for the file

Mirror `aliases_override.json` shape verbatim:

```jsonc
{
  "_comment": "human notes; underscore-prefixed keys are ignored",
  "_format": "{ rel_path: [alias, alias, ...] }",
  "concepts/some-template.md": [
    "first alias",
    "second alias",
    "multi-token alias"
  ],
  "tasks/another-template.md": ["..."]
}
```

The shape is documented in the user-corpus guide; the harness
report includes a "your overrides" section listing what was loaded.

### 2.4 Symmetry with `summaries_override.json`

For full symmetry, the same scoping pass should consider whether to
add `extra_summaries_file=` alongside `extra_aliases_file=`. The
parallel mechanism exists internally in `AttuneHelpCorpus`. **Decision
deferred:** the alias mechanism is the load-bearing one for the
sweep's lever; summaries can follow in v1.1.0 if usage justifies.

### 2.5 Alternatives considered

**Alternative A: Helper function, not a kwarg.**
`from attune_rag.corpus import load_aliases_from_file; corpus = DirectoryCorpus(root, extra_aliases=load_aliases_from_file("path"))`.
Cost: still asymmetric with `AttuneHelpCorpus`'s clean construction;
just moves the load step out by one frame. **Decision:** reject as
the primary path. The helper function ships *also* (as a public
function in `attune_rag.corpus.__init__`'s `__all__`) for users who
want the loader without the constructor sugar, but the kwarg is the
documented entry point.

**Alternative B: Auto-discovery (look for `aliases_override.json`
next to the corpus root).**
Cost: implicit behavior is a known footgun (the alias-expansion
sweep landed an explicit-only mechanism specifically to avoid this).
**Decision:** reject. Explicit kwarg only.

## 3. "Your own corpus" guide

### 3.1 Location

Proposed: top-level [`docs/USER_CORPUS_GUIDE.md`](../../USER_CORPUS_GUIDE.md).

Linked from:
- The package `README.md` "Public API" section.
- The `DirectoryCorpus` docstring.
- The harness `--help` output footer.
- `docs/POLICY.md` (the v1.0.0 commitment section — see the
  forthcoming POLICY.md behavioral-commitment paragraph).

### 3.2 Outline

```
# Building a quality retrieval corpus for attune-rag

## 1. Corpus structure
   1.1 Directory layout (concepts/, tasks/, references/, etc. — or
       your own taxonomy)
   1.2 File naming
   1.3 Frontmatter schema (link to editor/template_schema.json)

## 2. Frontmatter aliases
   2.1 What aliases are (the K from "K-alias-token-overlap")
   2.2 Multi-token alias intent (why MIN_ALIAS_OVERLAP=2 by default)
   2.3 Authoring patterns: feature names, common phrasings,
       error-message snippets
   2.4 The _tokenize() validation step (the bites→bit lesson)

## 3. The override file pattern
   3.1 When to use overrides vs frontmatter
   3.2 The override-then-promote workflow (override file is
       tactical; frontmatter is strategic)
   3.3 Example aliases_override.json (the schema above)
   3.4 Trimming overrides after promotion

## 4. The MIN_ALIAS_OVERLAP knob
   4.1 Default value and rationale
   4.2 When to flip to 1 (corpora without curated multi-token aliases)
   4.3 When to flip higher (very dense alias sets where overlap=2
       still allows phantom hits)
   4.4 Measuring the trade-off (point at the harness)

## 5. Stemmer gotchas
   5.1 The token pipeline (lowercase → tokenize → stem)
   5.2 The _MIN_STEM_LEN floor
   5.3 Common traps (bites→bit, vulnerabilities→vulnerabilit)
   5.4 The "run _tokenize() before authoring" discipline

## 6. Quality measurement (the harness)
   6.1 Authoring queries.yaml (the schema, the difficulty tiers)
   6.2 Authoring queries_paraphrased.yaml (no-token-overlap variants)
   6.3 Running the harness (CLI + Python API)
   6.4 Reading the report
   6.5 Wiring into CI (the watermark flag)
   6.6 The strict-dominance discipline (validating that any change
       to your corpus's aliases doesn't regress baseline)

## 7. The QueryExpander (when to use it)
   - For corpora *without* curated frontmatter aliases, the
     QueryExpander is a viable alternative path. Cite the D2
     measurement. Link to its re-framed docstring/README section.

## 8. Worked example
   - Walk a hypothetical "tutorials directory" through the full
     flow: corpus structure → first authoring pass → harness run
     → identify a miss → add an alias → re-run → quality lift
     measured.

## 9. The override mechanism's place
   - Pointer to attune-rag's own aliases_override.json — what it
     is, why we use it, when you might want yours to look similar
     (i.e. you're shipping a package that ships with a corpus and
     wants to iterate retrieval quality between releases).
```

This is a substantial document — likely 1000–1500 lines markdown.
The scoping pass should size it explicitly; v1.0.0 may ship a
**leaner v1** with the worked example deferred to v1.1.0 if calendar
pressure demands.

### 3.3 Voice

First-person plural, present tense, concrete. Lessons learned
(`bites → bit`) cited as discoveries from the alias-expansion
sweep — readers should see *how* the discipline was developed, not
just *what* it is. The strict-dominance pattern is named explicitly:
"every change we shipped to attune-help's retrieval ran the full
baseline diagnostic before commit; we recommend the same shape for
your corpus."

## 4. Documentation polish (Phase 5 M4)

After M1–M3 land, the polish step:

- README "Public API" section gets a "Building your own corpus"
  link.
- `DirectoryCorpus` docstring updated to point at the guide + the
  new `extra_aliases_file` kwarg.
- `attune-help`-bundled documentation (the README, the POLICY)
  cross-references the guide so attune-help users who want to
  *also* build a corpus have the path.
- Harness `--help` output points at the guide.
- A short README section under "Quality" that says:
  > "We measure attune-rag's retrieval quality against an 80-query
  > regression set on the bundled attune-help corpus. Paraphrased
  > R@3 is currently 100%, baseline R@3 is 100%. To run the same
  > measurement against your own corpus, see [USER_CORPUS_GUIDE.md](docs/USER_CORPUS_GUIDE.md)."

## 5. Interaction with other Phase 5 work

### 5.1 Sequencing within Phase 5

Proposed order (refined at scoping):

1. **Harness implementation (M1)** — needs no external dependencies.
   First because the guide cites it.
2. **`extra_aliases_file` kwarg (M2)** — independent of M1; can
   land in parallel.
3. **"Your own corpus" guide (M3)** — depends on M1 (cites the
   harness) and M2 (documents the kwarg).
4. **Documentation polish (M4)** — depends on M1, M2, M3.

### 5.2 Sequencing vs `perf-baseline-multi-run`

Both are Phase 5 work; they don't conflict. `perf-baseline-multi-run`
touches the perf-gate plumbing; user-corpus-onboarding touches new
public surface in `attune_rag.corpus` + a new `attune_rag.measure_corpus`
module + a new top-level doc. Independent. The full Phase 5 calendar
runs them in parallel, not sequentially.

### 5.3 Sequencing vs the v1.0.0 cut

The cut spec ([`api-v0.2.0-cut`](../api-v0.2.0-cut/) is the 0.2.0
analogue; the v1.0.0 cut will follow the same shape) ratifies the
public surface that exists at cut time. The new public symbols this
spec introduces — `attune_rag.measure_corpus`, the
`extra_aliases_file` kwarg, possibly a `load_aliases_from_file`
helper — must land *before* the v1.0.0 cut PR. Tasks.md ordering
during scoping reflects this.

### 5.4 Sequencing vs telemetry

Telemetry emission is v1.1.0 work per the Phase 5 scope decision.
This spec is independent — the quality harness measures retrieval
results, not query distribution. They share the same general theme
(observation over the retrieval pipeline) but don't share code.

## 6. Strict-dominance constraint

Per the alias-expansion sweep's discipline: every change in M1–M4
must hold the bundled `attune-help` corpus's baseline metrics
unchanged. The harness must produce **byte-identical** report
numbers on the bundled corpus before and after each implementation
PR — the bundled corpus is the regression test for the harness
itself.

If a refactor moves a number by even ±0.01, that's a real
regression in either the measurement or the retrieval path, and
gets investigated before the PR lands.
