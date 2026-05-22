# Building a quality retrieval corpus for attune-rag

> **Version:** v0 — documents the user-corpus path as of `attune-rag` **0.1.23**.
> v1 (cleaner ergonomics + a first-class measurement harness) ships at
> **v1.0.0** per the [`user-corpus-onboarding`](specs/user-corpus-onboarding/)
> spec. This guide is freeze-compatible: nothing here introduces new
> public API surface. It documents the working-today pattern.

attune-rag is a keyword retrieval framework. The bundled
[`AttuneHelpCorpus`](../src/attune_rag/corpus/attune_help.py) ships with
the attune-help package and is our exemplar — but the framework was
built to point at **your** markdown corpus. This guide walks through
the corpus authoring discipline that produced the bundled corpus's
**P@1=1.00 / R@3=1.00 / paraphrased-R@3=1.00** baseline, so you can
hit similar numbers on your own content.

## Why a guide

The pipeline shape is enumerated in
[README "Public API"](../README.md#public-api). The reproducibility
guarantees are in [`docs/POLICY.md`](POLICY.md). Neither answers the
question that determines whether retrieval actually works: **how do
you author the corpus so users' queries reach the right document?**

That's the authoring discipline. It's not in the API; it's in how
frontmatter aliases get written, when overrides get used vs.
promoted, which knobs get flipped on which corpus shape, and where
the stemmer hides traps. This guide documents it explicitly so you
don't have to reverse-engineer the alias-expansion-sweep PRs to
learn it.

## How to read this guide

- **§§1-2** are the foundation. If you only read two sections, read
  these. They cover the corpus shape and the alias mechanism that
  does most of the work.
- **§§3-5** are the operational knobs — when to use the override
  file pattern, when to flip `MIN_ALIAS_OVERLAP`, what stemmer traps
  to avoid. Read when you hit a specific problem.
- **§6** is the measurement loop — how to know whether your
  changes helped.
- **§§7, 9** are sidebars: when the `QueryExpander` is the right
  lever (§7), and where the override mechanism fits in larger
  shipping patterns (§9).
- The worked example (a hypothetical "tutorials directory" walked
  through the full flow) is deferred to v1.1.0 with iteration based
  on user feedback. v0 ships without it.

## 1. Corpus structure

attune-rag treats a corpus as **a directory tree of markdown
files**. There's no database, no embedding index, no preprocessing
step — just files on disk with frontmatter at the top.

### 1.1 Directory layout

```
my-corpus/
├── concepts/
│   ├── what-is-attune.md
│   ├── retrieval-pipeline.md
│   └── ...
├── tasks/
│   ├── setup-mcp-server.md
│   ├── ship-a-release.md
│   └── ...
├── references/
│   └── ...
└── quickstarts/
    └── ...
```

The top-level directory name (e.g. `concepts/`, `tasks/`,
`references/`) becomes the `category` field on each `RetrievalEntry`.
Files at the corpus root (no subdirectory) get category `""`.
attune-help uses the four categories above; you're free to use
whatever taxonomy fits your content.

### 1.2 File naming

- One markdown file per template / document.
- File extension `.md`. Other extensions are ignored.
- The corpus-relative path (e.g. `concepts/what-is-attune.md`) is
  the document's **stable identity**. Treat it like a slug — once
  shipped, renaming requires a rename-plan (see
  [`editor.plan_rename`](../README.md#public-api) for the supported
  refactor).

### 1.3 Frontmatter schema

Every file should open with a YAML frontmatter block:

```yaml
---
type: concept
name: What is attune
aliases:
  - what is attune
  - attune intro
  - the attune framework
tags: [framework, overview, attune]
related: [retrieval-pipeline]
summary: |
  attune-rag is a keyword retrieval framework over a directory of
  markdown files. The shape is `DirectoryCorpus` → `RagPipeline`.
---

# What is attune

(body content)
```

The full schema lives at
[`src/attune_rag/editor/template_schema.json`](../src/attune_rag/editor/template_schema.json).
The fields that matter for retrieval quality are **`aliases`** (the
load-bearing one — see §2), **`tags`** (categorical filters),
**`related`** (cross-link graph), and **`summary`** (the
short-form text the reranker sees when scoring candidates).

Body text matters too — it gets indexed and the first 1 200
characters get scored for content-overlap — but aliases are where
most authoring effort pays off.

### 1.4 Putting it together

```python
from pathlib import Path
from attune_rag import RagPipeline, DirectoryCorpus

corpus = DirectoryCorpus(Path("./my-corpus"))
pipeline = RagPipeline(corpus=corpus)
result = pipeline.run("How do I set up the MCP server?")

print(result.augmented_prompt)
# Now feed result.augmented_prompt to whatever LLM you use.
```

That's the whole framework end-to-end. Everything below is about
making the **retrieval** step land the right document at rank 1.

## 2. Frontmatter aliases

If §1 is "how to lay out the corpus," §2 is "how to make retrieval
actually work."

### 2.1 What aliases are

An **alias** is a phrase users might type that should land at *this
document*. The frontmatter `aliases:` list captures those phrasings
explicitly. When a user query comes in, the
[`KeywordRetriever`](../src/attune_rag/retrieval.py) tokenizes both
the query and each document's alias-union and credits each document
by how many query tokens overlap that document's aliases.

```yaml
aliases:
  - ship a release          ← matches "ship a release"
  - cut a new version       ← matches "cut a new version"
  - publish to PyPI         ← matches "publish to PyPI"
  - tag and push            ← matches "tag and push the release"
```

A query of "how do I ship a release?" tokenizes to `{ship,
releas}` (after stemming) and finds the alias-union containing
`{ship, releas, cut, new, version, publish, pypi, tag, push}` —
strong overlap → high retrieval score.

### 2.2 Multi-token alias intent (the `MIN_ALIAS_OVERLAP = 2` consequence)

The retriever's default is **`MIN_ALIAS_OVERLAP = 2`**. A query has
to overlap **at least two** distinct tokens with a document's
alias-union before any alias contribution counts. That gates out
single-common-token phantom hits.

**Practical consequence:** aliases should be **two-or-more-token
phrases**. Single-word aliases get filtered out by the threshold.

```yaml
aliases:
  - pypi                    ← single-token; alone, this won't fire
  - publish to pypi         ← two-token; this is the useful shape
  - pypi release            ← two-token; useful
```

Counter-example from the alias-sweep: an alias `"vers"` (intended
to catch `version`) on a release-prep template was contributing a
single-token hit that crowded out the rightful winner on the query
`"version bump and changelog"`. Removing the alias removed the
phantom.

If a single-word concept is genuinely the right alias, **pair it
with the natural extension**:

```yaml
aliases:
  - schema validation       ← rather than just "schema"
  - schema check
```

### 2.3 Authoring patterns

Aliases serve three intents, in priority order:

1. **The feature name + its common rephrasings.**
   ```yaml
   aliases:
     - alias expansion sweep   # the canonical name
     - alias expansion         # the common short form
     - paraphrase coverage     # the user-facing benefit
   ```
2. **Specific error messages or symptoms users will paste.**
   ```yaml
   aliases:
     - readme is wrong         # paste-likely
     - readme lies about code  # paste-likely
     - readme out of date
   ```
3. **Multi-word concepts that anchor topical category.**
   ```yaml
   aliases:
     - security review
     - vulnerability scan
     - secrets in code
   ```

### 2.4 The `_tokenize()` validation step (the `bites → bit` lesson)

attune-rag's tokenizer + stemmer collapse plurals and verb forms,
but **not always how you'd expect**. Always run your alias candidates
through `_tokenize()` before authoring to see what they actually
become.

```python
from attune_rag.retrieval import _tokenize
print(_tokenize("diff bites"))
# → {'diff', 'bit'}    (the -es suffix strips "bites" to "bit")

print(_tokenize("diff bite"))
# → {'diff', 'bite'}   (no suffix; "bite" stays as-is)
```

The first form (`"diff bites"`) and the second form (`"diff bite"`)
**do not unify**. Authoring `"diff bites"` as your alias when the
user query is `"diff bite ..."` results in zero overlap.

> 📝 **The lesson surfaced in the alias-expansion sweep
> ([`docs/retros/2026-05-21-alias-sweep.md`](retros/2026-05-21-alias-sweep.md)
> §4):** one alias landed in the first authoring pass that covered
> nothing because of exactly this trap. The retro recommends running
> `_tokenize()` on every multi-token candidate before adding it to
> the frontmatter.

Common traps:

| Surface form | Stems to | Notes |
|---|---|---|
| `bites` | `bit` | `-es` suffix matches |
| `bite` | `bite` | no suffix; doesn't unify with `bites` |
| `vulnerability` | `vulnerability` | (unchanged) |
| `vulnerabilities` | `vulnerabilit` | `-ies` → `-it` (matched in 0.1.22) |
| `cities` | `cities` | `_MIN_STEM_LEN = 3` floor protects 3-char stems |
| `city` | `city` | (unchanged) |

See §5 for the full stemmer reference.

## 3. The override file pattern

Sometimes you want to add aliases to a corpus you don't fully own —
e.g. you're an attune-rag *consumer*, not the attune-help maintainer,
and you've discovered a query class that misses. The override file
is for that case.

### 3.1 When to use overrides vs frontmatter

| Use overrides when… | Use frontmatter when… |
|---|---|
| You don't own the corpus repo | You own the corpus |
| You're iterating fast and don't want to ship an upstream release per alias batch | The alias is settled |
| You're A/B testing alias candidates against your query set | The alias survives your A/B |
| The corpus ships in a separate package and lags your iteration speed | The corpus ships in the same release cadence |

**The override file is tactical. Frontmatter is strategic.** The
recommended workflow is **override-then-promote** (§3.2).

### 3.2 The override-then-promote workflow

1. **Discover** a query class your corpus misses. Add 2-5 candidate
   aliases to your override file. Re-measure (see §6).
2. **Validate** that the aliases hold (per-query coverage up, baseline
   coverage unchanged — see §6 for the strict-dominance check).
3. **Promote upstream.** Open a PR on the corpus repo moving the
   override entries to the document's frontmatter. The override
   stays in your code until the upstream PR merges.
4. **Trim** your override file once the upstream release ships.

This is the discipline that produced the alias-expansion sweep
(0.1.23) — aliases lived in `aliases_override.json` in `attune-rag`
while attune-help#9 (the upstream promotion PR) was still in flight.

### 3.3 Override file schema

```jsonc
{
  "_comment": "Free-form note for the next reader; underscore-prefixed keys are ignored",
  "_format": "{ rel_path: [alias, alias, ...] }",

  "concepts/some-template.md": [
    "first authored alias",
    "second authored alias",
    "another multi-token alias"
  ],
  "tasks/another-template.md": [
    "and another"
  ]
}
```

Rules:

- Top-level keys are **corpus-relative paths**, same shape the corpus
  uses internally (forward slashes, no leading dot).
- Each value is an array of **alias strings**.
- Underscore-prefixed keys (`_comment`, `_format`, anything starting
  `_`) are silently ignored, so you can keep human notes inline.
- Aliases are **appended** to the document's frontmatter aliases —
  they don't replace anything.
- Within-template duplicates against frontmatter are silently deduped.
- **Cross-template alias collisions raise `DuplicateAliasError`** at
  corpus-load time — an alias can belong to exactly one document.

### 3.4 Loading the override file

attune-rag 0.1.23 ships the **`extra_aliases`** kwarg on
`DirectoryCorpus`. To load aliases from a JSON file today:

```python
import json
from pathlib import Path
from attune_rag import DirectoryCorpus

def _load_aliases(path: Path) -> dict[str, list[str]]:
    """Load alias overrides from a JSON file, filtering underscore-keys."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}

extras = _load_aliases(Path("./aliases_override.json"))
corpus = DirectoryCorpus(Path("./my-corpus"), extra_aliases=extras)
```

> 🚧 **v1.0.0 cleanup (forthcoming):** the
> [`user-corpus-onboarding`](specs/user-corpus-onboarding/) spec adds
> a first-class `extra_aliases_file="path.json"` kwarg on
> `DirectoryCorpus` plus a public `load_aliases_from_file` helper, so
> the snippet above collapses to a one-liner:
> `DirectoryCorpus(root, extra_aliases_file="./aliases_override.json")`.
> The boilerplate above is the v0.2.x pattern; both will work at
> v1.0.0.

### 3.5 Example: attune-rag's own override file

attune-rag ships its own override file at
[`src/attune_rag/corpus/aliases_override.json`](../src/attune_rag/corpus/aliases_override.json).
It's a working example of the schema. Look at it when you're
authoring your own.

The file's top-level `_comment` field documents its history
(promoted from the alias-expansion sweep) and its trim plan (entries
become redundant once attune-help#9 ships and can be removed).

## 4. The `MIN_ALIAS_OVERLAP` knob

§2.2 introduced the default: **`MIN_ALIAS_OVERLAP = 2`**. You can
flip it.

### 4.1 Default and rationale

The knob lives on `KeywordRetriever` as a class attribute:

```python
class KeywordRetriever:
    MIN_ALIAS_OVERLAP: int = 2
    # ...
```

It defaults to **2** because that's the floor at which multi-token
aliases (the design intent — `"ship a release"`, `"publish to PyPI"`)
still fire while single-common-token phantom hits get filtered.

### 4.2 When to flip to 1

Corpora **without curated multi-token aliases** — e.g. you've
inherited a corpus that uses single-word aliases like `[security,
api, mcp]` — won't benefit from `MIN_ALIAS_OVERLAP=2`; it'll just
gate all your aliases out.

Subclass + flip:

```python
from attune_rag.retrieval import KeywordRetriever

class LooserRetriever(KeywordRetriever):
    MIN_ALIAS_OVERLAP = 1

# ...wire into the pipeline via the constructor:
from attune_rag import RagPipeline
pipeline = RagPipeline(corpus=corpus, retriever=LooserRetriever())
```

> 🚧 **v1.0.0 cleanup (forthcoming):** the
> [`user-corpus-onboarding`](specs/user-corpus-onboarding/) guide
> v1 will document a constructor kwarg path. For v0 / v0.2.x,
> subclassing is the path.

### 4.3 When to flip higher

If your corpus has **very dense alias sets** where the 2-token floor
still admits phantom hits, flipping to 3 is the next step. This is
uncommon — for most corpora 2 is right — but it's available.

### 4.4 Measuring the trade-off

There is a per-query trade-off whenever you flip `MIN_ALIAS_OVERLAP`.
Don't guess; measure. §6 covers the measurement loop.

## 5. Stemmer gotchas

The stemmer is the single most surprising piece of attune-rag's
retrieval pipeline.

### 5.1 The token pipeline

```
"diff bites the test"
   ↓ lowercase
"diff bites the test"
   ↓ tokenize (split on non-alphanumerics + drop stopwords)
{"diff", "bites", "test"}
   ↓ stem (strip longest matching suffix; min stem length 3)
{"diff", "bit", "test"}
```

The actual implementation is in
[`attune_rag.retrieval._tokenize()`](../src/attune_rag/retrieval.py).
You can call it directly:

```python
from attune_rag.retrieval import _tokenize
print(_tokenize("diff bites the test"))
# → {'diff', 'bit', 'test'}
```

(The underscore prefix means it's internal-but-callable — not part
of the SemVer-frozen public surface, but stable enough to use for
alias validation. If you need it as a guaranteed-stable contract,
file an issue and we'll consider promoting it.)

### 5.2 The `_MIN_STEM_LEN` floor

`_MIN_STEM_LEN = 3` protects short tokens from over-stemming. A
suffix only strips if the resulting stem is at least 3 characters.

| Input | Suffix matched | Resulting stem |
|---|---|---|
| `cities` | `-ies` | `cit` → **but 2 chars** → falls back to `cities` |
| `bodies` | `-ies` | `bod` → 3 chars → `bod` |
| `bytes` | `-es` | `byt` → 3 chars → `byt` |
| `dies` | `-ies` | `d` → 1 char → `dies` (no strip) |

If your aliases lean on 3- or 4-character tokens, double-check
they don't fall foul of the floor.

### 5.3 Common traps

In rough order of how often they surprise corpus authors:

| Trap | Why it matters | Fix |
|---|---|---|
| `bites → bit`, `bite → bite` | They don't unify | Author both forms or use a longer phrase |
| `vulnerabilities → vulnerabilit`, `vulnerability → vulnerability` | They don't unify (until 0.1.22 added `-ity` / `-ities` to the stem table) | Already-fixed; mentioned for historical context. |
| `pypi`, `npm`, `cli` (3-char abbreviations) | Below the stem floor; they stay as-is | Author them verbatim |
| Punctuation in queries | Stripped at tokenize step | Don't put punctuation in aliases either |
| Stopwords (the, a, of, ...) | Dropped at tokenize step | Don't put stopwords in aliases either |

### 5.4 The "`_tokenize()` before authoring" discipline

The single most useful authoring habit is **running `_tokenize()` on
every alias candidate before adding it to the frontmatter or override
file**. It takes 5 seconds and catches every stemmer surprise above.

```python
# A 5-second authoring check:
from attune_rag.retrieval import _tokenize

candidates = [
    "ship a release",
    "publish to pypi",
    "diff bites",
]
for c in candidates:
    print(f"{c!r:30} → {sorted(_tokenize(c))}")
```

Output:

```
'ship a release'              → ['releas', 'ship']
'publish to pypi'             → ['publish', 'pypi']
'diff bites'                  → ['bit', 'diff']
```

The third one is the trap — you authored `"diff bites"` but
queries like `"diff bite something"` won't overlap.

## 6. Quality measurement

You've authored aliases. You've added overrides. How do you know
your retrieval is actually good?

### 6.1 The basic loop

attune-rag's bundled corpus is measured against a fixed query set:

- [`tests/golden/queries.yaml`](../tests/golden/queries.yaml) — 40
  "baseline" queries (the canonical phrasings).
- [`tests/golden/queries_paraphrased.yaml`](../tests/golden/queries_paraphrased.yaml)
  — 80 paraphrased queries (no-token-overlap variants of the
  baseline).

For each query, the ground truth is the corpus-relative path of the
document that should land at rank 1. Two metrics matter:

- **P@1** — Precision at rank 1. "Did the right document land at
  position 1?" (binary per query, averaged over the set.)
- **R@3** — Recall at rank 3. "Did the right document land in the
  top 3?" (binary per query, averaged over the set.)

For your own corpus, **author the same shape**:

```yaml
# my-queries.yaml
- query: How do I set up the MCP server?
  expected_top_1: tasks/setup-mcp-server.md
  difficulty: easy

- query: Configure MCP for Claude
  expected_top_1: tasks/setup-mcp-server.md
  difficulty: medium

- query: Set up the model context protocol server
  expected_top_1: tasks/setup-mcp-server.md
  difficulty: paraphrased
```

Authoring 20-40 baseline queries + 40-80 paraphrased variants gets
you to the measurement floor where shifts in P@1 / R@3 are
meaningful. Fewer queries → noisier numbers.

### 6.2 Running the measurement

attune-rag ships three equivalent entry points — pick the one that
matches your workflow. All three produce the same byte-identical
output given the same inputs.

**Console script (recommended for ad-hoc + CI):**

```bash
attune-rag-measure \
    --corpus-path ./my-corpus \
    --queries ./my-queries.yaml \
    --paraphrased ./my-paraphrased.yaml \
    --output report.md \
    --watermark-r3 0.85
```

Available after `pip install attune-rag` (no repo clone needed).
Non-zero exit on watermark fail makes it CI-suitable directly.
`--paraphrased` is optional; pass it when you've authored a
no-token-overlap regression set (see §6.1).

**Module entry (handy when you need a specific Python interpreter):**

```bash
python -m attune_rag.measure_corpus \
    --corpus-path ./my-corpus \
    --queries ./my-queries.yaml \
    --output report.md
```

**Python API (build your own pipelines + dashboards on top):**

```python
from pathlib import Path
from attune_rag.measure_corpus import measure

result = measure(
    corpus_path=Path("./my-corpus"),
    queries_path=Path("./my-queries.yaml"),
    paraphrased_path=Path("./my-paraphrased.yaml"),
)
print(f"P@1 = {result.p1:.4f}  R@3 = {result.r3:.4f}")

# Per-difficulty breakdown for surfacing the corpus's weak spots
for diff, stats in sorted(result.per_difficulty_breakdown.items()):
    print(f"  {diff}: P@1={stats['p1']:.2%} R@3={stats['r3']:.2%} (n={int(stats['n'])})")

# CI-suitable watermark check
import sys
if result.watermark_failures(r3_floor=0.85):
    sys.exit(1)

Path("report.md").write_bytes(result.report_markdown().encode("utf-8"))
```

`MeasureResult` is a frozen dataclass — safe to log, pass between
threads, or feed into downstream tooling. It carries the
aggregate scalars, per-query records (in YAML input order),
the per-difficulty breakdown, and the file SHA-256s for
provenance. See the
[`attune_rag.measure_corpus`](../src/attune_rag/measure_corpus.py)
docstrings for the full schema.

**Two-pass with rerank (opt-in, ~$0.05 per 80-query set):**

```bash
attune-rag-measure \
    --corpus-path ./my-corpus \
    --queries ./my-queries.yaml \
    --with-rerank \
    --output report.md
```

This runs both keyword and keyword-plus-rerank, side-by-side, so
you can see exactly **whether rerank earns its keep on your corpus**.
Sometimes it lifts a handful of marginal queries; sometimes it's
neutral (the bundled corpus is one such case — see
[`docs/specs/release-quality-baseline/baseline-with-rerank.md`](specs/release-quality-baseline/baseline-with-rerank.md)
for the N=1 measurement and [D5's diagnostic](specs/reranker-evaluation/diagnostic-1.md)
for the rigorous N=5 verdict). Either result is informative: a lift
tells you to leave rerank on for prod traffic; a neutral result tells
you the keyword path is doing its job and you can skip the API spend
for end users. Requires `ANTHROPIC_API_KEY` in the environment and
the `[claude]` extra installed.

**Backward-compat: `scripts/measure_corpus.py`.** The original
freeze-time entry point still works — it's now a thin shim that
calls into `attune_rag.measure_corpus:main`. Existing CI
invocations (`python scripts/measure_corpus.py ...`) keep working
unchanged. New code should prefer one of the three paths above.

**If you prefer to script the loop yourself**, the 20-line shape
that the harness is built on:

```python
import yaml
from pathlib import Path
from attune_rag import RagPipeline, DirectoryCorpus

corpus = DirectoryCorpus(Path("./my-corpus"))
pipeline = RagPipeline(corpus=corpus)

queries = yaml.safe_load(Path("./my-queries.yaml").read_text())["queries"]

hits_at_1 = 0
hits_at_3 = 0
for q in queries:
    result = pipeline.run(q["query"])
    top_paths = [h.template_path for h in result.citation.hits[:3]]
    expected = set(q["expected_in_top_3"])
    if top_paths and top_paths[0] in expected:
        hits_at_1 += 1
    if expected & set(top_paths):
        hits_at_3 += 1

n = len(queries)
print(f"P@1: {hits_at_1/n:.4f}  R@3: {hits_at_3/n:.4f}  (n={n})")
```

For most users `attune-rag-measure` is the better path — it
handles the YAML schema, SHA-256 provenance, deterministic
report rendering, and watermark-gating mechanics that you'd
otherwise re-implement.

### 6.3 The strict-dominance discipline

The alias-expansion sweep's load-bearing rule: **every change to
your corpus's aliases must hold previously-passing queries unchanged.**

When you add an alias to fix a missing query, re-run the full
measurement and confirm:

- The originally-failing query now passes ✓
- **Every previously-passing query still passes** ✓
- **Aggregate P@1 and R@3 did not drop** ✓

If aggregate drops, you've moved a phantom hit somewhere else.
**Don't ship the change.** Investigate which previously-passing
query now fails, fix the alias (often: it's two-token-overlapping
with another template by accident), re-measure.

This is what made the 13-PR sweep land with **zero baseline
regression** — every PR cleared the discipline before merge. See
[the retro](retros/2026-05-21-alias-sweep.md) §3 for the M12
near-regression that this discipline caught pre-merge.

### 6.4 Wiring into CI

`attune-rag-measure` is designed for direct CI use — non-zero
exit on watermark fail makes it a one-line gate:

```yaml
# .github/workflows/corpus-quality.yml (sketch)
on: [pull_request]
jobs:
  measure:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
      - run: pip install attune-rag
      - run: |
          attune-rag-measure \
              --corpus-path ./my-corpus \
              --queries ./my-queries.yaml \
              --paraphrased ./my-paraphrased.yaml \
              --watermark-r3 0.85 \
              --output measurement.md
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: corpus-measurement
          path: measurement.md
```

Exit non-zero when R@3 drops below your floor (0.85 is the bundled
attune-help floor; adjust to your corpus's measured baseline).
This converts the strict-dominance discipline into an automated gate.

## 7. The `QueryExpander` (when to use it)

The default `RagPipeline` includes a `QueryExpander` step before
retrieval. The expander is **Claude-Haiku-based** and rewrites the
incoming query into 3-5 alternative phrasings before retrieval runs
on all of them and merges results.

The expander is the **fallback retrieval lever** for corpora without
curated multi-token aliases. It bought ~51 percentage points of
paraphrased R@3 on the bundled corpus in the alias-expansion sweep's
D2 diagnostic — but cost ~10 percentage points of baseline P@1 (it
expands queries that didn't need expansion).

**When to use it:**

- Your corpus is brand-new and doesn't have curated aliases yet.
- Your corpus is large enough that aliasing every document is
  infeasible.
- Your query mix is high-paraphrase by nature (e.g. natural-language
  questions over a docs corpus rather than precise tool-name queries).

**When NOT to use it:**

- Your corpus has high-quality multi-token aliases and you've
  measured paraphrased R@3 ≥ 0.85 without it.
- You're cost-sensitive on Anthropic-API spend.

Toggle it off via `RagPipeline(use_query_expander=False)`. The
attune-rag default is on (~$0.0008/query at Haiku list price); for
bundled-corpus paraphrased queries, the alias mechanism alone now
gets R@3 to 100% — but the expander helps when aliases are sparse.

## 9. Where the override mechanism fits

If you're shipping a **package that ships with a corpus** (the
attune-help shape — your package bundles a markdown directory that
downstream users consume), the override mechanism is also your tool
for **iterating retrieval quality between releases**.

Pattern:

1. Your package ships v1.0 with frontmatter aliases.
2. A user reports a missing query class.
3. You add overrides to the *downstream consumer* (e.g.
   attune-rag's `aliases_override.json`) and validate against the
   user's query.
4. You promote the overrides into your package's frontmatter in
   the next minor release.
5. The downstream consumer trims its override entries at the next
   minor.

This is what `attune-rag` does with `attune-help`. The
[`docs/specs/alias-expansion-sweep/`](specs/alias-expansion-sweep/)
spec and its
[retro](retros/2026-05-21-alias-sweep.md) document the full
13-cluster run that established the pattern.

For a single-corpus authoring flow (you own the corpus and ship it
with your code), you don't need the override mechanism — just edit
frontmatter directly.

## Quick reference

### Authoring checklist

When adding an alias:

- [ ] Two or more tokens (or paired with a multi-token form)
- [ ] Run through `_tokenize()` to check stemming
- [ ] No stopwords; no punctuation
- [ ] Re-measure: target query passes, baseline holds, no regression
- [ ] If owning the corpus: lands in frontmatter. If not: lands in override file with a promote-upstream PR open.

### Knob reference

| Knob | Default | Lives at | Flip via |
|---|---|---|---|
| `MIN_ALIAS_OVERLAP` | 2 | `KeywordRetriever` class attribute | Subclass + override |
| `_MIN_STEM_LEN` | 3 | `attune_rag.retrieval` module constant | Edit + ship (internal) |
| `_STEM_SUFFIXES` | (see source) | `attune_rag.retrieval` module constant | Edit + ship (internal) |
| `use_query_expander` | True | `RagPipeline` constructor kwarg | Pass `use_query_expander=False` |
| `candidate_multiplier` | 3 | `RagPipeline` constructor kwarg | Pass `candidate_multiplier=N` |

### Forthcoming at v1.0.0

Tracked in [`user-corpus-onboarding`](specs/user-corpus-onboarding/):

- `DirectoryCorpus(root, extra_aliases_file="path.json")` kwarg — one-liner override loading.
- `attune_rag.corpus.load_aliases_from_file(path)` — public helper.
- `attune_rag.measure_corpus` module + `attune-rag-measure` CLI — packaged harness.
- Guide §6 upgrades to reference the packaged harness instead of the 20-line script pattern.
- Guide §4 upgrades to reference a constructor-kwarg path for `MIN_ALIAS_OVERLAP` if scoping picks one.

### Further reading

- [`docs/POLICY.md`](POLICY.md) — public-API and deprecation policy
- [`docs/retros/2026-05-21-alias-sweep.md`](retros/2026-05-21-alias-sweep.md) — the 13-cluster sweep that established the discipline this guide documents
- [`docs/specs/alias-expansion-sweep/`](specs/alias-expansion-sweep/) — D1-D4 diagnostic chain showing how the discipline was validated
- [`docs/specs/embedding-retriever/`](specs/embedding-retriever/) — permanent-defer report; documents why keyword + aliases beat embeddings for the bundled corpus and the scope-specific defer for arbitrary user corpora
- [`docs/specs/user-corpus-onboarding/`](specs/user-corpus-onboarding/) — the v1.0.0 spec that turns this guide's v0 patterns into first-class API

## Provenance

This guide synthesizes the lessons from the **2026-05-21
alias-expansion sweep** (PRs [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94)–[#108](https://github.com/Smart-AI-Memory/attune-rag/pull/108),
[#110](https://github.com/Smart-AI-Memory/attune-rag/pull/110)) that
shipped at attune-rag 0.1.23 — the run that took paraphrased R@3 from
28.75% to 100% on the bundled attune-help corpus with zero new
dependencies and zero baseline regression. The sweep's
[1-page retro](retros/2026-05-21-alias-sweep.md) is required reading
for understanding *why* the discipline this guide documents is the
discipline.

The full design rationale lives in
[`docs/specs/user-corpus-onboarding/`](specs/user-corpus-onboarding/).
This guide is the v0 forerunner — the user-facing artifact ahead of
v1.0.0's clean ergonomics.
