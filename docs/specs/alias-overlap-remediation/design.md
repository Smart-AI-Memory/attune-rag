# Design: alias-overlap-remediation

> **Status:** approved 2026-06-03. See [README](README.md).

## 1. Where the warning lives

`DirectoryCorpus.__init__`
([directory.py:50](../../../src/attune_rag/corpus/directory.py)) is
the single build-time chokepoint where every entry's final alias
set (frontmatter aliases + `extra_aliases` merge) is known.
`AttuneHelpCorpus` builds on `DirectoryCorpus`, so a check there
covers both the bundled and user-directory paths.

The detection runs **after** the `extra_aliases` merge so that a
corpus rescued by an override file (the documented multi-token
rescue path) does not warn.

## 2. Detection logic

For the effective floor `f = KeywordRetriever.MIN_ALIAS_OVERLAP`
(read from the class attribute; if `f <= 1` the floor is inert and
we never warn):

1. For each entry, tokenize each alias with the existing
   `_tokenize` (reuse — do not reimplement; import from
   `attune_rag.retrieval`).
2. An entry is **alias-floor-reachable** if it has at least one
   alias whose token count `>= f`. (With `f = 2`, that means at
   least one genuinely multi-token alias.)
3. An entry is **alias-degraded** if it has >=1 alias but **none**
   are floor-reachable — every alias it carries is too short to
   ever credit a hit under the floor.
4. Count `alias-degraded` entries. Entries with zero aliases are
   not degraded (they never relied on alias signal).

This keys on the *structural* fact ("this corpus's aliases can
never fire under the active floor"), which is exactly the silent
failure mode — not on any query.

## 3. Threshold + message

Warn when `alias_degraded_count >= max(1, 0.10 * entries_with_aliases)`
— i.e. any degraded entry warns once the corpus has few aliased
entries, scaling to "at least 10% of aliased entries are degraded"
for larger corpora. Rationale: a single degraded entry in a
100-entry corpus is likely intentional; 10%+ is a corpus-shape
signal. The exact constant lives in a module-level
`_ALIAS_WARN_FRACTION = 0.10` for tunability without an API change.

Message (single `logging.warning`, logger
`attune_rag.corpus.directory`):

```
N of M aliased entries have only single-token aliases and will
contribute zero alias signal under MIN_ALIAS_OVERLAP=2. If your
corpus uses single-word aliases, set MIN_ALIAS_OVERLAP=1 (see
USER_CORPUS_GUIDE section 4.2) or author multi-token aliases.
```

## 4. Suppression (R3)

- **Logging config:** the warning uses logger
  `attune_rag.corpus.directory`; users set that logger to
  `ERROR`+ to silence. Documented in the guide.
- **Constructor flag:** `DirectoryCorpus(..., warn_alias_overlap:
  bool = True)`. Setting `False` skips the scan entirely. This is
  a **private behavior toggle**, not counted against the surface
  budget — it is not added to any `__all__`, and
  `test_api_surface.py` snapshots `__all__`, not constructor
  signatures. (If the surface test also pins signatures, the spec
  re-routes the flag to a module-level `_WARN_ALIAS_OVERLAP_DEFAULT`
  with no signature change — verified in T1.)

## 5. Why not the public kwarg now

The ergonomic fix (`min_alias_overlap=` constructor kwarg threaded
`RagPipeline -> KeywordRetriever`) is real but spends the last of
5 v1.0.0 surface slots on a niche knob. Per
[`POLICY.md` §4.1](../../POLICY.md), the 5th slot is reserved for
something that "materially shapes the v1.0.0 framing." The subclass
path (`class X(KeywordRetriever): MIN_ALIAS_OVERLAP = 1`) already
works and is documented. Deferring the kwarg to the v1.0.0 scope
keeps this remediation freeze-legal and surface-neutral. See
[decisions.md](decisions.md) D1.

## 6. Test strategy

- **Unit:** a fixture corpus of single-token-only aliased entries
  → assert one warning fires (caplog), message names count +
  floor. A multi-token corpus → assert silent. `f = 1` → assert
  silent regardless. `warn_alias_overlap=False` → assert silent.
- **Strict-dominance:** run the bundled-corpus golden snapshot
  (`tests/golden/measure_corpus_bundled.golden.md`) — must be
  byte-identical (proves R4: zero behavior change). The bundled
  corpus is multi-token-tuned so it must also be silent.
- **Perf:** the build-time scan is O(aliases); confirm no
  regression on the `directory_corpus_load` advisory axis.
