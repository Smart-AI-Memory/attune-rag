# Phase-5 exit summary — attune-rag v1.0.0

**Closed:** 2026-08-10 (chair-directed early close; see "The watch"
below). Spec status: **complete**.

## The cut

v1.0.0 shipped 2026-08-10 (~05:33 UTC): tag `v1.0.0` at merge SHA
`1351d79`, publish run 31358524673 approved by Patrick at the `pypi`
gate, wheel + sdist + provenance verified on the simple index,
classifier `Production/Stable`. M0–M3 all closed same-day; two of the
four M0 consumer-pin tasks were VOIDED by archived repos
(attune-author, attune-gui) — recorded inline in tasks.md with the
validation evidence from the attempts.

## The watch (M4) — retired early, superseded by 1.1.0

The 7-day no-hotfix window opened 2026-08-10 and was **retired the
same day** when the chair pulled 1.1.0 forward (abstention-by-default
+ confidence-gated retrieval; tag `v1.1.0` at `284d0ce3`, live on
PyPI ~07:45 UTC). The honest reading, per M4.2's own framing:

- **Zero `1.0.z` hotfixes shipped and none were pending.** The only
  post-1.0.0 fixes before 1.1.0 were non-release items (rag#213
  workflow install-extra, rag#214 test-gate deflake) — neither
  touched shipped library behavior nor warranted a patch release.
- The 7-day threshold is a *signal-strength* device, not a gate; it
  was **superseded, not met** — 0 clean days observed, and the 1.0.0
  claim needed no walking back on the evidence available. The
  window's remaining signal value transfers to 1.1.0's ordinary
  post-release observation (daily briefing), with no formal watch.
- M4.2's retrospective question ("what would M1's audit have
  caught?") is vacuously closed: no hotfix, no root cause to log.

## Carried observations (non-blocking → ordinary backlog)

Five audit observations from M1.2 / review-pass-2.md went to the
1.0.x backlog; with 1.1.0 shipped they are ordinary backlog, none
release-gating: (1) `_ALIASES_BLOCK_RE.sub(count=1)` can strip a
body-level `aliases:` line when frontmatter has none (preview-only);
(2) `KeywordRetriever.__init__` shadows the `MIN_SCORE` class
constant via instance attribute (stylistic); (3–5) the three
review-pass-2 observations recorded in
[review-pass-2.md](review-pass-2.md).

Per M4.3's condition (open `post-1.0.0-watch/` only if M1.2/M1.3
left outstanding items): **no watch spec is opened** — M1.3 closed
with zero gap items and the observations above are non-blocking.

## Support-window note (POLICY §8)

§8 defines the 1.0.x support window as six months past the next
minor. 1.1.0 shipping 2026-08-10 starts that clock: **1.0.x is
supported through ~2027-02-10.**
