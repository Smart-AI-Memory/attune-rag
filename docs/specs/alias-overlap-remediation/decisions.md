# Decisions: alias-overlap-remediation

> **Status:** approved 2026-06-03. See [README](README.md).

## D1 — Warn + docs now; defer the public kwarg to v1.0.0

**Ratified with Patrick 2026-06-03.**

### Decision

Remediate the silent `MIN_ALIAS_OVERLAP = 2` regression with:

1. a build-time single-token-alias **warning** (zero
   public-surface cost — internal `logging.warning`), and
2. **consumer-impact docs** (`USER_CORPUS_GUIDE` §4.2 + downstream
   release-note pointers).

**Defer** the public `min_alias_overlap` constructor kwarg to the
v1.0.0 surface-scoping pass. The subclass override
(`class X(KeywordRetriever): MIN_ALIAS_OVERLAP = 1`) remains the
documented escape hatch meanwhile.

### Alternatives considered

- **Spend the 5th surface slot on a `min_alias_overlap=` kwarg.**
  Full ergonomic fix, but burns the last v1.0.0 bargaining chip on
  a niche knob and still needs the warning to kill silent failure.
  Rejected: the warning addresses the actual harm (silence) at
  zero surface cost; the kwarg is sugar.
- **Re-open `user-corpus-onboarding` scoping decision #5** to
  expand the budget, then add the kwarg. Rejected for this
  remediation: couples a stabilization fix to v1.0.0 surface
  re-scoping; heavier process for no added safety.
- **Flip the default to `MIN_ALIAS_OVERLAP = 1`.** Rejected: the
  bundled `AttuneHelpCorpus` was tuned for `2`; flipping would
  regress the corpus the default is measured against. The default
  is correct *for the bundled corpus*; the problem is silence for
  *un-tuned* corpora.

### Consequences

- Surface budget stays **4/5** — the last slot is preserved for
  v1.0.0 framing.
- The kwarg becomes a v1.0.0 backlog item, naturally scoped with
  the rest of the public-API graduation.
- Because the change is observability-only, the family-plan
  UX-regression guard is satisfied by construction (no scoring
  path touched).

## D2 — Suppression is a private toggle, not a symbol

The `warn_alias_overlap=False` opt-out is a constructor flag that
does **not** enter `__all__`. If `test_api_surface.py` is found to
pin constructor signatures (not just `__all__`), reroute to a
module-level `_WARN_ALIAS_OVERLAP_DEFAULT` constant so no public
signature changes. Verified during T1 implementation.
