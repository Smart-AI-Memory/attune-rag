# Risks: alias-overlap-remediation

> **Status:** approved 2026-06-03. See [README](README.md).

## R-1 — Warning noise on legitimately single-token corpora (LOW)

A corpus that *intends* single-token aliases and runs with
`MIN_ALIAS_OVERLAP = 1` never trips the warning (floor inert). A
corpus that keeps the default `2` *and* uses single-token aliases
is exactly the misconfiguration we want to surface — so the
"noise" is the signal. Mitigation: the `warn_alias_overlap=False`
flag + logger-level suppression (R3) give a clean opt-out.

## R-2 — Threshold mis-tuned (LOW)

The 10%-of-aliased-entries threshold (`_ALIAS_WARN_FRACTION`) is a
judgment call. Too low → noise; too high → misses partial
degradation. Mitigation: module-level constant, tunable without an
API change; default chosen conservatively (any degraded entry in a
small corpus warns).

## R-3 — Surface-test pins constructor signatures (LOW)

If `test_api_surface.py` snapshots signatures (not just `__all__`),
the `warn_alias_overlap` kwarg would trip it. Mitigation: D2 —
reroute to a module-level default constant; verify in T1 before
choosing the flag shape.

## R-4 — Behavior-change regression (VERY LOW)

The warning must not alter scoring. Mitigation: the bundled golden
snapshot byte-identical assertion in T2 is the hard gate; any
scoring drift fails CI.
