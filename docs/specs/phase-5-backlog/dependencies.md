# Phase-5 backlog — dependencies

What blocks Phase-5 work from starting. Cross-reference
[ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5.

## Blockers

1. **4-week API freeze closes.** Phase 4 W4 must complete. While the
   freeze is active, no items in [items.md](items.md) move — landing
   them mid-freeze either invalidates the freeze gate (perf items P1–P4
   would re-tune the σ baseline mid-window; methodology item M1
   rewrites the gate itself) or churns freeze-locked surface area
   (quality items Q3, Q4 touch the public `__all__` / schema discriminator
   that the freeze is meant to stabilise).
2. **0.2.0 cut lands.** The 0.2.0 public-surface release closes out
   Phase 3 / Phase 4 deliverables. Quality item Q2 (`_iter_entries`
   lift) is additionally gated on the `editor/_*.py` shim removal
   scheduled for 0.3.0; flag during triage if 0.3.0 shim removal
   moves into Phase 5.
3. **Phase 5 spec created.** `docs/specs/v1.0.0-release/` does not
   exist yet (per ROADMAP Phase 5 row: "Status: not started"). Items
   move from this backlog into that spec's `tasks.md` at Phase 5
   kickoff.

## Not blockers

- Phase 5 does not require attune-gui-side M5.3 closeout (queued
  separately per Phase 4 row).
- No external review is gated on this backlog — it's a working
  document, not an approval artifact.
