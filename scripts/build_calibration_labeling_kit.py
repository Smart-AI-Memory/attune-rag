"""Generate a hand-labeling kit from a `--compare-thinking --json` artifact.

Picks a small subset of queries from the calibration artifact —
biased toward queries where the thinking-off vs thinking-on judge
verdicts disagree the most (high-signal) plus a handful of
unchanged queries as controls — and emits a markdown file where
each query has the answer, the retrieved passages, the judge's
two verdicts, and a YAML stub for the human to fill in.

Run this once after a calibration run; hand the markdown to a
domain expert; then feed the labeled file plus the original
artifact into ``score_against_ground_truth.py``.

Usage::

    python scripts/build_calibration_labeling_kit.py \\
        --artifact artifacts/calibration/thinking-2026-05-15.json \\
        --out artifacts/calibration/ground-truth-2026-05-15.template.md \\
        --n-shifted 5 --n-controls 3

The ``--n-random`` flag adds a third bucket — N queries drawn
uniformly from the remaining (non-shifted, non-control) pool.
Used by Phase 2 of the v1.0 roadmap to separate "judge
disagreement" from "judge run-to-run variance" (see
``docs/specs/faithfulness-thinking-decision/design.md``).
Pair with ``--seed`` for reproducible draws.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def _per_query_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {q["id"]: q for q in report.get("per_query", [])}


def _select_queries(
    artifact: dict[str, Any],
    n_shifted: int,
    n_controls: int,
    shift_threshold: float,
    n_random: int = 0,
    rng: random.Random | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return three lists of query IDs: shifted, controls, random.

    "Shifted" = abs(score_on - score_off) >= ``shift_threshold``.
    "Control" = unchanged on score AND on claim count.
    "Random" = uniform draw from the remaining pool, disjoint from
    shifted and controls. Requires ``rng`` when ``n_random > 0``.

    Raises ``ValueError`` if ``n_random`` exceeds the remaining
    pool — the caller should either lower N or widen the source.
    """
    off = _per_query_by_id(artifact["faithfulness_thinking_off"])
    on = _per_query_by_id(artifact["faithfulness_thinking_on"])
    common = sorted(set(off) & set(on))

    scored = []
    for qid in common:
        delta = on[qid]["score"] - off[qid]["score"]
        claims_off = off[qid]["supported"] + off[qid]["unsupported"]
        claims_on = on[qid]["supported"] + on[qid]["unsupported"]
        scored.append((qid, abs(delta), claims_off == claims_on and abs(delta) < 1e-6))

    shifted = [
        qid
        for qid, mag, _unchanged in sorted(scored, key=lambda r: -r[1])
        if mag >= shift_threshold
    ][:n_shifted]
    controls = [qid for qid, _mag, unchanged in scored if unchanged][:n_controls]

    random_picks: list[str] = []
    if n_random > 0:
        if rng is None:
            raise ValueError("rng is required when n_random > 0")
        used = set(shifted) | set(controls)
        pool = [qid for qid in common if qid not in used]
        if n_random > len(pool):
            raise ValueError(
                f"--n-random={n_random} exceeds remaining pool of {len(pool)} "
                f"(after {len(shifted)} shifted + {len(controls)} controls); "
                f"lower --n-random or widen the artifact."
            )
        random_picks = sorted(rng.sample(pool, n_random))

    return shifted, controls, random_picks


def _format_passages(context: str) -> str:
    """Render the joined-passages string as a blockquote."""
    if not context:
        return "> _(no passages retrieved)_"
    lines = context.split("\n")
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _format_query_block(qid: str, off: dict[str, Any], on: dict[str, Any]) -> str:
    """Render one labeling section for ``qid``.

    Embeds the generator's answer text and the retrieved
    passages when present in the artifact (added in
    benchmark.py post-2026-05-15). When absent (older
    artifacts), surfaces a warning instead of the missing
    fields so the kit still works on legacy data.
    """
    query = off.get("query", "")
    answer = off.get("answer")
    context = off.get("context")
    if answer is None or context is None:
        body_block = _MD_LEGACY_WARNING
    else:
        body_block = _MD_ANSWER_CONTEXT_BLOCK.format(
            context=_format_passages(context),
            answer=answer,
        )
    return _MD_BLOCK_TEMPLATE.format(
        qid=qid,
        query=query,
        body_block=body_block,
        score_off=off["score"],
        sup_off=off["supported"],
        unsup_off=off["unsupported"],
        score_on=on["score"],
        sup_on=on["supported"],
        unsup_on=on["unsupported"],
        reasoning_off=off.get("reasoning", "(no reasoning captured)"),
        reasoning_on=on.get("reasoning", "(no reasoning captured)"),
        sup_claims_off="\n".join(f"  - {c}" for c in off.get("supported_claims", [])),
        unsup_claims_off="\n".join(f"  - {c}" for c in off.get("unsupported_claims", [])),
        sup_claims_on="\n".join(f"  - {c}" for c in on.get("supported_claims", [])),
        unsup_claims_on="\n".join(f"  - {c}" for c in on.get("unsupported_claims", [])),
    )


_MD_LEGACY_WARNING = """\
> ⚠️ This artifact predates answer/context capture. The kit
> only has the judge's claim lists as a proxy. Re-run the
> calibration with a current `attune-rag-benchmark` to get a
> richer kit.
"""


_MD_ANSWER_CONTEXT_BLOCK = """\
### Retrieved context

{context}

### Answer

{answer}
"""


_MD_BLOCK_TEMPLATE = """\
## {qid} — `{query}`

{body_block}

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | {score_off:.3f} | {sup_off} | {unsup_off} |
| Thinking on  | {score_on:.3f} | {sup_on} | {unsup_on} |

**Reasoning (thinking off):**

> {reasoning_off}

**Reasoning (thinking on):**

> {reasoning_on}

**Claims identified by judge (thinking off):**

Supported:
{sup_claims_off}

Unsupported:
{unsup_claims_off}

**Claims identified by judge (thinking on):**

Supported:
{sup_claims_on}

Unsupported:
{unsup_claims_on}

### Your labels

```yaml
id: {qid}
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a hand-labeling kit from a calibration JSON artifact."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--n-shifted",
        type=int,
        default=5,
        help="Pick this many queries with the largest |score_on - score_off|.",
    )
    parser.add_argument(
        "--n-controls",
        type=int,
        default=3,
        help="Pick this many queries where neither score nor claim count changed.",
    )
    parser.add_argument(
        "--shift-threshold",
        type=float,
        default=0.05,
        help="Minimum |score Δ| to count as 'shifted'.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=0,
        help=(
            "Pick this many queries uniformly at random from the pool "
            "remaining after shift and control selection. Used by Phase 2 "
            "to anchor judge-variance measurement on typical queries."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the random draw for --n-random (omit for non-deterministic).",
    )
    args = parser.parse_args(argv)

    if args.n_shifted < 0 or args.n_controls < 0 or args.n_random < 0:
        print("--n-shifted, --n-controls, --n-random must be >= 0", file=sys.stderr)
        return 2

    if not args.artifact.is_file():
        print(f"Artifact not found: {args.artifact}", file=sys.stderr)
        return 2

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    rng = random.Random(args.seed) if args.n_random > 0 else None
    try:
        shifted, controls, random_picks = _select_queries(
            artifact,
            n_shifted=args.n_shifted,
            n_controls=args.n_controls,
            shift_threshold=args.shift_threshold,
            n_random=args.n_random,
            rng=rng,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    selected = shifted + controls + random_picks
    if not selected:
        print("No queries selected — check artifact and thresholds.", file=sys.stderr)
        return 1

    off = _per_query_by_id(artifact["faithfulness_thinking_off"])
    on = _per_query_by_id(artifact["faithfulness_thinking_on"])

    seed_note = f"; seed = {args.seed}" if args.n_random > 0 and args.seed is not None else ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "# Faithfulness ground-truth labels\n",
        f"\nSource artifact: `{args.artifact}`\n",
        f"Selected {len(selected)} queries "
        f"({len(shifted)} shifted + {len(controls)} controls + "
        f"{len(random_picks)} random; "
        f"shift threshold = {args.shift_threshold:.2f}{seed_note}).\n",
        "\n",
        "## How to label\n",
        "\n",
        "For each query below, fill in the YAML block under "
        '"Your labels". The scoring script compares your '
        "`faithfulness_score` against both judge passes (off / on) "
        "to decide whether thinking-on aligns better with ground "
        "truth than thinking-off. See "
        "`docs/rag/faithfulness-thinking-calibration.md` for the "
        "decision matrix this feeds.\n",
        "\n",
        "---\n",
        "\n",
    ]
    for qid in selected:
        parts.append(_format_query_block(qid, off[qid], on[qid]))

    args.out.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote labeling kit: {args.out} ({len(selected)} queries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
