"""Score the FaithfulnessJudge (thinking off / on) against hand labels.

Reads a labeled markdown file produced by
``build_calibration_labeling_kit.py`` and the original
``--compare-thinking --json`` artifact. For each query the human
labeled, compares the human's ``faithfulness_score`` against the
judge's score on both passes and reports which pass aligned more
closely with the human verdict.

This is the empirical signal that gates the option-A vs option-B
decision in ``docs/rag/faithfulness-thinking-calibration.md``.

Usage::

    python scripts/score_against_ground_truth.py \\
        --labels artifacts/calibration/ground-truth-2026-05-15.md \\
        --artifact artifacts/calibration/thinking-2026-05-15.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_YAML_BLOCK_RE = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)

# Default tie threshold per design.md acceptance rubric. The label,
# off, and on must all sit within this distance for a "tie."
_RUBRIC_TIE_THRESHOLD = 0.025


def _extract_labels(markdown: str) -> list[dict[str, Any]]:
    """Pull every ```yaml ...``` block whose YAML carries an ``id`` field.

    The kit emits one labeling block per query, surrounded by other
    fenced code blocks. We only return blocks that parse as dicts
    with an ``id`` key — that distinguishes label blocks from any
    other YAML in the document.
    """
    labels: list[dict[str, Any]] = []
    for match in _YAML_BLOCK_RE.finditer(markdown):
        raw = match.group(1)
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and "id" in data:
            labels.append(data)
    return labels


def _is_labeled(label: dict[str, Any]) -> bool:
    """True iff the label has a usable faithfulness_score (not the stub)."""
    raw = label.get("faithfulness_score")
    if isinstance(raw, int | float) and 0.0 <= float(raw) <= 1.0:
        return True
    return False


def _per_query_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {q["id"]: q for q in report.get("per_query", [])}


def _classify(
    label_score: float,
    off_score: float,
    on_score: float,
    tied_window: float,
) -> str:
    """Legacy tie rule: tied if the two judge-to-label distances differ
    by less than ``tied_window``. Kept for backward compatibility with
    v1 / v2 runs that documented this rule.
    """
    delta_off = abs(off_score - label_score)
    delta_on = abs(on_score - label_score)
    if abs(delta_off - delta_on) < tied_window:
        return "tied"
    return "off" if delta_off < delta_on else "on"


def _classify_rubric(
    label_score: float,
    off_score: float,
    on_score: float,
    threshold: float = _RUBRIC_TIE_THRESHOLD,
) -> str:
    """Design.md rubric tie rule.

    Tied iff |off - on| < threshold AND |off - label| < threshold AND
    |on - label| < threshold. Otherwise the closer-to-label wins.
    """
    delta_off = abs(off_score - label_score)
    delta_on = abs(on_score - label_score)
    judges_agree = abs(off_score - on_score) < threshold
    if judges_agree and delta_off < threshold and delta_on < threshold:
        return "tied"
    return "off" if delta_off < delta_on else "on"


def _bootstrap_margin_ci(
    verdicts: list[str],
    iters: int,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap a 95 % CI on (wins_off − wins_on).

    Args:
        verdicts: per-query closer labels in ("off", "on", "tied").
        iters: number of resamples (B).
        seed: PRNG seed for reproducibility.

    Returns:
        (point_estimate, ci_low, ci_high). Quantiles are 2.5 % and 97.5 %.
    """
    n = len(verdicts)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    margins: list[int] = []
    indices = list(range(n))
    for _ in range(iters):
        sample = [verdicts[rng.choice(indices)] for _ in range(n)]
        wins_off = sample.count("off")
        wins_on = sample.count("on")
        margins.append(wins_off - wins_on)
    margins.sort()
    lo_idx = int(0.025 * iters)
    hi_idx = int(0.975 * iters)
    point = verdicts.count("off") - verdicts.count("on")
    return (float(point), float(margins[lo_idx]), float(margins[hi_idx]))


_WORD_RE = re.compile(r"[A-Za-z0-9_\-/]+")


def _content_words(text: str, min_len: int = 4) -> set[str]:
    """Tokenize ``text`` into lowercase content words.

    Keeps backtick/slash/underscore-bearing identifiers (e.g.
    ``/smart-test``, ``bug_predict``) and ordinary words of
    length >= min_len. Drops short connectives ("the", "and",
    "to") that overlap trivially across any prose.
    """
    return {tok.lower() for tok in _WORD_RE.findall(text) if len(tok) >= min_len}


def _phantom_claim_rate(
    qids: list[str],
    on_records: dict[str, dict[str, Any]],
    *,
    overlap_threshold: float = 0.4,
) -> tuple[float, int, int]:
    """Fraction of on-judge "unsupported" claims that don't share
    enough content with the answer to be paraphrases.

    For each on-judge unsupported claim:
      1. Tokenize claim and answer into content words (length
         >= 4, preserving slash/underscore identifiers).
      2. Compute overlap = |claim_words ∩ answer_words| /
         |claim_words|.
      3. Phantom iff overlap < ``overlap_threshold``.

    This captures the v1/v2 phantom pattern: the judge invents
    a claim about entities (other tools, concepts) that simply
    aren't in the answer at all. A literal-substring match was
    tried first but produced 100 % phantom rates because the
    judge legitimately paraphrases everything it extracts — so
    no claim is ever a literal substring of the answer.

    Returns:
        (rate, phantom_count, total_unsupported). Rate is 0.0 if
        ``total_unsupported`` is zero.

    Note: this is a heuristic, not a guarantee. It catches the
    obvious phantom pattern (claim names a tool not in the
    answer) but won't catch a phantom where the judge reuses
    the answer's vocabulary to fabricate a relationship that
    isn't there. The rubric reads phantom rate as a *signal*,
    not an exact measure.
    """
    phantoms = 0
    total = 0
    for qid in qids:
        rec = on_records.get(qid, {})
        answer = rec.get("answer") or ""
        answer_words = _content_words(answer)
        for claim in rec.get("unsupported_claims", []) or []:
            total += 1
            claim_words = _content_words(str(claim))
            if not claim_words:
                # Claim has no content words to compare; treat as non-phantom.
                continue
            if not answer_words:
                continue
            overlap = len(claim_words & answer_words) / len(claim_words)
            if overlap < overlap_threshold:
                phantoms += 1
    rate = (phantoms / total) if total > 0 else 0.0
    return (rate, phantoms, total)


def _apply_rubric(
    *,
    wins_off: int,
    wins_on: int,
    ci_low: float,
    ci_high: float,
    phantom_rate: float,
    margin_stdev: float | None,
) -> tuple[str, str]:
    """Apply the 6-rule rubric from design.md.

    Returns (verdict_label, prose_explanation). The verdict_label
    is one of: "off-forever", "on-flip", "off-with-followup",
    "auto-thresholded-scoped", "inconclusive-escalate".
    """
    if margin_stdev is not None and margin_stdev > 0.10:
        return (
            "inconclusive-escalate",
            f"margin_stdev = {margin_stdev:.4f} exceeds 0.10 threshold. "
            f"Labeled-sample signal is below the noise floor; escalate "
            f"to n = 40 (full golden set) before locking the decision.",
        )

    ci_excludes_zero = ci_low > 0 or ci_high < 0

    if ci_excludes_zero and wins_off > wins_on:
        return (
            "off-forever",
            f"95 % CI on (wins_off − wins_on) = [{ci_low:.0f}, {ci_high:.0f}] "
            f"excludes 0 with off ahead. Decision: B forever — keep "
            f"--thinking opt-in. Ship at 0.1.18.",
        )

    if ci_excludes_zero and wins_on > wins_off and phantom_rate < 0.10:
        return (
            "on-flip",
            f"95 % CI = [{ci_low:.0f}, {ci_high:.0f}] excludes 0 with on "
            f"ahead, AND phantom-claim rate = {phantom_rate:.1%} < 10 %. "
            f"Decision: A — flip --thinking ON by default. "
            f"Re-baseline thresholds per Phase 1. Ship at 0.2.0.",
        )

    if wins_on > wins_off and phantom_rate >= 0.10:
        return (
            "off-with-followup",
            f"On apparent advantage tainted by phantom-claim rate "
            f"{phantom_rate:.1%} >= 10 %. Decision: B holds with a "
            f"follow-up — keep OFF, investigate phantom-claim fix "
            f"before re-evaluating. Ship at 0.1.18.",
        )

    # CI includes 0 (or wins are too close).
    return (
        "off-forever",
        f"95 % CI = [{ci_low:.0f}, {ci_high:.0f}] includes 0 (no "
        f"statistically distinguishable advantage). Phantom-claim "
        f"rate = {phantom_rate:.1%}. Decision: B forever — keep "
        f"OFF; on shows no aggregate benefit at this sample size. "
        f"Auto-thresholded routing scoping deferred. Ship at 0.1.18.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare hand labels vs. judge verdicts (off/on).")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--tied-window",
        type=float,
        default=0.02,
        help="If |Δ_off − Δ_on| < this, count as tied (default 0.02). "
        "Legacy rule, retained for v1/v2 continuity.",
    )
    parser.add_argument(
        "--rubric-rule",
        choices=("legacy", "design"),
        default="legacy",
        help=(
            "Tie rule for the main table. 'legacy' = current script "
            "(distance-diff < tied-window). 'design' = design.md "
            "rubric (|off-on|, |off-label|, |on-label| all < 0.025). "
            "The bootstrap CI block at the end always uses 'design'."
        ),
    )
    parser.add_argument(
        "--control-ids",
        type=str,
        default="",
        help=(
            "Comma-separated query IDs to exclude from rubric "
            "numerator/denominator (e.g. drift-check controls). "
            "Still scored and printed in the table."
        ),
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10000,
        help="Bootstrap resamples for the 95 %% CI on (wins_off-wins_on).",
    )
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed for the bootstrap.")
    parser.add_argument(
        "--variance",
        type=Path,
        default=None,
        help=(
            "Optional path to a variance JSON from "
            "scripts/measure_judge_variance.py. When provided, "
            "the rubric's 'escalate if margin_stdev > 0.10' "
            "check is wired up."
        ),
    )
    args = parser.parse_args(argv)

    if not args.labels.is_file():
        print(f"Labels file not found: {args.labels}", file=sys.stderr)
        return 2
    if not args.artifact.is_file():
        print(f"Artifact not found: {args.artifact}", file=sys.stderr)
        return 2

    markdown = args.labels.read_text(encoding="utf-8")
    labels = _extract_labels(markdown)
    labeled = [lbl for lbl in labels if _is_labeled(lbl)]
    if not labeled:
        print(
            f"No usable labels found in {args.labels}. "
            f"({len(labels)} blocks parsed; none have a numeric "
            f"`faithfulness_score`.)",
            file=sys.stderr,
        )
        return 1

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    off = _per_query_by_id(artifact["faithfulness_thinking_off"])
    on = _per_query_by_id(artifact["faithfulness_thinking_on"])

    control_ids = {qid.strip() for qid in args.control_ids.split(",") if qid.strip()}

    rows: list[tuple[str, float, float, float, str, bool]] = []
    counts = {"off": 0, "on": 0, "tied": 0}
    skipped: list[str] = []
    for lbl in labeled:
        qid = lbl["id"]
        if qid not in off or qid not in on:
            skipped.append(qid)
            continue
        label_score = float(lbl["faithfulness_score"])
        off_score = off[qid]["score"]
        on_score = on[qid]["score"]
        if args.rubric_rule == "design":
            verdict = _classify_rubric(label_score, off_score, on_score)
        else:
            verdict = _classify(label_score, off_score, on_score, args.tied_window)
        is_control = qid in control_ids
        if not is_control:
            counts[verdict] += 1
        rows.append((qid, label_score, off_score, on_score, verdict, is_control))

    # Output.
    print(f"Labels parsed: {len(labels)}  Usable: {len(labeled)}  Compared: {len(rows)}")
    if skipped:
        print(f"Skipped (not in artifact): {', '.join(skipped)}")
    if control_ids:
        print(f"Controls (excluded from rubric): {', '.join(sorted(control_ids))}")
    print()
    print(
        f"{'id':<14}  {'label':>7}  {'off':>7}  {'on':>7}  {'Δoff':>7}  " f"{'Δon':>7}  closer  tag"
    )
    print("-" * 78)
    for qid, label, off_s, on_s, verdict, is_control in rows:
        d_off = abs(off_s - label)
        d_on = abs(on_s - label)
        tag = "ctrl" if is_control else ""
        print(
            f"{qid:<14}  {label:>7.3f}  {off_s:>7.3f}  {on_s:>7.3f}  "
            f"{d_off:>7.3f}  {d_on:>7.3f}  {verdict:<6}  {tag}"
        )

    rubric_n = sum(counts.values())
    denom = rubric_n or 1
    print()
    print(
        f"Aggregate alignment with human labels "
        f"({args.rubric_rule} tie rule; rubric n = {rubric_n}, "
        f"controls excluded):"
    )
    for key in ("off", "on", "tied"):
        print(f"  {key + '-closer':<14}  {counts[key]:>3}  ({counts[key] / denom:.0%})")

    # ---------------------------------------------------------------------
    # Rubric block: bootstrap CI + phantom-claim rate + verdict
    # ---------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Rubric verdict (design.md acceptance rubric, controls excluded)")
    print("=" * 78)

    # Always recompute under the design.md rule for the rubric, regardless
    # of --rubric-rule (which only controls the main table presentation).
    rubric_verdicts = [
        _classify_rubric(label, off_s, on_s)
        for qid, label, off_s, on_s, _v, is_control in rows
        if not is_control
    ]
    rubric_off = rubric_verdicts.count("off")
    rubric_on = rubric_verdicts.count("on")
    rubric_tied = rubric_verdicts.count("tied")
    print(
        f"\nDesign-rule counts: off = {rubric_off}, on = {rubric_on}, "
        f"tied = {rubric_tied} (of {len(rubric_verdicts)})"
    )

    point, ci_low, ci_high = _bootstrap_margin_ci(
        rubric_verdicts, args.bootstrap_iters, seed=args.seed
    )
    print(
        f"Bootstrap CI on (wins_off - wins_on), B = {args.bootstrap_iters}: "
        f"point = {point:+.0f}, 95 % CI = [{ci_low:+.0f}, {ci_high:+.0f}]"
    )

    rubric_qids = [qid for qid, *_, is_control in rows if not is_control]
    phantom_rate, phantoms, n_unsup = _phantom_claim_rate(rubric_qids, on)
    print(
        f"Phantom-claim rate (rubric queries only): {phantom_rate:.1%} "
        f"({phantoms} / {n_unsup} on-judge unsupported claims)"
    )

    margin_stdev: float | None = None
    if args.variance is not None:
        if args.variance.is_file():
            try:
                v = json.loads(args.variance.read_text(encoding="utf-8"))
                margin_stdev = float(v.get("aggregate", {}).get("margin_stdev"))
                print(
                    f"margin_stdev (from --variance): {margin_stdev:.4f} "
                    f"(threshold 0.10 escalates to n=40)"
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"WARNING: failed to read --variance: {exc}")
        else:
            print(f"WARNING: --variance file not found: {args.variance}")
    else:
        print("margin_stdev: (not supplied; pass --variance variance.json to wire up)")

    verdict_label, prose = _apply_rubric(
        wins_off=rubric_off,
        wins_on=rubric_on,
        ci_low=ci_low,
        ci_high=ci_high,
        phantom_rate=phantom_rate,
        margin_stdev=margin_stdev,
    )
    print(f"\nVerdict: {verdict_label}")
    print(f"  {prose}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
