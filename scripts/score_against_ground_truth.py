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
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_YAML_BLOCK_RE = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)


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
    if isinstance(raw, (int, float)) and 0.0 <= float(raw) <= 1.0:
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
    delta_off = abs(off_score - label_score)
    delta_on = abs(on_score - label_score)
    if abs(delta_off - delta_on) < tied_window:
        return "tied"
    return "off" if delta_off < delta_on else "on"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare hand labels vs. judge verdicts (off/on).")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--tied-window",
        type=float,
        default=0.02,
        help="If |Δ_off − Δ_on| < this, count as tied (default 0.02).",
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

    rows: list[tuple[str, float, float, float, str]] = []
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
        verdict = _classify(label_score, off_score, on_score, args.tied_window)
        counts[verdict] += 1
        rows.append((qid, label_score, off_score, on_score, verdict))

    # Output.
    print(f"Labels parsed: {len(labels)}  Usable: {len(labeled)}  Compared: {len(rows)}")
    if skipped:
        print(f"Skipped (not in artifact): {', '.join(skipped)}")
    print()
    print(f"{'id':<14}  {'label':>7}  {'off':>7}  {'on':>7}  {'Δoff':>7}  {'Δon':>7}  closer")
    print("-" * 70)
    for qid, label, off_s, on_s, verdict in rows:
        d_off = abs(off_s - label)
        d_on = abs(on_s - label)
        print(
            f"{qid:<14}  {label:>7.3f}  {off_s:>7.3f}  {on_s:>7.3f}  "
            f"{d_off:>7.3f}  {d_on:>7.3f}  {verdict}"
        )

    total = sum(counts.values()) or 1
    print()
    print("Aggregate alignment with human labels:")
    for key in ("off", "on", "tied"):
        print(f"  {key + '-closer':<14}  {counts[key]:>3}  ({counts[key] / total:.0%})")

    # Decision hint per the matrix in the calibration doc.
    print()
    if counts["on"] > counts["off"]:
        diff = counts["on"] - counts["off"]
        print(
            f"Signal: thinking-on aligns better on {diff} more "
            f"queries than thinking-off. Suggests Option A "
            f"(make --thinking the default) is supported if this "
            f"holds on a larger sample."
        )
    elif counts["off"] > counts["on"]:
        diff = counts["off"] - counts["on"]
        print(
            f"Signal: thinking-off aligns better on {diff} more "
            f"queries than thinking-on. Suggests Option B (keep "
            f"--thinking opt-in) or possibly Option C (retire) "
            f"depending on cost / signal ratio."
        )
    else:
        print(
            "Signal: tied. Pre-committed matrix points to Option B "
            "(keep opt-in) when on doesn't beat off."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
