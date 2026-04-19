"""Evaluation harness: faithfulness judging and prompt A/B.

Experiment 1 — faithfulness metric
    :class:`FaithfulnessJudge` scores how well an answer is
    grounded in retrieved passages. Uses Claude with forced
    tool-use for reliable structured output.

Experiment 2 — prompt surgery
    :mod:`attune_rag.eval.bench_prompts` runs every prompt
    variant in :data:`attune_rag.prompts.PROMPT_VARIANTS`
    against the golden query set, generates answers, and
    scores each with the judge. Prints a comparison table.

Both experiments require the ``[claude]`` extra.
"""

from __future__ import annotations

from .faithfulness import FaithfulnessJudge, FaithfulnessResult

__all__ = [
    "FaithfulnessJudge",
    "FaithfulnessResult",
]
