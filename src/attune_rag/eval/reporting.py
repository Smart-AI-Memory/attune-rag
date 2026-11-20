"""Offline, additive benchmark evidence. Partial observations are never gate results."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 1
CONTEXT_REFUSAL = "The provided context does not cover this question."


def run_manifest(queries_path: Path) -> dict[str, Any]:
    """Capture explicit provenance without credentials or arbitrary environment dumps."""
    code_sha = None
    tracked_changes = None
    source_root = Path(__file__).resolve().parents[3]
    source_module = source_root / "src/attune_rag/eval/reporting.py"
    # A wheel under another application's .venv must not inherit its Git HEAD.
    if source_module.resolve() == Path(__file__).resolve() and (source_root / ".git").exists():
        try:
            top = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=source_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
            if Path(top).resolve() == source_root:
                # Layout alone is insufficient: the executing module must belong
                # to the revision being reported, not be an untracked local copy.
                subprocess.run(
                    ["git", "cat-file", "-e", "HEAD:src/attune_rag/eval/reporting.py"],
                    cwd=source_root,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                code_sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source_root,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                ).stdout.strip()
                tracked_changes = bool(
                    subprocess.run(
                        ["git", "status", "--porcelain", "--untracked-files=no"],
                        cwd=source_root,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    ).stdout.strip()
                )
        except (OSError, subprocess.SubprocessError):
            code_sha = None
            tracked_changes = None
    query_sha = (
        hashlib.sha256(queries_path.read_bytes()).hexdigest() if queries_path.is_file() else None
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "code_sha": code_sha,
        "tracked_changes": tracked_changes,
        "code_sha_status": (
            "source_checkout_head" if code_sha else "unavailable_or_unverified_source_revision"
        ),
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "queries_sha256": query_sha,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "benchmark_attempt": os.environ.get("ATTUNE_BENCH_ATTEMPT"),
        "dependencies": sorted(
            ({"name": d.metadata["Name"], "version": d.version} for d in distributions()),
            key=lambda d: (d["name"] or "", d["version"]),
        ),
        "unavailable_values": "null means not exposed or not running in GitHub Actions",
    }


def corpus_provenance(corpus: Any) -> dict[str, Any]:
    """Use existing body fingerprints; do not invent a complete corpus identity."""
    from ..corpus.attune_help import AttuneHelpCorpus
    from ..corpus.directory import DirectoryCorpus

    directory = corpus._inner if isinstance(corpus, AttuneHelpCorpus) else corpus
    fingerprint = directory.version if isinstance(directory, DirectoryCorpus) else None
    return {
        "name": getattr(corpus, "name", None),
        "version": getattr(corpus, "version", None),
        "content_fingerprint": fingerprint,
        "fingerprint_status": (
            "available" if fingerprint is not None else "not_exposed_by_corpus_interface"
        ),
        "fingerprint_scope": (
            "SHA-256 prefix16 of loaded paths and bodies; excludes aliases and summaries"
            if fingerprint is not None
            else None
        ),
    }


def summarize_answers(rows: list[dict[str, Any]], requested: int) -> dict[str, Any]:
    """Separate observed claim ratios from unsupported answers and non-answers.

    Refusal appropriateness/usefulness cannot be inferred from a zero-claim judge
    result. Exact refusal text is an observable category, not that semantic verdict.
    Missing verdicts are excluded from scores and retain an explicit denominator.
    """
    scored = [r for r in rows if r["status"] == "completed"]
    for row in scored:
        supported, unsupported = row["supported_claims"], row["unsupported_claims"]
        if not all(
            isinstance(claims, list) and all(isinstance(c, str) for c in claims)
            for claims in (supported, unsupported)
        ):
            raise ValueError("Invalid claim lists in evaluation evidence")
        total = len(supported) + len(unsupported)
        expected = len(supported) / total if total else 1.0
        score = row["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
            or not math.isclose(score, expected)
        ):
            raise ValueError("Invalid claim ratio in evaluation evidence")
    generated = [r for r in rows if "answer" in r]
    unsupported_answers = sum(bool(r["unsupported_claims"]) for r in scored)
    zero_claim = [r for r in scored if not r["supported_claims"] and not r["unsupported_claims"]]
    n = len(scored)
    return {
        "complete": n == requested and len(rows) == requested,
        "requested_answers": requested,
        "generated_answers": len(generated),
        "scored_answers": n,
        "unscored_answers": requested - n,
        "observed_mean_per_answer_faithfulness": sum(r["score"] for r in scored) / n if n else None,
        "answers_with_unsupported_claims": unsupported_answers,
        "unsupported_answer_denominator": n,
        "unsupported_answer_rate": unsupported_answers / n if n else None,
        "supported_claims": sum(len(r["supported_claims"]) for r in scored),
        "unsupported_claims": sum(len(r["unsupported_claims"]) for r in scored),
        "zero_claim_answers": len(zero_claim),
        "zero_claim_denominator": n,
        "pipeline_abstentions": sum(r.get("abstained") is True for r in generated),
        "pipeline_abstention_denominator": len(generated),
        "exact_context_refusals": sum(r["answer"].strip() == CONTEXT_REFUSAL for r in generated),
        "empty_answers": sum(not r["answer"].strip() for r in generated),
        "other_zero_claim_answers": sum(
            bool(r["answer"].strip()) and r["answer"].strip() != CONTEXT_REFUSAL for r in zero_claim
        ),
        "refusal_appropriateness": "not_assessed",
        "answer_usefulness": "not_assessed",
    }
