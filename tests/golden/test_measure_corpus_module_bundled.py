"""Byte-identical R1 self-test for the module CLI (M1.5).

``python -m attune_rag.measure_corpus`` against the bundled
``AttuneHelpCorpus`` + bundled queries must produce a report
byte-identical to ``measure_corpus_bundled.golden.md``.

Companion to ``test_measure_corpus_bundled.py`` which tests the
backward-compat ``scripts/measure_corpus.py`` shim. This test
pins the module entry point directly so a regression that breaks
only the module (not the shim) gets caught.

Skipped when the ``[attune-help]`` extra is not installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("attune_help")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = Path(__file__).parent / "measure_corpus_bundled.golden.md"
_FROZEN_TIMESTAMP = "2026-05-22T00:00:00Z"


def test_module_bundled_byte_identical(tmp_path: Path) -> None:
    """``python -m attune_rag.measure_corpus`` produces byte-identical
    output against the bundled corpus.

    R1 strict-dominance: pins the entire report — aggregate + per-query
    tables + footer. Any drift fails until the golden file is updated.
    """
    out = tmp_path / "actual.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "attune_rag.measure_corpus",
            "--corpus-bundled",
            "--queries",
            "tests/golden/queries.yaml",
            "--paraphrased",
            "tests/golden/queries_paraphrased.yaml",
            "--frozen-timestamp",
            _FROZEN_TIMESTAMP,
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr

    actual = out.read_bytes()
    golden = _GOLDEN.read_bytes()
    if actual != golden:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                golden.decode("utf-8").splitlines(),
                actual.decode("utf-8").splitlines(),
                fromfile=_GOLDEN.name,
                tofile="actual",
                lineterm="",
            )
        )
        pytest.fail(
            f"attune_rag.measure_corpus output drifted from golden snapshot.\n"
            f"If this drift is intentional, update {_GOLDEN.name} in this PR.\n\n"
            f"{diff}"
        )


def test_module_bundled_r1_numbers() -> None:
    """The Python API (not just the CLI) reproduces the locked numbers.

    Spec correction note: ``tasks.md`` M1.5 originally cited
    ``P@1=1.00, R@3=1.00, paraphrased P@1=0.9125, paraphrased R@3=1.00``.
    Those were the rerank-*on* aspirational numbers; D5's verdict
    landed rerank-*off* as the harness default. The actual rerank-off
    numbers (verified against ``tests/golden/measure_corpus_bundled.golden.md``)
    are ``0.9000`` and ``0.9875`` for paraphrased.
    """
    from attune_rag.measure_corpus import measure

    result = measure(
        bundled=True,
        queries_path=_REPO_ROOT / "tests" / "golden" / "queries.yaml",
        paraphrased_path=_REPO_ROOT / "tests" / "golden" / "queries_paraphrased.yaml",
    )
    assert result.p1 == 1.0
    assert result.r3 == 1.0
    assert result.paraphrased_p1 == pytest.approx(0.9)
    assert result.paraphrased_r3 == pytest.approx(0.9875)
    assert result.n == 40
    assert result.paraphrased_n == 80
    assert result.rerank is False
