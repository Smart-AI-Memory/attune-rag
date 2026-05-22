"""Byte-identical golden-snapshot test for scripts/measure_corpus.py.

Runs the script against the bundled AttuneHelpCorpus + bundled
queries, asserts the output equals the committed golden file at
``tests/golden/measure_corpus_bundled.golden.md``.

This is the strict-dominance regression net: any change to the
corpus, the retriever, the scoring logic, or the report shape
fails this test until the golden file is updated in the same PR.

Skipped when the ``[attune-help]`` extra is not installed (same
guard as ``test_golden.py``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("attune_help")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "measure_corpus.py"
_GOLDEN = Path(__file__).parent / "measure_corpus_bundled.golden.md"
_QUERIES_REL = "tests/golden/queries.yaml"
_PARAPHRASED_REL = "tests/golden/queries_paraphrased.yaml"
_FROZEN_TIMESTAMP = "2026-05-22T00:00:00Z"


def test_bundled_corpus_byte_identical(tmp_path: Path) -> None:
    """Script output against bundled corpus must byte-equal the golden file.

    The golden file pins the *entire* report — aggregate numbers,
    per-query table, alias section — so a regression that changes any
    fraction of any cell trips the test.

    Updating the golden file is intentional: the PR that legitimately
    moves the bundled-baseline (new alias work, retriever upgrade, etc.)
    must (a) update the golden in the same PR and (b) call out
    before/after numbers in the PR body. The diff is the contract.
    """
    out = tmp_path / "actual.md"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--corpus-bundled",
            "--queries",
            _QUERIES_REL,
            "--paraphrased",
            _PARAPHRASED_REL,
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
        # Surface a useful diff in the failure message.
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                golden.decode("utf-8").splitlines(),
                actual.decode("utf-8").splitlines(),
                fromfile=str(_GOLDEN.name),
                tofile="actual",
                lineterm="",
            )
        )
        pytest.fail(
            f"measure_corpus.py output drifted from golden snapshot.\n"
            f"If this drift is intentional (corpus / retriever / scoring "
            f"change), update {_GOLDEN.name} in this PR and call out the "
            f"before/after in the PR body.\n\n{diff}"
        )
