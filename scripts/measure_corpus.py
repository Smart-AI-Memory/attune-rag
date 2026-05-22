"""Backward-compat shim for ``python -m attune_rag.measure_corpus``.

The measurement harness lives at
:mod:`attune_rag.measure_corpus` (promoted from this script per
user-corpus-onboarding M1). This script remains as a thin
backward-compat entry point so existing invocations like
``python scripts/measure_corpus.py ...`` keep working and so the
bundled-corpus golden snapshot test
(:mod:`tests.golden.test_measure_corpus_bundled`) doesn't churn.

New code should prefer one of:

- ``python -m attune_rag.measure_corpus ...``
- ``attune-rag-measure ...`` (``[project.scripts]`` entry)
- ``from attune_rag.measure_corpus import measure`` (Python API)
"""

from __future__ import annotations

import sys

from attune_rag.measure_corpus import main

if __name__ == "__main__":
    sys.exit(main())
