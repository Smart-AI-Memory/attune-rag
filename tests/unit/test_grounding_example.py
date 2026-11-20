"""Execute the documented offline journey verbatim."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_documented_grounding_example_uses_only_base_features(tmp_path):
    example = Path(__file__).resolve().parents[2] / "examples/grounding_evidence.py"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "CLAUDECODE"}
    }
    result = subprocess.run(
        [sys.executable, "-I", str(example)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["strict"]["answer"] == "The provided context does not cover this question."
    assert report["strict"]["abstained"] and report["strict"]["context"] == ""
    assert report["fallback"]["fallback_used"] and not report["fallback"]["abstained"]
    assert report["stub_calls"] == {"plain": 1, "native": 1}
    document = report["grounded"]["generation_documents"][0]
    assert document["text"] in report["grounded"]["context"]
    assert report["fixed_report_fixture"]["summary"]["unsupported_answer_rate"] == 1.0
    assert report["fixed_report_fixture"]["summary"]["observed_mean_per_answer_faithfulness"] == 0.5
