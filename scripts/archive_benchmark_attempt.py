"""Archive an attempt from the fixed public/synthetic CI benchmark invocation.

Called only by benchmark.yml; it has no user-corpus upload or network behavior.
Keep its diagnostic even when the benchmark could not produce a report.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    directory = args.output / f"attempt-{args.attempt}"
    directory.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "attempt": args.attempt,
        "benchmark_exit": args.exit_code,
        "report_present": args.dump.is_file(),
        "valid_json_object": False,
        "inputs": "CI bundled public/synthetic only",
    }
    if args.dump.is_file():
        shutil.copyfile(args.dump, directory / "benchmark.json")
        try:
            report = json.loads(args.dump.read_text(encoding="utf-8"))
            diagnostic["valid_json_object"] = isinstance(report, dict)
            if isinstance(report, dict):
                diagnostic["outcomes"] = report.get("outcomes")
        except (OSError, UnicodeError, ValueError):
            pass
    (directory / "attempt.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
