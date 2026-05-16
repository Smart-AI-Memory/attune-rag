#!/usr/bin/env bash
# Smoke-test the quality-gate plumbing without spending API tokens.
#
# Verifies that:
#   - A synthetic "good" dump → check_thresholds.py exits 0.
#   - A synthetic "bad"  dump → exits 1 and writes a PR comment
#     body containing the failing metric and the gate's marker.
#   - Validation errors on a malformed dump → exits 2 and no
#     comment body is written.
#
# Runs in CI as part of .github/workflows/benchmark.yml so the
# gate's own logic stays exercised on every PR even when the real
# benchmark passes (or is skipped on cost grounds).
#
# Also runnable locally:
#
#     bash scripts/smoke_check_gate.sh
#
# Exits 0 on success, non-zero on any assertion failure.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
THRESHOLDS="${ROOT}/docs/specs/release-quality-baseline/thresholds.json"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

GOOD="${WORK}/dump-good.json"
BAD="${WORK}/dump-bad.json"
BROKEN="${WORK}/dump-broken.json"
COMMENT="${WORK}/comment.md"

# Read the SHA so the dump's queries_path verifies; pull it out of
# thresholds.json to avoid duplicating the constant.
QUERIES_PATH="${ROOT}/tests/golden/queries.yaml"

# Good dump: metrics safely above thresholds.
python - "$GOOD" "$QUERIES_PATH" <<'PY'
import json, sys
out, queries = sys.argv[1:]
json.dump({
    "retrieval": {"precision_at_1": 0.95, "recall_at_k": 1.0, "k": 3},
    "faithfulness_legacy": {"mean_faithfulness": 0.985},
    "queries_path": queries,
}, open(out, "w"))
PY

# Bad dump: faithfulness well below threshold (0.85 < 0.9686).
python - "$BAD" "$QUERIES_PATH" <<'PY'
import json, sys
out, queries = sys.argv[1:]
json.dump({
    "retrieval": {"precision_at_1": 0.95, "recall_at_k": 1.0, "k": 3},
    "faithfulness_legacy": {"mean_faithfulness": 0.85},
    "queries_path": queries,
}, open(out, "w"))
PY

# Broken dump: missing precision_at_1.
python - "$BROKEN" "$QUERIES_PATH" <<'PY'
import json, sys
out, queries = sys.argv[1:]
json.dump({
    "retrieval": {"recall_at_k": 1.0, "k": 3},
    "faithfulness_legacy": {"mean_faithfulness": 0.985},
    "queries_path": queries,
}, open(out, "w"))
PY

assert_exit() {
    local expected="$1"; shift
    local label="$1"; shift
    set +e
    "$@"
    local rc=$?
    set -e
    if [ "$rc" -ne "$expected" ]; then
        echo "FAIL [$label]: expected exit $expected, got $rc"
        exit 1
    fi
    echo "OK   [$label]: exit $expected"
}

echo "== Good dump should pass =="
assert_exit 0 "good→0" \
    python "${ROOT}/scripts/check_thresholds.py" \
        --dump "$GOOD" --thresholds "$THRESHOLDS"

echo "== Bad dump should fail (exit 1) and write a comment =="
rm -f "$COMMENT"
assert_exit 1 "bad→1" \
    python "${ROOT}/scripts/check_thresholds.py" \
        --dump "$BAD" --thresholds "$THRESHOLDS" \
        --comment-out "$COMMENT"
if [ ! -s "$COMMENT" ]; then
    echo "FAIL: comment file was not written (or empty)"
    exit 1
fi
grep -q '<!-- attune-rag-quality-gate -->' "$COMMENT" || {
    echo "FAIL: comment missing marker"; exit 1; }
grep -q 'mean_faithfulness' "$COMMENT" || {
    echo "FAIL: comment missing failing metric"; exit 1; }
echo "OK   [bad→comment]: marker + failing metric present"

echo "== Broken dump should be a validation error (exit 2) =="
rm -f "$COMMENT"
assert_exit 2 "broken→2" \
    python "${ROOT}/scripts/check_thresholds.py" \
        --dump "$BROKEN" --thresholds "$THRESHOLDS" \
        --comment-out "$COMMENT"
if [ -e "$COMMENT" ]; then
    echo "FAIL: comment file written on validation error (should not be)"
    exit 1
fi
echo "OK   [broken→no-comment]: validation errors do not stack PR comments"

echo
echo "smoke_check_gate.sh: all assertions passed"
