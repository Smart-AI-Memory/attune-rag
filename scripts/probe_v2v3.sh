#!/usr/bin/env bash
# Combined V2 + V3 probe runner.
#
# Usage:
#   source ~/.attune/anthropic.env   # loads ANTHROPIC_API_KEY
#   bash ~/attune-rag/.claude/worktrees/native-citations-v2v3/scripts/probe_v2v3.sh
#
# Runs both V2 (cache_control) and V3 (doc-count ceiling) probes
# back-to-back and prints all output to stdout. Single command, no
# multi-line paste required.

set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "error: ANTHROPIC_API_KEY not set in this shell." >&2
    echo "       run:  source ~/.attune/anthropic.env"     >&2
    exit 2
fi

echo "ANTHROPIC_API_KEY loaded: ${ANTHROPIC_API_KEY:0:10}***"
echo

ROOT="$HOME/attune-rag/.claude/worktrees/native-citations-v2v3"
PY="$HOME/attune-rag/.venv/bin/python"

cd "$ROOT"

echo "=========================================="
echo " V2: cache_control on document blocks"
echo "=========================================="
PYTHONPATH=src "$PY" scripts/probe_v2_cache_control.py
v2_rc=$?

echo
echo "=========================================="
echo " V3: per-request document-count ceiling"
echo "=========================================="
PYTHONPATH=src "$PY" scripts/probe_v3_doc_count_ceiling.py
v3_rc=$?

echo
echo "=========================================="
echo " summary: v2_rc=$v2_rc  v3_rc=$v3_rc"
echo "=========================================="
