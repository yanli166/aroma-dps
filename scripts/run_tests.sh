#!/usr/bin/env bash
# Run all regression tests
# Usage: bash scripts/run_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export PYTHONPATH="src:p0_verification:$PYTHONPATH"

python -m pytest tests/test_regression.py -v --tb=short
