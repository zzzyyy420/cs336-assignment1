#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results
uv run pytest --timeout 10 -q ./tests --junitxml=results/test_results.xml
uv run python scripts/package_submission.py
echo "Source archive: submission/code.zip"
