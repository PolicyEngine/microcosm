#!/usr/bin/env bash
set -euo pipefail

npm ci --ignore-scripts --prefix tools/orrery-contract

files=()
while IFS= read -r file; do files+=("$file"); done < <(
  uv run --no-sync python tools/ci_test_plan.py list-job engine-free
)

/usr/bin/time -v uv run --no-sync pytest "${files[@]}" \
  -n 2 --dist loadfile -v --tb=short --maxfail=1 \
  --durations=25 -p no:cacheprovider
