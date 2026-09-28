#!/usr/bin/env bash
set -euo pipefail

: "${RUN_US:?RUN_US must be set}"
: "${RUN_UK:?RUN_UK must be set}"

groups=(engine-free-shared)
if [[ "$RUN_US" == "true" ]]; then groups+=(engine-free-us); fi
if [[ "$RUN_UK" == "true" ]]; then groups+=(engine-free-uk); fi

files=()
for group in "${groups[@]}"; do
  while IFS= read -r file; do files+=("$file"); done < <(
    uv run --no-sync python tools/ci_test_groups.py --list "$group"
  )
done

/usr/bin/time -v uv run --no-sync pytest "${files[@]}" \
  -n 2 --dist loadfile -v --tb=short --maxfail=1 \
  --durations=25 -p no:cacheprovider
