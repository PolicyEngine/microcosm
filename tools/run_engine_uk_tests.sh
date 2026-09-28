#!/usr/bin/env bash
set -euo pipefail

files=()
while IFS= read -r file; do files+=("$file"); done < <(
  uv run --no-sync python tools/ci_test_groups.py --list engine-uk
)

# Keep country-engine execution serial so each process stays within its
# configured memory limit.
/usr/bin/time -v uv run --no-sync pytest "${files[@]}" \
  -v --tb=short --maxfail=1 --durations=25 -p no:cacheprovider
