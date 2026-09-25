#!/usr/bin/env bash

set -euo pipefail

uv sync --all-packages --locked --extra uk
files=()
while IFS= read -r file; do files+=("$file"); done < <(
  uv run --no-sync python tools/ci_test_groups.py --list integration-uk
)
uv run --no-sync pytest "${files[@]}" \
  --run-integration -q -s -p no:cacheprovider

if [[ -z "${HF_STAGING_READ_TOKEN:-}" ]]; then
  echo \
    "HF_STAGING_READ_TOKEN is unavailable; skipping the optional private repository access check."
  exit 0
fi

HF_TOKEN="$HF_STAGING_READ_TOKEN" \
  uv run --no-sync python tools/provision_uk_staging_repository.py --verify-access
